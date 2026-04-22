import csv
import shutil
import sys

import pandas as pd 

from activetigger.datamodels import ProjectBaseModel, ProjectModel
from activetigger.functions import slugify
from activetigger.tasks.base_task import BaseTask

csv.field_size_limit(sys.maxsize)


class CreateProject(BaseTask):
    """
    Create a new project
    """

    kind = "create_project"

    def __init__(
        self,
        project_slug: str,
        params: ProjectBaseModel,
        username: str,
        data_all: str = "data_all.parquet",
        train_file: str = "train.parquet",
        valid_file: str = "valid.parquet",
        test_file: str = "test.parquet",
        features_file: str = "features.parquet",
        random_seed: int = 42,
    ):
        super().__init__()
        self.random_seed = params.seed if hasattr(params, "seed") else random_seed
        self.project_slug = project_slug
        self.params = params
        self.username = username
        self.data_all = data_all
        self.train_file = train_file
        self.test_file = test_file
        self.valid_file = valid_file
        self.features_file = features_file
    
    def _build_train_set(self,dataset:pd.DataFrame)->tuple[pd.DataFrame | None, list]:
        """
        Build Train Set from a given datset based on a selected strategy:
        """
        train_size=0
        train_rows = []
        trainset=None
        if dataset.empty:
            raise Exception("Dataset is corrupted or empty")
        print(len(dataset),flush=True)   
        if 0<self.params.n_train <=len(dataset):
            train_size=self.params.n_train       
            match self.params.train_selection:
                case "sequential":
                    trainset=dataset.iloc[:train_size].copy()
                case "force_label":
                    if len(self.params.cols_label)>0:
                        l_cols=self.params.cols_label
                        f_notna=dataset[dataset[l_cols].notna().any(axis=1)]
                        f_na =dataset[dataset[l_cols].isna().all(axis=1)]
                        if len(f_notna) >= train_size:
                            trainset = f_notna.sample(train_size, random_state=self.random_seed)
                        else:
                            n_train_random = train_size -len(f_notna)
                            trainset = pd.concat(
                                [
                                    f_notna,
                                    f_na.sample(n_train_random, random_state=self.random_seed),
                                ]
                            )
                case "stratification":
                    if len(self.params.cols_stratify)>0:
                        df_grouped = dataset.groupby(self.params.cols_stratify, group_keys=False)
                        nb_cat = len(df_grouped)
                        nb_elements_cat = round(train_size / nb_cat)
                        sampled_idx = df_grouped.apply(
                            lambda x: x.sample(min(len(x), nb_elements_cat), random_state=self.random_seed)
                        ).index.get_level_values(-1)
                        trainset = dataset.loc[sampled_idx]
                        self.params.n_train=len(trainset)
                case "random":
                    trainset = dataset.sample(train_size, random_state=self.random_seed)
            #save to parquet
            if trainset is not None:
                train_rows=list(trainset.index)
            else:
                raise Exception("trainset is None")
        return trainset,train_rows 
    
    def _build_evalset(self,dataset:pd.DataFrame,train_rows:list=[])->tuple[pd.DataFrame | None, pd.DataFrame | None, list, list]:
        """
        build evaluation sets
        
        """
        n_draw=self.params.n_test+self.params.n_valid
        #eval_strategy=self.holdout_selection
        #holdout_pool is init as the full corpus
        holdout_pool=None
        #For overlap checks with train rows
        valid_rows=[]
        test_rows=[]
        validset=None
        testset=None
        if n_draw == 0 and self.params.holdout_selection is None:
            self.params.s_val_idx = None
            self.params.s_test_idx = None
            return None, None, [], []
        if n_draw == 0 and self.params.holdout_selection is not None:
            raise Exception("holdout_selection must be None when n_test=0 and n_valid=0")
        if n_draw > 0 and self.params.holdout_selection is None:
            raise Exception("holdout_selection required when n_test or n_valid > 0")
        if self.params.holdout_selection == "sequential":
            if self.params.s_val_idx is None and self.params.n_valid > 0 and self.params.train_selection != "sequential":
                raise Exception("s_val_idx required when train is not sequential")
            if self.params.s_test_idx is None and self.params.n_test > 0 and self.params.train_selection != "sequential":
                raise Exception("s_test_idx required when train is not sequential")
        if self.params.holdout_selection in ["stratification", "random"]:
            self.params.s_val_idx = None
            self.params.s_test_idx = None
        else:
            if self.params.s_val_idx is not None and self.params.n_valid == 0:
                raise ValueError("s_val_idx provided but n_valid=0")
            if self.params.s_test_idx is not None and self.params.n_test == 0:
                raise ValueError("s_test_idx provided but n_test=0")
        #Drop train
        if train_rows :
            if self.params.train_selection==self.params.holdout_selection=="sequential":
                holdout_pool=dataset
        holdout_pool=dataset.drop(index=train_rows)

        match self.params.holdout_selection:
            case"sequential":              
                #in case user has a specific start index for validation or test set(s)
                if self.params.s_val_idx is None :
                    start_val=self.params.n_train
                else:
                    start_val=self.params.s_val_idx              
                if self.params.s_test_idx is None:
                    start_test=self.params.n_train+self.params.n_valid  
                else:   
                    start_test=self.params.s_test_idx 
                # Building Validation set
                end_val=start_val+self.params.n_valid if self.params.n_valid>0 else None
                end_test=start_test+self.params.n_test if self.params.n_test>0 else None
                if end_val is None or end_test is None:
                    end = end_val or end_test
                    start = start_val if end_val is not None else start_test
                    if end > len(holdout_pool):
                        raise ValueError(f"eval set end {end} exceeds dataset length {len(holdout_pool)}")
                    if self.params.train_selection == "sequential" and start < self.params.n_train:
                        raise ValueError(f"eval set overlaps train [0:{self.params.n_train}]")
                else:
                    if end_val > len(holdout_pool) or end_test > len(holdout_pool):
                        raise ValueError(f"eval sets exceed dataset length {len(holdout_pool)}")
                    if self.params.train_selection == "sequential":
                        if start_val < self.params.n_train or start_test < self.params.n_train:
                            raise ValueError(f"eval sets overlap train [0:{self.params.n_train}]")
                    if start_val in range(start_test, end_test) or start_test in range(start_val, end_val):
                        raise ValueError(f"val [{start_val}:{end_val}] and test [{start_test}:{end_test}] overlap")
                if end_val  is not None: 
                    validset=holdout_pool.iloc[start_val:end_val]
                if end_test is not None:
                    testset=holdout_pool.iloc[start_test:end_test]
            case "stratification":
                if len(self.params.cols_stratify) == 0:
                    raise ValueError("Missing Columns for stratification , need to define a least 1 column")
                else:
                    df_grouped = holdout_pool.groupby(self.params.cols_stratify, group_keys=False)
                    nb_cat = len(df_grouped)
                    nb_elements_cat = round(n_draw/nb_cat)   
                    if nb_elements_cat == 0:
                        raise ValueError(f"number of elements to draw={n_draw} too small for {nb_cat} categories, increase test/valid size(s)")
                    sampled_idx = df_grouped.apply(
                        lambda x: x.sample(min(len(x), nb_elements_cat), random_state=self.random_seed)
                    ).index.get_level_values(-1)
                    draw = holdout_pool.loc[sampled_idx]
                    #handle split
                    if draw is not None and not draw.empty:
                        if self.params.n_test > 0 and self.params.n_valid == 0:
                            #case:test 
                            testset=draw
                            self.params.n_test=len(draw)
                            print(f"setting up the len as {self.params.n_test}",flush=True)
                        if self.params.n_valid > 0 and self.params.n_test == 0:
                            #case:val
                            validset=draw
                            self.params.n_valid=len(draw)
                            print(f"setting up the len as {self.params.n_valid}",flush=True)
                            #case:both 
                        else:
                            if self.params.n_valid > 0 and self.params.n_test > 0:
                                valid_prop=round(len(draw)*self.params.n_valid/n_draw)
                                validset=draw[:valid_prop]
                                testset=draw[valid_prop:]
                                self.params.n_test=len(testset)
                                self.params.n_valid=len(validset)
                                print(f"setting up the len as {self.params.n_valid}",flush=True)
                                print(f"setting up the len as {self.params.n_test}",flush=True)
            case "random":
                draw = holdout_pool.sample(n_draw, random_state=self.random_seed)
                testset = draw.sample(self.params.n_test, random_state=self.random_seed)
                validset = draw.drop(index=testset.index)
        if validset is not None and not validset.empty:
            valid_rows=list(validset.index)
        if testset is not None and not testset.empty:
            test_rows=list(testset.index)
        return validset,testset,test_rows,valid_rows
        
    def _write_to_parquet_(self, d_set: pd.DataFrame, path: str = "", name: str = "train")-> None:
        if d_set is not None and not d_set.empty:
            try:
                if name in ["test", "valid"]:
                    d_set.to_parquet(self.params.dir.joinpath(path), index=True)
                    setattr(self.params, name, True)
                if name == "train":
                    d_set[["id_external", "text"] + self.params.cols_context].to_parquet(
                        self.params.dir.joinpath(self.train_file), index=True
                    )
            except Exception as e:
                if name in ["test", "valid"]:
                    setattr(self.params, name, False)
                raise Exception(f"Failed to write {name} parquet: {e}")
        else:
            if name in ["test", "valid"]:
                setattr(self.params, name, False)
                                   
    def __call__(
        self,
    ) -> tuple[ProjectModel, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        """
        Create the project with the given name and file
        Define an internal and external index
        return the project model and the train/valid/test datasets to import in the database
        """
        print(f"Start queue project {self.project_slug} for {self.username}")
        # check if the directory already exists + file (should with the data)
        if self.params.dir is None or not self.params.dir.exists():
            print("The directory does not exist and should", self.params.dir)
            raise Exception("The directory does not exist and should")

        # Step 1 : load all data, rename columns and define index
        processed_corpus = False
        if self.params.filename is None or self.params.from_toy_dataset:
            processed_corpus = True

        if (self.params.filename is not None) and not self.params.from_toy_dataset:
            # if a file was uploaded
            # load the uploaded file
            file_path = self.params.dir.joinpath(self.params.filename)
        else:
            # if the file was copied from a toy dataset / existing project
            file_path = self.params.dir.joinpath(self.data_all)

        if not file_path.exists():
            raise Exception("File not found, problem when uploading")

        if str(file_path).endswith(".csv"):
            try:
                content = pd.read_csv(
                    file_path, sep=None, low_memory=False, on_bad_lines="skip", engine="python"
                )
            except Exception:
                content = pd.read_csv(file_path, sep=None, on_bad_lines="skip", engine="python")
        elif str(file_path).endswith(".parquet"):
            content = pd.read_parquet(file_path)
        elif str(file_path).endswith(".xlsx"):
            content = pd.read_excel(file_path)
        else:
            raise Exception("File format not supported (only csv, xlsx and parquet)")

        # if no already processed
        if not processed_corpus:
            # rename columns both for data & params to avoid confusion (use internal dataset_ prefix)
            content.columns = ["dataset_" + i for i in content.columns]
            if self.params.col_id is not None:
                self.params.col_id = "dataset_" + self.params.col_id

            # change also the name in the parameters
            self.params.cols_text = ["dataset_" + i for i in self.params.cols_text if i]
            self.params.cols_context = ["dataset_" + i for i in self.params.cols_context if i]
            self.params.cols_label = ["dataset_" + i for i in self.params.cols_label if i]
            self.params.cols_stratify = ["dataset_" + i for i in self.params.cols_stratify if i]

            # remove completely empty lines
            content = content.dropna(how="all")

        # quickfix : case where the index is the row number
        if self.params.col_id == "row_number":
            self.params.col_id = "dataset_row_number"

        all_columns = list(content.columns)
        n_total = len(content)

        # test if the size of the sample requested is possible
        if n_total < self.params.n_test + self.params.n_valid + self.params.n_train:
            shutil.rmtree(self.params.dir)
            raise Exception(
                f"Not enough data for creating the train/valid/test dataset. Current : {len(content)} ; Selected : {self.params.n_test + self.params.n_valid + self.params.n_train}"
            )

        # create the internal/external index
        # case where the index is the row number
        if self.params.col_id == "dataset_row_number":
            content["id_internal"] = [str(i) for i in range(len(content))]
            content["id_external"] = content["id_internal"]

        # case the index is a column: check uniqueness then use slugify for internal if unique after slugify
        else:
            content["id_external"] = content[self.params.col_id].astype(str)
            if content["id_external"].nunique() != len(content):
                n_duplicates = len(content) - content["id_external"].nunique()
                raise Exception(
                    f"The selected ID column '{self.params.col_id.removeprefix('dataset_')}' "
                    f"contains {n_duplicates} duplicate values. "
                    f"Please choose a column with unique values or use 'Row number'."
                )
            col_slugified = content["id_external"].apply(slugify)
            if col_slugified.nunique() == len(content):
                content["id_internal"] = col_slugified
            else:
                content["id_internal"] = [str(i) for i in range(len(content))]

        content.set_index("id_internal", inplace=True)

        # convert columns that can be numeric or force text, exception for the text/labels
        for col in [i for i in content.columns if i not in self.params.cols_label]:
            try:
                content[col] = pd.to_numeric(content[col], errors="raise")
            except Exception:
                content[col] = content[col].astype(str).replace("nan", None)
        for col in self.params.cols_label:
            try:
                content[col] = content[col].astype(str).replace("nan", None)
            except Exception:
                # if the column is not convertible to string, keep it as is
                pass

        # create the text column, merging the different columns
        content["text"] = content[self.params.cols_text].apply(
            lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
        )

        # convert NA texts in empty string
        content["text"] = content["text"].fillna("")

        # save a complete copy of the dataset
        content.to_parquet(self.params.dir.joinpath(self.data_all), index=True)

        # ------------------------
        # End of the data cleaning
        # ------------------------
        # Step2: Building Train - valid/test (if they exist) based on cases
        #Init all Null / rows empty 
        trainset,valid_set,test_set=None,None,None
        train_rows,valid_rows,test_rows=[],[],[]
        n_to_draw=self.params.n_test+self.params.n_valid
        
        #case if both train is sequential(pre-preapred) : Train build → evaluation build              
        if self.params.train_selection=="sequential":
            trainset,train_rows=self._build_train_set(content)
            #print(f"train_rows{train_rows}",flush=True)
            valid_set,test_set,valid_rows,test_rows=self._build_evalset(content,train_rows=train_rows)
        else:
            valid_set,test_set,valid_rows,test_rows=self._build_evalset(content)
            content_=content.drop(index=test_rows+valid_rows)
            trainset,train_rows=self._build_train_set(content_)
        #write eval 
        self._write_to_parquet_(valid_set,self.valid_file,"valid")
        self._write_to_parquet_(test_set,self.test_file,"test")
        # write the trainset
        self._write_to_parquet_(trainset)        

        # save parameters (without the data)
        project = self.params.model_dump()

        # add elements for the parameters
        project["project_slug"] = self.project_slug
        project["all_columns"] = all_columns
        project["n_total"] = n_total

        # schemes/labels to import (in the main process)
        import_trainset = None
        import_testset = None
        import_validset = None
        if len(self.params.cols_label) > 0:
            import_trainset = trainset[self.params.cols_label].dropna(how="all")
            if test_set is not None and not self.params.clear_test:
                import_testset = test_set[self.params.cols_label].dropna(how="all")
            if valid_set is not None and not self.params.clear_valid:
                import_validset = valid_set[self.params.cols_label].dropna(how="all")

        # delete the initial file
        if self.params.filename is not None:
            try:
                self.params.dir.joinpath(self.params.filename).unlink()
            except OSError as e:
                print(f"Warning: could not delete uploaded file: {e}")

        print("Project created")

        return ProjectModel(**project), import_trainset, import_validset, import_testset