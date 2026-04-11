import csv
import shutil
import sys

import pandas as pd  # type: ignore[import]

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
    #===============================================================================================#
    # Defining 3 Main functions for Now :                   
    # 1- 'Build test_val" dataset" : responsible for the build of both test / validation dataset(s)
    # 2- buildtrainset : responsible for building the trainset 
    # 3- detect labels : this function will automatically catch label cols with no human 
    #     internvation and give suggestion to use it 
    # 4- adding WaX /U-WaX ---> 
    #================================================================================================#
    def build_trainset(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame | None, list]:
        """
        Build a Training Set based on all The possible train selection
        modes taken in count 
        Returns (trainset,list_of_train_row_indices)
        """
        if self.params.n_train<=0:
            raise Exception("invalid Training dataset Size")
        #========================Sequential train (first N rows)==============================
        if self.params.train_selection=="sequential":
            n=min(self.params.n_train,len(dataset))
            trainset = dataset.iloc[:n].copy()
        #========================Force label (prioritize already labeled rows)==========================
        elif self.params.train_selection=="force_label":
            if len(self.params.cols_label)>0:
                col=self.params.cols_label
                labeled=dataset[dataset[col].notna().any(axis=1)]
                unlabeled=dataset[dataset[col].isna().all(axis=1)]
                if len(labeled) >= self.params.n_train:
                    trainset=labeled.sample(self.params.n_train,random_state=self.random_seed)
                else:
                    n_random=self.params.n_train-len(labeled)
                    trainset = pd.concat([labeled,unlabeled.sample(n_random, random_state=self.random_seed)], ignore_index=False)
            else:
                raise Exception ("Train Selection is stratification without handing colums for that ")
        #===============================Stratified train======================================================
        elif self.params.train_selection=="stratify":
            if len(self.params.cols_stratify)>0 :
                df_grouped=dataset.groupby(self.params.cols_stratify, group_keys=False)
                nb_cat=len(df_grouped)
                nb_per_cat=round(self.params.n_train/nb_cat)
                sampled_idx = df_grouped.apply(
                    lambda x: x.sample(min(len(x), nb_per_cat), random_state=self.random_seed)
                ).index.get_level_values(-1)
                trainset = dataset.loc[sampled_idx]
            else:
                raise Exception ("Train Selection is stratification without handing colums for that ")
                
        # ===================================================================
        # Random (default / fallback)
        # ===================================================================
        else:
            self.params.train_selection = "random"  
            trainset = dataset.sample(
                self.params.n_train, 
                random_state=self.random_seed
            )

        # ====================== SAVE TRAINSET ======================
        if trainset is not None and len(trainset) > 0:
            cols_to_save = ["id_external", "text"] + self.params.cols_context
            trainset[cols_to_save].to_parquet(
                self.params.dir.joinpath(self.train_file), 
                index=True
            )

        train_rows = list(trainset.index) if trainset is not None else []
        #print(len(trainset))
        return trainset, train_rows
    def build_test_val_set(
        self,
        dataset: pd.DataFrame,
        train_rows: list | None = None,
        start_index_val: int | None = None,
        start_index_test: int | None = None,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list, list]:
        """
        Build valid and test sets with support for explicit start indices + overlap protection.
        """
        n_test = getattr(self.params, 'n_test', 0)
        n_val = getattr(self.params, 'n_valid', 0)
        n_draw = n_test + n_val

        validset = testset = None
        rows_valid: list = []
        rows_test: list = []

        if n_draw == 0:
            return None, None, [], []

        # Pool for random/stratified cases (excludes train to prevent leakage)
        if train_rows:
            holdout_pool = dataset.drop(index=train_rows)
            train_idx_set = set(train_rows)
        else:
            holdout_pool = dataset.copy()
            train_idx_set = set()

        # ===================================================================
        # 1. SEQUENTIAL MODE (holdout_selection is None)
        # ===================================================================
        if self.params.holdout_selection is None:

            # Determine start positions
            start_val = start_index_val if start_index_val is not None else self.params.n_train
            start_test = start_index_test if start_index_test is not None else (start_val + n_val)

            # --- Build Validation ---
            if n_val > 0:
                end_val = start_val + n_val
                if end_val > len(dataset):
                    raise Exception(f"Validation slice [{start_val}:{end_val}] out of bounds! "
                                  f"Dataset has {len(dataset)} rows.")
                validset = dataset.iloc[start_val:end_val]

            # --- Build Test ---
            if n_test > 0:
                end_test = start_test + n_test
                if end_test > len(dataset):
                    raise Exception(f"Test slice [{start_test}:{end_test}] out of bounds! "
                                  f"Dataset has {len(dataset)} rows.")
                testset = dataset.iloc[start_test:end_test]

            # ====================== OVERLAP CHECKS ======================
            valid_idx = set(validset.index) if validset is not None else set()
            test_idx = set(testset.index) if testset is not None else set()

            # Val overlaps with Train?
            if valid_idx & train_idx_set:
                overlap = len(valid_idx & train_idx_set)
                raise Exception(f"Validation set overlaps with train by {overlap} rows!")

            # Test overlaps with Train?
            if test_idx & train_idx_set:
                overlap = len(test_idx & train_idx_set)
                raise Exception(f"Test set overlaps with train by {overlap} rows!")

            # Test overlaps with Validation?
            if valid_idx & test_idx:
                overlap = len(valid_idx & test_idx)
                raise Exception(f"Test and Validation sets overlap by {overlap} rows!")

        # ===================================================================
        # STRATIFIED HOLD OUT
        # ===================================================================
        elif self.params.holdout_selection == "stratify" and len(self.params.cols_stratify) > 0:
            if len(holdout_pool) < n_draw:
                raise Exception(f"Not enough rows for stratified holdout. Need {n_draw}, have {len(holdout_pool)}")

            df_grouped = holdout_pool.groupby(self.params.cols_stratify, group_keys=False)
            nb_cat = len(df_grouped)
            nb_per_cat = max(1, round(n_draw / nb_cat))

            sampled_idx = df_grouped.apply(
                lambda x: x.sample(min(len(x), nb_per_cat), random_state=self.random_seed)
            ).index.get_level_values(-1)

            holdout = holdout_pool.loc[sampled_idx]

            if n_test > 0 and n_val > 0:
                testset = holdout.sample(n_test, random_state=self.random_seed)
                validset = holdout.drop(index=testset.index)
            elif n_test > 0:
                testset = holdout
            else:
                validset = holdout

        # ===================================================================
        # 3. RANDOM HOLD OUT (default)
        # ===================================================================
        else:
            if len(holdout_pool) < n_draw:
                raise Exception(f"Not enough rows for holdout. Need {n_draw}, have {len(holdout_pool)}")

            holdout = holdout_pool.sample(n_draw, random_state=self.random_seed)

            if n_test > 0 and n_val > 0:
                testset = holdout.sample(n_test, random_state=self.random_seed)
                validset = holdout.drop(index=testset.index)
            elif n_test > 0:
                testset = holdout
            else:
                validset = holdout

        # ====================== SAVE TO PARQUET ======================
        for name, ds, file_path_, flag in [
            ("valid", validset, self.valid_file, "valid"),
            ("test",  testset,  self.test_file,  "test")
        ]:
            if ds is not None and len(ds) > 0:
                ds.to_parquet(self.params.dir.joinpath(file_path_), index=True)
                setattr(self.params, flag, True)
                rows = list(ds.index)
                if name == "valid":
                    rows_valid = rows
                else:
                    rows_test = rows
        
        return validset, testset, rows_valid, rows_test

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
                content = pd.read_csv(file_path, low_memory=False, on_bad_lines="skip")
            except Exception:
                content = pd.read_csv(file_path, on_bad_lines="skip", engine="python")
        elif str(file_path).endswith(".parquet"):
            content = pd.read_parquet(file_path)
        elif str(file_path).endswith(".xlsx"):
            content = pd.read_excel(file_path)
        else:
            raise Exception("File format not supported (only csv, xlsx and parquet)")

        # if no already processed
        if not processed_corpus:
            # rename columns both for data & params to avoid confusion (use internal dataset_ prefix)
            content.columns = ["dataset_" + i for i in content.columns]  # type: ignore[assignment]
            if self.params.col_id is not None:
                self.params.col_id = "dataset_" + self.params.col_id

            # change also the name in the parameters
            self.params.cols_text = ["dataset_" + i for i in self.params.cols_text if i]
            self.params.cols_context = ["dataset_" + i for i in self.params.cols_context if i]
            self.params.cols_label = ["dataset_" + i for i in self.params.cols_label if i]
            self.params.cols_stratify = ["dataset_" + i for i in self.params.cols_stratify if i]

            # remove completely empty lines..
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

        # -----------------------
        # -
        # End of the data cleaning
        # ------------------------

        # ====================== SPLITTING LOGIC ======================
        rows_test: list = []
        rows_valid: list = []
        self.params.test = False
        self.params.valid = False
        trainset = None
        validset = None
        testset = None

        # ------------------------------------------------------------------
        # Sequential train → build train first, then holdout from remaining
        # ------------------------------------------------------------------
        if self.params.train_selection == "sequential":
            trainset, train_rows = self.build_trainset(content)

            validset, testset, rows_valid, rows_test = self.build_test_val_set(
                dataset=content,
                train_rows=train_rows,
                start_index_val=getattr(self.params, 'start_index_val', None),
                start_index_test=getattr(self.params, 'start_index_test', None),
            )

        # ------------------------------------------------------------------
        # Non-sequential (random/stratify/force_label) → build holdout first
        # ------------------------------------------------------------------
        else:
            # Build holdout on full data
            validset, testset, rows_valid, rows_test = self.build_test_val_set(content)

            # Then build train on the remaining rows (prevents leakage)
            excluded_idx = set(rows_valid + rows_test)
            remaining = content.drop(index=list(excluded_idx)) if excluded_idx else content
            trainset, _ = self.build_trainset(remaining)

        # ====================== SAVE PARAMETERS & LABELS ======================
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
            if trainset is not None:
                import_trainset = trainset[self.params.cols_label].dropna(how="all")

            if testset is not None and not getattr(self.params, 'clear_test', False):
                import_testset = testset[self.params.cols_label].dropna(how="all")

            if validset is not None and not getattr(self.params, 'clear_valid', False):
                import_validset = validset[self.params.cols_label].dropna(how="all")

        # delete the initial uploaded file
        if self.params.filename is not None:
            try:
                self.params.dir.joinpath(self.params.filename).unlink()
            except OSError as e:
                print(f"Warning: could not delete uploaded file: {e}")

        print("Project created")

        return ProjectModel(**project), import_trainset, import_validset, import_testset
