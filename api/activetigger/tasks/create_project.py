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
        # Step 2 : valid/test dataset : from the complete dataset + random/stratification
        # if both, draw together valid / test set and then separage
        rows_test = []
        rows_valid = []
        self.params.test = False
        self.params.valid = False
        testset = None
        validset = None

        # case there is a test or valid set, common draw
        if self.params.n_test + self.params.n_valid != 0:
            n_to_draw = self.params.n_test + self.params.n_valid
            # manage stratification
            if len(self.params.cols_stratify) == 0:
                draw = content.sample(n_to_draw, random_state=self.random_seed)
            else:
                df_grouped = content.groupby(self.params.cols_stratify, group_keys=False)
                nb_cat = len(df_grouped)
                nb_elements_cat = round(n_to_draw / nb_cat)
                sampled_idx = df_grouped.apply(
                    lambda x: x.sample(min(len(x), nb_elements_cat), random_state=self.random_seed)
                ).index.get_level_values(-1)
                draw = content.loc[sampled_idx]

            # divide between test and valid
            if self.params.n_test > 0 and self.params.n_valid == 0:
                testset = draw
                testset.to_parquet(self.params.dir.joinpath(self.test_file), index=True)
                self.params.test = True
                rows_test = list(testset.index)
            elif self.params.n_valid > 0 and self.params.n_test == 0:
                validset = draw
                validset.to_parquet(self.params.dir.joinpath(self.valid_file), index=True)
                self.params.valid = True
                rows_valid = list(validset.index)
            else:
                testset = draw.sample(self.params.n_test, random_state=self.random_seed)
                validset = draw.drop(index=testset.index)
                validset.to_parquet(self.params.dir.joinpath(self.valid_file), index=True)
                testset.to_parquet(self.params.dir.joinpath(self.test_file), index=True)
                self.params.test = True
                self.params.valid = True
                rows_valid = list(validset.index)
                rows_test = list(testset.index)

        # Step 3 : train dataset / different strategies

        # remove test rows
        content = content.drop(rows_test)
        content = content.drop(rows_valid)

        # case where there is no valid/test set and the selection is deterministic
        if (
            not self.params.random_selection
            and self.params.n_test == 0
            and self.params.n_valid == 0
        ):
            trainset = content[0 : self.params.n_train]
        # case to force the max of label from one column
        elif self.params.force_label and len(self.params.cols_label) > 0:
            f_notna = content[self.params.cols_label[0]].notna()
            f_na = content[self.params.cols_label[0]].isna()
            # different case regarding the number of labels
            if f_notna.sum() > self.params.n_train:
                trainset = content[f_notna].sample(
                    self.params.n_train, random_state=self.random_seed
                )
            else:
                n_train_random = self.params.n_train - f_notna.sum()
                trainset = pd.concat(
                    [
                        content[f_notna],
                        content[f_na].sample(n_train_random, random_state=self.random_seed),
                    ]
                )
        # case there is stratification on the trainset
        elif len(self.params.cols_stratify) > 0 and self.params.stratify_train:
            df_grouped = content.groupby(self.params.cols_stratify, group_keys=False)
            nb_cat = len(df_grouped)
            nb_elements_cat = round(self.params.n_train / nb_cat)
            sampled_idx = df_grouped.apply(
                lambda x: x.sample(min(len(x), nb_elements_cat), random_state=self.random_seed)
            ).index.get_level_values(-1)
            trainset = content.loc[sampled_idx]
        # default with random selection in the remaining elements
        else:
            trainset = content.sample(self.params.n_train, random_state=self.random_seed)

        # write the trainset
        trainset[["id_external", "text"] + self.params.cols_context].to_parquet(
            self.params.dir.joinpath(self.train_file), index=True
        )

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
            if testset is not None and not self.params.clear_test:
                import_testset = testset[self.params.cols_label].dropna(how="all")
            if validset is not None and not self.params.clear_valid:
                import_validset = validset[self.params.cols_label].dropna(how="all")

        # delete the initial file
        if self.params.filename is not None:
            try:
                self.params.dir.joinpath(self.params.filename).unlink()
            except OSError as e:
                print(f"Warning: could not delete uploaded file: {e}")

        print("Project created")

        return ProjectModel(**project), import_trainset, import_validset, import_testset
