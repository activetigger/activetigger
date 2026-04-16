from pathlib import Path

import pandas as pd

from activetigger.config import config
from activetigger.datamodels import ProjectModel, ProjectUpdateModel
from activetigger.tasks.base_task import BaseTask


class UpdateDatasets(BaseTask):
    """
    Update datasets with access to the raw dataset
    - Write new data on parquet files
    - Return modified project params
    """

    kind = "update_datasets"
    update: ProjectUpdateModel
    params: ProjectModel
    project_path: Path
    path_data_all: Path
    path_data_train: Path
    path_data_valid: Path
    path_data_test: Path

    def __init__(self, project_params: ProjectModel, update: ProjectUpdateModel) -> None:
        super().__init__()
        if project_params.dir is None:
            raise ValueError("Project path is None")
        self.project_path = project_params.dir
        self.params = project_params
        self.path_data_all = self.project_path / Path(config.data_all)
        self.path_data_train = self.project_path / Path(config.train_file)
        self.path_data_valid = self.project_path / Path(config.valid_file)
        self.path_data_test = self.project_path / Path(config.test_file)
        self.update = update

    def __call__(self) -> tuple[ProjectModel, bool]:
        try:
            # apply the changes
            resetFeatures = False
            if self.update.cols_context is not None and set(self.update.cols_context) != set(
                self.params.cols_context
            ):
                self.change_context()
            if self.update.cols_text is not None and set(self.update.cols_text) != set(
                self.params.cols_text
            ):
                self.change_text_column()
            if self.update.add_n_train is not None and self.update.add_n_train > 0:
                self.change_n_train()
                resetFeatures = True

            # return the fields to update
            return (self.params, resetFeatures)
        except Exception as e:
            raise e

    def change_context(self) -> None:
        """
        Change context columns
        """
        if self.update.cols_context is None:
            raise ValueError("No new context columns provided")
        df_all = pd.read_parquet(
            self.path_data_all,
            columns=self.update.cols_context,
        )
        df_train = pd.read_parquet(self.path_data_train)
        df_train.drop(columns=self.params.cols_context, inplace=True)
        df_train = pd.concat([df_train, df_all.loc[df_train.index]], axis=1)
        df_train.to_parquet(self.path_data_train)
        self.params.cols_context = self.update.cols_context

    def change_text_column(self) -> None:
        """
        Change text column
        """
        if self.update.cols_text is None:
            raise ValueError("No text columns provided")
        df_all = pd.read_parquet(
            self.path_data_all,
            columns=self.update.cols_text,
        )
        df_train = pd.read_parquet(self.path_data_train)
        df_sub = df_all.loc[df_train.index]
        df_sub["text"] = df_sub[self.update.cols_text].apply(
            lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
        )
        df_train["text"] = df_sub["text"]
        df_train.to_parquet(self.path_data_train)
        self.params.cols_text = self.update.cols_text

    def change_n_train(self) -> None:
        """
        Change number of training elements by adding new elements from all data
        """
        if self.update.add_n_train is None:
            raise ValueError("No number of training elements to add provided")
        df_train = pd.read_parquet(self.path_data_train)
        df_all = pd.read_parquet(
            self.path_data_all,
            columns=list(df_train.columns),
        )
        elements_index = list(df_train.index)
        if self.path_data_valid.exists():
            df_valid = pd.read_parquet(self.path_data_valid)
            elements_index += list(df_valid.index)
        if self.path_data_test.exists():
            df_test = pd.read_parquet(self.path_data_test)
            elements_index += list(df_test.index)

        # take elements that are not in index
        df_all = df_all[~df_all.index.isin(elements_index)]

        # sample elements
        elements_to_add = df_all.sample(self.update.add_n_train)

        # drop na elements to avoid problems
        elements_to_add = elements_to_add[elements_to_add["text"].notna()]

        df_train = pd.concat([df_train, elements_to_add])
        df_train.to_parquet(self.path_data_train)

        # update the params
        self.params.n_train = len(df_train)
