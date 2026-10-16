from pathlib import Path
from typing import TypedDict

import pandas as pd
from activetigger.config import config
from activetigger.datamodels import ProjectUpdateModel
from activetigger.functions import concat_text_columns


class UpdatedFields(TypedDict):
    cols_context: list[str]
    cols_text: list[str]
    n_train: int | None
    # the train rows changed: features, bertopic and projections are invalid
    reset_features: bool


def update_datasets(
    path_project: Path,
    cols_context: list[str],
    cols_text: list[str],
    update: ProjectUpdateModel,
) -> UpdatedFields:
    """
    Update datasets with access to the raw dataset
    - Write new data on parquet files
    - Return the modified project fields
    """
    path_data_all = path_project / Path(config.data_all)
    path_data_train = path_project / Path(config.train_file)
    path_data_valid = path_project / Path(config.valid_file)
    path_data_test = path_project / Path(config.test_file)

    result: UpdatedFields = {
        "cols_context": cols_context,
        "cols_text": cols_text,
        "n_train": None,
        "reset_features": False,
    }
    if update.cols_context is not None and set(update.cols_context) != set(cols_context):
        change_context(path_data_all, path_data_train, cols_context, update.cols_context)
        result["cols_context"] = update.cols_context
    if update.cols_text is not None and set(update.cols_text) != set(cols_text):
        change_text_column(path_data_all, path_data_train, update.cols_text)
        result["cols_text"] = update.cols_text
    if update.add_n_train is not None and update.add_n_train > 0:
        result["n_train"] = change_n_train(
            path_data_all, path_data_train, path_data_valid, path_data_test, update.add_n_train
        )
        result["reset_features"] = True
    return result


def change_context(
    path_data_all: Path, path_data_train: Path, old_cols: list[str], new_cols: list[str]
) -> None:
    """
    Change context columns
    """
    df_all = pd.read_parquet(path_data_all, columns=new_cols)
    df_train = pd.read_parquet(path_data_train)
    df_train.drop(columns=old_cols, inplace=True)
    df_train = pd.concat([df_train, df_all.loc[df_train.index]], axis=1)
    df_train.to_parquet(path_data_train)


def change_text_column(path_data_all: Path, path_data_train: Path, cols_text: list[str]) -> None:
    """
    Change text column
    """
    df_all = pd.read_parquet(path_data_all, columns=cols_text)
    df_train = pd.read_parquet(path_data_train)
    df_sub = df_all.loc[df_train.index]
    df_sub["text"] = concat_text_columns(df_sub, cols_text)
    df_train["text"] = df_sub["text"]
    df_train.to_parquet(path_data_train)


def change_n_train(
    path_data_all: Path,
    path_data_train: Path,
    path_data_valid: Path,
    path_data_test: Path,
    add_n_train: int,
) -> int:
    """
    Change number of training elements by adding new elements from all data
    """
    df_train = pd.read_parquet(path_data_train)
    df_all = pd.read_parquet(path_data_all, columns=list(df_train.columns))
    elements_index = list(df_train.index)
    if path_data_valid.exists():
        df_valid = pd.read_parquet(path_data_valid)
        elements_index += list(df_valid.index)
    if path_data_test.exists():
        df_test = pd.read_parquet(path_data_test)
        elements_index += list(df_test.index)

    # take elements that are not in index
    df_all = df_all[~df_all.index.isin(elements_index)]

    # sample elements
    elements_to_add = df_all.sample(add_n_train)

    # drop na elements to avoid problems
    elements_to_add = elements_to_add[elements_to_add["text"].notna()]

    df_train = pd.concat([df_train, elements_to_add])
    df_train.to_parquet(path_data_train)
    return len(df_train)
