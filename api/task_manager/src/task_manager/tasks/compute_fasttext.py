import os
from collections.abc import Callable
from pathlib import Path

import fasttext
import pandas as pd
from activetigger.functions import tokenize
from fasttext.util import download_model
from pandas import DataFrame, Series


def encode_fasttext(
    texts: Series,
    language: str,
    path_models: Path,
    path_progress: Path,
    model: str | None = "",
    is_aborted: Callable[[], bool] | None = None,
) -> DataFrame:
    """
    Fasttext sentence embeddings, shared by the celery task and the in-process
    callers (feature recomputation). Downloads the language model if none is
    given. Progress is written as a percentage to path_progress; the caller
    owns that file.
    """
    if not path_models.exists():
        raise Exception(f"path {str(path_models)} does not exist")

    # download_model saves into the current directory
    current_directory = os.getcwd()
    try:
        os.chdir(path_models)
        # if no model is specified, try to dl the language model
        if model is None or model == "" or model == "generic":
            model_name = download_model(language, if_exists="ignore")
        else:
            model_name = model
            if not Path(model_name).exists():
                raise FileNotFoundError(f"Model {model_name} not found")
    finally:
        os.chdir(current_directory)

    texts_tk = tokenize(texts)
    ft = fasttext.load_model(str(path_models.joinpath(model_name)))
    emb = []
    for t in texts_tk:
        emb.append(ft.get_sentence_vector(t.replace("\n", " ")))
        # every 100 texts: write progress and poll the abort flag (a redis round trip)
        if len(emb) % 100 == 0:
            progress_percent = len(emb) / len(texts_tk) * 100
            with open(path_progress, "w") as f:
                f.write(str(round(progress_percent, 1)))
            print(progress_percent)
            if is_aborted is not None and is_aborted():
                raise Exception("Process interrupted by user")

    df = pd.DataFrame(emb, index=texts.index)
    df.columns = ["ft%03d" % (x + 1) for x in range(len(df.columns))]
    return df
