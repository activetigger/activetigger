import os
from pathlib import Path

import fasttext
import pandas as pd
from fasttext.util import download_model
from pandas import DataFrame, Series

from activetigger.functions import tokenize
from activetigger.tasks.base_task import BaseTask


class ComputeFasttext(BaseTask):
    """
    Compute Fastext feature
    TODO : check of possible to avoid loop ?
    """

    kind = "compute_feature_fastext"

    def __init__(
        self,
        texts: Series,
        language: str,
        path_process: Path,
        path_models: Path,
        model: str = "",
    ):
        super().__init__()
        self.texts = texts
        self.path_process = path_process
        self.path_models = path_models
        self.language = language
        self.model = model
        self.event = None

    def __call__(self) -> DataFrame:
        """
        Compute fasttext embedding
        Download the model if needed
        Args:
            texts (pandas.Series): texts
            model (str): model to use
        Returns:
            pandas.DataFrame: embeddings
        """
        if not self.path_models.exists():
            raise Exception(f"path {str(self.path_models)} does not exist")

        progress_path = self.path_process.joinpath(self.unique_id)
        try:
            current_directory = os.getcwd()
            os.chdir(self.path_models)

            # if no model is specified, try to dl the language model
            if self.model is None or self.model == "" or self.model == "generic":
                model_name = download_model(self.language, if_exists="ignore")
            else:
                model_name = self.model
                if not Path(model_name).exists():
                    raise FileNotFoundError(f"Model {model_name} not found")

            os.chdir(current_directory)
            texts_tk = tokenize(self.texts)
            ft = fasttext.load_model(str(self.path_models.joinpath(model_name)))
            emb = []
            for t in texts_tk:
                emb.append(ft.get_sentence_vector(t.replace("\n", " ")))
                # manage progress
                if len(emb) % 100 == 0:
                    progress_percent = len(emb) / len(texts_tk) * 100
                    with open(progress_path, "w") as f:
                        f.write(str(round(progress_percent, 1)))
                    print(progress_percent)
                # check if the user want to stop the process
                if self.event is not None:
                    if self.event.is_set():
                        raise Exception("Process interrupted by user")

            # emb = [ft.get_sentence_vector(t.replace("\n", " ")) for t in texts_tk]
            df = pd.DataFrame(emb, index=self.texts.index)
            # WARN: this seems strange. Maybe replace with a more explicit syntax
            df.columns = ["ft%03d" % (x + 1) for x in range(len(df.columns))]
            return df
        finally:
            progress_path.unlink(missing_ok=True)
