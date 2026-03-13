import gc
import math
import multiprocessing
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pandas import DataFrame, Series
from sentence_transformers import SentenceTransformer

from activetigger.config import config

# from transformers import AutoConfig
from activetigger.tasks.base_task import BaseTask


class ComputeSbert(BaseTask):
    """
    Compute Embeddings features
    """

    kind = "compute_feature_embeddings"

    def __init__(
        self,
        texts: Series,
        path_process: Path,
        model: str,
        batch_size: int = 32,
        min_gpu: int = 6,
        max_tokens: int = 1024,
        path_progress: Path | None = None,
        event: Optional[multiprocessing.synchronize.Event] = None,
    ):
        super().__init__()
        self.texts = texts
        self.model = model
        self.model_args = {}
        if self.model in config.models_embeddings:
            self.model_args = config.models_embeddings[self.model]
        self.batch_size = batch_size
        self.min_gpu = min_gpu
        self.path_process = path_process
        self.max_tokens = int(max_tokens)
        self.event = event
        if path_progress:
            self.progress_file_temporary = False
            self.path_progress = path_progress
        else:
            self.path_progress = self.path_process.joinpath(self.unique_id)
            self.progress_file_temporary = True

    # def retrieve_model_max_length(self) -> int:
    #     try:
    #         model_max_length = AutoConfig.from_pretrained(
    #             self.model, trust_remote_code=True
    #         ).max_position_embeddings
    #     except Exception:
    #         print("Cannot retrieve model max length, fallback to user value")
    #         model_max_length = self.max_tokens

    #     return model_max_length

    def __call__(self) -> DataFrame:
        """
        Compute embeddings
        """
        # update temporary file with the current ID
        if self.progress_file_temporary:
            self.path_progress = self.path_process.joinpath(self.unique_id)

        # test the data
        if self.texts.isnull().sum() > 0:
            raise ValueError("There are missing values in the input data, so we can't proceed")

        if torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).total_memory / (1024**3) > self.min_gpu:
                device = torch.device("cuda")  # Use CUDA
            else:
                print("Not enough GPU memory, fallback to CPU")
                device = torch.device("cpu")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")  # Fallback to CPU

        sbert = None
        try:
            sbert = SentenceTransformer(self.model, device=str(device), trust_remote_code=True)
            max_seq_length = sbert.max_seq_length
            sbert.max_seq_length = int(min(self.max_tokens, max_seq_length))

            print("start computation")
            embeddings = []
            total_batches = math.ceil(len(self.texts) / self.batch_size)
            for i, start in enumerate(range(0, len(self.texts), self.batch_size), 1):
                # check if the user want to stop the process
                if self.event is not None:
                    if self.event.is_set():
                        print("Process interrupted by user")
                        raise Exception("Process interrupted by user")
                # create the batch
                batch_texts = list(self.texts.iloc[start : start + self.batch_size])
                embeddings.append(
                    sbert.encode(
                        batch_texts,
                        device=str(device),
                        normalize_embeddings=True,
                        **self.model_args,
                    )
                )

                # manage progress
                progress_percent = (i / total_batches) * 100
                with open(self.path_progress, "w") as f:
                    f.write(str(round(progress_percent, 1)))
                print(progress_percent)

            # shape the data
            emb = DataFrame(
                np.vstack(embeddings),
                index=self.texts.index,
                columns=["sb%03d" % (x + 1) for x in range(len(embeddings[0][0]))],
            )
            if self.progress_file_temporary:
                self.path_progress.unlink()
            return emb
        except Exception as e:
            raise e
        finally:
            if sbert is not None:
                del sbert
            del self.texts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
