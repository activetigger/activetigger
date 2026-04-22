import io
import uuid
from typing import Tuple

import pandas as pd  # type: ignore[import]

from activetigger.config import config
from activetigger.datamodels import EvalSetDataModel, ProjectBaseModel
from activetigger.functions import slugify
from activetigger.tasks.base_task import BaseTask


class AddEvalSet(BaseTask):
    """
    Add an evaluation set to the project
    """

    Kind = "add_evalset"

    def __init__(
        self,
        dataset: str,
        evalset: EvalSetDataModel,
        project: ProjectBaseModel,
        username: str,
        project_slug: str,
        index: pd.Index,
        scheme=None,
    ):

        super().__init__()
        self.evalset = evalset
        self.project_model = project
        self.dataset = dataset
        self.username = username
        self.index_all = index
        self.scheme = scheme
        self.project_slug = project_slug
        self.elements = []
        # self.Kind = f"{self.Kind}_{dataset}"

    def __stop_process_opportunity(self):
        if self.event is not None and self.event.is_set():
            file_name = config.test_file if self.dataset == "test" else config.valid_file
            if self.project_model.dir is not None:
                self.project_model.dir.joinpath(file_name).unlink(missing_ok=True)
            raise Exception("Adding evaluation set process interrupted by user")

    def __call__(self) -> Tuple[Tuple[str, str, str, str | None, list], ProjectBaseModel]:
        try:
            self.__stop_process_opportunity()
            csv_buffer = io.StringIO(self.evalset.csv)
            df = pd.read_csv(
                csv_buffer,
                sep=None,
                engine="python",
                dtype={self.evalset.col_id: str, **{col: str for col in self.evalset.cols_text}},
                nrows=self.evalset.n_eval,
            )
            if len(df) > 10000:
                raise Exception("You valid set is too large")
            # added a check if DF is empty to avoid errors
            if len(df) == 0:
                raise Exception("Your valid set is empty")
            #print("df cols", df.columns, flush=True)
            # stop Process
            self.__stop_process_opportunity()
            # create text column
            df["text"] = df[self.evalset.cols_text].apply(
                lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
            )
            if not self.evalset.col_label:
                df = df.rename(columns={self.evalset.col_id: "id"})
            else:
                if "label" in df.columns and self.evalset.col_label != "label":
                    df = df.rename(columns={"label": "_label_"})
                df = df.rename(
                    columns={
                        self.evalset.col_id: "id",
                        self.evalset.col_label: "label",
                    }
                )
                df["label"] = df["label"].apply(lambda x: None if pd.isna(x) else str(x))
            # deal with non-unique id
            df["id_external"] = df["id"].apply(str)
            if not ((df["id"].astype(str).apply(slugify)).nunique() == len(df)):
                df["id"] = [str(i) for i in range(len(df))]
                print("ID not unique, changed to default id")
            # =================== Compare with the general dataset (Overlaps) ===================
            #                    ============================================
            plain_full_index = {str(x).removeprefix("imported-") for x in self.index_all}
            overlapping_ids = set(df["id"]).intersection(set(plain_full_index))
            if overlapping_ids and len(overlapping_ids) > 0:
                prefix = f"c-ev-{uuid.uuid4().hex[:6]}-"
                df.loc[df["id"].isin(overlapping_ids), "id"] = [
                    prefix + str(i) for i in range(len(overlapping_ids))
                ]
                print(
                    f"{len(overlapping_ids)} IDs in the eval set already exist in the main dataset changed"
                )
            df["id"] = df["id"].apply(lambda x: f"imported-{str(x)}")
            df = df.set_index("id")
            # verify label columns be fore writing to parquet
            if self.evalset.col_label and self.evalset.scheme:
                # Check the label columns if they match the scheme or raise error
                for label in df["label"].dropna().unique():
                    if self.scheme and label not in self.scheme:
                        raise Exception(f"Label {label} not in the scheme {self.evalset.scheme}")
            # stop Process
            self.__stop_process_opportunity()

            # write to parquet
            if self.dataset in ("test", "valid") and len(df) > 0:
                file_name = config.test_file if self.dataset == "test" else config.valid_file
                if self.project_model.dir is not None:
                    df[["id_external", "text"]].to_parquet(
                        self.project_model.dir.joinpath(file_name)
                    )
                setattr(self.project_model, f"{self.dataset}", True)
                setattr(self.project_model, f"n_{self.dataset}", len(df))
            else:
                raise Exception("Dataset should be test or valid")
            # stop Process
            self.__stop_process_opportunity()

            # once written to parquet→import labels if specified + scheme // check if the labels are in the scheme
            current_dataset = getattr(self.project_model, self.dataset)
            if current_dataset:
                if self.evalset.col_label and self.evalset.scheme:
                    self.elements = [
                        {"element_id": element_id, "annotation": label, "comment": ""}
                        for element_id, label in df["label"].dropna().items()
                    ]
            args = (
                self.dataset,
                self.username,
                self.project_slug,
                self.evalset.scheme,
                self.elements,
            )
            return args, self.project_model
        except Exception as e:
            print(e)
            raise e
