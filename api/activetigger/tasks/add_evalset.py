import io
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd  # type: ignore[import]

from activetigger.config import config
from activetigger.datamodels import EvalSetDataModel, EvalSetImageModel, ProjectBaseModel
from activetigger.functions import concat_text_columns, slugify
from activetigger.functions_image import (
    ALLOWED_EXT,
    MAX_IMAGE_BYTES,
    MAX_ZIP_BYTES,
    filter_readable_images,
    generate_thumbnail,
)
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
            dtype_map: dict[str, type] = {col: str for col in self.evalset.cols_text}
            if self.evalset.col_id != "row_number":
                dtype_map[self.evalset.col_id] = str
            df = pd.read_csv(
                csv_buffer,
                sep=None,
                engine="python",
                dtype=dtype_map,
                nrows=self.evalset.n_eval,
            )
            if len(df) > 10000:
                raise Exception("You valid set is too large")
            # added a check if DF is empty to avoid errors
            if len(df) == 0:
                raise Exception("Your valid set is empty")
            # print("df cols", df.columns, flush=True)
            # stop Process
            self.__stop_process_opportunity()
            # create text column
            df["text"] = concat_text_columns(df, self.evalset.cols_text)
            if self.evalset.col_id == "row_number":
                df["id"] = [str(i) for i in range(len(df))]
                if self.evalset.col_label:
                    if "label" in df.columns and self.evalset.col_label != "label":
                        df = df.rename(columns={"label": "_label_"})
                    df = df.rename(columns={self.evalset.col_label: "label"})
                    df["label"] = df["label"].apply(lambda x: None if pd.isna(x) else str(x))
            elif not self.evalset.col_label:
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


class AddEvalSetImage(BaseTask):
    """
    Add an evaluation set for image projects.

    Accepts a zip of images (uploaded separately to project.dir/data/) plus
    an optional labels CSV/Parquet (also uploaded separately). Extracts to
    project.dir/images/eval_{dataset}/, drops unreadable images via the
    shared cleaner, and writes test/valid.parquet with the same
    [id_external, text] schema as text projects so Data.load_dataset works
    unchanged. Mirrors AddEvalSet's ID-collision handling.
    """

    Kind = "add_evalset"

    def __init__(
        self,
        dataset: str,
        evalset: EvalSetImageModel,
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
        self.elements: list[dict] = []

    def __stop_process_opportunity(self, images_dir: Path | None = None) -> None:
        if self.event is not None and self.event.is_set():
            if images_dir is not None and images_dir.exists():
                shutil.rmtree(images_dir, ignore_errors=True)
            file_name = config.test_file if self.dataset == "test" else config.valid_file
            if self.project_model.dir is not None:
                self.project_model.dir.joinpath(file_name).unlink(missing_ok=True)
            raise Exception("Adding evaluation set process interrupted by user")

    def __call__(self) -> Tuple[Tuple[str, str, str, str | None, list], ProjectBaseModel]:
        try:
            if self.dataset not in ("test", "valid"):
                raise Exception("Dataset should be test or valid")
            if self.project_model.dir is None:
                raise Exception("Project directory is missing")

            project_dir = self.project_model.dir
            data_dir = project_dir.joinpath("data")
            zip_path = data_dir.joinpath(self.evalset.filename)
            if not zip_path.exists() or not str(zip_path).lower().endswith(".zip"):
                raise Exception("Image eval set requires a .zip archive upload")
            if zip_path.stat().st_size > MAX_ZIP_BYTES:
                raise Exception(f"Zip file too large (max {MAX_ZIP_BYTES // (1024 * 1024)} MB)")

            images_dir = project_dir.joinpath("images", f"eval_{self.dataset}")
            thumbs_dir = images_dir.joinpath("thumbs")
            images_dir.mkdir(parents=True, exist_ok=True)
            thumbs_dir.mkdir(parents=True, exist_ok=True)

            self.__stop_process_opportunity(images_dir)

            rows: list[dict] = []
            with zipfile.ZipFile(zip_path, "r") as zf:
                infos = [i for i in zf.infolist() if not i.is_dir()]
                image_infos = [
                    i
                    for i in infos
                    if Path(i.filename).suffix.lower() in ALLOWED_EXT
                    and not Path(i.filename).name.startswith("._")
                    and "__MACOSX" not in Path(i.filename).parts
                ]
                if not image_infos:
                    raise Exception("Zip does not contain any .png/.jpg image")
                for info in image_infos:
                    if info.file_size > MAX_IMAGE_BYTES:
                        raise Exception(
                            f"Image '{info.filename}' exceeds per-file cap of "
                            f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB"
                        )

                if self.evalset.n_eval and self.evalset.n_eval > 0:
                    image_infos = image_infos[: self.evalset.n_eval]

                total = len(image_infos)
                progress_file = project_dir.joinpath(f"add_evalset_{self.dataset}_progress")
                for idx, info in enumerate(image_infos, 1):
                    if idx % 50 == 0:
                        self.__stop_process_opportunity(images_dir)
                    src_name = Path(info.filename).name
                    element_id = Path(info.filename).stem
                    target = images_dir.joinpath(src_name)
                    # avoid silent overwrites if the zip has duplicate basenames
                    if target.exists():
                        target = images_dir.joinpath(f"{uuid.uuid4().hex[:6]}_{src_name}")
                    with zf.open(info) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    rows.append({"id": element_id, "path": str(target)})

                    generate_thumbnail(target, thumbs_dir.joinpath(f"{slugify(element_id)}.jpg"))

                    try:
                        with open(progress_file, "w") as pf:
                            pf.write(str(round((idx / total) * 100, 1)))
                    except OSError:
                        pass

                try:
                    progress_file.unlink()
                except OSError:
                    pass

            # Pre-flight: drop unreadable images so the eval pool only carries
            # files the rest of the pipeline can actually load.
            readable_flags = filter_readable_images(
                [r["path"] for r in rows],
                stop_check=lambda: self.__stop_process_opportunity(images_dir),
            )
            n_dropped = 0
            kept: list[dict] = []
            for row, ok in zip(rows, readable_flags):
                if ok:
                    kept.append(row)
                else:
                    n_dropped += 1
                    try:
                        Path(row["path"]).unlink(missing_ok=True)
                    except OSError:
                        pass
            if n_dropped > 0:
                print(f"Dropped {n_dropped} unreadable/corrupt images", flush=True)
            rows = kept
            if not rows:
                raise Exception("Zip contained no readable images after cleaning")

            df = pd.DataFrame(rows)

            # Optional labels file: small CSV/Parquet uploaded separately.
            labels_filename = self.evalset.labels_filename
            if labels_filename:
                labels_path = data_dir.joinpath(labels_filename)
                if not labels_path.exists():
                    raise Exception(f"Labels file {labels_filename} not found")
                if str(labels_path).lower().endswith(".csv"):
                    labels_df = pd.read_csv(labels_path, sep=None, engine="python")
                elif str(labels_path).lower().endswith(".parquet"):
                    labels_df = pd.read_parquet(labels_path)
                elif str(labels_path).lower().endswith(".xlsx"):
                    labels_df = pd.read_excel(labels_path)
                else:
                    raise Exception("Labels file must be .csv, .parquet, or .xlsx")

                col_id = self.evalset.col_id or labels_df.columns[0]
                col_label = self.evalset.col_label
                if col_id not in labels_df.columns:
                    raise Exception(f"ID column '{col_id}' not in labels file")
                if col_label and col_label not in labels_df.columns:
                    raise Exception(f"Label column '{col_label}' not in labels file")

                # match on filename stem (same convention as CreateProjectImagexp)
                labels_df[col_id] = labels_df[col_id].astype(str).map(lambda s: Path(s).stem)
                cols = [col_id] + ([col_label] if col_label else [])
                labels_df = labels_df[cols].rename(columns={col_id: "id"})
                if col_label:
                    labels_df = labels_df.rename(columns={col_label: "label"})
                    labels_df["label"] = labels_df["label"].apply(
                        lambda x: None if pd.isna(x) else str(x)
                    )
                df = df.merge(labels_df, on="id", how="left")

            # ID handling — mirror AddEvalSet (uniqueness + train-set collision)
            df["id_external"] = df["id"].apply(str)
            if not ((df["id"].astype(str).apply(slugify)).nunique() == len(df)):
                df["id"] = [str(i) for i in range(len(df))]
                print("ID not unique, changed to default id")
            plain_full_index = {str(x).removeprefix("imported-") for x in self.index_all}
            overlapping_ids = set(df["id"]).intersection(plain_full_index)
            if overlapping_ids:
                prefix = f"c-ev-{uuid.uuid4().hex[:6]}-"
                df.loc[df["id"].isin(overlapping_ids), "id"] = [
                    prefix + str(i) for i in range(len(overlapping_ids))
                ]
                print(
                    f"{len(overlapping_ids)} IDs in the eval set already exist in the main dataset changed"
                )
            df["id"] = df["id"].apply(lambda x: f"imported-{str(x)}")
            df = df.set_index("id")
            df["text"] = df["path"].astype(str)

            if "label" in df.columns and self.evalset.scheme:
                for label in df["label"].dropna().unique():
                    if self.scheme and label not in self.scheme:
                        raise Exception(
                            f"Label {label} not in the scheme {self.evalset.scheme}"
                        )

            self.__stop_process_opportunity(images_dir)

            file_name = config.test_file if self.dataset == "test" else config.valid_file
            df[["id_external", "text"]].to_parquet(project_dir.joinpath(file_name))
            setattr(self.project_model, self.dataset, True)
            setattr(self.project_model, f"n_{self.dataset}", len(df))

            if "label" in df.columns:
                self.elements = [
                    {"element_id": element_id, "annotation": label, "comment": ""}
                    for element_id, label in df["label"].dropna().items()
                ]

            # cleanup: uploaded inputs are not needed after extraction
            try:
                zip_path.unlink()
            except OSError:
                pass
            if labels_filename:
                try:
                    data_dir.joinpath(labels_filename).unlink()
                except OSError:
                    pass

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
