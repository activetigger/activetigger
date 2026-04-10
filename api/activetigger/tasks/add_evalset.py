from ast import Tuple
import io
from typing import Tuple
import pandas as pd  # type: ignore[import]
from fastapi.encoders import jsonable_encoder  # type: ignore[import]

from activetigger.tasks.base_task import BaseTask
from activetigger.config import config
from activetigger.datamodels import EvalSetDataModel,ProjectBaseModel
from activetigger.functions import slugify


class AddEvalSet(BaseTask):
    """
    Add an evaluation set to the project
    """
    Kind = "add_evalset"

    def __init__(self,
                dataset: str,
                evalset: EvalSetDataModel,
                project: ProjectBaseModel,
                username: str,
                project_slug: str,
                index: pd.DataFrame,
                scheme=None,
                ):
        
        super().__init__()
        self.evalset = evalset
        self.Project = project
        self.dataset = dataset
        self.username = username
        self.index_all= index
        self.scheme = scheme
        self.project_slug = project_slug
        self.elements=None
    def __stop_process_opportunity(self):
        if self.event is not None and self.event.is_set():
            file_name = config.test_file if self.dataset == "test" else config.valid_file
            self.Project.dir.joinpath(file_name).unlink(missing_ok=True)
            raise Exception ("Adding evaluation set process interrupted by user")
    def __call__(self) -> Tuple[Tuple[str,str,str,str,list],ProjectBaseModel]:
        try:
            self.__stop_process_opportunity()
            csv_buffer=io.StringIO(self.evalset.csv)
            df=pd.read_csv(
                csv_buffer,
                dtype={self.evalset.col_id: str, **{col: str for col in self.evalset.cols_text}},
                nrows=self.evalset.n_eval,
            )
            if len(df) > 10000:
                raise Exception("You valid set is too large")
            #added a check if DF is empty to avoid errors 
            if len(df) == 0:
                raise Exception("Your valid set is empty")
            
            #stop Process
            self.__stop_process_opportunity()
            # create text column
            df["text"]=df[self.evalset.cols_text].apply(
                lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
            )
            if not self.evalset.col_label:
                df = df.rename(columns={self.evalset.col_id: "id"})
            else:
                df = df.rename(
                    columns={
                        self.evalset.col_id: "id",
                        self.evalset.col_label: "label",
                    }
                )
            df["label"] = df["label"].apply(lambda x: str(x) if pd.notna(x) else None)
            # deal with non-unique id
            df["id_external"] = df["id"].apply(str)
            if not ((df["id"].astype(str).apply(slugify)).nunique() == len(df)):
                df["id"] = [str(i) for i in range(len(df))]
                print("ID not unique, changed to default id")  
            #=================== Compare with the general dataset (Overlaps) ===================
            #                    ============================================        
            overlapping_ids = set(df["id"]).intersection(set(self.index_all))
            if overlapping_ids:
                df.loc[df["id"].isin(overlapping_ids), "id"] = [
                    'c-ev-'+str(i) for i in range(len(overlapping_ids))
                    ]
                print(f"{len(overlapping_ids)} IDs in the eval set already exist in the main dataset") 
            df["id"] = df["id"].apply(lambda x: f"imported-{str(x)}")
            df = df.set_index("id")
            if self.evalset.col_label and self.evalset.scheme:
                # Check the label columns if they match the scheme or raise error
                for label in df["label"].dropna().unique():
                    if label not in self.scheme:
                        raise Exception(f"Label {label} not in the scheme {self.evalset.scheme}")
            #stop Process
            self.__stop_process_opportunity()
            #write to parquet
            if self.dataset in ("test", "valid") and len(df) > 0:
                file_name = config.test_file if self.dataset == "test" else config.valid_file
                df[["id_external", "text"]].to_parquet(self.Project.dir.joinpath(file_name))
                self.Project.__dict__[f"{self.dataset}"] = True 
                self.Project.__dict__[f"n_{self.dataset}"] = len(df)
            else:
                raise Exception("Dataset should be test or valid")
            #stop Process
            self.__stop_process_opportunity()
            # import labels if specified + scheme // check if the labels are in the scheme
            if self.Project.__dict__[f"{self.dataset}"] :
                if self.evalset.col_label and self.evalset.scheme:
                    for label in df["label"].dropna().unique():
                        if label not in self.scheme:
                            raise Exception(f"Label {label} not in the scheme {self.evalset.scheme}")
                self.elements = [
                        {"element_id": element_id, "annotation": label, "comment": ""}
                        for element_id, label in df["label"].dropna().items()
                        ]
            args=(self.dataset,self.username,self.project_slug,self.evalset.scheme,self.elements)   
        except Exception as e:
            print(e)
            raise e 
        return args,self.Project

                    
                
                
                
        
        
        
        