from activetigger.tasks.base_task import BaseTask
from activetigger.datamodels import EvalSetDataModel,ProjectModel
import pandas as pd
import io
from activetigger.functions import slugify
from fastapi.encoders import jsonable_encoder
from activetigger.config import config





class AddEvalsets(BaseTask):
    """
    Class responsable for adding Evalset to the queue
    Added in the queue
    
    """
    
    kind="add_eval_set"
    def __init__(self,
                 project:ProjectModel,
                 dataset:str,
                 evalset:EvalSetDataModel,
                 username:str,
                 project_slug:str,
                 train_index:list[str],
                 scheme:dict[str,list],
                 
                 ):
        super().__init__()
        self.project=project
        self.dataset=dataset
        self.evalset=evalset
        self.username=username
        self.project_slug=project_slug
        self.train_indexs=train_index
        self.scheme=scheme
        self.elements=None
        self.warnings=[]
    def _add_evalset(
        self
    ) -> None:
        """
        Add a eval dataset (test or valid)

        The eval dataset should :
        - not contains NA
        - have a unique id different from the complete dataset
        - No overlap with trainset

        The id will be modified to indicate imported

        #TODO : put this task in the queue -> Done (Please after Approval remove this)
        
        """
        #### Addition :
        # 1- add overlap verification 
        # 3- ID columns is valid/unique
        # 4- No duplicate IDs within the eval set
        ###################
        if len(self.evalset.cols_text) == 0:
            raise Exception("No text column selected for the evalset")
        if self.project.dir is None:
            raise Exception("Cannot add eval data without a valid dir")

        if self.evalset.col_label == "":
            self.evalset.col_label = None

        if self.dataset not in ["test", "valid"]:
            raise Exception("Dataset should be test or valid")

        if self.dataset == "test" and self.project.test:
            raise Exception("There is already a test dataset")
        
        if self.dataset == "valid" and self.project.valid:
            raise Exception("There is already a valid dataset")
        ###--------Importing the CSV File
        csv_buffer = io.StringIO(self.evalset.csv)
        try:
            df = pd.read_csv(
                csv_buffer,
                dtype={self.evalset.col_id: str, **{col: str for col in self.evalset.cols_text}},
                nrows=self.evalset.n_eval,
            )
        #the CSV file parsing error is now caught
        except Exception as e:
            raise Exception(f"Can't Parse the CSV File :{e}")
        #Suppose the  File is empty 
        if len(df) == 0:
            raise Exception("The Chosen Evaluation Set is Empty")
        if len(df) > 10000:
            raise Exception(f"The Evaluation set is too large : {len(df)} rows > 10^4 ")
        #======UniqueId=====
        if df[self.evalset.col_id].isnull().any():
            raise Exception("The Column of ID contains Null values")
        n_dup=df[self.evalset.col_id].duplicated()
        if n_dup.any():
            raise Exception (f"Columns ID has {n_dup.sum()} duplicates")   
        # create text column
        df["text"] = df[self.evalset.cols_text].apply(
            lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
        )
        #So  we will check for overlaps:
        try :
            train_ids=set(map(slugify,self.train_indexs))
            eval_id=set(df[self.evalset.col_id].astype(str).map(slugify))
            ovelaps=train_ids.intersection(eval_id)
            if ovelaps:
                self.warnings.append(
                    f"{len(ovelaps)} eval ID(s) already exist in the train set — "
                    f"data leakage risk, metrics may be unreliable. "
                    f"Examples: {list(ovelaps)[:3]}"
                )
        except Exception as e:
            #The overlap error is hard failure since both the new eval set and train set
            #both contains any|all same values 
            if "evalID(s) already existi in the trainset" in str(e):#the sentence use in the error of overlap
                raise
            print(f"can't run ovelap check : {e}") 
        # Between raising Error Or warning : If Error -> would create a design default , if warning this will keeep the app working 
        # So in this case we will either drop overlps or give user chance to change eval set
        # Now change names
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
        # compare with the general dataset
        # check overlpas between Train and evaluation set is done before Renaming
        df["id_external"] = df["id"].apply(str)
        if not ((df["id"].astype(str).apply(slugify)).nunique() == len(df)):
            df["id"] = [str(i) for i in range(len(df))]
            print("ID not unique, changed to default id")
        # identify the dataset as imported and set the id
        df["id"] = df["id"].apply(lambda x: f"imported-{str(x)}")
        df = df.set_index("id")
        #elements = None
        # import labels if specified + scheme // check if the labels are in the scheme
        if self.evalset.col_label and self.evalset.scheme:
            # Check the label columns if they match the scheme or raise error
            #scheme = self.schemes.available()[self.evalset.scheme].labels
            for label in df["label"].dropna().unique():
                if label not in self.scheme:
                    raise Exception(f"Label {label} not in the scheme {self.evalset.scheme}")

            self.elements = [
                {"element_id": element_id, "annotation": label, "comment": ""}
                for element_id, label in df["label"].dropna().items()
            ]
            #This is annotation 
            # self.project.db_manager.projects_service.add_annotations(
            #     dataset=dataset,
            #     user_name=username,
            #     project_slug=project_slug,
            #     scheme=evalset.scheme,
            #     elements=elements,
            # )
            # print("Valid labels imported")

        # write the dataset
        if self.dataset == "test":
            df[["id_external", "text"]].to_parquet(self.project.dir.joinpath(config.test_file))
            self.project.test = True
            #self.project.data.load_dataset("test")
            self.project.n_test=len(df)
        elif self.dataset == "valid":
            df[["id_external", "text"]].to_parquet(self.project.dir.joinpath(config.valid_file))
            self.project.valid = True
            #self.project.data.load_dataset("valid")
            self.project.n_valid=len(df)
        # else:
        #     raise Exception("Dataset should be test or valid")
        # Lines[460-467] already dealt with it 

        # update the database
        # self.db_manager.projects_service.update_project(
        #     self.params.project_slug, jsonable_encoder(self.params)
        # )

        # # reset the features file
        # self.project.features.reset_features_file()
        # self.project.quickmodels.drop_models(which="all")
        return {
            "dataset":self.dataset,
            "n_eval_rows":len(df),
            "elements":self.elements,
            "scheme":self.evalset.scheme,
            "project_params":self.project,
            "project_slug":self.project_slug,
            "username":self.username,
            "warnings":self.warnings,   
        }
    def __call__(self):
        return self._add_evalset()
        