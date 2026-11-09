from pydantic import BaseModel


# Generic Task inputs
class TaskInputs(BaseModel):
    project_slug: str
    user: str
