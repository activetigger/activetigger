from datetime import datetime, timezone

from pydantic import BaseModel


# Generic Task inputs
class TaskInputs(BaseModel):
    project_slug: str
    user: str
    time: datetime = datetime.now(timezone.utc)
