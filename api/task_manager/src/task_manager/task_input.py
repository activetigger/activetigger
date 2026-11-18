from datetime import datetime, timezone
from typing import Generic, TypeVar

from pydantic import BaseModel

ProjectSlugGeneric = TypeVar("ProjectSlugGeneric", bound=str | None)


# Generic Task inputs
class TaskInputs(BaseModel, Generic[ProjectSlugGeneric]):
    project_slug: ProjectSlugGeneric
    user: str
    time: datetime = datetime.now(timezone.utc)
