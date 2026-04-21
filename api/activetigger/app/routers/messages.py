from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from activetigger.app.dependencies import (
    ServerAction,
    test_rights,
    verified_user,
)
from activetigger.datamodels import MessagesInModel, MessagesOutModel, UserInDBModel
from activetigger.orchestrator import get_orchestrator

router = APIRouter(tags=["messages"])


@router.get("/messages")
def get_messages(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    kind: str,
    from_user: str | None = None,
    for_user: str | None = None,
    for_project: str | None = None,
) -> list[MessagesOutModel]:
    """
    Get messages
    - all if root
    - only for oneself
    """
    try:
        orchestrator = get_orchestrator()
        if current_user.username == "root":
            return orchestrator.messages.get_messages(kind, from_user, for_user, for_project)
        else:
            return orchestrator.messages.get_messages(
                kind, from_user, current_user.username, for_project
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/messages")
def post_message(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    message: MessagesInModel,
) -> None:
    """
    Post a new message
    """
    test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    try:
        get_orchestrator().messages.add_message(
            user_name=current_user.username, kind=message.kind, content=message.content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/messages/delete")
def delete_message(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    message_id: int,
) -> None:
    """
    Delete a message
    """
    test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    try:
        get_orchestrator().messages.delete_message(message_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
