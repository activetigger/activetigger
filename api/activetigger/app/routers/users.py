from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from activetigger.app.dependencies import (
    ProjectAction,
    ServerAction,
    oauth2_scheme,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    AuthActions,
    AuthUserModel,
    ChangePasswordModel,
    NewUserModel,
    UserInDBModel,
    UserModel,
    UserStatistics,
)
from activetigger.orchestrator import get_orchestrator

router = APIRouter(tags=["users"])


@router.post("/users/disconnect", dependencies=[Depends(verified_user)], tags=["users"])
def disconnect_user(token: Annotated[str, Depends(oauth2_scheme)]) -> None:
    """
    Revoke user connexion
    """
    try:
        get_orchestrator().revoke_access_token(token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users/me", tags=["users"])
def read_users_me(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> UserModel:
    """
    Information on current user
    """
    try:
        return UserModel(username=current_user.username, status=current_user.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users", tags=["users"])
def existing_users(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> dict[str, UserModel]:
    """
    Get existing users
    """
    try:
        return get_orchestrator().users.existing_users(current_user.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users/recent", tags=["users"])
def recent_users() -> int:
    """
    Get the number of recently connected users
    """
    return len(get_orchestrator().db_manager.users_service.get_current_users(300))


@router.post("/users/create", dependencies=[Depends(verified_user)], tags=["users"])
def create_user(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    new_user: NewUserModel,
) -> None:
    """
    Create user
    """
    test_rights(ServerAction.MANAGE_USERS, current_user.username)
    try:
        get_orchestrator().users.add_user(new_user, current_user.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/users/delete", dependencies=[Depends(verified_user)], tags=["users"])
def delete_user(
    current_user: Annotated[UserInDBModel, Depends(verified_user)], user_to_delete: str
) -> None:
    """
    Delete user
    - root can delete all
    - users can only delete account they created
    """
    test_rights(ServerAction.MANAGE_USERS, current_user.username)
    try:
        get_orchestrator().users.delete_user(user_to_delete, current_user.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/users/changepwd", dependencies=[Depends(verified_user)], tags=["users"])
def change_password(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    changepwd: ChangePasswordModel,
) -> None:
    """
    Change our own password for an account
    """
    try:
        get_orchestrator().users.change_password(
            current_user.username, changepwd.pwdold, changepwd.pwd1, changepwd.pwd2
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/users/auth/{action}", dependencies=[Depends(verified_user)], tags=["users"])
def set_auth(
    action: AuthActions,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    auth: AuthUserModel,
) -> None:
    """
    Modify user auth on a specific project
    """
    test_rights(ProjectAction.UPDATE, current_user.username, auth.project_slug)
    if action == "add":
        if not auth.status:
            raise HTTPException(status_code=400, detail="Missing status")
        try:
            orchestrator = get_orchestrator()
            orchestrator.users.set_auth(auth)
            orchestrator.log_action(current_user.username, f"ADD AUTH USER: {auth.username}", "all")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        return None

    if action == "delete":
        try:
            orchestrator = get_orchestrator()
            orchestrator.users.delete_auth(auth.username, auth.project_slug)
            orchestrator.log_action(
                current_user.username, f"DELETE AUTH USER: {auth.username}", "all"
            )
        except Exception as e:
            raise HTTPException(status_code=500) from e

        return None

    raise HTTPException(status_code=400, detail="Action not found")


@router.get("/users/statistics", dependencies=[Depends(verified_user)], tags=["users"])
def get_statistics(
    current_user: Annotated[UserInDBModel, Depends(verified_user)], username: str
) -> UserStatistics:
    """
    Get statistics for specific user
    """
    test_rights(ServerAction.MANAGE_USERS, current_user.username)
    try:
        return get_orchestrator().users.get_statistics(username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/users/resetpwd", tags=["users"])
def reset_password(mail: str) -> None:
    try:
        get_orchestrator().users.reset_password(mail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
