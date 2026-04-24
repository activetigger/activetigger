import secrets
from datetime import datetime, timezone
from pathlib import Path

import yaml

from activetigger.config import config
from activetigger.datamodels import (
    AuthUserModel,
    DatasetModel,
    NewUserModel,
    ProjectModel,
    ProjectSummaryModel,
    UserInDBModel,
    UserModel,
    UsersStateModel,
    UserStatistics,
)
from activetigger.db.manager import DatabaseManager
from activetigger.functions import compare_to_hash, get_dir_size, get_hash
from activetigger.messages import Messages


class Users:
    """
    Managers users
    """

    db_manager: DatabaseManager
    users: dict
    failed_attemps: dict[str, list[datetime]]
    messages: Messages

    def __init__(
        self,
        db_manager: DatabaseManager,
        messages: Messages,
        file_users: str = "users.yaml",
    ):
        """
        Init users references
        """
        self.db_manager = db_manager
        self.messages = messages

        # add specific users parameters if they exist
        self.users = {}
        if Path(file_users).exists():
            with open(file_users) as f:
                self.users = yaml.safe_load(f)
        self.failed_attemps: dict = {}

    def log_failed_login_attempt(self, username: str) -> None:
        """
        Register the timestamp of a failed login attempt
        Raise an error if there are too many failed attempts in a short period
        """
        if username not in self.failed_attemps:
            self.failed_attemps[username] = []
        self.failed_attemps[username].append(datetime.now(timezone.utc))

    def check_failed_login_attempts(
        self, username: str, timewindow: int = 10, max_attempts: int = 3
    ) -> None:
        """
        Check if there are too many failed login attempts in a short period
        Raise an error if there are too many failed attempts in a short period
        """
        if username not in self.failed_attemps:
            return
        # filter attempts in the timewindow
        now = datetime.now(timezone.utc)
        self.failed_attemps[username] = [
            t for t in self.failed_attemps[username] if (now - t).total_seconds() < timewindow
        ]
        if len(self.failed_attemps[username]) >= max_attempts:
            raise Exception("Too many failed login attempts. Please try again after a few minutes.")

    def get_project_auth(self, project_slug: str) -> dict[str, str]:
        """
        Get user auth for a project
        """
        return self.db_manager.projects_service.get_project_auth(project_slug)

    def set_auth(self, auth: AuthUserModel) -> None:
        """
        Set user auth for a project
        """
        if auth.status is None:
            raise Exception("Missing status")
        self.get_user(auth.username)
        self.db_manager.projects_service.add_auth(auth.project_slug, auth.username, auth.status)

    def delete_auth(self, username: str, project_slug: str) -> None:
        """
        Delete user auth
        """
        if username == "root":
            raise Exception("Can't delete root user auth")
        self.get_user(username)
        self.db_manager.projects_service.delete_auth(project_slug, username)

    def get_auth_projects(self, username: str, auth: str | None = None) -> list:
        """
        Get user auth
        """
        return self.db_manager.projects_service.get_user_auth_projects(username, auth)

    def get_auth(self, username: str, project_slug: str = "all") -> list:
        """
        Get user auth
        Comments:
        - Either for all projects
        - Or one project
        """
        if project_slug == "all":
            auth = self.db_manager.projects_service.get_user_auth(username)
        else:
            auth = self.db_manager.projects_service.get_user_auth(username, project_slug)
        return auth

    def existing_users(self, username: str = "root", active: bool = True) -> dict[str, UserModel]:
        """
        Get existing users which have been created by one user
        (except root which can't be modified)
        TODO : better rules
        """
        if username == "root":
            return self.db_manager.users_service.get_users_created_by("all", active)
        else:
            # TODO : in the future, allows users in project related to the account ?
            return {}

    def add_user(
        self,
        new_user: NewUserModel,
        created_by: str,
    ) -> None:
        """
        Add user to database
        Comments:
            Default, users are managers
        """
        # test if the user doesn't exist, even among deactivated users
        if new_user.username in self.existing_users(active=False):
            raise Exception("Username already exists")
        hash_pwd = get_hash(new_user.password)
        self.db_manager.users_service.add_user(
            new_user.username,
            hash_pwd.decode("utf8"),
            new_user.status,
            created_by,
            contact=new_user.contact,
        )

    def delete_user(self, user_to_delete: str, username: str) -> None:
        """
        Deleting user
        """
        # test specific rights
        if user_to_delete == "root":
            raise Exception("Can't delete root user")
        if user_to_delete not in self.existing_users():
            raise Exception("Username does not exist")
        if user_to_delete not in self.existing_users(username):
            raise Exception("You don't have the right to delete this user")

        # delete the user
        self.db_manager.users_service.delete_user(user_to_delete)

    def get_user(self, name: str) -> UserInDBModel:
        """
        Get active user from database
        """
        if name not in self.existing_users():
            raise Exception("Username doesn't exist or is deactivated")
        user = self.db_manager.users_service.get_user(name)
        return UserInDBModel(username=name, hashed_password=user.key, status=user.description)

    def authenticate_user(self, username: str, password: str) -> UserInDBModel:
        """
        User authentification
        - Check too many failed login attempts
        - Check username/password
        """
        self.check_failed_login_attempts(username)
        try:
            user = self.get_user(username)
            if not compare_to_hash(password, user.hashed_password):
                raise Exception("Wrong password")
            return user
        except Exception:
            self.log_failed_login_attempt(username)
            raise Exception("Wrong username or password")

    def auth(self, username: str, project_slug: str) -> str | None:
        """
        Check auth for a specific project
        """
        user_auth = self.get_auth(username, project_slug)
        if len(user_auth) == 0:  # not associated
            return None
        return user_auth[0][1]

    def change_password(
        self, username: str, password_old: str, password1: str, password2: str
    ) -> None:
        """
        Change password for a user
        """
        if password1 != password2:
            raise Exception("Passwords don't match")
        user = self.get_user(username)
        if not compare_to_hash(password_old, user.hashed_password):
            raise Exception("Wrong password")
        hash_pwd = get_hash(password1)
        self.db_manager.users_service.change_password(username, hash_pwd.decode("utf8"))
        return None

    def force_change_password(self, username: str, password: str) -> None:
        """
        Force change password for a user (no old password needed)
        """
        hash_pwd = get_hash(password)
        self.db_manager.users_service.change_password(username, hash_pwd.decode("utf8"))

    def get_statistics(self, username: str) -> UserStatistics:
        """
        Get statistics for specific user
        """
        try:
            projects = {i[0]: i[1] for i in self.get_auth_projects(username)}
            return UserStatistics(username=username, projects=projects)
        except Exception as e:
            raise Exception(f"Error in getting statistics for {username}") from e

    def get_storage(self, username: str) -> float:
        """
        Get total size for user projects in GB
        """
        projects = self.db_manager.users_service.get_user_created_projects(username)
        size_mb = sum(
            [get_dir_size(f"{config.data_path}/projects/{project}") for project in projects]
        )
        return size_mb / 1024

    def get_storage_limit(self, username: str) -> float:
        """
        Get storage limit for user
        TODO : add a list of exceptions
        """
        # case of root
        if username == "root":
            return 500.0
        # derogation for specific users
        if username in self.users:
            return float(self.users[username]["storage_limit"])
        # default value
        return config.user_hdd_max

    def state(self, project_slug: str) -> UsersStateModel:
        """
        Get last annotation date for all users
        """
        r = self.db_manager.users_service.get_project_users_last_annotation(project_slug)
        return UsersStateModel(
            users=list(r.keys()),
            last_schemes={i: r[i].scheme for i in r},
        )

    def get_auth_datasets(self, username: str) -> list[DatasetModel]:
        """
        Get datasets authorized for the user (projects where is manager)
        """
        projects = self.get_auth_projects(username, auth="manager")
        return [
            DatasetModel(
                project_slug=p[2]["project_slug"],
                columns=p[2]["all_columns"],
                n_rows=p[2]["n_total"],
            )
            for p in projects
        ]

    def get_user_projects(self, username: str) -> list[ProjectSummaryModel]:
        """
        Get projects authorized for the user
        """
        projects_auth = self.get_auth_projects(username)
        projects = []
        for i in list(reversed(projects_auth)):
            # get the project slug
            project_slug = i[0]
            user_right = i[1]
            parameters = self.db_manager.projects_service.get_project(project_slug)
            if parameters is None:
                continue
            parameters = ProjectModel(**parameters["parameters"])
            created_by = i[3]
            created_at = i[4].strftime("%Y-%m-%d %H:%M:%S")
            try:
                size = round(get_dir_size(config.data_path + "/projects/" + i[0]), 1)
            except Exception as e:
                print(e)
                size = 0.0
            last_activity = self.db_manager.logs_service.get_last_activity_project(i[0])

            # create the project summary model
            projects.append(
                ProjectSummaryModel(
                    project_slug=project_slug,
                    user_right=user_right,
                    parameters=parameters,
                    created_by=created_by,
                    created_at=created_at,
                    size=size,
                    last_activity=last_activity,
                )
            )
        return projects

    def get_all_projects(self, username: str) -> list[ProjectSummaryModel]:
        """
        Get all existing projects regardless of auth (admin view).
        user_right reflects the given user's auth on each project, or "none".
        """
        slugs = self.db_manager.projects_service.existing_projects()
        projects = []
        for slug in slugs:
            project = self.db_manager.projects_service.get_project(slug)
            if project is None:
                continue
            parameters = ProjectModel(**project["parameters"])
            created_by = project["user_name"]
            created_at = project["time_created"].strftime("%Y-%m-%d %H:%M:%S")
            try:
                size = round(get_dir_size(config.data_path + "/projects/" + slug), 1)
            except Exception as e:
                print(e)
                size = 0.0
            last_activity = self.db_manager.logs_service.get_last_activity_project(slug)
            auth = self.db_manager.projects_service.get_user_auth(username, slug)
            user_right = auth[0][1] if auth else "none"
            projects.append(
                ProjectSummaryModel(
                    project_slug=slug,
                    user_right=user_right,
                    parameters=parameters,
                    created_by=created_by,
                    created_at=created_at,
                    size=size,
                    last_activity=last_activity,
                )
            )
        projects.sort(key=lambda p: p.created_at, reverse=True)
        return projects

    def reset_password(self, mail: str) -> None:
        """
        Reset password for a user with the given email
        """
        # Check if mail is connected to a user
        user_name = self.db_manager.users_service.get_user_by_mail(mail)

        # Generate a random password
        new_password = secrets.token_hex(16)

        # Send the mail to the user with the new password
        self.messages.send_mail_reset_password(user_name, mail, new_password)

        # Update the user's password in the database
        self.force_change_password(user_name, new_password)
