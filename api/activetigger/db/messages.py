from datetime import datetime, timezone

from sqlalchemy.orm import Session as SessionType
from sqlalchemy.orm import sessionmaker

from activetigger.db.models import Messages


class MessagesService:
    Session: sessionmaker[SessionType]

    def __init__(self, sessionmaker: sessionmaker[SessionType]):
        self.Session = sessionmaker

    def add_message(
        self,
        user_name: str,
        content: str,
        kind: str,
        property: dict | None = None,
        for_project: str | None = None,
        for_user: str | None = None,
    ):
        """
        kind: system, project, user
        for_user : user name if kind is user
        for_project : project slug if kind is project
        """
        session = self.Session()
        if not property:
            property = {}
        message = Messages(
            created_by=user_name,
            time=datetime.now(timezone.utc),
            content=content,
            kind=kind,
            property=property,
            for_project=for_project,
            for_user=for_user,
        )
        session.add(message)
        session.commit()
        session.close()

    def add_messages_bulk(
        self,
        user_name: str,
        content: str,
        kind: str,
        recipients: list[str],
        for_project: str | None = None,
        property: dict | None = None,
    ) -> int:
        """
        Insert one message row per recipient in a single transaction.
        Used by both project-distribution (N rows) and DM (1 row) flows.
        Returns the number of rows inserted.
        """
        if not recipients:
            return 0
        if not property:
            property = {}
        now = datetime.now(timezone.utc)
        session = self.Session()
        try:
            session.add_all(
                [
                    Messages(
                        created_by=user_name,
                        time=now,
                        content=content,
                        kind=kind,
                        property=property,
                        for_project=for_project,
                        for_user=recipient,
                    )
                    for recipient in recipients
                ]
            )
            session.commit()
            return len(recipients)
        finally:
            session.close()

    def delete_message(self, id: int):
        """
        Delete a message by its ID.
        """
        session = self.Session()
        message = session.query(Messages).filter(Messages.id == id).first()
        if message:
            session.delete(message)
            session.commit()
        session.close()

    def delete_message_for_user(self, id: int, user_name: str) -> bool:
        """
        Delete a message only if it is addressed to the given user
        (i.e. row.for_user == user_name). Returns False if no such row
        exists, so the caller can map that to a 403/404.
        """
        session = self.Session()
        try:
            message = (
                session.query(Messages)
                .filter(Messages.id == id, Messages.for_user == user_name)
                .first()
            )
            if message is None:
                return False
            session.delete(message)
            session.commit()
            return True
        finally:
            session.close()

    def get_messages_system(self, from_user: str | None = None) -> list[Messages]:
        """
        Get all system messages ordered by time desc.
        Optionally filter by creator.
        """
        session = self.Session()
        query = session.query(Messages).filter(Messages.kind == "system")

        if from_user:
            query = query.filter(Messages.created_by == from_user)

        messages = query.order_by(Messages.time.desc()).all()
        session.close()
        return messages

    def get_messages_for_project(
        self, project_slug: str, from_user: str | None = None
    ) -> list[Messages]:
        """
        Get all project messages for a specific project ordered by time desc
        """
        session = self.Session()
        messages = (
            session.query(Messages)
            .filter(Messages.kind == "project", Messages.for_project == project_slug)
            .order_by(Messages.time.desc())
            .all()
        )
        session.close()
        return messages

    def get_messages_for_user(self, user_name: str, from_user: str | None = None) -> list[Messages]:
        """
        Get all user messages for a specific user ordered by time desc
        """
        session = self.Session()
        messages = (
            session.query(Messages)
            .filter(Messages.kind == "user", Messages.for_user == user_name)
            .order_by(Messages.time.desc())
            .all()
        )
        session.close()
        return messages

    def get_inbox_for_user(self, user_name: str) -> list[Messages]:
        """
        Inbox for a user: every message row addressed to them, whether it is a
        direct message (kind='user') or a project-distribution copy
        (kind='project'). Ordered by time desc.
        """
        session = self.Session()
        try:
            return (
                session.query(Messages)
                .filter(
                    Messages.for_user == user_name,
                    Messages.kind.in_(("user", "project")),
                )
                .order_by(Messages.time.desc())
                .all()
            )
        finally:
            session.close()

    def get_codebook_messages(self, user_name: str, project_slug: str) -> list[Messages]:
        """
        Project-distribution copies still owned by `user_name` for a given
        project. Used to surface project messages on the Codebook page.
        """
        session = self.Session()
        try:
            return (
                session.query(Messages)
                .filter(
                    Messages.kind == "project",
                    Messages.for_project == project_slug,
                    Messages.for_user == user_name,
                )
                .order_by(Messages.time.desc())
                .all()
            )
        finally:
            session.close()
