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
