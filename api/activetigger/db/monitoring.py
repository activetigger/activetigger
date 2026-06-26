from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session as SessionType
from sqlalchemy.orm import sessionmaker

from activetigger.db.models import Annotations, Logs, Monitoring


class MonitoringService:
    Session: sessionmaker[SessionType]

    def __init__(self, sessionmaker: sessionmaker[SessionType]):
        self.Session = sessionmaker

    def add_process(
        self,
        process_name: str,
        kind: str,
        parameters: dict = {},
        events: dict = {},
        project_slug: str | None = None,
        user_name: str | None = None,
    ):
        """
        Add a new process monitoring entry.
        status: running, stopped, error
        details: additional information about the process
        """
        session = self.Session()
        process = Monitoring(
            process_name=process_name,
            kind=kind,
            time=datetime.now(timezone.utc),
            parameters=parameters,
            events=events,
            project_slug=project_slug,
            user_name=user_name,
            duration=None,
        )
        session.add(process)
        session.commit()
        session.close()

    def get_element_by_process(self, process_name: str) -> Monitoring | None:
        """
        Get the latest monitoring entry for a given process name.
        """
        session = self.Session()
        element = (
            session.query(Monitoring)
            .filter(Monitoring.process_name == process_name)
            .order_by(Monitoring.time.desc())
            .first()
        )
        session.close()
        return element

    def update_process(
        self,
        process_name: str,
        events: dict | None = None,
        parameters: dict | None = None,
        duration: float | None = None,
    ):
        """
        Update the latest monitoring entry for a given process name.
        """
        session = self.Session()

        process = (
            session.query(Monitoring)
            .filter(Monitoring.process_name == process_name)
            .order_by(Monitoring.time.desc())
            .first()
        )

        if process:
            if events is not None:
                process.events = events
            if parameters is not None:
                process.parameters = parameters
            if duration is not None:
                process.duration = duration
            session.commit()
        session.close()

    def get_completed_processes(
        self, kind: str, username: str | None, limit: int = 100
    ) -> list[Monitoring]:
        """
        Get completed processes of a given kind (duration is not None)
        - for all users
        - for a specific user
        """
        session = self.Session()
        if kind == "all" and username is None:
            processes = (
                session.query(Monitoring)
                .filter(
                    Monitoring.duration.isnot(None),
                )
                .order_by(Monitoring.time.desc())
                .limit(limit)
                .all()
            )
        elif username is None:
            processes = (
                session.query(Monitoring)
                .filter(
                    Monitoring.kind == kind,
                    Monitoring.duration.isnot(None),
                )
                .order_by(Monitoring.time.desc())
                .limit(limit)
                .all()
            )
        else:
            processes = (
                session.query(Monitoring)
                .filter(
                    Monitoring.kind == kind,
                    Monitoring.user_name == username,
                    Monitoring.duration.isnot(None),
                )
                .order_by(Monitoring.time.desc())
                .limit(limit)
                .all()
            )
        session.close()
        return processes

    def get_recent_activity(
        self, days: int = 7
    ) -> tuple[list[tuple[datetime, str]], list[tuple[datetime, str]]]:
        """
        Get raw (time, user_name) pairs from annotations and logs over the last `days` days.
        Aggregation by hour is done by the caller to stay portable across SQLite/Postgres.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        session = self.Session()
        annotations = (
            session.query(Annotations.time, Annotations.user_name)
            .filter(Annotations.time >= cutoff)
            .all()
        )
        logs = (
            session.query(Logs.time, Logs.user_name)
            .filter(Logs.time >= cutoff)
            .all()
        )
        session.close()
        return (
            [(t, u) for t, u in annotations],
            [(t, u) for t, u in logs],
        )
