from datetime import datetime, timedelta, timezone

from sqlalchemy import distinct, func
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

    def get_hourly_activity_counts(
        self, days: int = 7
    ) -> tuple[dict[datetime, int], dict[datetime, int]]:
        """
        Aggregated hourly counts over the last `days` days, computed in SQL so the
        result set is bounded by ~(days * 24) rows instead of every annotation/log row.

        Returns (annotations_by_hour, distinct_users_by_hour). Hours are UTC,
        aligned on the hour start.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        session = self.Session()
        try:
            dialect = session.bind.dialect.name if session.bind is not None else ""

            # Build a portable hour-bucket expression.
            # Postgres: date_trunc returns a timestamp; SQLite has no date_trunc,
            # so we use strftime which returns a 'YYYY-MM-DD HH' text token.
            if dialect == "postgresql":
                hour_ann = func.date_trunc("hour", Annotations.time)
                hour_log = func.date_trunc("hour", Logs.time)
            else:
                hour_ann = func.strftime("%Y-%m-%d %H", Annotations.time)
                hour_log = func.strftime("%Y-%m-%d %H", Logs.time)

            annotations_rows = (
                session.query(hour_ann.label("h"), func.count().label("c"))
                .filter(Annotations.time >= cutoff)
                .group_by(hour_ann)
                .all()
            )
            logs_rows = (
                session.query(
                    hour_log.label("h"),
                    func.count(distinct(Logs.user_name)).label("c"),
                )
                .filter(Logs.time >= cutoff)
                .group_by(hour_log)
                .all()
            )
        finally:
            session.close()

        def to_utc_hour(h) -> datetime:
            if isinstance(h, str):
                # SQLite path: 'YYYY-MM-DD HH'
                dt = datetime.strptime(h, "%Y-%m-%d %H")
            else:
                dt = h
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)

        annotations_by_hour = {to_utc_hour(h): int(c) for h, c in annotations_rows}
        users_by_hour = {to_utc_hour(h): int(c) for h, c in logs_rows}
        return annotations_by_hour, users_by_hour
