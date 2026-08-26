import uuid

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from activetigger.config import config
from activetigger.db import DBException
from activetigger.db.generations import GenerationsService
from activetigger.db.languagemodels import ModelsService
from activetigger.db.logs import LogsService
from activetigger.db.messages import MessagesService
from activetigger.db.models import Base
from activetigger.db.monitoring import MonitoringService
from activetigger.db.projects import ProjectsService
from activetigger.db.users import UsersService
from activetigger.functions import get_hash, get_root_pwd


def set_sqlite_pragma(dbapi_connection, _):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    # WAL lets readers proceed while a writer holds the lock; the default
    # rollback-journal mode blocks every read during any write, which under
    # multi-user load surfaces as "database is locked" errors
    cursor.execute("PRAGMA journal_mode=WAL")
    # with WAL, NORMAL is durable enough and avoids an fsync per commit
    cursor.execute("PRAGMA synchronous=NORMAL")
    # wait up to 15s for a lock instead of failing after the 5s default
    cursor.execute("PRAGMA busy_timeout=15000")
    cursor.close()


class DatabaseManager:
    """
    Database management with SQLAlchemy
    """

    engine: Engine
    SessionMaker: sessionmaker[Session]
    default_user: str
    users_service: UsersService
    projects_service: ProjectsService

    def __init__(self):
        # priority to environ if set
        db_url = config.database_url

        # connect the session
        print(f"connecting to DB ${db_url}")
        self.engine = create_engine(db_url)

        # enable foreign key verification in sqlite
        if db_url.startswith("sqlite"):
            event.listen(self.engine, "connect", set_sqlite_pragma)

        self.SessionMaker = sessionmaker(bind=self.engine)
        self.default_user = "server"
        self.users_service = UsersService(self.SessionMaker)
        self.projects_service = ProjectsService(self.SessionMaker)
        self.generations_service = GenerationsService(self.SessionMaker)
        self.language_models_service = ModelsService(self.SessionMaker)
        self.logs_service = LogsService(self.SessionMaker)
        self.messages_service = MessagesService(self.SessionMaker)
        self.monitoring_service = MonitoringService(self.SessionMaker)

        # Create tables under an advisory lock to prevent race conditions
        # when multiple uvicorn workers start concurrently.
        # pg_advisory_xact_lock is released when the transaction commits,
        # ensuring the second process sees the committed tables.
        if db_url.startswith("sqlite"):
            Base.metadata.create_all(self.engine)
        else:
            with self.engine.begin() as conn:
                conn.execute(text("SELECT pg_advisory_xact_lock(1)"))
                Base.metadata.create_all(conn)

        # create_all only creates missing tables — it never adds indexes to
        # tables that already exist, and this repo does not run alembic. These
        # indexes back the hottest read paths (token lookup on every request,
        # activity queries on the ever-growing logs table); without them each
        # query is a full-table scan that gets slower over the instance's life.
        with self.engine.begin() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_tokens_token ON tokens (token)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_logs_time ON logs (time)"))
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_logs_project_slug_time "
                    "ON logs (project_slug, time)"
                )
            )

        self._create_default_users()

    def _create_default_users(self) -> None:
        """Create default users if they don't exist yet."""
        for user, create_fn in [
            ("system", self.create_system_session),
            ("root", self.create_root_session),
            ("demo", self.create_demo_session),
        ]:
            try:
                _ = self.users_service.get_user(user)
            except DBException:
                try:
                    create_fn()
                except IntegrityError:
                    pass  # another worker created it first

    def create_root_session(self) -> None:
        """
        Create root session
        :return: None
        """
        pwd: str = config.root_password if config.root_password is not None else get_root_pwd()
        hash_pwd: bytes = get_hash(pwd)
        self.users_service.add_user("root", hash_pwd.decode("utf8"), "root", "system")

    def create_system_session(self) -> None:
        """
        Create root session
        :return: None
        """
        # use a random password
        pwd: str = str(uuid.uuid4())
        hash_pwd: bytes = get_hash(pwd)
        self.users_service.add_user("system", hash_pwd.decode("utf8"), "system", "system")

    def create_demo_session(self) -> None:
        """
        Create demo session
        :return: None
        """
        # use a random password
        pwd: str = "demo"
        hash_pwd: bytes = get_hash(pwd)
        self.users_service.add_user("demo", hash_pwd.decode("utf8"), "demo", "system")
