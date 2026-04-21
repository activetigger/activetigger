import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    # convert to JSON the types [dict[str], Any] to be able to directly store objets
    # allow to search on JSON
    type_annotation_map = {dict[str, Any]: JSON}
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(column_0_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class Projects(Base):
    __tablename__: str = "projects"

    project_slug: Mapped[str] = mapped_column(primary_key=True)
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    time_modified: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )
    parameters: Mapped[dict[str, Any]]
    user_name: Mapped[str] = mapped_column(ForeignKey("users.user_name", ondelete="CASCADE"))
    user: Mapped["Users"] = relationship("Users")
    schemes: Mapped[list["Schemes"]] = relationship(
        "Schemes", cascade="all,delete,delete-orphan", back_populates="project"
    )
    auths: Mapped[list["Auths"]] = relationship(
        "Auths", cascade="all,delete,delete-orphan", back_populates="project"
    )
    # logs: Mapped[list["Logs"]] = relationship(
    #    "Logs", cascade="all,delete,delete-orphan", back_populates="project"
    # )
    generations: Mapped[list["Generations"]] = relationship(
        "Generations", cascade="all,delete,delete-orphan", back_populates="project"
    )
    features: Mapped[list["Features"]] = relationship(
        "Features", cascade="all,delete,delete-orphan", back_populates="project"
    )
    gen_models: Mapped[list["GenModels"]] = relationship(
        "GenModels", cascade="all, delete-orphan", back_populates="project"
    )
    prompts: Mapped[list["Prompts"]] = relationship(
        "Prompts", cascade="all,delete,delete-orphan", back_populates="project"
    )


class Users(Base):
    __tablename__ = "users"

    user_name: Mapped[str] = mapped_column(primary_key=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    key: Mapped[str]
    description: Mapped[str]
    contact: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str]
    projects: Mapped[list[Projects]] = relationship(
        back_populates="user", cascade="all,delete,delete-orphan"
    )
    deactivated: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))


class Schemes(Base):
    __tablename__ = "schemes"
    __table_args__ = (
        PrimaryKeyConstraint("project_slug", "name", name="uq_project_slug_name_schemes"),
    )

    name: Mapped[str]
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    time_modified: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )
    user_name: Mapped[str] = mapped_column(ForeignKey("users.user_name"))
    user: Mapped[Users] = relationship()
    project_slug: Mapped[str] = mapped_column(
        ForeignKey("projects.project_slug", ondelete="CASCADE")
    )
    project: Mapped[Projects] = relationship(back_populates="schemes")
    models: Mapped[list["Models"]] = relationship(passive_deletes=True, back_populates="scheme")
    params: Mapped[dict[str, Any]]


class Annotations(Base):
    __tablename__ = "annotations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_slug", "scheme_name"],
            ["schemes.project_slug", "schemes.name"],
            name="fkc_project_slug_scheme_name",
            ondelete="CASCADE",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    dataset: Mapped[str]
    user_name: Mapped[str] = mapped_column(ForeignKey("users.user_name"))
    user: Mapped[Users] = relationship()
    project_slug: Mapped[str]
    element_id: Mapped[str]
    scheme_name: Mapped[str]
    scheme: Mapped[Schemes] = relationship()
    annotation: Mapped[str | None]
    comment: Mapped[str | None] = mapped_column(Text)
    selection: Mapped[str | None] = mapped_column(Text)


class Auths(Base):
    __tablename__ = "auth"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_name: Mapped[str] = mapped_column(ForeignKey("users.user_name"))
    user: Mapped[Users] = relationship()
    project_slug: Mapped[str] = mapped_column(
        ForeignKey("projects.project_slug", ondelete="CASCADE")
    )
    project: Mapped[Projects] = relationship(back_populates="auths")
    status: Mapped[str]
    created_by: Mapped[str | None]


class Logs(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    user_name: Mapped[str] = mapped_column(ForeignKey("users.user_name"))
    user: Mapped[Users] = relationship()
    project_slug: Mapped[str]
    action: Mapped[str | None]
    connect: Mapped[str | None]


class Tokens(Base):
    __tablename__ = "tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    token: Mapped[str]
    status: Mapped[str]
    time_revoked: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))


class GenModels(Base):
    __tablename__ = "gen_models"
    __table_args__ = (
        UniqueConstraint(
            "project_slug",
            "name",
            name="fkc_project_slug_name",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_slug: Mapped[str] = mapped_column(
        ForeignKey("projects.project_slug", ondelete="CASCADE")
    )
    user_name: Mapped[str] = mapped_column(ForeignKey("users.user_name"))
    project: Mapped[Projects] = relationship(back_populates="gen_models")
    slug: Mapped[str]
    name: Mapped[str]
    api: Mapped[str]
    endpoint: Mapped[str | None]
    credentials: Mapped[str | None]


class Generations(Base):
    __tablename__ = "generations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    user_name: Mapped[str] = mapped_column(ForeignKey("users.user_name"))
    user: Mapped[Users] = relationship()
    project_slug: Mapped[str] = mapped_column(
        ForeignKey("projects.project_slug", ondelete="CASCADE")
    )
    project: Mapped[Projects] = relationship(back_populates="generations")
    element_id: Mapped[str]
    model_id: Mapped[int] = mapped_column(ForeignKey("gen_models.id"))
    model: Mapped[GenModels] = relationship()
    prompt: Mapped[str]
    answer: Mapped[str]
    batch: Mapped[str | None]


class Features(Base):
    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    user_name: Mapped[str] = mapped_column(ForeignKey("users.user_name"))
    user: Mapped[Users] = relationship()
    project_slug: Mapped[str] = mapped_column(
        ForeignKey("projects.project_slug", ondelete="CASCADE")
    )
    project: Mapped[Projects] = relationship(back_populates="features")
    name: Mapped[str]
    kind: Mapped[str]
    parameters: Mapped[dict[str, Any]]
    data: Mapped[dict[str, Any]]


class Models(Base):
    __tablename__ = "models"
    __table_args__ = (
        PrimaryKeyConstraint("project_slug", "name", name="uq_project_slug_name_models"),
        ForeignKeyConstraint(
            ["project_slug", "scheme_name"],
            ["schemes.project_slug", "schemes.name"],
            name="fkc_project_slug_scheme_name",
            ondelete="CASCADE",
        ),
    )
    name: Mapped[str]
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    time_modified: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )
    user_name: Mapped[str] = mapped_column(ForeignKey("users.user_name"))
    user: Mapped[Users] = relationship()
    project_slug: Mapped[str]
    scheme: Mapped[Schemes] = relationship(back_populates="models")
    scheme_name: Mapped[str]
    kind: Mapped[str]
    parameters: Mapped[dict[str, Any]]
    path: Mapped[str]
    status: Mapped[str]
    statistics: Mapped[str | None]
    test: Mapped[str | None]


class Prompts(Base):
    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    time_modified: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )
    user_name: Mapped[str] = mapped_column(ForeignKey("users.user_name"))
    user: Mapped[Users] = relationship()
    project_slug: Mapped[str] = mapped_column(
        ForeignKey("projects.project_slug", ondelete="CASCADE")
    )
    project: Mapped[Projects] = relationship(back_populates="prompts")
    value: Mapped[str]
    parameters: Mapped[dict[str, Any]]


class Messages(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_by: Mapped[str]
    kind: Mapped[str]
    content: Mapped[str]
    property: Mapped[dict[str, Any]] = mapped_column(JSON)
    for_project: Mapped[str | None] = mapped_column(
        ForeignKey("projects.project_slug", ondelete="CASCADE")
    )
    for_user: Mapped[str | None] = mapped_column(ForeignKey("users.user_name"))


class Monitoring(Base):
    __tablename__ = "monitoring"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_name: Mapped[str | None]
    process_name: Mapped[str]
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    project_slug: Mapped[str] = mapped_column(
        ForeignKey("projects.project_slug", ondelete="CASCADE")
    )
    kind: Mapped[str]
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON)
    events: Mapped[dict[str, Any]] = mapped_column(JSON)
    duration: Mapped[float | None]
