import datetime
from datetime import timezone
from typing import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session as SessionType
from sqlalchemy.orm import joinedload, sessionmaker

from activetigger.config import config
from activetigger.datamodels import (
    GenCredentialsInput,
    GenCredentialsOut,
    GenerationParams,
    GenPipelineCreate,
    GenPipelineOut,
    PostprocessStep,
)
from activetigger.db.models import GenCredentials, Generations, GenPipelines
from activetigger.errors import NotFoundError
from activetigger.functions import decrypt, encrypt


def _credentials_out(entry: GenCredentials) -> GenCredentialsOut:
    return GenCredentialsOut(
        id=entry.id,
        kind=entry.kind,
        name=entry.name,
        endpoint=entry.endpoint,
        models=entry.models,
        last_tested=entry.last_tested,
    )


def _pipeline_out(pipeline: GenPipelines) -> GenPipelineOut:
    return GenPipelineOut(
        id=pipeline.id,
        name=pipeline.name,
        user_name=pipeline.user_name,
        scheme_name=pipeline.scheme_name,
        credentials_id=pipeline.credentials_id,
        credentials_name=pipeline.credentials.name,
        endpoint=pipeline.credentials.endpoint,
        model_slug=pipeline.model_slug,
        parameters=GenerationParams(**pipeline.parameters),
        prompt=pipeline.prompt,
        postprocess=[PostprocessStep(**s) for s in pipeline.postprocess.get("steps", [])],
        time=pipeline.time,
    )


class GenerationsService:
    """
    Database access for generative pipelines: credentials (user or
    instance level), pipelines, and runs.
    """

    Session: sessionmaker[SessionType]

    def __init__(self, sessionmaker: sessionmaker[SessionType]):
        self.Session = sessionmaker

    # ------------------------------------------------------------------
    # credentials
    # ------------------------------------------------------------------

    def list_credentials(self, user_name: str) -> list[GenCredentialsOut]:
        """
        Credentials visible to a user: their own entries + instance entries
        """
        with self.Session() as session:
            entries = session.scalars(
                select(GenCredentials)
                .where(
                    (GenCredentials.user_name == user_name) | (GenCredentials.kind == "instance")
                )
                .order_by(GenCredentials.kind, GenCredentials.name)
            ).all()
            return [_credentials_out(e) for e in entries]

    def get_credentials(self, credentials_id: int, user_name: str | None = None) -> GenCredentials:
        """
        Get one entry with the api_key decrypted.
        If user_name is given, restrict to entries visible to this user.
        """
        with self.Session() as session:
            entry = session.scalars(select(GenCredentials).filter_by(id=credentials_id)).first()
            if entry is None:
                raise NotFoundError("Credentials not found")
            if user_name is not None and entry.kind == "user" and entry.user_name != user_name:
                raise NotFoundError("Credentials not found")
            entry.api_key = decrypt(entry.api_key, config.secret_key)
            return entry

    def add_credentials(self, user_name: str, credentials: GenCredentialsInput) -> int:
        """
        Save a user entry; an existing entry with the same name is replaced
        """
        with self.Session.begin() as session:
            entry = session.scalars(
                select(GenCredentials).filter_by(
                    kind="user", user_name=user_name, name=credentials.name
                )
            ).first()
            if entry is None:
                entry = GenCredentials(
                    kind="user",
                    user_name=user_name,
                    name=credentials.name,
                    endpoint=credentials.endpoint,
                    api_key=encrypt(credentials.api_key, config.secret_key),
                )
                session.add(entry)
            else:
                entry.endpoint = credentials.endpoint
                entry.api_key = encrypt(credentials.api_key, config.secret_key)
                entry.last_tested = None
            session.flush()
            return entry.id

    def delete_credentials(self, credentials_id: int, user_name: str) -> None:
        """
        Delete a user entry (instance entries are managed by the yaml).
        Refused if a pipeline still uses it.
        """
        with self.Session.begin() as session:
            entry = session.scalars(
                select(GenCredentials).filter_by(
                    id=credentials_id, kind="user", user_name=user_name
                )
            ).first()
            if entry is None:
                raise NotFoundError("Credentials not found")
            used_by = session.scalars(
                select(GenPipelines).filter_by(credentials_id=credentials_id)
            ).first()
            if used_by is not None:
                raise Exception(
                    f"These credentials are used by the pipeline '{used_by.name}' "
                    f"of project '{used_by.project_slug}'"
                )
            session.delete(entry)

    def set_last_tested(self, credentials_id: int) -> None:
        with self.Session.begin() as session:
            entry = session.scalars(select(GenCredentials).filter_by(id=credentials_id)).first()
            if entry is not None:
                entry.last_tested = datetime.datetime.now(timezone.utc)

    def sync_instance_credentials(self, entries: dict[str, dict]) -> None:
        """
        Sync the instance-level entries with the generative.yaml content:
        upsert by name; entries removed from the yaml are deleted when no
        pipeline references them, otherwise kept with a warning.
        """
        with self.Session.begin() as session:
            existing = {
                e.name: e
                for e in session.scalars(select(GenCredentials).filter_by(kind="instance")).all()
            }
            for name, params in entries.items():
                entry = existing.pop(name, None)
                if entry is None:
                    entry = GenCredentials(kind="instance", user_name=None, name=name)
                    session.add(entry)
                entry.endpoint = params["endpoint"]
                entry.api_key = encrypt(params.get("key", ""), config.secret_key)
                entry.models = params.get("models")
            for name, entry in existing.items():
                used_by = session.scalars(
                    select(GenPipelines).filter_by(credentials_id=entry.id)
                ).first()
                if used_by is None:
                    session.delete(entry)
                else:
                    print(
                        f"Instance credentials '{name}' removed from generative.yaml "
                        f"but still used by pipeline '{used_by.name}': kept in database"
                    )

    # ------------------------------------------------------------------
    # pipelines
    # ------------------------------------------------------------------

    def add_pipeline(self, project_slug: str, user_name: str, pipeline: GenPipelineCreate) -> int:
        with self.Session.begin() as session:
            exists = session.scalars(
                select(GenPipelines).filter_by(project_slug=project_slug, name=pipeline.name)
            ).first()
            if exists is not None:
                raise Exception("A pipeline with this name already exists")
            new_pipeline = GenPipelines(
                project_slug=project_slug,
                user_name=user_name,
                name=pipeline.name,
                scheme_name=pipeline.scheme_name,
                credentials_id=pipeline.credentials_id,
                model_slug=pipeline.model_slug,
                parameters=pipeline.parameters.model_dump(),
                prompt=pipeline.prompt,
                postprocess={"steps": [s.model_dump() for s in pipeline.postprocess]},
            )
            session.add(new_pipeline)
            session.flush()
            return new_pipeline.id

    def get_pipelines(self, project_slug: str) -> list[GenPipelineOut]:
        with self.Session() as session:
            pipelines = session.scalars(
                select(GenPipelines)
                .filter_by(project_slug=project_slug)
                .options(joinedload(GenPipelines.credentials))
                .order_by(GenPipelines.time.desc())
            ).all()
            return [_pipeline_out(p) for p in pipelines]

    def get_pipeline(self, project_slug: str, pipeline_id: int) -> GenPipelines:
        """
        Get one pipeline with its credentials loaded (api_key still encrypted)
        """
        with self.Session() as session:
            pipeline = session.scalars(
                select(GenPipelines)
                .filter_by(project_slug=project_slug, id=pipeline_id)
                .options(joinedload(GenPipelines.credentials))
            ).first()
            if pipeline is None:
                raise NotFoundError("Pipeline not found")
            return pipeline

    def delete_pipeline(self, project_slug: str, pipeline_id: int) -> None:
        with self.Session.begin() as session:
            session.execute(
                delete(Generations).filter_by(project_slug=project_slug, pipeline_id=pipeline_id)
            )
            session.execute(
                delete(GenPipelines).filter_by(project_slug=project_slug, id=pipeline_id)
            )

    # ------------------------------------------------------------------
    # runs
    # ------------------------------------------------------------------

    def add_run(
        self,
        pipeline_id: int,
        project_slug: str,
        user_name: str,
        dataset: str,
        mode: str,
        n_elements: int,
        path: str,
    ) -> int:
        with self.Session.begin() as session:
            run = Generations(
                pipeline_id=pipeline_id,
                project_slug=project_slug,
                user_name=user_name,
                dataset=dataset,
                mode=mode,
                n_elements=n_elements,
                status="running",
                path=path,
            )
            session.add(run)
            session.flush()
            return run.id

    def get_runs(self, project_slug: str) -> Sequence[Generations]:
        """
        Runs of a project, most recent first, with their pipeline loaded
        """
        with self.Session() as session:
            return session.scalars(
                select(Generations)
                .filter_by(project_slug=project_slug)
                .options(joinedload(Generations.pipeline))
                .order_by(Generations.time.desc())
            ).all()

    def get_run(self, project_slug: str, run_id: int) -> Generations:
        with self.Session() as session:
            run = session.scalars(
                select(Generations).filter_by(project_slug=project_slug, id=run_id)
            ).first()
            if run is None:
                raise NotFoundError("Generation run not found")
            return run

    def update_run_status(self, run_id: int, status: str, n_elements: int | None = None) -> None:
        with self.Session.begin() as session:
            run = session.scalars(select(Generations).filter_by(id=run_id)).first()
            if run is None:
                return
            run.status = status
            if n_elements is not None:
                run.n_elements = n_elements

    def delete_run(self, project_slug: str, run_id: int) -> None:
        with self.Session.begin() as session:
            session.execute(delete(Generations).filter_by(project_slug=project_slug, id=run_id))
