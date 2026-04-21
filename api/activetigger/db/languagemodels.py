import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from activetigger.datamodels import ModelDescriptionModel
from activetigger.db import DBException
from activetigger.db.models import (
    Models,
)


class ModelsService:
    """
    Database service for models
    """

    SessionMaker: sessionmaker[Session]

    def __init__(self, sessionmaker: sessionmaker[Session]):
        self.SessionMaker = sessionmaker

    def available_models(self, project_slug: str, kind: str) -> list[ModelDescriptionModel]:
        """
        Get available models in database
        """
        with self.SessionMaker() as session:
            models = session.execute(
                select(Models.name, Models.parameters, Models.path, Models.scheme_name, Models.time)
                .filter_by(
                    project_slug=project_slug,
                    status="trained",
                    kind=kind,
                )
                .order_by(Models.time.desc())
            ).all()
        return [
            ModelDescriptionModel(
                name=m.name,
                kind=kind,
                scheme=m.scheme_name,
                path=m.path,
                parameters=m.parameters,
                time=str(m.time),
            )
            for m in models
        ]

    def add_model(
        self,
        kind: str,
        project: str,
        name: str,
        user: str,
        status: str,
        scheme: str,
        params: dict[str, Any],
        path: str,
        retrain: bool = False,
    ):
        session = self.SessionMaker()

        # test if the name does not exist
        models = (
            session.query(Models).filter(Models.name == name, Models.project_slug == project).all()
        )
        if (len(models) == 1) and retrain:
            models[0].time = datetime.datetime.now(timezone.utc)
        elif len(models) > 0:
            # A model already exists, we can't have another one
            raise Exception("Model already exists")
        else:
            # it's a new model that must be saved into the database
            model = Models(
                project_slug=project,
                time=datetime.datetime.now(timezone.utc),
                kind=kind,
                name=name,
                user_name=user,
                parameters=params,
                scheme_name=scheme,
                status=status,
                path=path,
            )
            session.add(model)
        session.commit()
        session.close()

    def model_exists(self, project_slug: str, name: str):
        session = self.SessionMaker()
        models = (
            session.query(Models)
            .filter(Models.name == name, Models.project_slug == project_slug)
            .all()
        )
        session.close()
        return len(models) > 0

    def delete_model(self, project_slug: str, name: str):
        session = self.SessionMaker()
        # test if the name does not exist
        models = (
            session.query(Models)
            .filter(Models.name == name, Models.project_slug == project_slug)
            .all()
        )
        if len(models) == 0:
            print("Model does not exist")
            return False
        # delete the model
        session.query(Models).filter(
            Models.name == name, Models.project_slug == project_slug
        ).delete()
        session.commit()
        session.close()
        return True

    def get_model(self, project_slug: str, name: str):
        session = self.SessionMaker()
        model = (
            session.query(Models)
            .filter(Models.name == name, Models.project_slug == project_slug)
            .first()
        )
        session.close()
        return model

    def rename_model(self, project_slug: str, old_name: str, new_name: str):
        session = self.SessionMaker()

        # test if the name does not exist
        models = (
            session.query(Models)
            .filter(Models.name == new_name, Models.project_slug == project_slug)
            .all()
        )
        if len(models) > 0:
            raise Exception("Model already exists")
        # get and rename
        model = (
            session.query(Models)
            .filter(Models.name == old_name, Models.project_slug == project_slug)
            .first()
        )
        if model is None:
            raise DBException("Model not found")

        model.name = new_name
        model_path = Path(model.path)
        model.path = str(model_path.parent / model_path.name.replace(old_name, new_name))
        session.commit()
        session.close()

    def set_model_params(self, project_slug: str, name: str, flag: str, value):
        session = self.SessionMaker()
        model = (
            session.query(Models)
            .filter(Models.name == name, Models.project_slug == project_slug)
            .first()
        )
        if model is None:
            session.close()
            raise DBException("Model not found")

        parameters = model.parameters.copy()
        parameters[flag] = value
        model.parameters = parameters
        session.commit()
        session.close()
