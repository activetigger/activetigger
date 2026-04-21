import datetime
from datetime import timezone
from typing import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session as SessionType
from sqlalchemy.orm import joinedload, sessionmaker

from activetigger.config import config
from activetigger.datamodels import (
    GenerationCreationModel,
    PromptModel,
)
from activetigger.db.models import Generations, GenModels, Prompts
from activetigger.functions import decrypt, encrypt


class GenerationsService:
    Session: sessionmaker[SessionType]

    def __init__(self, sessionmaker: sessionmaker[SessionType]):
        self.Session = sessionmaker

    def add_generated(
        self,
        user_name: str,
        project_slug: str,
        element_id: str,
        model_id: int,
        prompt: str,
        answer: str,
        batch: str | None = None,
    ):
        session = self.Session()
        generation = Generations(
            user_name=user_name,
            time=datetime.datetime.now(timezone.utc),
            project_slug=project_slug,
            element_id=element_id,
            model_id=model_id,
            prompt=prompt,
            answer=answer,
            batch=batch,
        )
        session.add(generation)
        session.commit()
        session.close()

    def get_generated(
        self, project_slug: str, user_name: str, n_elements: int | None = None
    ) -> list[list]:
        """
        Get elements from generated table by order desc
        """
        with self.Session() as session:
            if n_elements is None:
                generated = session.scalars(
                    select(Generations)
                    .filter_by(project_slug=project_slug, user_name=user_name)
                    .options(joinedload(Generations.model))  # join with the model table
                    .order_by(Generations.time.desc())
                ).all()
            else:
                generated = session.scalars(
                    select(Generations)
                    .filter_by(project_slug=project_slug, user_name=user_name)
                    .options(joinedload(Generations.model))  # join with the model table
                    .order_by(Generations.time.desc())
                    .limit(n_elements)
                ).all()
            return [
                [
                    el.time,
                    el.element_id,
                    el.batch,
                    el.prompt,
                    el.answer,
                    el.model.name if el.model else None,
                ]
                for el in generated
            ]

    def get_batch(self, batch_id: str) -> list[Generations] | None:
        """
        Get all generations for a given batch ID
        """
        with self.Session() as session:
            generations = session.scalars(
                select(Generations).filter_by(batch=batch_id).order_by(Generations.time.desc())
            ).all()
        return list(generations) if generations is not None else None

    def get_project_gen_models(self, project_slug: str) -> Sequence[GenModels]:
        """
        Get the GenAI model configured for the given project

        Returns a list of GenerationModel
        """
        with self.Session() as session:
            models = session.scalars(select(GenModels).filter_by(project_slug=project_slug)).all()
        return models

    def get_gen_model(self, model_id: int) -> GenModels:
        with self.Session() as session:
            result = session.scalars(select(GenModels).filter_by(id=model_id)).first()
            if result is None:
                raise Exception("Generation model not found")
            result.credentials = decrypt(result.credentials, config.secret_key)
            return result

    def add_project_gen_model(
        self, project_slug: str, model: GenerationCreationModel, user_name: str
    ) -> int:
        """
        Add a new GenAI model for the given project
        """
        with self.Session() as session:
            if model.credentials is None:
                model.credentials = ""
            if model.name is None or model.name == "":
                raise Exception("You should provide a name")
            new_model = GenModels(
                project_slug=project_slug,
                slug=model.slug,
                name=model.name,
                api=model.api,
                endpoint=model.endpoint,
                credentials=encrypt(model.credentials, config.secret_key),
                user_name=user_name,
            )
            session.add(new_model)
            session.commit()
            session.refresh(new_model)
            return new_model.id

    def delete_project_gen_model(self, project_slug: str, model_id: int) -> None:
        """
        Delete a GenAI model from the given project
        """
        with self.Session.begin() as session:
            session.execute(delete(GenModels).filter_by(project_slug=project_slug, id=model_id))

    def add_prompt(
        self,
        project_slug: str,
        user_name: str,
        text: str,
        parameters: dict = {},
    ) -> None:
        with self.Session() as session:
            prompt = Prompts(
                project_slug=project_slug,
                user_name=user_name,
                value=text,
                parameters=parameters,
            )
            session.add(prompt)
            session.commit()

    def delete_prompt(self, prompt_id: int) -> None:
        with self.Session.begin() as session:
            session.execute(delete(Prompts).filter_by(id=prompt_id))

    def get_prompts(self, project_slug: str) -> list[PromptModel]:
        """
        Get all prompts for a project
        """
        with self.Session() as session:
            elements = session.scalars(select(Prompts).filter_by(project_slug=project_slug)).all()
        return [
            PromptModel(
                id=el.id,
                text=el.value,
                parameters=el.parameters,
            )
            for el in elements
        ]

    def drop_generated(self, project_slug: str, user_name: str) -> None:
        with self.Session.begin() as session:
            session.execute(
                delete(Generations).filter_by(project_slug=project_slug, user_name=user_name)
            )

        return None
