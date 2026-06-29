import re
from typing import cast

import pandas as pd
from pandas import DataFrame

from activetigger.config import config
from activetigger.datamodels import (  # ignore[import]
    ExportGenerationsParams,
    GenerationAvailableModel,
    GenerationComputing,
    GenerationComputingOut,
    GenerationCreationModel,
    GenerationModel,
    GenerationModelApi,
    GenerationsProjectStateModel,
    PromptInputModel,
    PromptModel,
)
from activetigger.db.generations import GenerationsService
from activetigger.db.manager import DatabaseManager
from activetigger.functions import remove_punctuation, replace_accented_chars

# Maps the `type` key used in generative.yaml to the internal `api` identifier
# used by the generation clients.
GENERATIVE_YAML_TYPE_MAP = {
    "openapi": "OpenAICompatible",
    "openai-compatible": "OpenAICompatible",
    "openai": "OpenAI",
    "ollama": "Ollama",
    "openrouter": "OpenRouter",
    "huggingface": "HuggingFace",
}


class Generations:
    """
    Class to manage generation data
    """

    computing: list
    generations_service: GenerationsService

    @staticmethod
    def get_available_models() -> list[GenerationModelApi]:
        """
        Get the available models for generation

        Currently, this is hardwired in code
        """
        return [
            GenerationModelApi(
                name="Ollama",
                models=[],
            ),
            GenerationModelApi(name="OpenRouter", models=[]),
            GenerationModelApi(name="OpenAICompatible", models=[]),
            GenerationModelApi(
                name="OpenAI",
                models=[
                    GenerationAvailableModel(
                        slug="gpt-4o-mini", api="OpenAI", name="ChatGPT 4o mini"
                    ),
                    GenerationAvailableModel(slug="gpt-4o", api="OpenAI", name="ChatGPT 4o"),
                    GenerationAvailableModel(
                        slug="gpt-4.1-mini", api="OpenAI", name="ChatGPT 4.1 mini"
                    ),
                    GenerationAvailableModel(slug="gpt-4.1", api="OpenAI", name="ChatGPT 4.1"),
                    GenerationAvailableModel(
                        slug="gpt-4.1-nano", api="OpenAI", name="ChatGPT 4.1 nano"
                    ),
                ],
            ),
            GenerationModelApi(name="HuggingFace", models=[]),
        ]

    def __init__(self, db_manager: DatabaseManager, computing: list[GenerationComputing]) -> None:
        self.generations_service = db_manager.generations_service
        self.computing = computing

    def add(
        self,
        user: str,
        project_slug: str,
        element_id: str,
        model_id: int,
        prompt: str,
        answer: str,
        batch: str | None = None,
    ) -> None:
        """
        Add a generated element in the database
        """
        self.generations_service.add_generated(
            user_name=user,
            project_slug=project_slug,
            element_id=element_id,
            model_id=model_id,
            prompt=prompt,
            answer=answer,
            batch=batch,
        )
        return None

    def get_generated(
        self, project_slug: str, user_name: str, params: ExportGenerationsParams
    ) -> DataFrame:
        """
        Get generated elements from the database
        """
        result = self.generations_service.get_generated(
            project_slug=project_slug, user_name=user_name
        )
        df = pd.DataFrame(
            result, columns=pd.Index(["time", "index", "batch", "prompt", "answer", "model_name"])
        )
        df["time"] = pd.to_datetime(df["time"])
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        df["time"] = df["time"].dt.tz_convert("Europe/Paris")
        df["answer"] = self.filter(df["answer"], params.filters)
        return df

    def training(self) -> dict[str, GenerationComputingOut]:
        """
        Get state current generation computing
        """
        return {
            e.user: GenerationComputingOut(
                model_id=e.model_id,
                progress=e.get_progress() if e.get_progress is not None else 0,
            )
            for e in self.computing
            if e.kind == "generation"
        }

    def prompt_exists(self, project_slug: str, name: str) -> bool:
        """
        Check if a prompt already exists
        """
        all_prompts = self.get_prompts(project_slug)
        return any([prompt.parameters["name"] == name for prompt in all_prompts])

    def model_exists(self, project_slug: str, name: str) -> bool:
        """
        Check if a model already exists
        """
        all_models = self.generations_service.get_project_gen_models(project_slug)
        return any([model.name == name for model in all_models])

    def save_prompt(self, prompt: PromptInputModel, username: str, project_slug: str) -> None:
        """
        Save a prompt in the database
        """

        # if no name, use the beginning of the text
        if prompt.name is not None:
            name = prompt.name
        else:
            name = prompt.text[0 : min(30, len(prompt.text))]

        # check if the name is already used
        if self.prompt_exists(project_slug, name):
            raise Exception("A prompt with this name already exists")

        self.generations_service.add_prompt(
            user_name=username,
            project_slug=project_slug,
            text=prompt.text,
            parameters={"name": name},
        )
        return None

    def delete_prompt(self, prompt_id: int) -> None:
        """
        Delete a prompt from the database
        """
        self.generations_service.delete_prompt(prompt_id)
        return None

    def get_prompts(self, project_slug: str) -> list[PromptModel]:
        """
        Get the list of prompts for the user
        """
        return self.generations_service.get_prompts(project_slug)

    def drop_generated(self, project_slug: str, user_name: str) -> None:
        """
        Drop all elements from prediction for a user
        """
        self.generations_service.drop_generated(project_slug, user_name)
        return None

    def filter(self, answers: pd.Series, filters) -> pd.Series:
        """
        Apply filters
        """
        if "remove_punct" in filters:
            answers = answers.apply(remove_punctuation)
        if "remove_spaces" in filters:
            answers = answers.str.replace(r"\s+", " ")
        if "lowercase" in filters:
            answers = answers.str.lower()
        if "strip" in filters:
            answers = answers.str.strip()
        if "replace_accents" in filters:
            answers = cast(pd.Series, answers.apply(replace_accented_chars))
        return answers

    def state(self) -> GenerationsProjectStateModel:
        return GenerationsProjectStateModel(training=self.training())

    def available_models(self, project_slug: str) -> list[GenerationModel]:
        """
        Get the available models for generation
        """
        r = self.generations_service.get_project_gen_models(project_slug)
        return [GenerationModel(**i.__dict__) for i in r]

    def add_model(self, project_slug: str, model: GenerationCreationModel, user_name: str) -> int:
        """
        Add a GenAI model to the project
        """
        if self.model_exists(project_slug, model.name):
            raise Exception("A model with this name already exists")
        return self.generations_service.add_project_gen_model(project_slug, model, user_name)

    def add_default_models(self, project_slug: str, user_name: str) -> None:
        """
        Pre-populate a project with the generative models declared in
        generative.yaml. Per-model errors are logged but don't abort the call.

        Expected YAML schema:
            models:
              <display-name>:
                type: openapi          # required, see GENERATIVE_YAML_TYPE_MAP
                url: https://...       # endpoint (alias: endpoint)
                key: <api-key>         # credentials (alias: credentials)
                model: <model-id>      # model identifier sent to the API
                                       # (aliases: slug; defaults to <display-name>)
        """
        models = getattr(config, "models_generative", {}) or {}
        for name, params in models.items():
            params = params or {}
            try:
                mtype = str(params.get("type", "")).strip().lower()
                api = GENERATIVE_YAML_TYPE_MAP.get(mtype)
                if api is None:
                    print(f"Skipping generative model {name!r}: unknown type {mtype!r}")
                    continue
                slug = params.get("model") or params.get("slug") or name
                model = GenerationCreationModel(
                    slug=str(slug),
                    api=api,
                    name=str(name),
                    endpoint=params.get("url") or params.get("endpoint"),
                    credentials=params.get("key") or params.get("credentials"),
                )
                if self.model_exists(project_slug, model.name):
                    continue
                self.generations_service.add_project_gen_model(project_slug, model, user_name)
            except Exception as e:
                print(f"Failed to add default generative model {name!r}: {e}")

    def delete_model(self, project_slug: str, model_id: int) -> None:
        """
        Delete a GenAI model from the project
        """
        self.generations_service.delete_project_gen_model(project_slug, model_id)
        return None

    @staticmethod
    def check_prompts(prompt: str, cols: list[str]) -> None:
        """
        Check if all prompts are valid
        "[[XXX]]" in the prompt correspond to a column
        """
        for tag_like in re.findall("[\[]{2}\w{1,}[\]]{2}", prompt):
            tag_name = tag_like[2:-2]  # tag minus "[[" and "]]""
            if tag_name in ["TEXT", *cols]:
                continue
            else:
                raise Exception(f"The tag {tag_like} is not part of the columns")
