"""
Generative pipelines manager
"""

import json
import re
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import openai
import pandas as pd
from pandas import DataFrame, Series
from pydantic import BaseModel
from rapidfuzz.distance import Levenshtein

from activetigger.config import config
from activetigger.datamodels import (
    EmptyParams,
    ExactMatchParams,
    FuzzyMatchParams,
    GenCredentialsInput,
    GenCredentialsOut,
    GenCredentialsTestOut,
    GeneratedRow,
    GenerationComputing,
    GenerationComputingOut,
    GenerationParams,
    GenerationsProjectStateModel,
    GenPipelineCreate,
    GenPipelineOut,
    GenRunOut,
    GenSandboxOut,
    JsonKeyParams,
    PostprocessStep,
    RegexAssignParams,
    RegexSubParams,
    TableOutModel,
)
from activetigger.db.generations import GenerationsService
from activetigger.db.manager import DatabaseManager
from activetigger.db.models import GenPipelines
from activetigger.functions import decrypt, remove_punctuation

# subdirectory of the project directory where run outputs are stored
GENERATIONS_DIR = "generations"


class PostprocessNA(Exception):
    """
    Raised when a post-treatment step cannot produce a scheme-compatible output
    """


def _shorten(text: str, n: int = 50) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _step_strip(text: str, params: EmptyParams, labels: list[str] | None):
    return text.strip(), None


def _step_lowercase(text: str, params: EmptyParams, labels: list[str] | None):
    return text.lower(), None


def _step_uppercase(text: str, params: EmptyParams, labels: list[str] | None):
    return text.upper(), None


def _step_remove_punctuation(text: str, params: EmptyParams, labels: list[str] | None):
    return remove_punctuation(text), None


def _step_regex_sub(text: str, params: RegexSubParams, labels: list[str] | None):
    flags = re.IGNORECASE if params.ignore_case else 0
    return re.sub(params.pattern, params.replacement, text, flags=flags), None


def _step_json_extract(text: str, params: EmptyParams, labels: list[str] | None):
    """
    Extract the first valid JSON object or array found in the text
    """
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char in "{[":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return json.dumps(value, ensure_ascii=False), None
            except json.JSONDecodeError:
                continue
    raise PostprocessNA("no JSON found in the output")


def _step_json_key(text: str, params: JsonKeyParams, labels: list[str] | None):
    """
    Extract the value of a key from the current text parsed as JSON
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        raise PostprocessNA("output is not valid JSON (use json_extract first?)")
    if not isinstance(value, dict) or params.key not in value:
        raise PostprocessNA(f"key '{params.key}' not found in the JSON output")
    value = value[params.key]
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), None


def _step_exact_match(text: str, params: ExactMatchParams, labels: list[str] | None):
    """
    Terminal step: the whole output must match a label of the scheme
    """
    if not labels:
        raise PostprocessNA("exact_match needs a scheme with labels")
    candidate = text.strip()
    for label in labels:
        if candidate == label or (params.ignore_case and candidate.lower() == label.lower()):
            return text, label
    raise PostprocessNA(f"output '{_shorten(candidate)}' does not match any label")


def _step_fuzzy_match(text: str, params: FuzzyMatchParams, labels: list[str] | None):
    """
    Terminal step: match the closest label within a Levenshtein distance
    """
    if not labels:
        raise PostprocessNA("fuzzy_match needs a scheme with labels")
    candidate = text.strip()
    if params.ignore_case:
        distances = [(Levenshtein.distance(candidate.lower(), la.lower()), la) for la in labels]
    else:
        distances = [(Levenshtein.distance(candidate, la), la) for la in labels]
    best_distance, best_label = min(distances)
    if best_distance <= params.max_distance:
        return text, best_label
    raise PostprocessNA(
        f"output '{_shorten(candidate)}' is at distance {best_distance} from the closest label"
    )


def _step_regex_assign(text: str, params: RegexAssignParams, labels: list[str] | None):
    """
    If the pattern is present, assign the given label; otherwise continue
    """
    flags = re.IGNORECASE if params.ignore_case else 0
    if re.search(params.pattern, text, flags):
        return text, params.label
    return text, None


# name -> (function, params model)
POSTPROCESS_STEPS: dict[str, tuple[Callable, type[BaseModel]]] = {
    "strip": (_step_strip, EmptyParams),
    "lowercase": (_step_lowercase, EmptyParams),
    "uppercase": (_step_uppercase, EmptyParams),
    "remove_punctuation": (_step_remove_punctuation, EmptyParams),
    "regex_sub": (_step_regex_sub, RegexSubParams),
    "json_extract": (_step_json_extract, EmptyParams),
    "json_key": (_step_json_key, JsonKeyParams),
    "exact_match": (_step_exact_match, ExactMatchParams),
    "fuzzy_match": (_step_fuzzy_match, FuzzyMatchParams),
    "regex_assign": (_step_regex_assign, RegexAssignParams),
}


class Generations:
    """
    Manage the generative pipelines of a project
    """

    project_slug: str
    path: Path | None
    computing: list
    generations_service: GenerationsService

    def __init__(
        self,
        project_slug: str,
        path: Path | None,
        computing: list,
        db_manager: DatabaseManager,
    ) -> None:
        self.project_slug = project_slug
        self.path = path.joinpath(GENERATIONS_DIR) if path is not None else None
        if self.path is not None:
            self.path.mkdir(exist_ok=True)
        self.computing = computing
        self.generations_service = db_manager.generations_service
        self.recover_orphan_runs()

    def training(self) -> dict[str, GenerationComputingOut]:
        """
        Currently running generations
        """
        return {
            e.user: GenerationComputingOut(
                run_id=e.run_id,
                pipeline_id=e.pipeline_id,
                pipeline_name=e.pipeline_name,
                progress=e.get_progress() if e.get_progress is not None else 0,
            )
            for e in self.computing
            if isinstance(e, GenerationComputing) and e.kind == "generation"
        }

    def available(self) -> list[GenPipelineOut]:
        """
        Pipelines of the project
        """
        return self.generations_service.get_pipelines(self.project_slug)

    def state(self) -> GenerationsProjectStateModel:
        return GenerationsProjectStateModel(
            training=self.training(),
            available=self.available(),
        )

    @staticmethod
    def list_user_credentials(
        db_manager: DatabaseManager, user_name: str
    ) -> list[GenCredentialsOut]:
        return db_manager.generations_service.list_credentials(user_name)

    @staticmethod
    def save_user_credentials(
        db_manager: DatabaseManager, user_name: str, credentials: GenCredentialsInput
    ) -> GenCredentialsTestOut:
        """
        Save a user entry (same name = replaced) and test it
        """
        if credentials.name.strip() == "":
            raise Exception("You should provide a name")
        credentials_id = db_manager.generations_service.add_credentials(user_name, credentials)
        return Generations.test_user_credentials(db_manager, credentials_id, user_name)

    @staticmethod
    def test_user_credentials(
        db_manager: DatabaseManager, credentials_id: int, user_name: str
    ) -> GenCredentialsTestOut:
        """
        Try a call on the endpoint; record the success in last_tested
        """
        service = db_manager.generations_service
        entry = service.get_credentials(credentials_id, user_name)
        try:
            Generations.test_credentials(entry.endpoint, entry.api_key)
            service.set_last_tested(credentials_id)
            return GenCredentialsTestOut(id=credentials_id, success=True)
        except Exception as e:
            return GenCredentialsTestOut(id=credentials_id, success=False, detail=str(e))

    @staticmethod
    def delete_user_credentials(
        db_manager: DatabaseManager, credentials_id: int, user_name: str
    ) -> None:
        db_manager.generations_service.delete_credentials(credentials_id, user_name)

    @staticmethod
    def models_for_credentials(
        db_manager: DatabaseManager, credentials_id: int, user_name: str
    ) -> list[str]:
        """
        The list declared in generative.yaml when present, otherwise /v1/models
        """
        entry = db_manager.generations_service.get_credentials(credentials_id, user_name)
        if entry.models:
            return entry.models
        return Generations.list_models(entry.endpoint, entry.api_key)

    def get_pipeline(self, pipeline_id: int) -> GenPipelines:
        return self.generations_service.get_pipeline(self.project_slug, pipeline_id)

    @staticmethod
    def check_prompt(prompt: str, cols_context: list[str]) -> None:
        """
        Check the "[[XXX]]" tags of a prompt template
        """
        for tag in re.findall(r"\[\[(\w+)\]\]", prompt):
            if tag not in ["TEXT", *cols_context]:
                raise Exception(f"The tag [[{tag}]] is not [[TEXT]] or a context column")

    @staticmethod
    def render_prompt(prompt: str, row: Series, cols_context: list[str]) -> str:
        """
        Fill the [[TEXT]] and [[context column]] tags with the row content.
        """
        if "[[TEXT]]" not in prompt:
            prompt += "\n\n[[TEXT]]"
        for col in cols_context:
            value = row.get(col, "")
            prompt = prompt.replace(f"[[{col}]]", "" if pd.isna(value) else str(value))
        return prompt.replace("[[TEXT]]", str(row["text"]))

    def add_pipeline(
        self,
        pipeline: GenPipelineCreate,
        user_name: str,
        schemes_kinds: dict[str, str],
        cols_context: list[str],
    ) -> int:
        """
        Validate and save a new pipeline (schemes_kinds: scheme name -> kind)
        """
        if pipeline.name.strip() == "":
            raise Exception("You should provide a name")
        self.check_prompt(pipeline.prompt, cols_context)
        self.validate_steps(pipeline.postprocess)
        if pipeline.scheme_name is not None:
            kind = schemes_kinds.get(pipeline.scheme_name)
            if kind is None:
                raise Exception(f"Scheme {pipeline.scheme_name} does not exist")
            if kind != "multiclass":
                raise Exception("Only multiclass schemes are supported for now")
        # check the credentials exist and are visible to the user
        self.generations_service.get_credentials(pipeline.credentials_id, user_name)
        return self.generations_service.add_pipeline(self.project_slug, user_name, pipeline)

    def delete_pipeline(self, pipeline_id: int) -> None:
        """
        Delete a pipeline, its runs and their files
        """
        for run in self.generations_service.get_runs(self.project_slug):
            if run.pipeline_id == pipeline_id:
                self._delete_run_files(Path(run.path))
        self.generations_service.delete_pipeline(self.project_slug, pipeline_id)

    def load_pipeline(
        self, pipeline_id: int
    ) -> tuple[GenPipelines, GenerationParams, list[PostprocessStep], str]:
        """
        Get a pipeline with its parameters parsed and the api key decrypted
        """
        pipeline = self.generations_service.get_pipeline(self.project_slug, pipeline_id)
        params = GenerationParams(**pipeline.parameters)
        steps = [PostprocessStep(**s) for s in pipeline.postprocess.get("steps", [])]
        api_key = decrypt(pipeline.credentials.api_key, config.secret_key)
        return pipeline, params, steps, api_key

    @staticmethod
    def run_on_row(
        row: Series,
        element_id: str,
        prompt_template: str,
        model_slug: str,
        endpoint: str,
        api_key: str,
        params: GenerationParams,
        steps: list[PostprocessStep],
        labels: list[str] | None,
        cols_context: list[str],
    ) -> GeneratedRow:
        """
        Run the full pipeline on one element
        """
        prompt = Generations.render_prompt(prompt_template, row, cols_context)
        try:
            raw = Generations.generate(prompt, model_slug, params, endpoint, api_key)
        except Exception as e:
            return GeneratedRow(
                element_id=element_id, prompt=prompt, raw="", error=f"generation failed: {e}"
            )
        predicted, error = Generations.apply_postprocess(raw, steps, labels)
        return GeneratedRow(
            element_id=element_id, prompt=prompt, raw=raw, predicted=predicted, error=error
        )

    def sandbox(
        self,
        pipeline_id: int,
        df: DataFrame,
        cols_context: list[str],
        labels: list[str] | None,
    ) -> GenSandboxOut:
        """
        Run a pipeline synchronously on a few elements; nothing is stored
        """
        pipeline, params, steps, api_key = self.load_pipeline(pipeline_id)
        rows = [
            self.run_on_row(
                row,
                str(element_id),
                pipeline.prompt,
                pipeline.model_slug,
                pipeline.credentials.endpoint,
                api_key,
                params,
                steps,
                labels,
                cols_context,
            )
            for element_id, row in df.iterrows()
        ]
        return GenSandboxOut(rows=rows, n_na=sum(1 for r in rows if r.predicted is None))

    @staticmethod
    def preview_postprocess(
        steps: list[PostprocessStep], raws: list[str], labels: list[str] | None
    ) -> GenSandboxOut:
        """
        Re-apply a candidate list of steps on raw outputs already generated
        (lets the user tune the post-treatment without paying generation)
        """
        Generations.validate_steps(steps)
        rows = []
        for raw in raws:
            predicted, error = Generations.apply_postprocess(raw, steps, labels)
            rows.append(
                GeneratedRow(element_id="", prompt="", raw=raw, predicted=predicted, error=error)
            )
        return GenSandboxOut(rows=rows, n_na=sum(1 for r in rows if r.predicted is None))

    def new_run_path(self) -> Path:
        """
        A fresh parquet path for a run output
        """
        if self.path is None:
            raise Exception("No directory available to store generations")
        return self.path.joinpath(f"run_{uuid4().hex[:12]}.parquet")

    @staticmethod
    def write_run_input(df: DataFrame, path_output: Path) -> Path:
        """
        Write the sampled elements next to the run output for the task
        """
        path_input = path_output.with_name(path_output.stem + "_input.parquet")
        df.to_parquet(path_input)
        return path_input

    def runs(self) -> list[GenRunOut]:
        """
        Runs of the project, with the NA count read from the output file
        """
        outs = []
        for run in self.generations_service.get_runs(self.project_slug):
            n_na = None
            if run.status in ("done", "interrupted"):
                try:
                    predicted = pd.read_parquet(run.path, columns=["predicted"])["predicted"]
                    n_na = int(predicted.isna().sum())
                except Exception:
                    pass
            outs.append(
                GenRunOut(
                    id=run.id,
                    pipeline_id=run.pipeline_id,
                    pipeline_name=run.pipeline.name,
                    user_name=run.user_name,
                    dataset=run.dataset,
                    mode=run.mode,
                    n_elements=run.n_elements,
                    status=run.status,
                    n_na=n_na,
                    time=run.time,
                )
            )
        return outs

    def run_data(self, run_id: int) -> DataFrame:
        """
        Full outputs of a run (element_id, prompt, raw, predicted, error)
        """
        run = self.generations_service.get_run(self.project_slug, run_id)
        return pd.read_parquet(run.path)

    def run_table(self, run_id: int, limit: int = 100, offset: int = 0) -> TableOutModel:
        """
        Paginated view of a run's outputs
        """
        df = self.run_data(run_id)
        page = df.iloc[offset : offset + limit]
        return TableOutModel(items=page.to_dict(orient="records"), total=len(df))

    def finish_run(self, run_id: int, status: str, n_elements: int | None = None) -> None:
        self.generations_service.update_run_status(run_id, status, n_elements)

    @staticmethod
    def _delete_run_files(path_output: Path) -> None:
        """
        A run's output parquet, jsonl recovery buffer and sampled input
        """
        path_output.unlink(missing_ok=True)
        path_output.with_suffix(".jsonl").unlink(missing_ok=True)
        path_output.with_name(path_output.stem + "_input.parquet").unlink(missing_ok=True)

    def delete_run(self, run_id: int) -> None:
        """
        Delete a run and its files
        """
        run = self.generations_service.get_run(self.project_slug, run_id)
        if run.status == "running":
            raise Exception("This run is still computing: stop the process first")
        self._delete_run_files(Path(run.path))
        self.generations_service.delete_run(self.project_slug, run_id)

    def recover_orphan_runs(self) -> None:
        """
        At project load nothing is computing, so runs still marked "running"
        were interrupted by a server stop. Their recovery jsonl (written by
        the task as it goes) is converted to the run parquet when present,
        otherwise the run is marked as error.
        """
        for run in self.generations_service.get_runs(self.project_slug):
            if run.status != "running":
                continue
            jsonl = Path(run.path).with_suffix(".jsonl")
            rows = []
            if jsonl.exists():
                try:
                    with open(jsonl) as f:
                        rows = [GeneratedRow(**json.loads(line)) for line in f if line.strip()]
                except Exception as e:
                    print(f"Failed to read generation checkpoint {jsonl}: {e}")
            if rows:
                DataFrame([r.model_dump() for r in rows]).to_parquet(run.path, index=False)
                jsonl.unlink(missing_ok=True)
                self.generations_service.update_run_status(run.id, "interrupted", len(rows))
            else:
                self.generations_service.update_run_status(run.id, "error")

    @staticmethod
    def _client(endpoint: str, api_key: str) -> openai.OpenAI:
        # some servers (eg. a local vllm or ollama) accept any non-empty key
        return openai.OpenAI(base_url=endpoint, api_key=api_key or "EMPTY", timeout=120)

    @staticmethod
    def list_models(endpoint: str, api_key: str) -> list[str]:
        """
        List the models exposed by an endpoint (/v1/models)
        """
        client = Generations._client(endpoint, api_key)
        return sorted(model.id for model in client.models.list())

    @staticmethod
    def test_credentials(endpoint: str, api_key: str) -> None:
        """
        Check that the endpoint answers with these credentials.
        Listing models costs nothing and requires authentication on hosted
        APIs. Raises with the provider message on failure.
        """
        client = Generations._client(endpoint, api_key)
        client.models.list()

    @staticmethod
    def generate(
        prompt: str,
        model_slug: str,
        params: GenerationParams,
        endpoint: str,
        api_key: str,
    ) -> str:
        """
        One completion call, sending only the parameters that are set
        """
        options: dict[str, Any] = {}
        for name in ("temperature", "seed", "top_p", "presence_penalty", "frequency_penalty"):
            value = getattr(params, name)
            if value is not None:
                options[name] = value
        if params.max_tokens is not None:
            options["max_tokens"] = params.max_tokens
        if params.thinking is not None:
            # understood by reasoning models, ignored by most other servers
            options["reasoning_effort"] = "medium" if params.thinking else "minimal"
        client = Generations._client(endpoint, api_key)
        messages: Any = [{"role": "user", "content": prompt}]
        try:
            # the SDK overloads reject a **dict of user-chosen parameters
            response = client.chat.completions.create(
                model=model_slug, messages=messages, **options
            )  # ty: ignore[no-matching-overload]
        except openai.BadRequestError as e:
            # newer OpenAI models refuse max_tokens in favor of max_completion_tokens
            if "max_tokens" in options and "max_completion_tokens" in str(e):
                options["max_completion_tokens"] = options.pop("max_tokens")
                response = client.chat.completions.create(
                    model=model_slug, messages=messages, **options
                )  # ty: ignore[no-matching-overload]
            else:
                raise
        content = response.choices[0].message.content
        if content is None:
            raise Exception("The model returned an empty answer")
        return content

    @staticmethod
    def validate_steps(steps: list[PostprocessStep]) -> list[tuple[Callable, BaseModel]]:
        """
        Check that each step exists and its parameters are valid.
        """
        compiled = []
        for step in steps:
            if step.name not in POSTPROCESS_STEPS:
                raise Exception(
                    f"Unknown post-treatment step '{step.name}' "
                    f"(available: {', '.join(POSTPROCESS_STEPS)})"
                )
            function, params_model = POSTPROCESS_STEPS[step.name]
            try:
                params = params_model(**step.params)
            except Exception as e:
                raise Exception(f"Invalid parameters for step '{step.name}': {e}")
            compiled.append((function, params))
        return compiled

    @staticmethod
    def apply_postprocess(
        raw: str, steps: list[PostprocessStep], labels: list[str] | None
    ) -> tuple[str | None, str | None]:
        """
        Apply the post-treatment steps to a raw output.
        """
        text = raw
        try:
            for function, params in Generations.validate_steps(steps):
                text, label = function(text, params, labels)
                if label is not None:
                    return label, None
        except PostprocessNA as e:
            return None, str(e)
        if labels is None:
            return text, None
        return None, "no step assigned a label (add a matching step at the end)"
