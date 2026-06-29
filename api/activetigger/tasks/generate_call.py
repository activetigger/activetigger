import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pandas import DataFrame, Series

from activetigger.datamodels import (
    GenerationModel,
    GenerationResult,
)
from activetigger.generation.client import GenerationModelClient
from activetigger.generation.huggingface import HuggingFace
from activetigger.generation.ollama import Ollama
from activetigger.generation.openai import OpenAI
from activetigger.generation.openapi import OpenAPI
from activetigger.generation.openrouter import OpenRouter
from activetigger.tasks.base_task import BaseTask

# Flush in-memory results to a recovery file every N rows so a crash loses at
# most FLUSH_EVERY - 1 paid generations instead of the whole batch.
FLUSH_EVERY = 10


class GenerateCall(BaseTask):
    """
    Generate call for api
    """

    kind = "generate_call"

    def __init__(
        self,
        path_process: Path | None,
        username: str,
        project_slug: str,
        df: DataFrame,
        model: GenerationModel,
        prompt: str,
        cols_context: list[str],
        dataset: str,
        prompt_name: str,
        n_workers: int = 1,
    ):
        super().__init__()
        if path_process is None:
            path_process = Path(".")
        self.path_process = path_process
        self.username = username
        self.project_slug = project_slug
        self.df = df
        self.model = model
        self.prompt = prompt
        self.cols_context = cols_context
        self.dataset = dataset
        self.prompt_name = prompt_name
        self.n_workers = max(1, int(n_workers))

    def _write_progress(self, progress: int):
        """
        Write progress in the file
        """
        with open(self.path_process.joinpath(self.unique_id), "w") as f:
            f.write(f"{progress}")
        print(f"Progress: {progress}")

    def _jsonl_path(self) -> Path:
        return self.path_process.joinpath(f"gen_{self.unique_id}.jsonl")

    def _flush_to_jsonl(self, chunk: list[GenerationResult], batch: str) -> None:
        with open(self._jsonl_path(), "a") as f:
            for r in chunk:
                payload = r.model_dump()
                payload["batch"] = batch
                f.write(json.dumps(payload) + "\n")

    @staticmethod
    def get_progress_callback(path_file):
        """
        Get progress callback
        """

        def callback() -> int | None:
            try:
                with open(path_file, "r") as f:
                    r = f.read()
                return int(r)
            except Exception as e:
                print(e)
                return None

        return callback

    def _make_client(self) -> GenerationModelClient:
        """
        Build the generation client matching the configured API.
        """
        if self.model.api == "Ollama":
            if self.model.endpoint is None:
                raise Exception("You need to give an endpoint for the Ollama model")
            return Ollama(self.model.endpoint)
        if self.model.api == "OpenAI":
            if self.model.credentials is None:
                raise Exception("You need to give your API key to call an OpenAI model")
            return OpenAI(self.model.credentials)
        if self.model.api == "HuggingFace":
            return HuggingFace(credentials=self.model.credentials, endpoint=self.model.endpoint)
        if self.model.api == "OpenRouter":
            return OpenRouter(credentials=self.model.credentials)
        if self.model.api == "ilaas":
            return OpenAPI(
                endpoint="https://llm.ilaas.fr/v1/chat/completions",
                credentials=self.model.credentials,
            )
        if self.model.api == "OpenAICompatible":
            if not self.model.endpoint:
                raise Exception("You need to provide an endpoint for the OpenAI-compatible model")
            return OpenAPI(endpoint=self.model.endpoint, credentials=self.model.credentials)
        raise Exception(f"Unknown model API: {self.model.api}")

    def __call__(self):
        """
        Manage batch generation request. Returns the list of GenerationResult.
        Calls are issued through a thread pool sized by n_workers (1 = sequential).
        """
        errors: list[Exception] = []
        results: list[GenerationResult] = []
        batch = f"{self.dataset}_{self.prompt_name}_{self.unique_id}"
        last_flushed = 0
        progress_path = self.path_process.joinpath(self.unique_id)

        gen_model = self._make_client()
        rows = list(self.df.iterrows())
        total = len(rows)

        def process_row(index, row) -> GenerationResult:
            prompt_with_text = self.__replace_tags_with_text(row, self.prompt, self.cols_context)
            response = gen_model.generate(prompt_with_text, self.model.slug)
            return GenerationResult(
                user=self.username,
                project_slug=self.project_slug,
                model_id=self.model.id,
                element_id=str(index),
                prompt=prompt_with_text,
                answer=response,
            )

        try:
            self._write_progress(0)
            completed = 0
            interrupted = False
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(process_row, idx, row) for idx, row in rows]
                try:
                    for future in as_completed(futures):
                        if self.event is not None and self.event.is_set():
                            interrupted = True
                            for f in futures:
                                f.cancel()
                            break
                        try:
                            results.append(future.result())
                        except Exception as e:
                            errors.append(e)
                        completed += 1
                        self._write_progress(int((completed / total) * 100))
                        if len(results) - last_flushed >= FLUSH_EVERY:
                            self._flush_to_jsonl(results[last_flushed:], batch)
                            last_flushed = len(results)
                except Exception:
                    for f in futures:
                        f.cancel()
                    raise

            if interrupted:
                if len(results) > last_flushed:
                    self._flush_to_jsonl(results[last_flushed:], batch)
                raise Exception("Process was interrupted")

            return results
        finally:
            progress_path.unlink(missing_ok=True)

    def __replace_tags_with_text(self, row: Series, prompt: str, context_columns: list[str]) -> str:
        """This function takes in the prompt with tags (eg: "Hello please insert
        here with the [[dataset_year]]) slice the prompt where the [[TAGS]] are
        and replace the holes with the corresponding text. If a tag appears
        multiple times the content is inserted as often as the tag appears."""

        def format(tag_name: str) -> str:
            return f"[[{tag_name}]]"

        def unformat(tag: str) -> str:
            return tag[2:-2]

        # Retrieve the locations of the tags in the prompt
        indexes = {}
        tags_list = [format(tag_name) for tag_name in ["TEXT", *context_columns]]
        for tag in tags_list:
            start, iteration = 0, 1
            while prompt.find(tag, start) != -1:
                tag_location = prompt.find(tag, start)
                indexes[(tag, iteration)] = tag_location
                start = tag_location + 1
                iteration += 1

        # if the text tag was not found, add it in the end
        if ("[[TEXT]]", 1) not in indexes:
            prompt += "\n\n[[TEXT]]"
            indexes[("[[TEXT]]", 1)] = len(prompt) - len("[[TEXT]]")

        # Sort the indexes so that the holes in the prompt will match the tags
        # https://realpython.com/sort-python-dictionary/#sorting-dictionaries-in-python
        indexes = dict(sorted(indexes.items(), key=lambda x: x[1]))

        sliced_prompt = self.__slice_prompt(indexes, prompt)

        # Insert the contents
        complete_prompt = sliced_prompt[0]
        for i, (tag, iteration) in enumerate(indexes.keys()):
            if tag == "[[TEXT]]":
                complete_prompt += str(row["text"])
            else:
                complete_prompt += str(row[unformat(tag)])
            complete_prompt += sliced_prompt[i + 1]
        return complete_prompt

    def __slice_prompt(self, indexes: dict, prompt: str) -> list[str]:
        """Takes in the prompt and the location of the tags and return the prompt
        as a list of slices where each hole correspond to a tag."""
        # Create a list of splits
        splits = [0]
        for tag, iteration in indexes:
            splits += [indexes[(tag, iteration)], indexes[(tag, iteration)] + len(tag)]
        splits += [len(prompt)]

        # cut the prompt
        sliced_prompt, slice_start, slice_end = [], 0, 1
        while slice_end < len(splits):
            sliced_prompt += [prompt[splits[slice_start] : splits[slice_end]]]
            slice_start += 2
            slice_end += 2

        return sliced_prompt
