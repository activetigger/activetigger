import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from activetigger.datamodels import GenerationModel, GenerationResult
from activetigger.generation.client import GenerationModelClient
from activetigger.generation.huggingface import HuggingFace
from activetigger.generation.ollama import Ollama
from activetigger.generation.openai import OpenAI
from activetigger.generation.openapi import OpenAPI
from activetigger.generation.openrouter import OpenRouter
from pandas import DataFrame, Series

# Flush in-memory results to a recovery file every N rows so a crash loses at
# most FLUSH_EVERY - 1 paid generations instead of the whole batch.
FLUSH_EVERY = 10


def make_client(model: GenerationModel) -> GenerationModelClient:
    """
    Build the generation client matching the configured API.
    """
    if model.api == "Ollama":
        if model.endpoint is None:
            raise Exception("You need to give an endpoint for the Ollama model")
        return Ollama(model.endpoint)
    if model.api == "OpenAI":
        if model.credentials is None:
            raise Exception("You need to give your API key to call an OpenAI model")
        return OpenAI(model.credentials)
    if model.api == "HuggingFace":
        return HuggingFace(credentials=model.credentials, endpoint=model.endpoint)
    if model.api == "OpenRouter":
        return OpenRouter(credentials=model.credentials)
    if model.api == "OpenAICompatible":
        if not model.endpoint:
            raise Exception("You need to provide an endpoint for the OpenAI-compatible model")
        return OpenAPI(endpoint=model.endpoint, credentials=model.credentials)
    raise Exception(f"Unknown model API: {model.api}")


def _write_progress(path_progress: Path, progress: int) -> None:
    with open(path_progress, "w") as f:
        f.write(f"{progress}")
    print(f"Progress: {progress}")


def _flush_to_jsonl(path_jsonl: Path, chunk: list[GenerationResult], batch: str) -> None:
    with open(path_jsonl, "a") as f:
        for r in chunk:
            payload = r.model_dump()
            payload["batch"] = batch
            f.write(json.dumps(payload) + "\n")


def generate_call(
    df: DataFrame,
    model: GenerationModel,
    prompt: str,
    cols_context: list[str],
    username: str,
    project_slug: str,
    batch: str,
    path_progress: Path,
    path_jsonl: Path,
    n_workers: int = 1,
    is_aborted: Callable[[], bool] | None = None,
) -> tuple[int, int]:
    """
    Manage batch generation request. Every result is written to path_jsonl
    (read back by the API), returns (n_results, n_errors).
    Calls are issued through a thread pool sized by n_workers (1 = sequential).
    """
    n_workers = max(1, int(n_workers))
    errors: list[Exception] = []
    results: list[GenerationResult] = []
    last_flushed = 0

    gen_model = make_client(model)
    rows = list(df.iterrows())
    total = len(rows)

    def process_row(index, row) -> GenerationResult:
        prompt_with_text = replace_tags_with_text(row, prompt, cols_context)
        response = gen_model.generate(prompt_with_text, model.slug)
        return GenerationResult(
            user=username,
            project_slug=project_slug,
            model_id=model.id,
            element_id=str(index),
            prompt=prompt_with_text,
            answer=response,
        )

    try:
        _write_progress(path_progress, 0)
        completed = 0
        interrupted = False
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_row, idx, row) for idx, row in rows]
            try:
                for future in as_completed(futures):
                    if is_aborted is not None and is_aborted():
                        interrupted = True
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        results.append(future.result())
                    except Exception as e:
                        errors.append(e)
                    completed += 1
                    _write_progress(path_progress, int((completed / total) * 100))
                    if len(results) - last_flushed >= FLUSH_EVERY:
                        _flush_to_jsonl(path_jsonl, results[last_flushed:], batch)
                        last_flushed = len(results)
            except Exception:
                for f in futures:
                    f.cancel()
                raise

        if len(results) > last_flushed:
            _flush_to_jsonl(path_jsonl, results[last_flushed:], batch)

        if interrupted:
            raise Exception("Process interrupted by user")

        # surface failures instead of silently returning a partial/empty batch
        if errors and not results:
            raise Exception(f"All {len(errors)} generation calls failed: {errors[0]}")
        if errors:
            print(f"Generation batch {batch}: {len(errors)}/{total} calls failed: {errors[0]}")

        return len(results), len(errors)
    finally:
        path_progress.unlink(missing_ok=True)


def replace_tags_with_text(row: Series, prompt: str, context_columns: list[str]) -> str:
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

    sliced_prompt = _slice_prompt(indexes, prompt)

    # Insert the contents
    complete_prompt = sliced_prompt[0]
    for i, (tag, iteration) in enumerate(indexes.keys()):
        if tag == "[[TEXT]]":
            complete_prompt += str(row["text"])
        else:
            complete_prompt += str(row[unformat(tag)])
        complete_prompt += sliced_prompt[i + 1]
    return complete_prompt


def _slice_prompt(indexes: dict, prompt: str) -> list[str]:
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
