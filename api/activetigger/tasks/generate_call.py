import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from activetigger.datamodels import (
    GeneratedRow,
    GenerationParams,
    GenRunSummary,
    PostprocessStep,
)
from activetigger.generations import Generations
from activetigger.tasks.base_task import BaseTask

# Flush results to a recovery file every N rows so a crash loses at most
# FLUSH_EVERY - 1 paid generations instead of the whole batch.
FLUSH_EVERY = 10


class GenerateCall(BaseTask):
    """
    Run a generative pipeline on a dataset.

    The sampled elements are read from path_input. The rows (element_id, prompt, raw,
    predicted, error) are written to the run's parquet file; the returned
    GenRunSummary lets the main process flip the run status. A jsonl file
    next to the parquet is the crash recovery buffer (recovered at project
    load, deleted on completion).
    """

    kind = "generate_call"

    def __init__(
        self,
        path_process: Path | None,
        run_id: int,
        path_input: Path,
        prompt: str,
        model_slug: str,
        endpoint: str,
        api_key: str,
        params: GenerationParams,
        steps: list[PostprocessStep],
        labels: list[str] | None,
        cols_context: list[str],
        path_output: Path,
        n_workers: int = 1,
        delete_input: bool = True,
    ):
        super().__init__()
        self.path_process = path_process if path_process is not None else Path(".")
        self.run_id = run_id
        self.path_input = path_input
        self.delete_input = delete_input
        self.prompt = prompt
        self.model_slug = model_slug
        self.endpoint = endpoint
        self.api_key = api_key
        self.params = params
        self.steps = steps
        self.labels = labels
        self.cols_context = cols_context
        self.path_output = path_output
        self.n_workers = max(1, int(n_workers))

    def _write_progress(self, progress: int) -> None:
        with open(self.path_process.joinpath(self.unique_id), "w") as f:
            f.write(f"{progress}")

    @staticmethod
    def get_progress_callback(path_file):
        def callback() -> int | None:
            try:
                with open(path_file, "r") as f:
                    return int(f.read())
            except Exception:
                return None

        return callback

    def _flush_to_jsonl(self, chunk: list[GeneratedRow]) -> None:
        with open(self.path_output.with_suffix(".jsonl"), "a") as f:
            for row in chunk:
                f.write(json.dumps(row.model_dump()) + "\n")

    def __call__(self) -> GenRunSummary:
        results: list[GeneratedRow] = []
        last_flushed = 0
        interrupted = False
        items = list(pd.read_parquet(self.path_input).iterrows())
        total = len(items)
        progress_path = self.path_process.joinpath(self.unique_id)

        def process_row(element_id, row) -> GeneratedRow:
            return Generations.run_on_row(
                row,
                str(element_id),
                self.prompt,
                self.model_slug,
                self.endpoint,
                self.api_key,
                self.params,
                self.steps,
                self.labels,
                self.cols_context,
            )

        try:
            self._write_progress(0)
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(process_row, eid, row) for eid, row in items]
                try:
                    for future in as_completed(futures):
                        if self.event is not None and self.event.is_set():
                            interrupted = True
                            for f in futures:
                                f.cancel()
                            break
                        results.append(future.result())
                        self._write_progress(int((len(results) / total) * 100))
                        if len(results) - last_flushed >= FLUSH_EVERY:
                            self._flush_to_jsonl(results[last_flushed:])
                            last_flushed = len(results)
                except Exception:
                    for f in futures:
                        f.cancel()
                    raise
        finally:
            if results:
                pd.DataFrame([r.model_dump() for r in results]).to_parquet(
                    self.path_output, index=False
                )
            self.path_output.with_suffix(".jsonl").unlink(missing_ok=True)
            if self.delete_input:
                self.path_input.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)

        return GenRunSummary(
            run_id=self.run_id,
            n_elements=len(results),
            n_na=sum(1 for r in results if r.predicted is None),
            interrupted=interrupted,
        )
