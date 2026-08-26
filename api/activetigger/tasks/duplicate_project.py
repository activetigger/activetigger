import shutil
from pathlib import Path

from activetigger.tasks.base_task import BaseTask


class DuplicateProject(BaseTask):
    """
    Copy a project directory (and its static export dir, if present) into a
    new directory. DB cloning is done by the orchestrator in the main process
    once this task finishes — workers run in spawned processes and don't share
    the SQLAlchemy session manager.
    """

    kind = "duplicate_project"

    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        source_static_dir: str | None = None,
        target_static_dir: str | None = None,
    ):
        super().__init__()
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.source_static_dir = source_static_dir
        self.target_static_dir = target_static_dir

    def __call__(self) -> bool:
        if not Path(self.source_dir).exists():
            raise Exception(f"Source project directory missing: {self.source_dir}")
        if Path(self.target_dir).exists():
            raise Exception(f"Target project directory already exists: {self.target_dir}")
        shutil.copytree(self.source_dir, self.target_dir)
        if (
            self.source_static_dir
            and self.target_static_dir
            and Path(self.source_static_dir).exists()
        ):
            try:
                shutil.copytree(self.source_static_dir, self.target_static_dir)
            except Exception as e:
                # static export files are non-critical — log and continue
                print(f"Failed to copy static dir for duplication: {e}")
        return True
