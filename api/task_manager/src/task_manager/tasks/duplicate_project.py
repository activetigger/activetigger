import shutil
from pathlib import Path


def duplicate_project_files(
    source_dir: Path,
    target_dir: Path,
    source_static_dir: Path | None = None,
    target_static_dir: Path | None = None,
) -> None:
    """
    Copy a project directory (and its static export dir, if present) into a
    new directory. DB cloning is done by the API in the callback.
    """
    if not source_dir.exists():
        raise Exception(f"Source project directory missing: {source_dir}")
    if target_dir.exists():
        raise Exception(f"Target project directory already exists: {target_dir}")
    shutil.copytree(source_dir, target_dir)
    if source_static_dir and target_static_dir and source_static_dir.exists():
        try:
            shutil.copytree(source_static_dir, target_static_dir)
        except Exception as e:
            # static export files are non-critical — log and continue
            print(f"Failed to copy static dir for duplication: {e}")
