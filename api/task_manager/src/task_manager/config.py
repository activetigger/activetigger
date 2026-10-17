import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

# same .env as the API (api/.env in local dev); docker passes the real environment
load_dotenv()

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
worker_loglevel = os.environ.get("WORKER_LOGLEVEL", "INFO")
storage_path = os.environ.get("STORAGE_PATH", "/tmp")
cache_model_path = os.environ.get("TASKMANAGER_CACHE_MODEL_PATH", "/tmp")


def _concurrency(key: str, default: int) -> int:
    """Pool size from the env; at least 1 (celery reads 0 as "one per core")"""
    try:
        return max(1, int(os.environ.get(key, default)))
    except ValueError:
        return default


cpu_worker_concurrency = _concurrency("N_WORKERS_CPU", 5)
gpu_worker_concurrency = _concurrency("N_WORKERS_GPU", 1)

# orchestrator API reached by task callbacks; the default matches a local
# `python -m activetigger` (port 5000), docker sets ORCHESTRATOR_API_URL=http://api:4000
api_url = os.environ.get(
    "ORCHESTRATOR_API_URL", f"http://localhost:{os.environ.get('API_PORT', '5000')}"
)
api_task_success_route = f"{api_url}/tasks/done"
api_task_failure_route = f"{api_url}/tasks/failed"

# header carrying the shared secret that authenticates workers on the callback routes
TASK_TOKEN_HEADER = "X-Task-Token"


def load_or_create_task_callback_token(data_path: str) -> str:
    """
    Shared secret between the API and the workers: TASK_CALLBACK_TOKEN from the
    env, else <data_path>/.task_callback_token, created atomically on first use
    (the API and the workers may start at the same time).
    """
    env_token = os.environ.get("TASK_CALLBACK_TOKEN")
    if env_token:
        return env_token
    token_file = Path(data_path) / ".task_callback_token"
    if token_file.exists():
        stored = token_file.read_text().strip()
        if stored:
            return stored
    token_file.parent.mkdir(parents=True, exist_ok=True)
    new_token = secrets.token_urlsafe(32)
    tmp_file = token_file.with_name(f"{token_file.name}.{os.getpid()}.tmp")
    tmp_file.write_text(new_token)
    tmp_file.chmod(0o600)
    try:
        # link fails if the file was created meanwhile: the other process wins
        os.link(tmp_file, token_file)
    except FileExistsError:
        new_token = token_file.read_text().strip()
    finally:
        tmp_file.unlink(missing_ok=True)
    return new_token


task_callback_token = load_or_create_task_callback_token(os.environ.get("DATA_PATH", "."))
