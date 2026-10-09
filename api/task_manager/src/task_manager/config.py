import os

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
worker_loglevel = os.environ.get("WORKER_LOGLEVEL", "DEBUG")
storage_path = os.environ.get("STORAGE_PATH", "/tmp")
cache_model_path = os.environ.get("TASKMANAGER_CACHE_MODEL_PATH", "/tmp")
cpu_worker_concurrency = os.environ.get("N_WORKERS_CPU", "1")
gpu_worker_concurrency = os.environ.get("N_WORKERS_GPU", "1")

# orchestrator API reached by task callbacks
api_url = os.environ.get("API_URL", f"http://api:{os.environ.get('API_PORT', '4000')}")
api_task_success_route = f"{api_url}/tasks/done"
api_task_failure_route = f"{api_url}/tasks/failed"
