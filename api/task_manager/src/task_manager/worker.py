#
# Worker entry points
#


from task_manager import config
from task_manager.auto_callback_task import QueueName
from task_manager.celery import celery_app


def start_cpu_worker_pool():
    """
    Start worker.s for the CPU queue
    """
    args = [
        "worker",
        f"--loglevel={config.worker_loglevel}",
        f"--concurrency={config.cpu_worker_concurrency}",
        # a child is forked from the already-imported parent (cheap) and exits
        # after one task, so the memory a task allocated is returned to the OS
        "--max-tasks-per-child=1",
        f"--queues={QueueName.CPU}",
        "--hostname=worker_cpu@%h",
        # "--purge",
    ]
    celery_app.worker_main(args)


def start_gpu_worker_pool():
    """
    Start the worker for the GPU queue.

    Heavy tasks (training, embeddings, prediction) are routed to this queue
    whether or not a GPU exists and run one at a time (concurrency 1 by
    default), which bounds memory use. Without a GPU they simply run on CPU.

    Each task runs in a fresh forked child that exits afterwards: a solo pool
    would keep the model memory (allocator arenas, CUDA context) in the long-
    lived worker process. CUDA must therefore never be initialised in the
    parent, i.e. at import time of a task module.
    """
    args = [
        "worker",
        f"--loglevel={config.worker_loglevel}",
        # we might need to create one queue by GPU device
        f"--concurrency={config.gpu_worker_concurrency}",
        "--pool=prefork",
        "--max-tasks-per-child=1",
        f"--queues={QueueName.GPU}",
        "--hostname=worker_gpu@%h",
        # "--purge",
    ]
    celery_app.worker_main(args)
