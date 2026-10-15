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
        f"--queues={QueueName.CPU}",
        "--hostname=worker_cpu@%h",
        # "--purge",
    ]
    celery_app.worker_main(args)


def start_gpu_worker_pool():
    """
    Start the worker for the GPU queue.

    Heavy tasks (training, embeddings, prediction) are routed to this queue
    whether or not a GPU exists: the solo pool runs them one at a time, which
    bounds memory use. Without a GPU they simply run on CPU.
    """
    args = [
        "worker",
        f"--loglevel={config.worker_loglevel}",
        # we might need to create one queue by GPU device having each pool solo
        f"--concurrency={config.gpu_worker_concurrency}",
        "--pool=solo",
        f"--queues={QueueName.GPU}",
        "--hostname=worker_gpu@%h",
        # "--purge",
    ]
    celery_app.worker_main(args)
