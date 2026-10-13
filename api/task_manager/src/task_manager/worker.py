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
    # without a GPU worker pool, GPU-routed tasks (training) would sit in the
    # GPU queue forever: consume both queues and run them on CPU
    queues = QueueName.CPU if config.gpu_available else f"{QueueName.CPU},{QueueName.GPU}"
    args = [
        "worker",
        f"--loglevel={config.worker_loglevel}",
        f"--concurrency={config.cpu_worker_concurrency}",
        f"--queues={queues}",
        "--hostname=worker_cpu@%h",
        # "--purge",
    ]
    celery_app.worker_main(args)


def start_gpu_worker_pool():
    """
    Start worker.s for the GPU queue
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
