#
# Worker entry points
#

import os
import traceback

# torch.cuda.is_available() poisons every later fork (children cannot use CUDA)
# unless it goes through NVML; third-party imports call it in the worker parent.
# Must be set before torch is imported anywhere in this process.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

from celery.signals import worker_process_init  # noqa: E402

from task_manager import config  # noqa: E402
from task_manager.auto_callback_task import QueueName  # noqa: E402
from task_manager.celery import celery_app  # noqa: E402


@worker_process_init.connect
def _check_cuda_fork(**_) -> None:
    """
    In a freshly forked child: warn when the parent already initialised CUDA
    (every GPU task would then fail with "Cannot re-initialize CUDA")
    """
    try:
        import torch
    except ImportError:
        return
    if torch.cuda._is_in_bad_fork():
        print(
            "ERROR: CUDA was initialised in the worker parent, GPU tasks will fail in this child. "
            "Set TASKMANAGER_TRACE_CUDA_INIT=1 to log where it happens."
        )


def _trace_cuda_init() -> None:
    """
    Opt-in diagnostic: print the stack of the first CUDA initialisation in this
    process (TASKMANAGER_TRACE_CUDA_INIT=1), to find what poisons the fork
    """
    if os.environ.get("TASKMANAGER_TRACE_CUDA_INIT") != "1":
        return
    import torch

    def wrap(owner, name):
        original = getattr(owner, name, None)
        if original is None:  # CPU-only torch build
            return

        def wrapped(*args, **kwargs):
            print(
                f"[cuda-trace pid {os.getpid()}] {name} first called from:\n"
                + "".join(traceback.format_stack(limit=30))
            )
            setattr(owner, name, original)
            return original(*args, **kwargs)

        setattr(owner, name, wrapped)

    wrap(torch._C, "_cuda_getDeviceCount")
    wrap(torch.cuda, "_lazy_init")


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

    Prefork with one task per child: each task runs in a fresh forked child
    that exits afterwards, so the model memory (allocator arenas, CUDA
    context) is returned to the OS, a child killed by the kernel (OOM) is
    reported through WorkerLostError, and a stop can terminate the child.
    Constraints: CUDA must never be initialised in the parent, i.e. at import
    time of a task module, and a child is daemonic, so a task cannot start
    sub-processes (DataLoader num_workers must be 0).
    Solo (--pool=solo) is the alternative if prefork causes trouble.
    """
    _trace_cuda_init()
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
