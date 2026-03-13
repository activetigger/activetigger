####
# Queue management for ActiveTigger
# This module manages the queue of tasks for CPU and GPU processing using Loky.
# There is no multi-GPU support in this version.
####

import asyncio
import datetime
import multiprocessing
import uuid

# manage the executor
from loky import get_reusable_executor  # type: ignore[import]

from activetigger.datamodels import QueueStateTaskModel, QueueTaskModel
from activetigger.tasks.base_task import BaseTask

multiprocessing.set_start_method("spawn", force=True)


class Queue:
    """
    Managining parallel processes for computation

    Use Loky as a backend for one common executor

    With 2 waiting queues to differentiate between CPU and GPU jobs
    (to limit concurrency in GPU memory usage)
    """

    max_processes: int = 20
    nb_workers: int
    nb_workers_cpu: int
    nb_workers_gpu: int
    current: list[QueueTaskModel]
    last_restart: datetime.datetime

    def __init__(self, nb_workers_cpu: int = 3, nb_workers_gpu: int = 1) -> None:
        """
        Initiating the queue
        :param nb_workers_cpu: Number of CPU workers
        :param nb_workers_gpu: Number of GPU workers
        :return: None
        """
        self.nb_workers_cpu = nb_workers_cpu
        self.nb_workers_gpu = nb_workers_gpu
        self.nb_workers = nb_workers_cpu + nb_workers_gpu
        self.current = []

        # manager for cross-process event signaling
        self.manager = multiprocessing.Manager()

        # create the executor
        self.executor = get_reusable_executor(
            max_workers=self.nb_workers, timeout=14400
        )  # 4 hours timeout for hung workers

        # launch a regular update on the queue
        self.task = asyncio.create_task(self._update_queue(timeout=0.5))

    def __del__(self) -> None:
        """
        Destructor to close the queue
        """
        if hasattr(self, "task"):
            self.task.cancel()
        if hasattr(self, "manager"):
            self.manager.shutdown()

    def _dispatch_pending_tasks(self) -> None:
        """
        Check for pending tasks and submit them to the executor.
        Runs in a thread to avoid blocking the event loop
        (executor.submit may spawn worker processes).
        """
        # active tasks in the queue
        nb_active_processes_gpu = len(
            [i for i in self.current if i.queue == "gpu" and i.state == "running"]
        )
        nb_active_processes_cpu = len(
            [i for i in self.current if i.queue == "cpu" and i.state == "running"]
        )

        # pending tasks in the queue
        task_gpu = [i for i in self.current if i.queue == "gpu" and i.state == "pending"]
        task_cpu = [i for i in self.current if i.queue == "cpu" and i.state == "pending"]

        # a worker available and possible to have gpu
        if (
            nb_active_processes_gpu < self.nb_workers_gpu
            and (nb_active_processes_gpu + nb_active_processes_cpu) < self.nb_workers
            and len(task_gpu) > 0
        ):
            executor = get_reusable_executor(max_workers=(self.nb_workers), timeout=600)
            task_gpu[0].future = executor.submit(task_gpu[0].task)
            task_gpu[0].state = "running"

        # a worker available and possible to have cpu
        if (
            nb_active_processes_cpu < self.nb_workers_cpu
            and (nb_active_processes_gpu + nb_active_processes_cpu) < self.nb_workers
            and len(task_cpu) > 0
        ):
            executor = get_reusable_executor(max_workers=(self.nb_workers), timeout=600)
            task_cpu[0].future = executor.submit(task_cpu[0].task)
            task_cpu[0].state = "running"

        # if there is nothing in the queue, shutdown the executor to free GPU memory
        if (
            nb_active_processes_cpu + nb_active_processes_gpu == 0
            and len(task_gpu) == 0
            and len(task_cpu) == 0
        ):
            executor = get_reusable_executor(max_workers=(self.nb_workers), timeout=600)
            executor.shutdown(wait=False)

    async def _update_queue(self, timeout: float = 1) -> None:
        """
        Update the queue every X seconds.
        Add new tasks to the executor if there are available workers.
        """
        while True:
            try:
                await asyncio.to_thread(self._dispatch_pending_tasks)
            except asyncio.CancelledError:
                print("Queue update task cancelled.")
                return
            except Exception as e:
                print(f"Error in queue dispatch: {e}")
            await asyncio.sleep(timeout)

    def add_task(self, kind: str, project_slug: str, task: BaseTask, queue: str = "cpu") -> str:
        """
        Add a task in the queue, first as pending in the current list
        """
        # test if the queue is not full
        if len(self.current) > self.max_processes:
            raise Exception("Queue is full. Wait for process to finish.")

        # generate a unique id
        unique_id = str(uuid.uuid4())
        task.unique_id = unique_id

        # set an event to inform the end of the process
        event = self.manager.Event()
        task.event = event

        # add it in the current processes
        self.current.append(
            QueueTaskModel(
                unique_id=unique_id,
                kind=kind,
                project_slug=project_slug,
                state="pending",
                future=None,
                event=event,
                starting_time=datetime.datetime.now(),
                queue=queue,
                task=task,
            )
        )

        return unique_id

    def get(self, unique_id: str) -> QueueTaskModel | None:
        """
        Get a running process
        """
        element = [i for i in self.current if i.unique_id == unique_id]
        if len(element) == 0:
            return None
        return element[0]

    def kill(self, unique_id: str) -> None:
        """
        Send a kill process with the event manager
        """
        element = [i for i in self.current if i.unique_id == unique_id]
        if len(element) == 0:
            raise Exception("Process not found")
        element[0].event.set()  # TODO update status to flag the killing
        self.delete(unique_id)  # TODO move this to the cleaning method

    def delete(self, ids: str | list) -> None:
        """
        Delete completed elements from the stack
        """
        if isinstance(ids, str):
            ids = [ids]
        for i in [t for t in self.current if t.unique_id in ids]:
            if i.future is None or not i.future.done():
                print("Deleting a unfinished process")
            self.current.remove(i)

    def state(self) -> list[QueueStateTaskModel]:
        """
        Return state of the queue
        """
        return [
            QueueStateTaskModel(
                unique_id=process.unique_id,
                state="done" if process.future and process.future.done() else process.state,
                exception=str(process.future.exception())
                if process.future and process.future.done() and process.future.exception()
                else None,
                kind=process.kind,
            )
            for process in self.current
        ]

    def get_nb_waiting_processes(self, queue: str = "cpu") -> int:
        """
        Number of waiting processes
        """
        return len([f for f in self.current if f.queue == queue and f.state == "pending"])

    def display_info(self, renew: int = 20) -> None:
        """
        Check if the exector still works
        if not, recreate it
        """
        print(self.state())
        print(
            "waiting",
            self.get_nb_waiting_processes("cpu"),
            self.get_nb_waiting_processes("gpu"),
        )
        return None

    def clean_old_processes(self, timeout: int = 2) -> None:
        """
        Remove old processes
        """
        n = len(self.current)
        self.current = [
            i
            for i in self.current
            if (datetime.datetime.now() - i.starting_time).total_seconds() / 3600 < timeout
        ]
        if n != len(self.current):
            print(f"Cleaned {n - len(self.current)} processes")
        return None

    def restart(self) -> None:
        """
        Restart the queue by getting the executor and closing it
        """
        executor = get_reusable_executor(max_workers=(self.nb_workers), timeout=600)
        executor.shutdown(wait=False)
        self.current = []
