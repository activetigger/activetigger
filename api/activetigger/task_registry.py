import threading
import time
from typing import Iterator, cast

from celery import chord, uuid
from celery.contrib.abortable import AbortableAsyncResult, AbortableTask
from celery.result import AsyncResult
from task_manager.active_tigger_task import (
    ActiveTiggerTask,
    QueueName,
    TaskScope,
)
from task_manager.celery import celery_app
from task_manager.task_input import TaskInputs
from task_manager.task_names import TaskNames
from task_manager.tasks.callback_tasks import FailureCallbackTask, SuccessCallbackTask

from activetigger.datamodels import (
    EventsModel,
    TaskComputing,
    TaskEndedEvent,
    TaskEndedReason,
    TaskEndObserver,
    TaskProgressState,
)
from activetigger.monitoring import Monitoring
from activetigger.tasks.task_manager_client import celery_task_state, celery_tasks_in_flight


class TaskRegistry:
    """
    Single owner of the in-memory picture of tasks dispatched to workers.

    Protects the state by construction:
    - the underlying dicts are name-mangled private;
    - every accessor returns a snapshot (list copy), never the live dict;
    - all mutation goes through explicit methods, serialized by a lock
      (callbacks run in the FastAPI threadpool, the update loop in a thread).

    This class must stay a LEAF module: imports only datamodels / task_manager,
    never orchestrator / project / features / prompts.
    """

    def __init__(self, monitoring: Monitoring) -> None:
        self._lock = threading.RLock()
        self._computing_tasks: dict[str, TaskComputing] = {}
        self.monitoring = monitoring

        # orchestrator callbacks
        self._observers: list[TaskEndObserver] = []

        # reconciliation is a BACKSTOP, the chord
        # bodies guarantee callback delivery while any worker lives. It only
        # fires for tasks orphaned by infra loss (worker container gone with
        # the task, redis backend flushed) — nothing else will ever report them.
        self._reconcile_interval = 60
        # the chord body retries delivery for ~1 min (exponential backoff +
        # jitter); grace is comfortably above that so a slow-but-delivered
        # callback is never dropped
        self._reconcile_grace = 180
        self._last_reconcile = 0.0
        self._orphaned_since: dict[str, float] = {}

    # ---------- public API ------------

    def start_task(
        self, task_name: TaskNames, inputs: TaskInputs, queue: QueueName | None = None
    ) -> str:
        """
        Send a task to the workers. With a process, register it in computing first:
        a task that fails or finishes at once calls back before the dispatcher returns.
        """
        if task_name not in celery_app.tasks:
            raise Exception(
                f"task {task_name} not registered in celery app. Possible tasks are {celery_app.tasks.keys()}"
            )
        # create TaskComputing from tasks input
        task_class = cast(ActiveTiggerTask, celery_app.tasks[task_name])

        task_id = uuid()
        callback_task_id = uuid()
        computing_task = TaskComputing(
            id=task_id,
            callback_id=callback_task_id,
            kind=task_name,
            user=inputs.user,
            project_slug=inputs.project_slug,
            inputs=inputs,
        )
        print(f"add new task in registry {computing_task}")
        with self._lock:
            self._computing_tasks[task_id] = computing_task

        # TODO: monitoring does not support orchestrator scope tasks
        print(
            f"create task {computing_task.kind} {computing_task.project_slug} {computing_task.id}"
        )
        if task_class.scope == "project" and computing_task.project_slug:
            # registered before the send: a fast task calls back at once
            self.monitoring.register_process(
                process_name=task_id,
                project_slug=computing_task.project_slug,
                kind=task_name,
                parameters={},
                user_name=inputs.user,
            )

        # queue overrides the task default (e.g. GPU for a wtpsplit preparation)

        try:
            main_task = (
                task_class.s(inputs=inputs.model_dump(mode="json"))
                .set(queue=queue or task_class.queue, task_id=task_id)
                .on_error(FailureCallbackTask().s(task_name=task_name, main_task_id=task_id))
            )
            callback_task = (
                SuccessCallbackTask()
                .s(task_name=task_name, main_task_id=task_id)
                .set(task_id=callback_task_id)
            )

            chord(main_task)(callback_task)

        except Exception as e:
            print(e)
            self.on_task_end(computing_task.id, "error")
            raise
        return task_id

    def on_task_end(
        self,
        task_id: str,
        reason: TaskEndedReason,
        events: EventsModel | None = None,
        error_message: str | None = None,
    ) -> None:
        """
        When task ends, this method must be called to delete the task representation from the orchestrator memory
        """
        with self._lock:
            computing_task = self._pop_computing_task(task_id)
        if computing_task is None:
            return
        print(f"terminating {computing_task}")
        # TODO: monitoring does not support orchestrator scope tasks
        if TaskRegistry.task_scope(computing_task.kind) == "project":
            self.monitoring.close_process(task_id, events)
        print(f"clean files {computing_task}")
        self._clean_temp_files(computing_task)
        print(f"emit task _end {computing_task}")
        self._emit_task_end(
            TaskEndedEvent(
                computing_task=computing_task,
                reason=reason,
                events=events,
                error_message=error_message,
            )
        )

    @staticmethod
    def task_scope(task_name: TaskNames) -> TaskScope:
        task_class = cast(ActiveTiggerTask, celery_app.tasks[task_name])
        return task_class.scope

    @staticmethod
    def is_task_abortable(task_name: TaskNames) -> bool:
        task_class = cast(ActiveTiggerTask, celery_app.tasks[task_name])
        return isinstance(task_class, AbortableTask)

    def stop_task(self, computing_task: TaskComputing):
        if TaskRegistry.is_task_abortable(computing_task.kind):
            task = AbortableAsyncResult(computing_task.id, app=celery_app)
            print(f"Aborting {computing_task.id} {type(task).__name__}")
            # abort() is a fallback flag; revoke drops a task still in the queue
            task.abort()
            celery_app.control.revoke(computing_task.id)
        else:
            print(f"Revoke task {computing_task.id}")
            celery_app.control.revoke(computing_task.id, terminate=True, signal="SIGKILL")

    # Stop one task by its id + project if appropriate
    def stop_task_by_id(self, task_id: str):
        computing_task = self.get_computing_task(task_id)

        if computing_task is None:
            # we don't raise here as it looks like the job has already been done
            print(f"Could not find task {task_id} in task_registry")
        else:
            self.stop_task(computing_task)

    def stop_user_tasks(
        self,
        username: str,
        only_project_slug: str | None = None,
        only_task_names: list[TaskNames] | None = None,
    ) -> None:
        """
        Stop all tasks created by a user with two options to restrict to project and/or kind
        """

        def should_stop_computing_task(ct: TaskComputing):
            return (
                ct.user == username
                and (only_project_slug is None or ct.project_slug == only_project_slug)
                and (only_task_names is None or ct.kind in only_task_names)
            )

        with self._lock:
            # find matching computing_tasks
            computing_tasks_to_stop: list[TaskComputing] = []
            for ct in self._computing_tasks.values():
                if should_stop_computing_task(ct):
                    computing_tasks_to_stop.append(ct)

        for ct in computing_tasks_to_stop:
            self.stop_task(ct)

    def stop_project_tasks(self, project_slug: str, username: str = "system") -> None:
        """
        Stop every task of a project (deleted, cleaned or reset)
        """
        # copy: stop_process removes entries from the list
        for computing_task in self.project_computing_tasks(project_slug):
            self.stop_task(computing_task)

    def stop_all_tasks(self):
        """
        Stop every task
        """
        for ct in self.all_computing_tasks():
            self.stop_task(ct)

    # get computing_task object from its id and scope
    def get_computing_task(self, task_id: str) -> TaskComputing:

        with self._lock:
            # orchestrator scope
            if task_id in self._computing_tasks:
                return self._computing_tasks[task_id]
            else:
                raise Exception(f"on_task_end called on an untracked task {task_id}")

    def projects_with_running_tasks(self) -> set[str]:
        with self._lock:
            return set(
                computing_task.project_slug
                for computing_task in self._computing_tasks.values()
                if computing_task.project_slug is not None
            )

    def project_creation_ongoing_computing_tasks(
        self,
    ) -> set[str]:
        with self._lock:
            return set(
                (
                    ct.project_slug
                    for ct in self._computing_tasks.values()
                    if ct.kind == "create_project" and ct.project_slug
                )
            )

    def project_duplication_ongoing_computing_tasks(
        self,
    ) -> set[str]:
        with self._lock:
            return set(
                (
                    ct.project_slug
                    for ct in self._computing_tasks.values()
                    if ct.kind == "duplicate_project" and ct.project_slug
                )
            )

    def all_computing_tasks(self) -> Iterator[TaskComputing]:
        """Snapshot iterator over every task (both scopes)."""
        with self._lock:
            all_cts = list(self._computing_tasks.values())
        return iter(all_cts)

    def project_computing_tasks(
        self, project_slug: str, user: str | None = None, kinds: list[TaskNames] | None = None
    ) -> list[TaskComputing]:
        with self._lock:
            return [
                ct
                for ct in self._computing_tasks.values()
                if (not user or ct.user == user)
                and (not kinds or ct.kind in kinds)
                and ct.project_slug == project_slug
            ]

    def task_progress(self, task_id: str) -> TaskProgressState | None:
        r = AsyncResult(task_id, app=celery_app)
        if r.state == "PROGRESS":
            return r.info
        return None

    # ---------------- internals ----------------

    def _pop_computing_task(self, task_id: str) -> TaskComputing | None:
        """Caller must hold the lock (or accept the small race and tolerate
        a None return). Idempotent: a stop and a late callback can race."""
        if task_id in self._computing_tasks:
            return self._computing_tasks.pop(task_id, None)
        return None

    def _clean_temp_files(self, computing_task: TaskComputing):
        task_class = cast(ActiveTiggerTask, celery_app.tasks[computing_task.kind])
        # clean temp files
        if task_class.clean_temp_files:
            task_class.clean_temp_files(computing_task)

    # ---------------- subscription API ----------------

    def subscribe_task_end(self, observer: TaskEndObserver) -> None:
        """Register a hook fired after any task leaves the registry.

        Called with the registry lock NOT held, after state has been
        updated. Observers may call registry methods freely.
        """
        self._observers.append(observer)

    def _emit_task_end(self, event: TaskEndedEvent) -> None:
        """Fire observers, isolated from each other."""
        for observer in list(self._observers):
            try:
                observer(event)
            except Exception as e:
                print(f"TaskRegistry task-end observer failed: {e}")

    # ---------------- sync with celery ----------------

    def maybe_reconcile(self) -> None:
        """Cheap periodic sweep; call from the update loop."""
        if time.time() - self._last_reconcile < self._reconcile_interval:
            return
        self._last_reconcile = time.time()
        self._reconcile_tasks()

    def _reconcile_tasks(self) -> None:
        """
        Backstop for tasks orphaned by infra loss. With chord-mediated
        callbacks, a task leaves the registry when the API acknowledges its
        success/failure callback. An entry can therefore only linger if:

        - its chord died with the infrastructure (worker container holding the
          main task or a callback body disappeared, redis flushed): the task
          is neither queued nor held by any worker, or
        - its celery state is terminal (SUCCESS/FAILURE/REVOKED) past the
          grace period, meaning the whole retry envelope of the callback body
          elapsed without delivery.

        Both mean genuine loss: report loudly, not as routine cleanup.
        """
        if celery_app.conf.task_always_eager:
            return
        in_flight = celery_tasks_in_flight()
        if in_flight is None:
            # cannot verify anything: skip this round, retry next interval
            return
        now = time.time()

        with self._lock:
            tracked = {ct.id: (ct.project_slug, ct.kind) for ct in self._computing_tasks.values()}
            # forget ids whose task left the registry meanwhile
            for task_id in [t for t in self._orphaned_since if t not in tracked]:
                del self._orphaned_since[task_id]

        for task_id, (slug, kind) in tracked.items():
            state = celery_task_state(task_id)
            # a task still queued or held by a worker is not orphaned;
            # a pending state likewise (not started or in a callback retry)
            if task_id in in_flight or state in ("PENDING", "PROGRESS", "RETRY", "STARTED"):
                self._orphaned_since.pop(task_id, None)
                continue

            # terminal (SUCCESS/FAILURE/REVOKED) or vanished: orphan candidate
            since = self._orphaned_since.setdefault(task_id, now)
            if now - since < self._reconcile_grace:
                continue

            del self._orphaned_since[task_id]
            reason = (
                f"celery state {state}, not held by any worker"
                if state != "PENDING"
                else "unknown to celery: task and result lost (redis flushed?)"
            )
            # infra loss is exceptional now: make it visible
            print(
                f"ORPHANED TASK — dropping {task_id} ({kind}, {slug}): {reason}. "
                f"No callback can ever arrive; investigate the workers/redis."
            )
            self.on_task_end(
                task_id,
                "lost",
                error_message=f"Task {kind} {task_id} lost ({reason})",
            )
