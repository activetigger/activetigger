import os
import traceback
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from task_manager.celery import celery_app
from task_manager.tasks import callback_tasks
from task_manager.tasks.callback_tasks import (
    FailureCallbackPayload,
    SuccessCallbackPayload,
)

# tasks run synchronously in the test process: no broker, no worker.
# Must come before importing the app: building the orchestrator purges the
# celery queues, which would hit the real redis (and the dev workers) otherwise.
celery_app.conf.update(
    task_always_eager=True,
    task_store_eager_result=True,
    broker_url="memory://",
    result_backend="cache+memory://",
)

from activetigger.app.main import app  # noqa: E402
from activetigger.config import config  # noqa: E402
from activetigger.tasks import all_callbacks  # noqa: E402
from activetigger.tasks.task_callback import ProjectNotTrackedError  # noqa: E402


def _deliver_callback(url: str, payload: SuccessCallbackPayload | FailureCallbackPayload) -> bool:
    """
    Replace the HTTP callback of the workers: the TestClient app listens on no
    port, so the result goes straight to the callback class (same JSON round trip)
    """
    callback = all_callbacks.task_callbacks[payload.task_name]
    try:
        if url.endswith("/done") and isinstance(payload, SuccessCallbackPayload):
            callback.on_complete(payload)
        elif isinstance(payload, FailureCallbackPayload):
            callback.on_failure(payload)
    except ProjectNotTrackedError as e:
        print(f"callback dropped: {e}")
        return False
    except Exception:
        traceback.print_exc()
        return False
    return True


@pytest.fixture(scope="session", autouse=True)
def in_process_task_callbacks() -> Generator[None, None, None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(callback_tasks, "_post_callback", _deliver_callback)
        yield


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def superuser_headers(client: TestClient) -> dict[str, str]:
    """
    Fixture to get the headers for the superuser.
    """

    login_data = {
        "username": config.default_user,
        "password": os.environ.get("ROOT_PASSWORD", "l3tm31n!"),
    }
    r = client.post("/api/token", data=login_data)
    tokens = r.json()
    a_token = tokens["access_token"]
    headers = {"Authorization": f"Bearer {a_token}"}
    return headers
