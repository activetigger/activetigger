import json
import os
import traceback
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from task_manager.celery import celery_app
from task_manager.utils import TaskFailureReportForCallback

from activetigger.app.main import app
from activetigger.config import config
from activetigger.tasks import all_callbacks
from activetigger.tasks.task_callback import ProjectNotTrackedError
from task_manager import utils as task_utils

# tasks run synchronously in the test process: no broker, no worker
celery_app.conf.update(
    task_always_eager=True,
    task_store_eager_result=True,
    broker_url="memory://",
    result_backend="cache+memory://",
)


def _deliver_callback(url: str, payload: Any) -> bool:
    """
    Replace the HTTP callback of the workers: the TestClient app listens on no
    port, so the result goes straight to the callback class (same JSON round trip)
    """
    body = json.loads(payload.model_dump_json())
    callback = all_callbacks.task_callbacks[body["task_name"]]
    try:
        if url.endswith("/done"):
            callback.on_complete(body["task_id"], body["results"])
        else:
            callback.on_failure(body["task_id"], TaskFailureReportForCallback(**body["report"]))
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
        mp.setattr(task_utils, "_post_callback", _deliver_callback)
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
