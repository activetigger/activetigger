import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from activetigger.app.main import app
from activetigger.config import config


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
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
