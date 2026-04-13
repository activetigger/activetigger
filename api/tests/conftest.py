from collections.abc import Generator
from fastapi.testclient import TestClient
import os
import pytest

from activetigger.config import config
from activetigger.app.main import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def superuser_token_headers(client: TestClient) -> dict[str, str]:
    login_data = {
        "username": config.default_user,
        "password": os.environ.get("ROOT_PASSWORD", "l3tm31n!"),
    }
    r = client.post(f"/api/token", data=login_data)
    tokens = r.json()
    a_token = tokens["access_token"]
    headers = {"Authorization": f"Bearer {a_token}"}
    return headers
