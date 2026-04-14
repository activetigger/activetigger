import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient


def test_get_users_superuser_me(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    r = client.get(f"/api/users/me", headers=superuser_token_headers)
    current_user = r.json()
    assert current_user
    assert current_user["username"] == "root"
    assert current_user["status"] == "root"
