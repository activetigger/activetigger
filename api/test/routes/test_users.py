import time

from fastapi.testclient import TestClient

from test.utils import create_user


def test_users_superuser_me(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing the /users/me endpoint for the superuser.
    """

    r = client.get("/api/users/me", headers=superuser_headers)
    current_user = r.json()
    assert current_user
    assert current_user["username"] == "root"
    assert current_user["status"] == "root"


def test_users_creation(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing the creation of a user and the /users/me endpoint for that user.
    """

    username = f"testuser-{int(time.time())}"
    headers = create_user(client, superuser_headers, username)
    r = client.get("/api/users/me", headers=headers)
    current_user = r.json()

    assert current_user
    assert current_user["username"] == username
    assert current_user["status"] == "active"


def test_users_list(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing the users list.
    """

    r = client.get("/api/users", headers=superuser_headers)
    users_map = r.json()
    nb_users = len(users_map)

    assert nb_users >= 2
    assert users_map["root"]
    assert users_map["root"]["username"] == "root"

    assert users_map["demo"]
    assert users_map["demo"]["username"] == "demo"
