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


def test_users_credentials(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing add/list/delete of saved endpoint/credentials pairs.
    """

    username = f"testuser-cred-{int(time.time())}"
    headers = create_user(client, superuser_headers, username)

    r = client.get("/api/users/credentials", headers=headers)
    assert r.status_code == 200
    assert r.json() == []

    r = client.post(
        "/api/users/credentials",
        headers=headers,
        json={
            "name": "my-key",
            "api": "OpenRouter",
            "endpoint": "https://api.example.com",
            "credentials": "sk-secret",
        },
    )
    assert r.status_code == 200

    r = client.get("/api/users/credentials", headers=headers)
    entries = r.json()
    assert entries == [
        {"name": "my-key", "api": "OpenRouter", "endpoint": "https://api.example.com"}
    ]
    # the secret never appears in any response
    assert "sk-secret" not in r.text

    # entries are private to their owner
    r = client.get("/api/users/credentials", headers=superuser_headers)
    assert all(e["name"] != "my-key" for e in r.json())

    r = client.post("/api/users/credentials/delete", headers=headers, params={"name": "my-key"})
    assert r.status_code == 200
    r = client.get("/api/users/credentials", headers=headers)
    assert r.json() == []

    r = client.post("/api/users/credentials/delete", headers=headers, params={"name": "my-key"})
    assert r.status_code == 400


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
