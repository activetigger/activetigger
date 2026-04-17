import time

from fastapi.testclient import TestClient

from test.utils import create_project


def test_projects_creation(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing the projects endpoint for the superuser.
    """

    project_name = f"Test-{int(time.time())}"
    project = create_project(client, superuser_headers, project_name)

    assert project
    assert project["project_name"] == project_name
