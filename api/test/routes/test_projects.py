import time

from fastapi.testclient import TestClient

from test.utils import create_project, delete_project


def test_projects_creation(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing the projects endpoint for the superuser.
    """

    project_name = f"Test-{int(time.time())}"
    project = create_project(client, superuser_headers, project_name)

    assert project
    assert project["project_name"] == project_name


def test_projects_delete(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing deleting a project.
    """

    project_name = f"Test-delete-{int(time.time())}"
    project = create_project(client, superuser_headers, project_name)
    project_slug = project["project_slug"]

    # Delete the project
    delete_project(client, superuser_headers, project_slug)

    # Verify the project no longer exists
    response = client.get(
        f"/api/projects/status?project_name={project_slug}",
        headers=superuser_headers,
    )
    assert response.status_code == 200
    assert response.json() == "not existing"
