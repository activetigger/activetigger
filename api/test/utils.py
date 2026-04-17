import time
from pathlib import Path

from fastapi.testclient import TestClient

from activetigger.datamodels import NewUserModel, ProjectBaseModel

TIMEOUT = 15


def create_user(
    client: TestClient, superuser_header: dict[str, str], username: str
) -> dict[str, str]:
    """
    Helper function to create a user and return the headers for that user.
    """
    data = NewUserModel(
        username=username,
        password="l3tm31n!",
        contact="activetigger@yopmail.com",
        status="active",
    )
    client.post("/api/users/create", json=data.model_dump(), headers=superuser_header)

    login_data = {
        "username": username,
        "password": "l3tm31n!",
    }
    lr = client.post("/api/token", data=login_data)
    tokens = lr.json()
    a_token = tokens["access_token"]
    headers = {"Authorization": f"Bearer {a_token}"}
    return headers


def create_project(
    client: TestClient, superuser_header: dict[str, str], project_name: str
) -> dict[str, str]:
    """
    Helper function to create a project and return the headers for that project.
    """

    # Upload file
    file_path = Path(__file__).resolve().parent / "assets" / "gwsd_train_test.csv"
    with open(file_path, "rb") as f:
        response = client.post(
            f"/api/files/add/project?project_name={project_name}",
            headers=superuser_header,
            files={"file": (file_path.name, f, "text/csv")},
            data={"name": "file", "filename": file_path.name},
        )
        assert response.status_code == 200

    # Create the project
    data = ProjectBaseModel(
        project_name=project_name,
        filename="gwsd_train_test.csv",
        col_id="row_number",
        cols_text=["sentence", "label"],
        cols_context=[],
        cols_label=[],
        n_train=100,
        n_test=0,
        n_valid=0,
        language="en",
        clear_test=False,
        random_selection=True,
        force_label=False,
        seed=1290,
        stratify_train=False,
        stratify_test=False,
        from_project=None,
        from_toy_dataset=False,
    )

    r = client.post("/api/projects/new", json=data.model_dump(), headers=superuser_header)
    assert r.status_code == 200, r.text
    project_slug = r.json()

    # Waiting for the project creation to be finished
    start = time.time()
    while True:
        response_status = client.get(
            f"/api/projects/status?project_name={project_slug}",
            headers=superuser_header,
        )
        assert response_status.status_code == 200, response_status.text

        status = response_status.json()
        if status == "existing":
            break
        elif status == "not existing":
            raise RuntimeError("Project not found")
        elif status == "creating":
            time.sleep(1)
        else:
            raise RuntimeError(status)

        if time.time() - start > TIMEOUT:
            raise TimeoutError("Project creation timeout")

    # Get the project by its slug
    response_project = client.get(f"/api/projects/{project_slug}", headers=superuser_header)

    return response_project.json()["params"]


def delete_project(client: TestClient, superuser_header: dict[str, str], project_slug: str) -> None:
    """
    Helper function to delete a project.
    """
    response = client.post(
        f"/api/projects/delete?project_slug={project_slug}",
        headers=superuser_header,
    )
    assert response.status_code == 200, response.text
