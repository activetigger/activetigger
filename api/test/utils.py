import time
from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from activetigger.datamodels import (
    AnnotationModel,
    NewUserModel,
    NextInModel,
    ProjectBaseModel,
    SchemeModel,
)

TIMEOUT = 60


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


def create_scheme(
    client: TestClient,
    superuser_header: dict[str, str],
    project_slug: str,
    scheme_name: str,
    kind: str = "multiclass",
    labels: list[str] | None = None,
) -> None:
    """
    Helper function to create a scheme on a project.
    """
    data = SchemeModel(
        project_slug=project_slug,
        name=scheme_name,
        kind=kind,
        labels=labels or [],
    )
    r = client.post(
        f"/api/schemes/add?project_slug={project_slug}",
        json=data.model_dump(),
        headers=superuser_header,
    )
    assert r.status_code == 200, r.text


def add_label(
    client: TestClient,
    superuser_header: dict[str, str],
    project_slug: str,
    scheme: str,
    label: str,
) -> None:
    """
    Helper function to add a label to a scheme.
    """
    r = client.post(
        f"/api/schemes/label/add?project_slug={project_slug}&scheme={scheme}&label={label}",
        headers=superuser_header,
    )
    assert r.status_code == 200, r.text


def get_next_element_id(
    client: TestClient,
    superuser_header: dict[str, str],
    project_slug: str,
    scheme: str,
    history: list[str],
) -> str:
    """
    Helper function to fetch the next untagged element id for a scheme.
    """
    data = NextInModel(scheme=scheme, history=history)
    r = client.post(
        f"/api/elements/next?project_slug={project_slug}",
        json=data.model_dump(mode="json"),
        headers=superuser_header,
    )
    assert r.status_code == 200, r.text
    return r.json()["element_id"]


def get_project_state(
    client: TestClient,
    superuser_header: dict[str, str],
    project_slug: str,
    expect: Callable[[dict], bool] | None = None,
    timeout: float = 5.0,
    interval: float = 0.5,
) -> dict:
    """
    Helper function to fetch project state, optionally polling until ``expect``
    is satisfied. ``project.state()`` is cached server-side for ~2s, so callers
    that want to observe a recent scheme/label change need to poll.
    """
    start = time.time()
    while True:
        r = client.get(f"/api/projects/{project_slug}", headers=superuser_header)
        assert r.status_code == 200, r.text
        state = r.json()
        if expect is None or expect(state):
            return state
        if time.time() - start > timeout:
            return state
        time.sleep(interval)


def annotate_element(
    client: TestClient,
    superuser_header: dict[str, str],
    project_slug: str,
    scheme: str,
    element_id: str,
    label: str,
    dataset: str = "train",
) -> None:
    """
    Helper function to push a single annotation.
    """
    data = AnnotationModel(
        project_slug=project_slug,
        dataset=dataset,
        scheme=scheme,
        element_id=element_id,
        label=label,
    )
    r = client.post(
        f"/api/annotation/add?project_slug={project_slug}",
        json=data.model_dump(mode="json"),
        headers=superuser_header,
    )
    assert r.status_code == 200, r.text
