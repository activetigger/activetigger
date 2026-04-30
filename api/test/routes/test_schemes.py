import time

from fastapi.testclient import TestClient

from test.utils import (
    add_label,
    annotate_element,
    create_project,
    create_scheme,
    get_next_element_id,
    get_project_state,
)


def test_scheme_creation(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing the creation of a new scheme on a project.
    """

    project_name = f"Test-scheme-{int(time.time())}"
    project = create_project(client, superuser_headers, project_name)
    project_slug = project["project_slug"]
    scheme_name = "scheme1"

    create_scheme(client, superuser_headers, project_slug, scheme_name)

    state = get_project_state(
        client,
        superuser_headers,
        project_slug,
        expect=lambda s: scheme_name in s["schemes"]["available"],
    )
    assert scheme_name in state["schemes"]["available"]


def test_scheme_add_labels(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing the addition of two labels (A and B) to a scheme.
    """

    project_name = f"Test-labels-{int(time.time())}"
    project = create_project(client, superuser_headers, project_name)
    project_slug = project["project_slug"]
    scheme_name = "scheme1"

    create_scheme(client, superuser_headers, project_slug, scheme_name)
    add_label(client, superuser_headers, project_slug, scheme_name, "A")
    add_label(client, superuser_headers, project_slug, scheme_name, "B")

    state = get_project_state(
        client,
        superuser_headers,
        project_slug,
        expect=lambda s: {"A", "B"}.issubset(
            s["schemes"]["available"].get(scheme_name, {}).get("labels", [])
        ),
    )
    labels = state["schemes"]["available"][scheme_name]["labels"]
    assert "A" in labels
    assert "B" in labels


def test_scheme_annotation(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing the annotation of 10 elements as A and 10 elements as B.
    """

    project_name = f"Test-annot-{int(time.time())}"
    project = create_project(client, superuser_headers, project_name)
    project_slug = project["project_slug"]
    scheme_name = "scheme1"

    create_scheme(client, superuser_headers, project_slug, scheme_name)
    add_label(client, superuser_headers, project_slug, scheme_name, "A")
    add_label(client, superuser_headers, project_slug, scheme_name, "B")

    history: list[str] = []
    for label in ["A"] * 10 + ["B"] * 10:
        element_id = get_next_element_id(
            client, superuser_headers, project_slug, scheme_name, history
        )
        annotate_element(client, superuser_headers, project_slug, scheme_name, element_id, label)
        history.append(element_id)

    r = client.get(
        f"/api/projects/{project_slug}/statistics?scheme={scheme_name}",
        headers=superuser_headers,
    )
    assert r.status_code == 200, r.text
    stats = r.json()
    assert stats["train_annotated_n"] == 20
    distribution = stats["train_annotated_distribution"]
    assert distribution.get("A") == 10
    assert distribution.get("B") == 10
