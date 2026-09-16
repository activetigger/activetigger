"""
Route tests for generative pipelines (issue #1100)
"""

import time
from pathlib import Path

from fastapi.testclient import TestClient

from activetigger.config import config
from test.utils import add_label, create_project, create_scheme, create_user, delete_project

# nothing listens there: connection is refused instantly, so the test call
# made when saving credentials fails fast without network access
DEAD_ENDPOINT = "http://127.0.0.1:1/v1"


def test_generate_credentials(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing add/list/test/delete of user credentials (secrets stay backend-side)
    """

    username = f"testuser-gencred-{int(time.time())}"
    headers = create_user(client, superuser_headers, username)

    r = client.get("/api/generate/credentials", headers=headers)
    assert r.status_code == 200
    assert r.json() == []

    r = client.post(
        "/api/generate/credentials",
        headers=headers,
        json={"name": "my-key", "endpoint": DEAD_ENDPOINT, "api_key": "sk-secret"},
    )
    assert r.status_code == 200
    saved = r.json()
    # saving works even if the endpoint does not answer, the test just fails
    assert saved["success"] is False
    credentials_id = saved["id"]

    r = client.get("/api/generate/credentials", headers=headers)
    entries = r.json()
    assert len(entries) == 1
    assert entries[0]["name"] == "my-key"
    assert entries[0]["kind"] == "user"
    assert entries[0]["endpoint"] == DEAD_ENDPOINT
    assert entries[0]["last_tested"] is None
    # the secret never appears in any response
    assert "sk-secret" not in r.text

    # saving again with the same name replaces the entry
    r = client.post(
        "/api/generate/credentials",
        headers=headers,
        json={"name": "my-key", "endpoint": DEAD_ENDPOINT, "api_key": "sk-other"},
    )
    assert r.status_code == 200
    assert r.json()["id"] == credentials_id
    r = client.get("/api/generate/credentials", headers=headers)
    assert len(r.json()) == 1

    # entries are private to their owner
    r = client.get("/api/generate/credentials", headers=superuser_headers)
    assert all(e["name"] != "my-key" for e in r.json())

    # re-test an entry explicitly
    r = client.post(
        "/api/generate/credentials/test",
        headers=headers,
        params={"credentials_id": credentials_id},
    )
    assert r.status_code == 200
    assert r.json()["success"] is False

    # another user cannot test or delete someone else's entry
    r = client.post(
        "/api/generate/credentials/test",
        headers=superuser_headers,
        params={"credentials_id": credentials_id},
    )
    assert r.status_code == 404

    r = client.post(
        "/api/generate/credentials/delete",
        headers=headers,
        params={"credentials_id": credentials_id},
    )
    assert r.status_code == 200
    r = client.get("/api/generate/credentials", headers=headers)
    assert r.json() == []

    r = client.post(
        "/api/generate/credentials/delete",
        headers=headers,
        params={"credentials_id": credentials_id},
    )
    assert r.status_code == 404


def test_generate_pipelines(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing pipeline creation/validation/listing, postprocess preview,
    sandbox behaviour on a dead endpoint, and deletions.
    """

    project_name = f"Test-genpipe-{int(time.time())}"
    project = create_project(client, superuser_headers, project_name, cols_context=["guid"])
    project_slug = project["project_slug"]
    try:
        scheme_name = "scheme-gen"
        create_scheme(client, superuser_headers, project_slug, scheme_name)
        add_label(client, superuser_headers, project_slug, scheme_name, "Positive")
        add_label(client, superuser_headers, project_slug, scheme_name, "Negative")

        r = client.post(
            "/api/generate/credentials",
            headers=superuser_headers,
            json={"name": "pipe-key", "endpoint": DEAD_ENDPOINT, "api_key": ""},
        )
        credentials_id = r.json()["id"]

        # the step registry is exposed with parameter schemas
        r = client.get("/api/generate/postprocess/steps", headers=superuser_headers)
        assert r.status_code == 200
        assert "exact_match" in r.json()
        assert "pattern" in r.json()["regex_sub"]["properties"]

        pipeline = {
            "name": "pipe1",
            "scheme_name": scheme_name,
            "credentials_id": credentials_id,
            "model_slug": "test-model",
            "prompt": "Classify this text (source [[dataset_guid]]): [[TEXT]]",
            "parameters": {"temperature": 0},
            "postprocess": [{"name": "strip"}, {"name": "exact_match"}],
        }
        r = client.post(
            f"/api/generate/pipelines?project_slug={project_slug}",
            headers=superuser_headers,
            json=pipeline,
        )
        assert r.status_code == 200
        pipeline_id = r.json()

        # unknown step and unknown prompt tag are refused
        bad = dict(pipeline, name="bad1", postprocess=[{"name": "does_not_exist"}])
        r = client.post(
            f"/api/generate/pipelines?project_slug={project_slug}",
            headers=superuser_headers,
            json=bad,
        )
        assert r.status_code == 500
        bad = dict(pipeline, name="bad2", prompt="Classify [[UNKNOWN]]")
        r = client.post(
            f"/api/generate/pipelines?project_slug={project_slug}",
            headers=superuser_headers,
            json=bad,
        )
        assert r.status_code == 500

        r = client.get(
            f"/api/generate/pipelines?project_slug={project_slug}", headers=superuser_headers
        )
        assert r.status_code == 200
        pipelines = r.json()
        assert len(pipelines) == 1
        assert pipelines[0]["name"] == "pipe1"
        assert pipelines[0]["credentials_name"] == "pipe-key"
        assert pipelines[0]["postprocess"][1]["name"] == "exact_match"

        # postprocess preview: tune steps on already-generated raw outputs
        r = client.post(
            f"/api/generate/postprocess/preview?project_slug={project_slug}",
            headers=superuser_headers,
            json={
                "scheme_name": scheme_name,
                "steps": [{"name": "strip"}, {"name": "exact_match"}],
                "raws": ["  positive ", "no label here"],
            },
        )
        assert r.status_code == 200
        preview = r.json()
        assert preview["rows"][0]["predicted"] == "Positive"
        assert preview["rows"][1]["predicted"] is None
        assert preview["n_na"] == 1

        # sandbox: the dead endpoint fails per element, the batch still answers
        r = client.post(
            f"/api/generate/pipelines/{pipeline_id}/sandbox?project_slug={project_slug}",
            headers=superuser_headers,
            json={"n_elements": 2},
        )
        assert r.status_code == 200
        sandbox = r.json()
        assert len(sandbox["rows"]) == 2
        assert all(row["error"].startswith("generation failed") for row in sandbox["rows"])
        assert sandbox["n_na"] == 2
        # the context column tag is filled with the element's value
        assert all("[[dataset_guid]]" not in row["prompt"] for row in sandbox["rows"])
        assert all("source " in row["prompt"] for row in sandbox["rows"])

        # run on the dataset: queued task, status flipped by the orchestrator
        r = client.post(
            f"/api/generate/pipelines/{pipeline_id}/start?project_slug={project_slug}",
            headers=superuser_headers,
            json={"n_elements": 2},
        )
        assert r.status_code == 200, r.text
        run_id = r.json()

        runs = []
        for _ in range(60):
            r = client.get(
                f"/api/generate/runs?project_slug={project_slug}", headers=superuser_headers
            )
            runs = r.json()
            if runs and runs[0]["status"] != "running":
                break
            time.sleep(1)
        assert len(runs) == 1
        assert runs[0]["id"] == run_id
        assert runs[0]["status"] == "done"
        # the dead endpoint fails every element: rows are kept with the error
        assert runs[0]["n_elements"] == 2
        assert runs[0]["n_na"] == 2

        r = client.get(
            f"/api/generate/runs/{run_id}/elements?project_slug={project_slug}",
            headers=superuser_headers,
        )
        assert r.status_code == 200
        table = r.json()
        assert table["total"] == 2
        assert all(item["error"].startswith("generation failed") for item in table["items"])

        r = client.get(
            f"/api/export/generations?project_slug={project_slug}&run_id={run_id}",
            headers=superuser_headers,
        )
        assert r.status_code == 200
        assert "element_id" in r.text

        r = client.post(
            "/api/generate/runs/delete",
            headers=superuser_headers,
            params={"project_slug": project_slug, "run_id": run_id},
        )
        assert r.status_code == 200, r.text
        r = client.get(f"/api/generate/runs?project_slug={project_slug}", headers=superuser_headers)
        assert r.json() == []

        # the complete dataset only runs whole
        r = client.post(
            f"/api/generate/pipelines/{pipeline_id}/start?project_slug={project_slug}",
            headers=superuser_headers,
            json={"dataset": "all", "n_elements": 2},
        )
        assert r.status_code == 500
        assert "runs whole" in r.json()["detail"]

        # full-dataset run (no limit, mode all): the task reads the project
        # train file directly instead of a written sample, and must not
        # delete it once done
        r = client.post(
            f"/api/generate/pipelines/{pipeline_id}/start?project_slug={project_slug}",
            headers=superuser_headers,
            json={"n_workers": 5},
        )
        assert r.status_code == 200, r.text
        full_run_id = r.json()

        runs = []
        for _ in range(120):
            r = client.get(
                f"/api/generate/runs?project_slug={project_slug}", headers=superuser_headers
            )
            runs = [run for run in r.json() if run["id"] == full_run_id]
            if runs and runs[0]["status"] != "running":
                break
            time.sleep(1)
        assert runs[0]["status"] == "done"
        assert runs[0]["n_elements"] == 100  # the whole train set

        project_dir = Path(config.data_path) / "projects" / project_slug
        assert (project_dir / "train.parquet").exists()
        # no temporary input file left behind
        assert list((project_dir / "generations").glob("*_input.parquet")) == []

        # a run with only NA outputs cannot become a scheme
        r = client.post(
            f"/api/generate/runs/{full_run_id}/to-scheme?project_slug={project_slug}",
            headers=superuser_headers,
            params={"project_slug": project_slug, "scheme_name": "scheme-from-gen"},
        )
        assert r.status_code == 500
        assert "No predicted labels" in r.json()["detail"]

        r = client.post(
            "/api/generate/runs/delete",
            headers=superuser_headers,
            params={"project_slug": project_slug, "run_id": full_run_id},
        )
        assert r.status_code == 200, r.text
        assert (project_dir / "train.parquet").exists()

        # credentials in use cannot be deleted
        r = client.post(
            "/api/generate/credentials/delete",
            headers=superuser_headers,
            params={"credentials_id": credentials_id},
        )
        assert r.status_code == 500

        r = client.post(
            "/api/generate/pipelines/delete",
            headers=superuser_headers,
            params={"project_slug": project_slug, "pipeline_id": pipeline_id},
        )
        assert r.status_code == 200, r.text
        r = client.get(
            f"/api/generate/pipelines?project_slug={project_slug}", headers=superuser_headers
        )
        assert r.json() == []

        r = client.post(
            "/api/generate/credentials/delete",
            headers=superuser_headers,
            params={"credentials_id": credentials_id},
        )
        assert r.status_code == 200
    finally:
        delete_project(client, superuser_headers, project_slug)
