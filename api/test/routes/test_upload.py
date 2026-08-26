import io
import time
from pathlib import Path

from fastapi.testclient import TestClient

from activetigger.datamodels import ProjectBaseModel
from test.utils import (
    create_project,
    create_scheme,
    delete_project,
    get_project_state,
    upload_staged,
)

CSV_CONTENT = (
    "doc,content\n" + "\n".join(f"id{i},some text {i}" for i in range(50)) + "\n"
).encode()


def _start(
    client: TestClient,
    headers: dict[str, str],
    filename: str = "test.csv",
    total_size: int | None = None,
    total_chunks: int = 2,
):
    return client.post(
        "/api/upload/start",
        headers=headers,
        json={
            "filename": filename,
            "total_size": total_size if total_size is not None else len(CSV_CONTENT),
            "total_chunks": total_chunks,
        },
    )


def _send_chunks(
    client: TestClient,
    headers: dict[str, str],
    upload_id: str,
    content: bytes,
    total_chunks: int,
):
    chunk_size = -(-len(content) // total_chunks)  # ceil division
    for i in range(total_chunks):
        part = content[i * chunk_size : (i + 1) * chunk_size]
        r = client.post(
            f"/api/upload/chunk?upload_id={upload_id}&index={i}",
            headers=headers,
            files={"file": ("blob", io.BytesIO(part), "application/octet-stream")},
        )
        assert r.status_code == 200, r.text


def test_chunked_upload_lifecycle(client: TestClient, superuser_headers: dict[str, str]):
    """
    start -> chunks -> finish assembles the exact original bytes
    """
    r = _start(client, superuser_headers, total_chunks=3)
    assert r.status_code == 200, r.text
    upload_id = r.json()["upload_id"]
    assert r.json()["filename"] == "test.csv"

    _send_chunks(client, superuser_headers, upload_id, CSV_CONTENT, total_chunks=3)

    r = client.post(f"/api/upload/finish?upload_id={upload_id}", headers=superuser_headers)
    assert r.status_code == 200, r.text
    assert r.json()["size"] == len(CSV_CONTENT)
    assert r.json()["filename"] == "test.csv"

    # cleanup
    r = client.delete(f"/api/upload/{upload_id}", headers=superuser_headers)
    assert r.status_code == 200, r.text


def test_finish_with_missing_chunk_fails(client: TestClient, superuser_headers: dict[str, str]):
    r = _start(client, superuser_headers, total_chunks=3)
    upload_id = r.json()["upload_id"]
    # send only the first chunk
    _send_chunks(client, superuser_headers, upload_id, CSV_CONTENT[:10], total_chunks=1)

    r = client.post(f"/api/upload/finish?upload_id={upload_id}", headers=superuser_headers)
    assert r.status_code == 400
    assert "missing" in r.json()["detail"]
    client.delete(f"/api/upload/{upload_id}", headers=superuser_headers)


def test_size_mismatch_fails(client: TestClient, superuser_headers: dict[str, str]):
    r = _start(client, superuser_headers, total_size=len(CSV_CONTENT) + 5, total_chunks=2)
    upload_id = r.json()["upload_id"]
    _send_chunks(client, superuser_headers, upload_id, CSV_CONTENT, total_chunks=2)

    r = client.post(f"/api/upload/finish?upload_id={upload_id}", headers=superuser_headers)
    assert r.status_code == 400
    assert "mismatch" in r.json()["detail"]
    client.delete(f"/api/upload/{upload_id}", headers=superuser_headers)


def test_bad_extension_rejected_at_start(client: TestClient, superuser_headers: dict[str, str]):
    r = _start(client, superuser_headers, filename="malware.exe")
    assert r.status_code == 400


def test_chunk_index_out_of_range(client: TestClient, superuser_headers: dict[str, str]):
    r = _start(client, superuser_headers, total_chunks=2)
    upload_id = r.json()["upload_id"]
    r = client.post(
        f"/api/upload/chunk?upload_id={upload_id}&index=5",
        headers=superuser_headers,
        files={"file": ("blob", io.BytesIO(b"xx"), "application/octet-stream")},
    )
    assert r.status_code == 400
    client.delete(f"/api/upload/{upload_id}", headers=superuser_headers)


def test_unknown_session_rejected(client: TestClient, superuser_headers: dict[str, str]):
    r = client.post(
        "/api/upload/chunk?upload_id=" + "0" * 32 + "&index=0",
        headers=superuser_headers,
        files={"file": ("blob", io.BytesIO(b"xx"), "application/octet-stream")},
    )
    assert r.status_code == 404
    r = client.post("/api/upload/finish?upload_id=" + "0" * 32, headers=superuser_headers)
    assert r.status_code == 404
    # malformed id
    r = client.post("/api/upload/finish?upload_id=../../etc", headers=superuser_headers)
    assert r.status_code == 400


def test_unauthenticated_rejected(client: TestClient):
    r = client.post(
        "/api/upload/start",
        json={"filename": "test.csv", "total_size": 10, "total_chunks": 1},
    )
    assert r.status_code == 401


def test_toolbox_upload_from_staged_file(client: TestClient, superuser_headers: dict[str, str]):
    """
    End-to-end: chunked upload then consume it in the toolbox endpoint
    """
    upload_id = upload_staged(client, superuser_headers, CSV_CONTENT, "test.csv", chunk_size=512)

    r = client.post(f"/api/toolbox/upload?upload_id={upload_id}", headers=superuser_headers)
    assert r.status_code == 200, r.text
    session = r.json()
    assert session["columns"] == ["doc", "content"]
    assert session["n_rows"] == 50

    # the staged session is consumed
    r = client.post(f"/api/toolbox/upload?upload_id={upload_id}", headers=superuser_headers)
    assert r.status_code == 404


def test_consumer_requires_upload_id(client: TestClient, superuser_headers: dict[str, str]):
    """
    Consumer endpoints no longer accept direct multipart files:
    upload_id is a required parameter
    """
    r = client.post("/api/toolbox/upload", headers=superuser_headers)
    assert r.status_code == 422
    r = client.post(
        "/api/toolbox/upload",
        headers=superuser_headers,
        files={"file": ("test.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
    )
    assert r.status_code == 422


def test_annotations_import_from_staged_file(client: TestClient, superuser_headers: dict[str, str]):
    """
    End-to-end: import annotations on the trainset from a staged csv
    """
    project = create_project(client, superuser_headers, f"test-annot-upload-{int(time.time())}")
    project_slug = project["project_slug"]
    try:
        create_scheme(client, superuser_headers, project_slug, "upload-scheme")

        # fetch two real trainset elements to annotate (the trainset is a
        # random sample, so external ids cannot be assumed)
        r = client.post(
            f"/api/elements/table?project_slug={project_slug}",
            headers=superuser_headers,
            json={"scheme": "upload-scheme", "min": 0, "max": 2},
        )
        assert r.status_code == 200, r.text
        ids = [row["id_external"] for row in r.json()["items"][:2]]
        assert len(ids) == 2

        csv_content = f"id,label\n{ids[0]},cat-a\n{ids[1]},cat-b\n".encode()
        upload_id = upload_staged(client, superuser_headers, csv_content, "annotations.csv")
        payload = {
            "col_id": "id",
            "col_label": "label",
            "scheme": "upload-scheme",
            "upload_id": upload_id,
            "filename": "annotations.csv",
        }
        r = client.post(
            f"/api/annotation/file?project_slug={project_slug}",
            headers=superuser_headers,
            json=payload,
        )
        assert r.status_code == 200, r.text

        # the labels have been created in the scheme from the file content
        state = get_project_state(
            client,
            superuser_headers,
            project_slug,
            expect=lambda s: {"cat-a", "cat-b"}.issubset(
                set(s["schemes"]["available"].get("upload-scheme", {}).get("labels", []))
            ),
        )
        labels = state["schemes"]["available"]["upload-scheme"]["labels"]
        assert {"cat-a", "cat-b"}.issubset(set(labels))

        # the staged session is consumed on success
        r = client.post(
            f"/api/annotation/file?project_slug={project_slug}",
            headers=superuser_headers,
            json=payload,
        )
        assert r.status_code == 404
    finally:
        delete_project(client, superuser_headers, project_slug)


def test_project_creation_from_staged_file(client: TestClient, superuser_headers: dict[str, str]):
    """
    End-to-end issue #1103 flow: chunked upload -> /projects/new with
    upload_id -> project exists.
    """
    asset = Path(__file__).resolve().parent.parent / "assets" / "gwsd_train_test.csv"
    content = asset.read_bytes()
    upload_id = upload_staged(client, superuser_headers, content, asset.name)

    project_name = f"test-staged-upload-{int(time.time())}"
    data = ProjectBaseModel(
        project_name=project_name,
        upload_id=upload_id,
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
        stratify_eval=False,
        from_project=None,
        from_toy_dataset=False,
    )
    r = client.post("/api/projects/new", json=data.model_dump(), headers=superuser_headers)
    assert r.status_code == 200, r.text
    project_slug = r.json()

    start = time.time()
    while True:
        status = client.get(
            f"/api/projects/status?project_name={project_slug}",
            headers=superuser_headers,
        ).json()
        if status == "existing":
            break
        if status == "creating":
            assert time.time() - start < 60, "Project creation timeout"
            time.sleep(1)
            continue
        raise RuntimeError(status)

    delete_project(client, superuser_headers, project_slug)


def test_project_creation_requires_upload_id(client: TestClient, superuser_headers: dict[str, str]):
    """
    /projects/new without an upload_id (and no copy source) is rejected
    """
    data = ProjectBaseModel(
        project_name=f"test-no-upload-{int(time.time())}",
        col_id="row_number",
        cols_text=["sentence"],
        n_train=10,
        n_test=0,
        n_valid=0,
        language="en",
    )
    r = client.post("/api/projects/new", json=data.model_dump(), headers=superuser_headers)
    assert r.status_code == 400
    assert "upload_id" in r.json()["detail"]
