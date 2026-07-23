import io
import time
import zipfile

import docx
import pandas as pd
from fastapi.testclient import TestClient

from test.utils import upload_staged

TIMEOUT = 180


def _toolbox_upload(client: TestClient, headers: dict[str, str], content: bytes, filename: str):
    """
    Upload a file to the preparation tool through the chunked-upload protocol.
    """
    upload_id = upload_staged(client, headers, content, filename)
    return client.post(f"/api/toolbox/upload?upload_id={upload_id}", headers=headers)


def _wait_for_task(
    client: TestClient, headers: dict[str, str], session_id: str, task_id: str
) -> dict:
    start = time.time()
    while True:
        r = client.get(
            f"/api/toolbox/status?session_id={session_id}&task_id={task_id}",
            headers=headers,
        )
        assert r.status_code == 200, r.text
        status = r.json()
        if status["status"] == "done":
            return status
        if status["status"] in ["failed", "not found"]:
            raise RuntimeError(status)
        if time.time() - start > TIMEOUT:
            raise TimeoutError("Split task timeout")
        time.sleep(1)


def test_prepare_csv_chunk_and_export(client: TestClient, superuser_headers: dict[str, str]):
    """
    Upload a csv, split it in character chunks, export it in the 3 formats
    """
    csv_content = (
        "doc,content,source\na,one two three four five six,press\nb,seven eight,web\n"
    ).encode()
    r = _toolbox_upload(client, superuser_headers, csv_content, "test.csv")
    assert r.status_code == 200, r.text
    session = r.json()
    assert session["columns"] == ["doc", "content", "source"]
    assert session["n_rows"] == 2
    assert len(session["preview"]) == 2

    r = client.post(
        "/api/toolbox/split",
        headers=superuser_headers,
        json={
            "session_id": session["session_id"],
            "cols_text": ["content"],
            "col_id": "doc",
            "cols_keep": ["source"],
            "method": "chunk",
            "chunk_size": 15,
        },
    )
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]

    status = _wait_for_task(client, superuser_headers, session["session_id"], task_id)
    assert status["n_rows"] > 2
    assert status["preview"][0]["id"] == "a_1"
    assert status["preview"][0]["source"] == "press"

    for format in ["csv", "xlsx", "parquet"]:
        r = client.get(
            f"/api/toolbox/export?session_id={session['session_id']}&format={format}",
            headers=superuser_headers,
        )
        assert r.status_code == 200, r.text
    # check the parquet export content
    exported = pd.read_parquet(io.BytesIO(r.content))
    assert list(exported.columns) == ["id", "text", "source"]
    assert list(exported["id"])[:2] == ["a_1", "a_2"]


def test_prepare_zip_wtpsplit(client: TestClient, superuser_headers: dict[str, str]):
    """
    Upload a zip of txt/docx files and split it in sentences with wtpsplit
    """
    document = docx.Document()
    document.add_paragraph("A paragraph in a word file. With two sentences.")
    docx_buffer = io.BytesIO()
    document.save(docx_buffer)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("doc1.txt", "First sentence. Second sentence.")
        zf.writestr("folder/doc2.txt", "Another document here.")
        zf.writestr("doc3.docx", docx_buffer.getvalue())
    zip_buffer.seek(0)

    r = _toolbox_upload(client, superuser_headers, zip_buffer.getvalue(), "corpus.zip")
    assert r.status_code == 200, r.text
    session = r.json()
    assert session["columns"] == ["id", "text"]
    assert session["n_rows"] == 3
    ids = {row["id"] for row in session["preview"]}
    assert ids == {"doc1", "doc2", "doc3"}

    r = client.post(
        "/api/toolbox/split",
        headers=superuser_headers,
        json={
            "session_id": session["session_id"],
            "cols_text": ["text"],
            "col_id": "id",
            "cols_keep": [],
            "method": "wtpsplit",
            "granularity": "sentence",
            "language": "en",
        },
    )
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]

    status = _wait_for_task(client, superuser_headers, session["session_id"], task_id)
    # at least one row per document, and at least one document split in sentences
    # (the exact segmentation depends on the wtpsplit model)
    assert status["n_rows"] is not None and status["n_rows"] >= 4
    assert status["preview"][0]["id"] == "doc1_1"
    assert status["preview"][0]["text"].startswith("First sentence.")


def test_prepare_split_invalid_params(client: TestClient, superuser_headers: dict[str, str]):
    """
    Invalid regex or unknown session are rejected
    """
    r = _toolbox_upload(client, superuser_headers, b"a,b\n1,2\n", "test.csv")
    assert r.status_code == 200, r.text
    session_id = r.json()["session_id"]

    # invalid regex
    r = client.post(
        "/api/toolbox/split",
        headers=superuser_headers,
        json={
            "session_id": session_id,
            "cols_text": ["a"],
            "method": "regex",
            "regex_pattern": "[unclosed",
        },
    )
    assert r.status_code == 400

    # unknown text column
    r = client.post(
        "/api/toolbox/split",
        headers=superuser_headers,
        json={
            "session_id": session_id,
            "cols_text": ["missing"],
            "method": "chunk",
            "chunk_size": 100,
        },
    )
    assert r.status_code == 400

    # negative min_chars
    r = client.post(
        "/api/toolbox/split",
        headers=superuser_headers,
        json={
            "session_id": session_id,
            "cols_text": ["a"],
            "method": "chunk",
            "chunk_size": 100,
            "min_chars": -1,
        },
    )
    assert r.status_code == 400

    # malformed session id
    r = client.post(
        "/api/toolbox/split",
        headers=superuser_headers,
        json={
            "session_id": "../../../etc",
            "cols_text": ["a"],
            "method": "chunk",
            "chunk_size": 100,
        },
    )
    assert r.status_code == 400


def test_prepare_stop_task(client: TestClient, superuser_headers: dict[str, str]):
    """
    A running split task can be stopped by its owner
    """
    r = _toolbox_upload(client, superuser_headers, b"a,b\n1,some text to split\n", "test.csv")
    assert r.status_code == 200, r.text
    session = r.json()

    # wtpsplit is slow enough (model loading) to be stopped before it finishes
    r = client.post(
        "/api/toolbox/split",
        headers=superuser_headers,
        json={
            "session_id": session["session_id"],
            "cols_text": ["b"],
            "method": "wtpsplit",
            "granularity": "sentence",
        },
    )
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]

    r = client.post(f"/api/toolbox/stop?task_id={task_id}", headers=superuser_headers)
    assert r.status_code == 200, r.text

    r = client.get(
        f"/api/toolbox/status?session_id={session['session_id']}&task_id={task_id}",
        headers=superuser_headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "failed"
    assert "cancelled" in r.json()["error"].lower()

    # stopping an unknown task returns 404
    r = client.post("/api/toolbox/stop?task_id=unknown", headers=superuser_headers)
    assert r.status_code == 404


def test_prepare_requires_auth(client: TestClient):
    r = client.post("/api/toolbox/upload")
    assert r.status_code == 401
