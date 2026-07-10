import time

from fastapi.testclient import TestClient

from test.utils import create_project, delete_project, get_project_state


def test_textometrics(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing the textometrics computation flow: not computed -> compute -> available.
    """

    project_name = f"Test-textometrics-{int(time.time())}"
    project = create_project(client, superuser_headers, project_name)
    project_slug = project["project_slug"]

    try:
        # not computed yet
        r = client.get(f"/api/projects/{project_slug}/textometrics", headers=superuser_headers)
        assert r.status_code == 200, r.text
        assert r.json() is None

        state = get_project_state(client, superuser_headers, project_slug)
        assert state["textometrics"] == {"available": False, "training": {}}

        # launch the computation
        r = client.post(
            f"/api/projects/{project_slug}/textometrics/compute", headers=superuser_headers
        )
        assert r.status_code == 200, r.text

        # wait for the computation to finish
        state = get_project_state(
            client,
            superuser_headers,
            project_slug,
            expect=lambda s: s["textometrics"]["available"],
            timeout=60,
        )
        assert state["textometrics"]["available"]
        assert state["textometrics"]["training"] == {}

        # get the statistics
        r = client.get(f"/api/projects/{project_slug}/textometrics", headers=superuser_headers)
        assert r.status_code == 200, r.text
        textometrics = r.json()
        statistics = textometrics["statistics"]
        n_train = statistics["words_per_doc"]["summary"]["count"]
        assert n_train == 100
        assert statistics["tokens_per_doc"]["summary"]["count"] == n_train
        assert statistics["words_per_doc"]["summary"]["mean"] > 0
        assert sum(statistics["words_per_doc"]["histogram"]["counts"]) == n_train
        assert len(statistics["most_frequent_words"]) > 0
        assert (
            statistics["most_frequent_words"][0]["count"]
            >= (statistics["most_frequent_words"][-1]["count"])
        )
    finally:
        delete_project(client, superuser_headers, project_slug)
