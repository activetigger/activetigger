import time

from fastapi.testclient import TestClient

from test.utils import create_project, delete_project, get_project_state


def test_lexicometrics(client: TestClient, superuser_headers: dict[str, str]) -> None:
    """
    Testing the lexicometrics computation flow: not computed -> compute -> available.
    """

    project_name = f"Test-lexicometrics-{int(time.time())}"
    project = create_project(client, superuser_headers, project_name)
    project_slug = project["project_slug"]

    try:
        # not computed yet
        r = client.get(f"/api/projects/{project_slug}/lexicometrics", headers=superuser_headers)
        assert r.status_code == 200, r.text
        assert r.json() is None

        state = get_project_state(client, superuser_headers, project_slug)
        assert state["lexicometrics"] == {"available": False, "training": {}}

        # launch the computation
        r = client.post(
            f"/api/projects/{project_slug}/lexicometrics/compute", headers=superuser_headers
        )
        assert r.status_code == 200, r.text

        # wait for the computation to finish
        state = get_project_state(
            client,
            superuser_headers,
            project_slug,
            expect=lambda s: s["lexicometrics"]["available"],
            timeout=60,
        )
        assert state["lexicometrics"]["available"]
        assert state["lexicometrics"]["training"] == {}

        # get the statistics
        r = client.get(f"/api/projects/{project_slug}/lexicometrics", headers=superuser_headers)
        assert r.status_code == 200, r.text
        lexicometrics = r.json()
        statistics = lexicometrics["statistics"]
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

        # tf-idf by word: capped vocabulary, top documents sorted by score
        tfidf_words = statistics["tfidf_words"]
        assert tfidf_words is not None and 0 < len(tfidf_words) <= 300
        for word in tfidf_words:
            scores = [d["score"] for d in word["top_documents"]]
            assert 0 < len(scores) <= 25
            assert all(s > 0 for s in scores)
            assert scores == sorted(scores, reverse=True)

        # tf-idf by document: present (corpus below the cap), consistent ids
        tfidf_documents = statistics["tfidf_documents"]
        assert tfidf_documents is not None and len(tfidf_documents) == n_train
        document_ids = {d["element_id"] for d in tfidf_documents}
        for document in tfidf_documents:
            scores = [w["score"] for w in document["top_words"]]
            assert len(scores) <= 10
            assert scores == sorted(scores, reverse=True)
        assert all(
            d["element_id"] in document_ids for word in tfidf_words for d in word["top_documents"]
        )
    finally:
        delete_project(client, superuser_headers, project_slug)
