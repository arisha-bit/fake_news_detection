"""
Pytest tests for Phase 4 — Evidence Retrieval.

Covers:
- Valid search returns ranked evidence items
- top_k parameter respected
- Empty query raises 422
- Index not found raises 503
- Results have correct structure
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.user import User


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_user():
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.role = "user"
    return user


@pytest.fixture
def auth_override(mock_user):
    from app.core.security import get_current_user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield mock_user
    app.dependency_overrides.clear()


@pytest.fixture
def db_override():
    from app.db.dependencies import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    yield mock_db
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Fake evidence results
# ---------------------------------------------------------------------------

def _make_evidence(n: int = 3) -> list[dict]:
    return [
        {
            "rank": i + 1,
            "title": f"Article title {i + 1}",
            "snippet": f"This is a snippet for article {i + 1}.",
            "label": "REAL" if i % 2 == 0 else "FAKE",
            "similarity": round(0.95 - i * 0.05, 4),
            "subject": "politics",
            "date": "2024-01-01",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidEvidenceSearch:
    def test_returns_evidence_items(self, client, auth_override, db_override):
        fake_results = _make_evidence(3)

        with patch(
            "app.api.evidence.search_evidence",
            return_value=fake_results,
        ):
            response = client.post(
                "/evidence/search",
                json={"query": "NASA confirmed water on Mars.", "top_k": 3},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 3
        assert len(data["evidence"]) == 3

    def test_evidence_item_has_required_fields(
        self, client, auth_override, db_override
    ):
        fake_results = _make_evidence(1)

        with patch("app.api.evidence.search_evidence", return_value=fake_results):
            response = client.post(
                "/evidence/search",
                json={"query": "Some claim about politics."},
            )

        item = response.json()["evidence"][0]
        assert "rank" in item
        assert "title" in item
        assert "snippet" in item
        assert "label" in item
        assert "similarity" in item

    def test_top_k_passed_to_service(self, client, auth_override, db_override):
        captured = {}

        def fake_search(query, top_k):
            captured["top_k"] = top_k
            return _make_evidence(top_k)

        with patch("app.api.evidence.search_evidence", side_effect=fake_search):
            client.post(
                "/evidence/search",
                json={"query": "Test claim.", "top_k": 7},
            )

        assert captured["top_k"] == 7

    def test_results_are_ranked_ascending(self, client, auth_override, db_override):
        fake_results = _make_evidence(5)

        with patch("app.api.evidence.search_evidence", return_value=fake_results):
            response = client.post(
                "/evidence/search",
                json={"query": "Government health policy."},
            )

        ranks = [item["rank"] for item in response.json()["evidence"]]
        assert ranks == sorted(ranks)


class TestEdgeCases:
    def test_empty_query_returns_422(self, client, auth_override, db_override):
        from fastapi import HTTPException

        with patch(
            "app.api.evidence.search_evidence",
            side_effect=HTTPException(status_code=422, detail="Empty query"),
        ):
            response = client.post(
                "/evidence/search",
                json={"query": ""},
            )
        assert response.status_code == 422

    def test_index_not_built_returns_503(self, client, auth_override, db_override):
        from fastapi import HTTPException

        with patch(
            "app.api.evidence.search_evidence",
            side_effect=HTTPException(
                status_code=503, detail="Evidence index not found."
            ),
        ):
            response = client.post(
                "/evidence/search",
                json={"query": "Some claim about something."},
            )
        assert response.status_code == 503

    def test_top_k_max_validation(self, client, auth_override, db_override):
        # top_k > 20 should be rejected by Pydantic schema validation
        response = client.post(
            "/evidence/search",
            json={"query": "Some claim.", "top_k": 99},
        )
        assert response.status_code == 422


class TestRetrievalServiceUnit:
    def test_search_evidence_raises_422_on_empty(self):
        from fastapi import HTTPException

        from app.services.retrieval_service import search_evidence

        with pytest.raises(HTTPException) as exc_info:
            search_evidence("")

        assert exc_info.value.status_code == 422

    def test_search_evidence_raises_503_when_no_index(self, tmp_path):
        """Ensure 503 is raised when FAISS index files are missing."""
        from fastapi import HTTPException

        import app.services.retrieval_service as svc

        # Reset singleton so it tries to reload
        original_index = svc._faiss_index
        original_meta = svc._metadata
        svc._faiss_index = None
        svc._metadata = None

        # Point paths to non-existent files
        original_faiss = svc.FAISS_INDEX_PATH
        original_meta_path = svc.METADATA_PATH
        svc.FAISS_INDEX_PATH = tmp_path / "nonexistent.index"
        svc.METADATA_PATH = tmp_path / "nonexistent.pkl"

        try:
            with pytest.raises(HTTPException) as exc_info:
                search_evidence("Some query text here.")
            assert exc_info.value.status_code == 503
        finally:
            # Restore
            svc._faiss_index = original_index
            svc._metadata = original_meta
            svc.FAISS_INDEX_PATH = original_faiss
            svc.METADATA_PATH = original_meta_path
