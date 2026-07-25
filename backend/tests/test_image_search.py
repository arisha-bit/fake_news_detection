"""
Pytest tests for Phase 5 — Reverse Image Verification.

Covers:
- Valid image upload returns matches
- top_k parameter respected
- Reuse detection flag when similarity high
- Invalid image extension rejected
- Missing index raises 503
- Results have correct structure
"""

import io
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


def _make_image_file(filename: str = "test.jpg", content_type: str = "image/jpeg"):
    return ("file", (filename, io.BytesIO(b"fake-image-bytes"), content_type))


def _make_matches(n: int = 3, high_similarity: bool = False) -> list[dict]:
    base_sim = 0.90 if high_similarity else 0.65
    return [
        {
            "rank": i + 1,
            "title": f"Article title {i + 1}",
            "potential_source": "politics",
            "label": "REAL" if i % 2 == 0 else "FAKE",
            "similarity": round(base_sim - i * 0.05, 4),
            "date": "2024-01-01",
            "snippet": f"Snippet for article {i + 1}.",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidReverseImageSearch:
    def test_returns_matches_for_valid_image(
        self, client, auth_override, db_override
    ):
        fake_results = _make_matches(3)

        with (
            patch(
                "app.api.images.reverse_image_search",
                return_value=(fake_results, False),
            ),
            patch(
                "app.utils.file_utils.save_upload",
                return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg"),
            ),
            patch("app.utils.file_utils.delete_temp_file"),
        ):
            response = client.post(
                "/images/reverse-search",
                files=[_make_image_file()],
                params={"top_k": 3},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 3
        assert len(data["matches"]) == 3

    def test_match_item_has_required_fields(
        self, client, auth_override, db_override
    ):
        fake_results = _make_matches(1)

        with (
            patch(
                "app.api.images.reverse_image_search",
                return_value=(fake_results, False),
            ),
            patch(
                "app.utils.file_utils.save_upload",
                return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg"),
            ),
            patch("app.utils.file_utils.delete_temp_file"),
        ):
            response = client.post(
                "/images/reverse-search",
                files=[_make_image_file()],
            )

        match = response.json()["matches"][0]
        assert "rank" in match
        assert "title" in match
        assert "potential_source" in match
        assert "label" in match
        assert "similarity" in match

    def test_reuse_detection_flag_when_high_similarity(
        self, client, auth_override, db_override
    ):
        fake_results = _make_matches(2, high_similarity=True)

        with (
            patch(
                "app.api.images.reverse_image_search",
                return_value=(fake_results, True),  # reuse detected
            ),
            patch(
                "app.utils.file_utils.save_upload",
                return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg"),
            ),
            patch("app.utils.file_utils.delete_temp_file"),
        ):
            response = client.post(
                "/images/reverse-search",
                files=[_make_image_file()],
            )

        data = response.json()
        assert data["possible_reuse_detected"] is True

    def test_top_k_query_param_passed(self, client, auth_override, db_override):
        captured = {}

        def fake_search(img_path, top_k):
            captured["top_k"] = top_k
            return _make_matches(top_k), False

        with (
            patch("app.api.images.reverse_image_search", side_effect=fake_search),
            patch(
                "app.utils.file_utils.save_upload",
                return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg"),
            ),
            patch("app.utils.file_utils.delete_temp_file"),
        ):
            client.post(
                "/images/reverse-search",
                files=[_make_image_file()],
                params={"top_k": 8},
            )

        assert captured["top_k"] == 8


class TestInvalidImageFormat:
    def test_rejects_pdf(self, client, auth_override):
        response = client.post(
            "/images/reverse-search",
            files=[("file", ("doc.pdf", io.BytesIO(b"data"), "application/pdf"))],
        )
        assert response.status_code in (400, 415)


class TestEdgeCases:
    def test_index_not_built_returns_503(self, client, auth_override, db_override):
        from fastapi import HTTPException

        with (
            patch(
                "app.utils.file_utils.save_upload",
                return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg"),
            ),
            patch("app.utils.file_utils.delete_temp_file"),
            patch(
                "app.api.images.reverse_image_search",
                side_effect=HTTPException(
                    status_code=503, detail="CLIP index not found."
                ),
            ),
        ):
            response = client.post(
                "/images/reverse-search",
                files=[_make_image_file()],
            )

        assert response.status_code == 503

    def test_top_k_max_validation(self, client, auth_override, db_override):
        # top_k > 20 should be rejected by query validation
        with (
            patch(
                "app.utils.file_utils.save_upload",
                return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg"),
            ),
            patch("app.utils.file_utils.delete_temp_file"),
        ):
            response = client.post(
                "/images/reverse-search",
                files=[_make_image_file()],
                params={"top_k": 50},
            )
        assert response.status_code == 422


class TestImageServiceUnit:
    def test_reverse_image_search_raises_404_on_missing_file(self):
        from fastapi import HTTPException

        from app.services.image_service import reverse_image_search

        with pytest.raises(HTTPException) as exc_info:
            reverse_image_search("/nonexistent/path.jpg")

        assert exc_info.value.status_code == 404

    def test_reverse_image_search_raises_503_when_no_index(self, tmp_path):
        from fastapi import HTTPException

        import app.services.image_service as svc

        # Reset singletons
        original_index = svc._faiss_index
        original_meta = svc._metadata
        svc._faiss_index = None
        svc._metadata = None

        # Point to non-existent paths
        original_faiss = svc.FAISS_INDEX_PATH
        original_meta_path = svc.METADATA_PATH
        svc.FAISS_INDEX_PATH = tmp_path / "nonexistent.index"
        svc.METADATA_PATH = tmp_path / "nonexistent.pkl"

        try:
            with pytest.raises(HTTPException) as exc_info:
                reverse_image_search("/tmp/some_image.jpg")
            assert exc_info.value.status_code == 503
        finally:
            svc._faiss_index = original_index
            svc._metadata = original_meta
            svc.FAISS_INDEX_PATH = original_faiss
            svc.METADATA_PATH = original_meta_path
