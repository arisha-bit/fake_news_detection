"""
Tests for multimodal image verification (Parts 2 & 5).

Covers:
- POST /predict/image returns classification result
- POST /predict/image raises 503 when model not trained
- POST /verify/image returns full multimodal response
- POST /verify/image degrades gracefully when image model missing
- Invalid file format rejected
- compute_verdict logic unit tests
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

    def _refresh(obj):
        if not hasattr(obj, "id") or obj.id is None:
            obj.id = uuid.uuid4()

    mock_db.refresh.side_effect = _refresh
    app.dependency_overrides[get_db] = lambda: mock_db
    yield mock_db
    app.dependency_overrides.clear()


def _fake_image():
    return ("file", ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg"))


# ---------------------------------------------------------------------------
# Fake branch results
# ---------------------------------------------------------------------------

FAKE_OCR = {
    "ocr_text": "Scientists confirm shocking discovery about moon.",
    "text_prediction": "FAKE",
    "text_confidence": 0.88,
}

FAKE_IMG = {
    "image_prediction": "FAKE",
    "image_confidence": 0.91,
    "image_class_probabilities": {"FAKE": 0.91, "REAL": 0.09},
}

FAKE_CLIP = {
    "similar_articles": [
        {
            "rank": 1,
            "title": "Moon conspiracy theory debunked",
            "label": "FAKE",
            "similarity": 0.88,
            "potential_source": "tabloid",
            "date": "2023-01-01",
            "snippet": "This claim has been fact-checked...",
        }
    ],
    "clip_reuse_detected": True,
}


# ---------------------------------------------------------------------------
# POST /predict/image
# ---------------------------------------------------------------------------

class TestImageClassification:
    def test_returns_classification_result(self, client, auth_override, db_override):
        with (
            patch("app.api.image_predict.run_image_classification_branch", return_value=FAKE_IMG),
            patch("app.utils.file_utils.save_upload",
                  return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg")),
            patch("app.utils.file_utils.delete_temp_file"),
        ):
            response = client.post("/predict/image", files=[_fake_image()])

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "FAKE"
        assert data["confidence"] == 0.91
        assert "class_probabilities" in data

    def test_raises_503_when_model_not_trained(self, client, auth_override, db_override):
        from fastapi import HTTPException

        unavailable = {
            "image_prediction": None,
            "image_confidence": None,
            "image_class_probabilities": {},
        }
        with (
            patch("app.api.image_predict.run_image_classification_branch",
                  return_value=unavailable),
            patch("app.utils.file_utils.save_upload",
                  return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg")),
            patch("app.utils.file_utils.delete_temp_file"),
        ):
            response = client.post("/predict/image", files=[_fake_image()])

        assert response.status_code == 503

    def test_rejects_pdf(self, client, auth_override):
        response = client.post(
            "/predict/image",
            files=[("file", ("doc.pdf", io.BytesIO(b"data"), "application/pdf"))],
        )
        assert response.status_code in (400, 415)


# ---------------------------------------------------------------------------
# POST /verify/image
# ---------------------------------------------------------------------------

class TestImageVerification:
    def test_returns_full_verification_response(
        self, client, auth_override, db_override
    ):
        with (
            patch("app.api.image_predict.run_ocr_branch", return_value=FAKE_OCR),
            patch("app.api.image_predict.run_image_classification_branch",
                  return_value=FAKE_IMG),
            patch("app.api.image_predict.run_clip_branch", return_value=FAKE_CLIP),
            patch("app.utils.file_utils.save_upload",
                  return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg")),
            patch("app.utils.file_utils.delete_temp_file"),
            patch("app.api.image_predict.save_prediction",
                  return_value=MagicMock(id=uuid.uuid4())),
        ):
            response = client.post("/verify/image", files=[_fake_image()])

        assert response.status_code == 200
        data = response.json()
        assert "overall_verdict" in data
        assert "text_prediction" in data
        assert "image_prediction" in data
        assert "similar_articles" in data
        assert "reasoning" in data

    def test_both_fake_gives_likely_fake_verdict(
        self, client, auth_override, db_override
    ):
        with (
            patch("app.api.image_predict.run_ocr_branch", return_value=FAKE_OCR),
            patch("app.api.image_predict.run_image_classification_branch",
                  return_value=FAKE_IMG),
            patch("app.api.image_predict.run_clip_branch", return_value=FAKE_CLIP),
            patch("app.utils.file_utils.save_upload",
                  return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg")),
            patch("app.utils.file_utils.delete_temp_file"),
            patch("app.api.image_predict.save_prediction",
                  return_value=MagicMock(id=uuid.uuid4())),
        ):
            response = client.post("/verify/image", files=[_fake_image()])

        assert response.json()["overall_verdict"] == "LIKELY FAKE"

    def test_degrades_gracefully_when_image_model_unavailable(
        self, client, auth_override, db_override
    ):
        img_unavailable = {
            "image_prediction": None,
            "image_confidence": None,
            "image_class_probabilities": {},
        }
        with (
            patch("app.api.image_predict.run_ocr_branch", return_value=FAKE_OCR),
            patch("app.api.image_predict.run_image_classification_branch",
                  return_value=img_unavailable),
            patch("app.api.image_predict.run_clip_branch", return_value=FAKE_CLIP),
            patch("app.utils.file_utils.save_upload",
                  return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg")),
            patch("app.utils.file_utils.delete_temp_file"),
            patch("app.api.image_predict.save_prediction",
                  return_value=MagicMock(id=uuid.uuid4())),
        ):
            response = client.post("/verify/image", files=[_fake_image()])

        # Should still return 200 — degrades gracefully
        assert response.status_code == 200
        data = response.json()
        assert data["image_prediction"] == "UNAVAILABLE"


# ---------------------------------------------------------------------------
# compute_verdict unit tests
# ---------------------------------------------------------------------------

class TestComputeVerdict:
    def test_both_fake(self):
        from app.services.image_verify_service import compute_verdict
        verdict, reasoning = compute_verdict("FAKE", "FAKE", False)
        assert verdict == "LIKELY FAKE"
        assert len(reasoning) > 0

    def test_both_real(self):
        from app.services.image_verify_service import compute_verdict
        verdict, _ = compute_verdict("REAL", "REAL", False)
        assert verdict == "LIKELY REAL"

    def test_image_fake_text_real(self):
        from app.services.image_verify_service import compute_verdict
        verdict, _ = compute_verdict("REAL", "FAKE", False)
        assert verdict == "LIKELY MISLEADING"

    def test_image_real_text_fake(self):
        from app.services.image_verify_service import compute_verdict
        verdict, _ = compute_verdict("FAKE", "REAL", False)
        assert verdict == "LIKELY FAKE"

    def test_text_only_fake(self):
        from app.services.image_verify_service import compute_verdict
        verdict, _ = compute_verdict("FAKE", None, False)
        assert verdict == "LIKELY FAKE"

    def test_both_none_uncertain(self):
        from app.services.image_verify_service import compute_verdict
        verdict, _ = compute_verdict(None, None, False)
        assert verdict == "UNCERTAIN"

    def test_clip_reuse_adds_reasoning(self):
        from app.services.image_verify_service import compute_verdict
        _, reasoning = compute_verdict("FAKE", "FAKE", True)
        assert any("reuse" in r.lower() or "context" in r.lower() for r in reasoning)
