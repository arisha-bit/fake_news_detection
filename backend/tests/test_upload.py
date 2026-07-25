"""
Pytest tests for the OCR image upload feature.

Covers:
- Valid image upload (mocked OCR + prediction)
- Unsupported file extension
- Unsupported MIME type
- OCR failure propagation
- Prediction integration (OCR text reaches prediction service)
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
    """Minimal User object sufficient for auth dependency override."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.role = "user"
    return user


@pytest.fixture
def auth_override(mock_user):
    """Override get_current_user so tests don't need a real JWT."""
    from app.core.security import get_current_user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield mock_user
    app.dependency_overrides.clear()


@pytest.fixture
def db_override():
    """Override get_db with a MagicMock session."""
    from app.db.dependencies import get_db

    mock_db = MagicMock()

    # Simulate DB add/commit/refresh chain
    def _refresh(obj):
        if not hasattr(obj, "id") or obj.id is None:
            obj.id = uuid.uuid4()

    mock_db.refresh.side_effect = _refresh

    app.dependency_overrides[get_db] = lambda: mock_db
    yield mock_db
    app.dependency_overrides.clear()


def _make_image_file(filename: str = "test.jpg", content_type: str = "image/jpeg"):
    """Return a minimal fake image as an UploadFile-compatible tuple."""
    return ("file", (filename, io.BytesIO(b"fake-image-bytes"), content_type))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidUpload:
    def test_returns_prediction_on_valid_image(
        self, client, auth_override, db_override
    ):
        fake_text = "Scientists discover new vaccine effective against flu."
        fake_prediction = {"prediction": "REAL", "confidence": 0.97}

        with (
            patch("app.api.upload.routes.extract_text_from_image", return_value=fake_text),
            patch("app.api.upload.routes._run_model", return_value=fake_prediction),
            patch("app.utils.file_utils.save_upload", return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg")),
            patch("app.utils.file_utils.delete_temp_file"),
        ):
            response = client.post(
                "/upload/image",
                files=[_make_image_file()],
            )

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "REAL"
        assert data["extracted_text"] == fake_text
        assert "keywords" in data
        assert "confidence" in data


class TestInvalidExtension:
    def test_rejects_pdf_extension(self, client, auth_override):
        response = client.post(
            "/upload/image",
            files=[("file", ("document.pdf", io.BytesIO(b"data"), "application/pdf"))],
        )
        assert response.status_code in (400, 415)

    def test_rejects_txt_extension(self, client, auth_override):
        response = client.post(
            "/upload/image",
            files=[("file", ("notes.txt", io.BytesIO(b"text"), "text/plain"))],
        )
        assert response.status_code in (400, 415)


class TestOCRFailure:
    def test_propagates_ocr_422(self, client, auth_override, db_override):
        from fastapi import HTTPException

        with (
            patch("app.utils.file_utils.save_upload", return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg")),
            patch("app.utils.file_utils.delete_temp_file"),
            patch(
                "app.api.upload.routes.extract_text_from_image",
                side_effect=HTTPException(status_code=422, detail="No text found"),
            ),
        ):
            response = client.post(
                "/upload/image",
                files=[_make_image_file()],
            )

        assert response.status_code == 422


class TestPredictionIntegration:
    def test_ocr_text_is_passed_to_run_model(
        self, client, auth_override, db_override
    ):
        fake_text = "Breaking: local election results revealed."
        captured = {}

        def fake_run_model(model_choice, text):
            captured["text"] = text
            return {"prediction": "FAKE", "confidence": 0.88}

        with (
            patch("app.api.upload.routes.extract_text_from_image", return_value=fake_text),
            patch("app.api.upload.routes._run_model", side_effect=fake_run_model),
            patch("app.utils.file_utils.save_upload", return_value=("abc.jpg", "test.jpg", "/tmp/abc.jpg")),
            patch("app.utils.file_utils.delete_temp_file"),
        ):
            client.post("/upload/image", files=[_make_image_file()])

        assert captured.get("text") == fake_text


# ---------------------------------------------------------------------------
# Phase 2 — PDF upload tests
# ---------------------------------------------------------------------------

class TestPDFValidUpload:
    def test_returns_prediction_on_valid_pdf(
        self, client, auth_override, db_override
    ):
        fake_text = "Government announces new infrastructure policy."
        fake_prediction = {"prediction": "REAL", "confidence": 0.93}

        with (
            patch("app.api.upload.routes.extract_text_from_pdf", return_value=fake_text),
            patch("app.api.upload.routes._run_model", return_value=fake_prediction),
            patch(
                "app.utils.file_utils.save_upload",
                return_value=("abc.pdf", "article.pdf", "/tmp/abc.pdf"),
            ),
            patch("app.utils.file_utils.delete_temp_file"),
        ):
            response = client.post(
                "/upload/pdf",
                files=[("file", ("article.pdf", io.BytesIO(b"%PDF-fake"), "application/pdf"))],
            )

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "REAL"
        assert data["extracted_text"] == fake_text


class TestPDFInvalidFormat:
    def test_rejects_image_as_pdf(self, client, auth_override):
        response = client.post(
            "/upload/pdf",
            files=[("file", ("photo.jpg", io.BytesIO(b"data"), "image/jpeg"))],
        )
        assert response.status_code in (400, 415)

    def test_rejects_txt_as_pdf(self, client, auth_override):
        response = client.post(
            "/upload/pdf",
            files=[("file", ("doc.txt", io.BytesIO(b"text"), "text/plain"))],
        )
        assert response.status_code in (400, 415)


class TestPDFExtractionFailure:
    def test_propagates_empty_pdf_422(self, client, auth_override, db_override):
        from fastapi import HTTPException

        with (
            patch(
                "app.utils.file_utils.save_upload",
                return_value=("abc.pdf", "article.pdf", "/tmp/abc.pdf"),
            ),
            patch("app.utils.file_utils.delete_temp_file"),
            patch(
                "app.api.upload.routes.extract_text_from_pdf",
                side_effect=HTTPException(
                    status_code=422, detail="No text found in PDF"
                ),
            ),
        ):
            response = client.post(
                "/upload/pdf",
                files=[("file", ("article.pdf", io.BytesIO(b"%PDF-fake"), "application/pdf"))],
            )

        assert response.status_code == 422
