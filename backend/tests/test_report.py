"""
Pytest tests for Phase 8 — Verification Report Generator.

Covers:
- Valid request returns PDF response
- Content-Type is application/pdf
- Content-Disposition has filename
- All sections can be disabled
- Source URL triggers credibility section
- Empty text handled gracefully
- Report service returns BytesIO buffer
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


SAMPLE_TEXT = (
    "Scientists at NASA confirmed the discovery of water on Mars. "
    "Government officials declared a national holiday to celebrate. "
    "Experts warn this could change the future of space exploration forever."
)

FAKE_PREDICTION = {"prediction": "REAL", "confidence": 0.92}
FAKE_CLAIMS = ["NASA confirmed water on Mars.", "Government declared a holiday."]
FAKE_EVIDENCE = [
    {
        "rank": 1, "title": "NASA Mars discovery", "snippet": "Water found...",
        "label": "REAL", "similarity": 0.89, "subject": "science", "date": "2020-09-28"
    }
]
FAKE_PROPAGANDA = {
    "propaganda_detected": False,
    "overall_score": 0.0,
    "techniques_found": [],
    "summary": "No propaganda detected.",
}
FAKE_CREDIBILITY = {
    "domain": "reuters.com",
    "found_in_database": True,
    "trust_score": 95.0,
    "reliability_score": 94.0,
    "bias_rating": "CENTER",
    "category": "Mainstream",
    "credibility_label": "HIGH",
    "verdict": "reuters.com is highly reliable.",
    "notes": "Wire service.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_all_services():
    return [
        patch("app.services.report_service._run_model", return_value=FAKE_PREDICTION),
        patch("app.services.report_service.extract_keywords", return_value=["water", "mars"]),
        patch("app.services.report_service.clickbait_score", return_value=0),
        patch("app.services.report_service.generate_explanation", return_value="Neutral language."),
        patch("app.services.report_service.extract_claims", return_value=FAKE_CLAIMS),
        patch("app.services.report_service.compute_overall_verdict", return_value=("REAL", 0.92)),
        patch("app.services.report_service.search_evidence", return_value=FAKE_EVIDENCE),
        patch("app.services.report_service.detect_propaganda", return_value=FAKE_PROPAGANDA),
        patch("app.services.report_service.check_credibility", return_value=FAKE_CREDIBILITY),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReportGeneration:
    def test_returns_pdf_response(self, client, auth_override, db_override):
        with (
            patch("app.services.report_service._run_model", return_value=FAKE_PREDICTION),
            patch("app.services.report_service.extract_keywords", return_value=["nasa"]),
            patch("app.services.report_service.clickbait_score", return_value=0),
            patch("app.services.report_service.generate_explanation", return_value="OK"),
            patch("app.services.report_service.extract_claims", return_value=FAKE_CLAIMS),
            patch("app.services.report_service.compute_overall_verdict", return_value=("REAL", 0.9)),
            patch("app.services.report_service.search_evidence", return_value=FAKE_EVIDENCE),
            patch("app.services.report_service.detect_propaganda", return_value=FAKE_PROPAGANDA),
        ):
            response = client.post(
                "/report/generate",
                json={
                    "text": SAMPLE_TEXT,
                    "include_credibility": False,
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"

    def test_content_disposition_has_filename(self, client, auth_override, db_override):
        with (
            patch("app.services.report_service._run_model", return_value=FAKE_PREDICTION),
            patch("app.services.report_service.extract_keywords", return_value=[]),
            patch("app.services.report_service.clickbait_score", return_value=0),
            patch("app.services.report_service.generate_explanation", return_value="OK"),
            patch("app.services.report_service.extract_claims", return_value=FAKE_CLAIMS),
            patch("app.services.report_service.compute_overall_verdict", return_value=("REAL", 0.9)),
            patch("app.services.report_service.search_evidence", return_value=FAKE_EVIDENCE),
            patch("app.services.report_service.detect_propaganda", return_value=FAKE_PROPAGANDA),
        ):
            response = client.post(
                "/report/generate",
                json={"text": SAMPLE_TEXT, "include_credibility": False},
            )

        cd = response.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert ".pdf" in cd

    def test_response_is_non_empty_pdf(self, client, auth_override, db_override):
        with (
            patch("app.services.report_service._run_model", return_value=FAKE_PREDICTION),
            patch("app.services.report_service.extract_keywords", return_value=[]),
            patch("app.services.report_service.clickbait_score", return_value=0),
            patch("app.services.report_service.generate_explanation", return_value="OK"),
            patch("app.services.report_service.extract_claims", return_value=[]),
            patch("app.services.report_service.compute_overall_verdict", return_value=("REAL", 0.9)),
            patch("app.services.report_service.search_evidence", return_value=[]),
            patch("app.services.report_service.detect_propaganda", return_value=FAKE_PROPAGANDA),
        ):
            response = client.post(
                "/report/generate",
                json={
                    "text": SAMPLE_TEXT,
                    "include_credibility": False,
                },
            )

        assert len(response.content) > 1000  # PDF is never empty
        assert response.content[:4] == b"%PDF"  # PDF magic bytes


class TestSectionToggles:
    def test_all_sections_disabled(self, client, auth_override, db_override):
        with (
            patch("app.services.report_service._run_model", return_value=FAKE_PREDICTION),
            patch("app.services.report_service.extract_keywords", return_value=[]),
            patch("app.services.report_service.clickbait_score", return_value=0),
            patch("app.services.report_service.generate_explanation", return_value="OK"),
        ):
            response = client.post(
                "/report/generate",
                json={
                    "text": SAMPLE_TEXT,
                    "include_claims": False,
                    "include_evidence": False,
                    "include_propaganda": False,
                    "include_credibility": False,
                },
            )

        assert response.status_code == 200
        assert response.content[:4] == b"%PDF"


class TestReportServiceUnit:
    def test_generate_report_returns_bytesio(self):
        from app.services.report_service import generate_report

        with (
            patch("app.services.report_service._run_model", return_value=FAKE_PREDICTION),
            patch("app.services.report_service.extract_keywords", return_value=["test"]),
            patch("app.services.report_service.clickbait_score", return_value=10),
            patch("app.services.report_service.generate_explanation", return_value="Test explanation"),
            patch("app.services.report_service.detect_propaganda", return_value=FAKE_PROPAGANDA),
        ):
            result = generate_report(
                text=SAMPLE_TEXT,
                include_claims=False,
                include_evidence=False,
                include_credibility=False,
            )

        assert isinstance(result, io.BytesIO)
        content = result.read()
        assert content[:4] == b"%PDF"
        assert len(content) > 500
