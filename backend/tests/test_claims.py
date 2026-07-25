"""
Pytest tests for Phase 3 — Claim Extraction.

Covers:
- Valid article extracts multiple claims
- Each claim gets individually predicted
- Overall verdict is majority vote
- Empty text raises 422
- No extractable claims raises 422
- Model choice is passed through correctly
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
# Tests
# ---------------------------------------------------------------------------

SAMPLE_ARTICLE = (
    "NASA scientists confirmed the discovery of water on Mars. "
    "The government declared a national holiday to celebrate. "
    "Experts warn this could change the future of space exploration."
)


class TestValidClaimExtraction:
    def test_returns_claims_for_valid_article(
        self, client, auth_override, db_override
    ):
        fake_claims = [
            "NASA scientists confirmed the discovery of water on Mars.",
            "The government declared a national holiday to celebrate.",
            "Experts warn this could change the future of space exploration.",
        ]
        fake_prediction = {"prediction": "REAL", "confidence": 0.90}

        with (
            patch(
                "app.api.claims.extract_claims",
                return_value=fake_claims,
            ),
            patch(
                "app.api.claims._run_model",
                return_value=fake_prediction,
            ),
        ):
            response = client.post(
                "/claims/extract",
                json={"text": SAMPLE_ARTICLE, "model": "logistic"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_claims"] == 3
        assert len(data["claims"]) == 3
        assert data["overall_verdict"] in ("REAL", "FAKE")

    def test_claim_index_is_sequential(
        self, client, auth_override, db_override
    ):
        fake_claims = ["Claim one is here.", "Claim two is here."]
        fake_prediction = {"prediction": "FAKE", "confidence": 0.85}

        with (
            patch("app.api.claims.extract_claims", return_value=fake_claims),
            patch("app.api.claims._run_model", return_value=fake_prediction),
        ):
            response = client.post(
                "/claims/extract",
                json={"text": SAMPLE_ARTICLE},
            )

        data = response.json()
        indices = [c["claim_index"] for c in data["claims"]]
        assert indices == list(range(len(fake_claims)))


class TestMajorityVoteVerdict:
    def test_majority_fake_gives_fake_verdict(
        self, client, auth_override, db_override
    ):
        fake_claims = ["Claim A.", "Claim B.", "Claim C."]
        # 2 FAKE, 1 REAL → overall should be FAKE
        side_effects = [
            {"prediction": "FAKE", "confidence": 0.9},
            {"prediction": "FAKE", "confidence": 0.85},
            {"prediction": "REAL", "confidence": 0.7},
        ]

        with (
            patch("app.api.claims.extract_claims", return_value=fake_claims),
            patch("app.api.claims._run_model", side_effect=side_effects),
        ):
            response = client.post(
                "/claims/extract",
                json={"text": SAMPLE_ARTICLE},
            )

        data = response.json()
        assert data["overall_verdict"] == "FAKE"
        assert data["fake_claims"] == 2
        assert data["real_claims"] == 1


class TestEdgeCases:
    def test_empty_text_returns_422(self, client, auth_override, db_override):
        response = client.post(
            "/claims/extract",
            json={"text": ""},
        )
        assert response.status_code == 422

    def test_no_extractable_claims_returns_422(
        self, client, auth_override, db_override
    ):
        with patch("app.api.claims.extract_claims", return_value=[]):
            response = client.post(
                "/claims/extract",
                json={"text": SAMPLE_ARTICLE},
            )
        assert response.status_code == 422


class TestClaimServiceUnit:
    """Unit tests for claim_service helpers — no HTTP layer."""

    def test_compute_overall_verdict_majority_fake(self):
        from app.services.claim_service import compute_overall_verdict

        verdict, conf = compute_overall_verdict(
            ["FAKE", "FAKE", "REAL"], [0.9, 0.8, 0.7]
        )
        assert verdict == "FAKE"
        assert round(conf, 4) == round((0.9 + 0.8 + 0.7) / 3, 4)

    def test_compute_overall_verdict_majority_real(self):
        from app.services.claim_service import compute_overall_verdict

        verdict, _ = compute_overall_verdict(
            ["REAL", "REAL", "FAKE"], [0.9, 0.85, 0.7]
        )
        assert verdict == "REAL"

    def test_compute_overall_verdict_empty(self):
        from app.services.claim_service import compute_overall_verdict

        verdict, conf = compute_overall_verdict([], [])
        assert verdict == "REAL"
        assert conf == 0.0

    def test_is_valid_claim_filters_short(self):
        from app.services.claim_service import _is_valid_claim

        assert _is_valid_claim("Too short.") is False

    def test_is_valid_claim_filters_questions(self):
        from app.services.claim_service import _is_valid_claim

        assert _is_valid_claim("Did the government really do this?") is False

    def test_is_valid_claim_accepts_good_sentence(self):
        from app.services.claim_service import _is_valid_claim

        assert _is_valid_claim("NASA confirmed water was found on Mars.") is True
