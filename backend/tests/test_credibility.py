"""
Pytest tests for Phase 6 — Source Credibility Scoring.

Covers:
- Known HIGH credibility domain
- Known VERY_LOW credibility domain
- Unknown domain returns found_in_database=False
- Full URL extracts domain correctly
- Subdomain resolves to root domain
- www prefix stripped correctly
- Blank input raises 400
- Correct credibility label returned
- Verdict string populated
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
# API Tests
# ---------------------------------------------------------------------------

class TestKnownHighCredibilitySource:
    def test_reuters_returns_high(self, client, auth_override, db_override):
        response = client.post(
            "/credibility/check",
            json={"url": "https://reuters.com/article/123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["found_in_database"] is True
        assert data["credibility_label"] == "HIGH"
        assert data["trust_score"] >= 90
        assert data["domain"] == "reuters.com"

    def test_apnews_returns_high(self, client, auth_override, db_override):
        response = client.post(
            "/credibility/check",
            json={"url": "apnews.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["credibility_label"] == "HIGH"


class TestKnownLowCredibilitySource:
    def test_infowars_returns_very_low(self, client, auth_override, db_override):
        response = client.post(
            "/credibility/check",
            json={"url": "infowars.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["credibility_label"] == "VERY_LOW"
        assert data["trust_score"] < 20

    def test_breitbart_returns_very_low(self, client, auth_override, db_override):
        response = client.post(
            "/credibility/check",
            json={"url": "https://www.breitbart.com/politics/story"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["credibility_label"] == "VERY_LOW"


class TestUnknownDomain:
    def test_unknown_domain_not_found(self, client, auth_override, db_override):
        response = client.post(
            "/credibility/check",
            json={"url": "https://totallyunknownnewssite12345.com/article"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["found_in_database"] is False
        assert data["trust_score"] is None
        assert data["credibility_label"] is None
        assert "not found" in data["verdict"].lower()


class TestDomainExtraction:
    def test_www_prefix_stripped(self, client, auth_override, db_override):
        response = client.post(
            "/credibility/check",
            json={"url": "www.bbc.com"},
        )
        assert response.status_code == 200
        assert response.json()["domain"] == "bbc.com"

    def test_full_url_extracts_domain(self, client, auth_override, db_override):
        response = client.post(
            "/credibility/check",
            json={"url": "https://www.nytimes.com/2024/01/01/politics/article.html"},
        )
        assert response.status_code == 200
        assert response.json()["domain"] == "nytimes.com"

    def test_bare_domain_works(self, client, auth_override, db_override):
        response = client.post(
            "/credibility/check",
            json={"url": "snopes.com"},
        )
        assert response.status_code == 200
        assert response.json()["domain"] == "snopes.com"


class TestEdgeCases:
    def test_blank_url_returns_400(self, client, auth_override, db_override):
        response = client.post(
            "/credibility/check",
            json={"url": ""},
        )
        assert response.status_code == 400

    def test_verdict_is_always_populated(self, client, auth_override, db_override):
        for url in ["reuters.com", "infowars.com", "unknowndomain999.com"]:
            response = client.post(
                "/credibility/check",
                json={"url": url},
            )
            assert response.status_code == 200
            assert len(response.json()["verdict"]) > 0


# ---------------------------------------------------------------------------
# Service Unit Tests
# ---------------------------------------------------------------------------

class TestCredibilityServiceUnit:
    def test_extract_domain_full_url(self):
        from app.services.credibility_service import _extract_domain
        assert _extract_domain("https://www.bbc.com/news") == "bbc.com"

    def test_extract_domain_bare(self):
        from app.services.credibility_service import _extract_domain
        assert _extract_domain("reuters.com") == "reuters.com"

    def test_extract_domain_www(self):
        from app.services.credibility_service import _extract_domain
        assert _extract_domain("www.apnews.com") == "apnews.com"

    def test_score_to_label_high(self):
        from app.services.credibility_service import _score_to_label
        assert _score_to_label(90) == "HIGH"

    def test_score_to_label_very_low(self):
        from app.services.credibility_service import _score_to_label
        assert _score_to_label(10) == "VERY_LOW"

    def test_score_to_label_medium(self):
        from app.services.credibility_service import _score_to_label
        assert _score_to_label(55) == "MEDIUM"

    def test_lookup_subdomain_resolves(self):
        from app.services.credibility_service import _lookup
        db = {"bbc.com": {"trust_score": 90}}
        assert _lookup("news.bbc.com", db) is not None

    def test_lookup_unknown_returns_none(self):
        from app.services.credibility_service import _lookup
        db = {"bbc.com": {"trust_score": 90}}
        assert _lookup("unknownsite.xyz", db) is None
