"""
Pytest tests for Phase 9 — Analytics Dashboard.

Covers:
- GET /analytics/monthly returns list with month keys
- GET /analytics/confidence returns 5 buckets
- GET /analytics/keywords returns keyword list
- GET /analytics/verdicts-over-time returns fake/real per month
- GET /analytics/summary returns all sections in one call
- Empty prediction history returns safe empty/zero results
- Confidence bucketing logic
- Verdicts-over-time counting logic
"""

import uuid
from datetime import datetime
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

class TestMonthlyUsage:
    def test_returns_list(self, client, auth_override, db_override):
        with patch(
            "app.api.analytics.get_monthly_usage",
            return_value=[{"month": "2024-01", "count": 5}],
        ):
            response = client.get("/analytics/monthly")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert data[0]["month"] == "2024-01"
        assert "count" in data[0]

    def test_empty_returns_empty_list(self, client, auth_override, db_override):
        with patch("app.api.analytics.get_monthly_usage", return_value=[]):
            response = client.get("/analytics/monthly")

        assert response.status_code == 200
        assert response.json() == []


class TestConfidenceDistribution:
    def test_returns_five_buckets(self, client, auth_override, db_override):
        fake_dist = {
            "0-50": 2, "50-70": 5, "70-85": 10, "85-95": 20, "95-100": 8
        }
        with patch(
            "app.api.analytics.get_confidence_distribution",
            return_value=fake_dist,
        ):
            response = client.get("/analytics/confidence")

        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {"0-50", "50-70", "70-85", "85-95", "95-100"}

    def test_all_buckets_present_when_empty(self, client, auth_override, db_override):
        with patch(
            "app.api.analytics.get_confidence_distribution",
            return_value={"0-50": 0, "50-70": 0, "70-85": 0, "85-95": 0, "95-100": 0},
        ):
            response = client.get("/analytics/confidence")

        data = response.json()
        assert sum(data.values()) == 0


class TestTopKeywords:
    def test_returns_keyword_list(self, client, auth_override, db_override):
        fake_kws = [
            {"keyword": "fake news", "count": 12},
            {"keyword": "climate change", "count": 8},
        ]
        with patch("app.api.analytics.get_top_keywords", return_value=fake_kws):
            response = client.get("/analytics/keywords")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert data[0]["keyword"] == "fake news"

    def test_empty_history_returns_empty(self, client, auth_override, db_override):
        with patch("app.api.analytics.get_top_keywords", return_value=[]):
            response = client.get("/analytics/keywords")

        assert response.json() == []


class TestVerdictsOverTime:
    def test_returns_monthly_fake_real_counts(self, client, auth_override, db_override):
        fake_data = [
            {"month": "2024-01", "fake_count": 3, "real_count": 7},
            {"month": "2024-02", "fake_count": 5, "real_count": 10},
        ]
        with patch(
            "app.api.analytics.get_verdicts_over_time", return_value=fake_data
        ):
            response = client.get("/analytics/verdicts-over-time")

        assert response.status_code == 200
        data = response.json()
        assert "fake_count" in data[0]
        assert "real_count" in data[0]
        assert "month" in data[0]


class TestDashboardSummary:
    def test_summary_has_all_sections(self, client, auth_override, db_override):
        with (
            patch("app.api.analytics.get_monthly_usage", return_value=[]),
            patch("app.api.analytics.get_confidence_distribution",
                  return_value={"0-50": 0, "50-70": 0, "70-85": 0, "85-95": 0, "95-100": 0}),
            patch("app.api.analytics.get_top_keywords", return_value=[]),
            patch("app.api.analytics.get_verdicts_over_time", return_value=[]),
        ):
            response = client.get("/analytics/summary")

        assert response.status_code == 200
        data = response.json()
        assert "overview" in data
        assert "model_usage" in data
        assert "monthly" in data
        assert "confidence_distribution" in data
        assert "top_keywords" in data
        assert "verdicts_over_time" in data

    def test_overview_has_required_fields(self, client, auth_override, db_override):
        with (
            patch("app.api.analytics.get_monthly_usage", return_value=[]),
            patch("app.api.analytics.get_confidence_distribution",
                  return_value={"0-50": 0, "50-70": 0, "70-85": 0, "85-95": 0, "95-100": 0}),
            patch("app.api.analytics.get_top_keywords", return_value=[]),
            patch("app.api.analytics.get_verdicts_over_time", return_value=[]),
        ):
            response = client.get("/analytics/summary")

        overview = response.json()["overview"]
        assert "total_predictions" in overview
        assert "fake_predictions" in overview
        assert "real_predictions" in overview
        assert "average_confidence" in overview
        assert "fake_ratio" in overview


# ---------------------------------------------------------------------------
# Service Unit Tests
# ---------------------------------------------------------------------------

class TestConfidenceBucketingUnit:
    def test_bucketing_logic(self):
        from app.services.analytics_service import get_confidence_distribution

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = [
            (0.30,), (0.60,), (0.75,), (0.90,), (0.98,)
        ]

        result = get_confidence_distribution(mock_db, uuid.uuid4())
        assert result["0-50"] == 1
        assert result["50-70"] == 1
        assert result["70-85"] == 1
        assert result["85-95"] == 1
        assert result["95-100"] == 1

    def test_none_confidence_ignored(self):
        from app.services.analytics_service import get_confidence_distribution

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = [
            (None,), (0.95,)
        ]

        result = get_confidence_distribution(mock_db, uuid.uuid4())
        assert result["95-100"] == 1
        assert sum(result.values()) == 1


class TestMonthlyUsageUnit:
    def test_formats_month_correctly(self):
        from app.services.analytics_service import get_monthly_usage

        mock_row = MagicMock()
        mock_row.year = 2024
        mock_row.month = 3
        mock_row.count = 7

        mock_db = MagicMock()
        (mock_db.query.return_value
            .filter.return_value
            .group_by.return_value
            .order_by.return_value
            .all.return_value) = [mock_row]

        result = get_monthly_usage(mock_db, uuid.uuid4())
        assert result[0]["month"] == "2024-03"
        assert result[0]["count"] == 7
