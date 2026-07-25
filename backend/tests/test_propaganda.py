"""
Pytest tests for Phase 7 — Propaganda Detection.

Covers:
- Clean neutral text returns no detection
- Fear appeal text triggers Fear technique
- Clickbait text triggers Clickbait technique
- Conspiracy framing text triggers Conspiracy technique
- Multiple techniques in one text
- Overall score scales with detections
- Empty text handled gracefully
- Matched phrases returned
"""

import uuid
from unittest.mock import MagicMock

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
# Test texts
# ---------------------------------------------------------------------------

NEUTRAL_TEXT = (
    "Researchers at the university published a study on renewable energy. "
    "The findings suggest that solar panels have improved efficiency. "
    "The study was peer-reviewed and published in a reputable journal."
)

FEAR_TEXT = (
    "This is a catastrophic crisis. Your family is in danger. "
    "If we don't act now, the deadly threat will destroy everything. "
    "Experts warn of a terrifying collapse. The panic is spreading."
)

CLICKBAIT_TEXT = (
    "You won't believe what scientists discovered! This shocking miracle cure "
    "is exposed! The secret they don't want you to know. "
    "Unbelievable and guaranteed instant results!"
)

CONSPIRACY_TEXT = (
    "They don't want you to know the truth. This is a massive cover-up. "
    "Wake up, sheeple! The hidden agenda is orchestrated by the deep state. "
    "The mainstream media suppresses this evidence."
)

MIXED_TEXT = (
    "You won't believe this shocking discovery! There is a terrifying conspiracy "
    "to destroy our freedom. They don't want you to know the truth. "
    "Everyone who knows the facts is outraged. This is a catastrophic threat."
)


# ---------------------------------------------------------------------------
# API Tests
# ---------------------------------------------------------------------------

class TestNeutralText:
    def test_neutral_article_no_propaganda(self, client, auth_override, db_override):
        response = client.post(
            "/propaganda/analyse",
            json={"text": NEUTRAL_TEXT},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["propaganda_detected"] is False
        assert data["overall_score"] == 0.0
        assert len(data["techniques_found"]) == 0


class TestSingleTechniqueDetection:
    def test_fear_appeal_detected(self, client, auth_override, db_override):
        response = client.post(
            "/propaganda/analyse",
            json={"text": FEAR_TEXT},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["propaganda_detected"] is True
        assert data["overall_score"] > 0
        technique_names = [t["technique"] for t in data["techniques_found"]]
        assert "Fear Appeal" in technique_names

    def test_clickbait_detected(self, client, auth_override, db_override):
        response = client.post(
            "/propaganda/analyse",
            json={"text": CLICKBAIT_TEXT},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["propaganda_detected"] is True
        technique_names = [t["technique"] for t in data["techniques_found"]]
        assert "Clickbait" in technique_names

    def test_conspiracy_detected(self, client, auth_override, db_override):
        response = client.post(
            "/propaganda/analyse",
            json={"text": CONSPIRACY_TEXT},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["propaganda_detected"] is True
        technique_names = [t["technique"] for t in data["techniques_found"]]
        assert "Conspiracy Framing" in technique_names


class TestMultipleTechniques:
    def test_mixed_text_detects_multiple(self, client, auth_override, db_override):
        response = client.post(
            "/propaganda/analyse",
            json={"text": MIXED_TEXT},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["propaganda_detected"] is True
        assert len(data["techniques_found"]) >= 2
        assert data["overall_score"] > 0.3  # should be moderate to high

    def test_overall_score_scales_with_techniques(
        self, client, auth_override, db_override
    ):
        # Neutral should be 0
        r1 = client.post("/propaganda/analyse", json={"text": NEUTRAL_TEXT})
        score_neutral = r1.json()["overall_score"]

        # Clickbait should be > 0
        r2 = client.post("/propaganda/analyse", json={"text": CLICKBAIT_TEXT})
        score_clickbait = r2.json()["overall_score"]

        # Mixed (multiple techniques) should be highest
        r3 = client.post("/propaganda/analyse", json={"text": MIXED_TEXT})
        score_mixed = r3.json()["overall_score"]

        assert score_neutral == 0.0
        assert score_clickbait > score_neutral
        assert score_mixed > score_clickbait


class TestTechniqueDetails:
    def test_technique_has_required_fields(self, client, auth_override, db_override):
        response = client.post(
            "/propaganda/analyse",
            json={"text": FEAR_TEXT},
        )
        assert response.status_code == 200
        techniques = response.json()["techniques_found"]
        if techniques:
            t = techniques[0]
            assert "technique" in t
            assert "confidence" in t
            assert "matched_phrases" in t
            assert "description" in t

    def test_matched_phrases_populated(self, client, auth_override, db_override):
        response = client.post(
            "/propaganda/analyse",
            json={"text": CLICKBAIT_TEXT},
        )
        techniques = response.json()["techniques_found"]
        assert any(len(t["matched_phrases"]) > 0 for t in techniques)


class TestEdgeCases:
    def test_empty_text_handled(self, client, auth_override, db_override):
        response = client.post(
            "/propaganda/analyse",
            json={"text": ""},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["propaganda_detected"] is False

    def test_summary_always_present(self, client, auth_override, db_override):
        for text in [NEUTRAL_TEXT, FEAR_TEXT, MIXED_TEXT, ""]:
            response = client.post(
                "/propaganda/analyse",
                json={"text": text},
            )
            assert "summary" in response.json()
            assert len(response.json()["summary"]) > 0


# ---------------------------------------------------------------------------
# Service Unit Tests
# ---------------------------------------------------------------------------

class TestPropagandaServiceUnit:
    def test_detect_propaganda_neutral(self):
        from app.services.propaganda_service import detect_propaganda

        result = detect_propaganda(NEUTRAL_TEXT)
        assert result["propaganda_detected"] is False

    def test_detect_propaganda_fear(self):
        from app.services.propaganda_service import detect_propaganda

        result = detect_propaganda(FEAR_TEXT)
        assert result["propaganda_detected"] is True
        names = [t["technique"] for t in result["techniques_found"]]
        assert "Fear Appeal" in names

    def test_empty_text_returns_safe_result(self):
        from app.services.propaganda_service import detect_propaganda

        result = detect_propaganda("")
        assert result["propaganda_detected"] is False
        assert result["overall_score"] == 0.0
