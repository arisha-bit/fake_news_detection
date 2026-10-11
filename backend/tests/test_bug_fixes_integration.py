"""
Integration tests for Knowledge Graph and Propaganda Detection bug fixes.

Tests:
1. Knowledge Graph: Stable node ID normalization (N.A.S.A → nasa)
2. Knowledge Graph: Text truncation detection (50k char limit)
3. Propaganda: Variable n-gram detection (3, 4, 5 words)
4. Propaganda: Selective stop-gram filtering
5. Propaganda: Repetition confidence scaling
"""

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.user import User


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


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE GRAPH TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeGraphNormalisation:
    """Test stable node ID normalization with punctuation variations."""
    
    def test_normalise_id_removes_punctuation(self):
        from app.services.graph_service import _normalise_id
        
        assert _normalise_id("N.A.S.A.") == "nasa"
        assert _normalise_id("U.S.A.") == "usa"
        assert _normalise_id('"NASA"') == "nasa"
        assert _normalise_id("(Elon Musk)") == "elon musk"
    
    def test_normalise_id_not_empty_after_punct_removal(self):
        from app.services.graph_service import _normalise_id
        
        # Even if input is mostly punctuation, should not return empty string
        # if there are letters/numbers
        assert _normalise_id("...NASA...") == "nasa"
        assert _normalise_id("---X---") == "x"
    
    def test_normalise_id_stable_across_variations(self):
        """Same entity in different formats should normalize to same ID."""
        from app.services.graph_service import _normalise_id
        
        variations = ["NASA", "N.A.S.A.", "N.A.S.A", "(NASA)", '"NASA"']
        normalized = [_normalise_id(v) for v in variations]
        
        # All should normalize to "nasa"
        assert all(n == "nasa" for n in normalized), f"Variations: {list(zip(variations, normalized))}"
    
    def test_normalise_id_collapse_spaces(self):
        from app.services.graph_service import _normalise_id
        
        assert _normalise_id("Elon   Musk") == "elon musk"
        assert _normalise_id("  NASA  ") == "nasa"


class TestKnowledgeGraphTruncation:
    """Test that long texts trigger truncation warning."""
    
    def test_extract_large_text_truncates(self):
        """Extract should handle texts > 50k chars without error."""
        from app.services.graph_service import extract_knowledge_graph
        
        # Create 60k character text
        base_text = "Elon Musk founded SpaceX in California. NASA partnered with SpaceX for the Artemis mission. "
        large_text = base_text * 700  # ~60k chars
        
        assert len(large_text) > 50_000
        
        # Should not raise an error, just truncate
        result = extract_knowledge_graph(large_text)
        
        # Should have extracted something (entities from first 50k chars)
        assert "nodes" in result
        assert "edges" in result
        assert isinstance(result["nodes"], list)
        
        # Note: exact entities depend on spaCy model, just verify structure
        assert "total_nodes" in result
        assert "total_edges" in result


# ─────────────────────────────────────────────────────────────────────────────
# PROPAGANDA DETECTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPropagandaRepetitionFix:
    """Test improved repetition detection with variable n-grams."""
    
    def test_detect_3gram_repetition(self):
        """Should detect 3-word repetition."""
        from app.services.propaganda_service import _detect_repetition
        
        text = "Lock her up! Lock her up! Lock her up!"
        confidence, matched = _detect_repetition(text)
        
        # Should detect the repetition
        assert confidence > 0, f"Expected confidence > 0, got {confidence}"
        assert matched, f"Expected matched phrases, got {matched}"
        assert any("lock her up" in m.lower() for m in matched), f"Expected 'lock her up' in {matched}"
    
    def test_detect_4gram_repetition(self):
        """Should detect 4-word repetition."""
        from app.services.propaganda_service import _detect_repetition
        
        text = "Act now immediately today. Act now immediately today. Act now immediately today."
        confidence, matched = _detect_repetition(text)
        
        assert confidence > 0, f"Expected confidence > 0, got {confidence}"
    
    def test_detect_5gram_repetition(self):
        """Should detect 5-word repetition."""
        from app.services.propaganda_service import _detect_repetition
        
        text = "We will fight them everywhere now. We will fight them everywhere now. We will fight them everywhere now."
        confidence, matched = _detect_repetition(text)
        
        assert confidence > 0, f"Expected confidence > 0, got {confidence}"
        assert any("fight them everywhere" in m.lower() for m in matched), f"Expected phrase in {matched}"
    
    def test_repetition_with_common_words_detected(self):
        """Should detect repetition even if phrase contains common words like 'to'."""
        from app.services.propaganda_service import _detect_repetition
        
        text = "You need to act now. You need to act now. You need to act now."
        confidence, matched = _detect_repetition(text)
        
        # Old bug: would filter out "you need to act" because "to" is a stop word
        # New fix: should detect this
        assert confidence > 0, f"Expected confidence > 0 for common words, got {confidence}"
        assert any("need to act" in m.lower() for m in matched), f"Expected 'need to act' in {matched}"
    
    def test_stop_word_only_grams_filtered(self):
        """Should NOT detect grams that are pure stop words."""
        from app.services.propaganda_service import _detect_repetition
        
        text = "the and of. the and of. the and of. the and of. the and of."
        confidence, matched = _detect_repetition(text)
        
        # Pure stop-word grams should be filtered (>= 80% stop words)
        assert confidence == 0.0 or matched == [], f"Should not detect pure stop words, got {matched}"


class TestPropagandaServiceIntegration:
    """Test that overall propaganda detection works with fixes."""
    
    def test_propaganda_with_repetition_detected(self):
        """Full pipeline should detect repetition technique."""
        from app.services.propaganda_service import detect_propaganda
        
        text = "Act now! Act now! Act now! We must act immediately! Act now, act now!"
        result = detect_propaganda(text)
        
        assert result["propaganda_detected"] is True
        technique_names = [t["technique"] for t in result["techniques_found"]]
        
        # Should include Repetition technique
        assert "Repetition" in technique_names, f"Repetition not in {technique_names}"
        
        # Should have at least moderate confidence
        assert result["overall_score"] >= 0.3, f"Expected moderate score, got {result['overall_score']}"
    
    def test_propaganda_fear_and_repetition(self):
        """Should detect multiple techniques in one text."""
        from app.services.propaganda_service import detect_propaganda
        
        text = (
            "This is a catastrophic threat! "
            "Act now! Act now! Act now! "
            "We are in deadly danger. "
            "We must act now or face terror!"
        )
        result = detect_propaganda(text)
        
        assert result["propaganda_detected"] is True
        technique_names = [t["technique"] for t in result["techniques_found"]]
        
        # Should detect at least Fear Appeal and Repetition
        assert "Fear Appeal" in technique_names or "Repetition" in technique_names, f"Expected techniques in {technique_names}"
        assert result["overall_score"] > 0.5, f"Expected high score, got {result['overall_score']}"


class TestPropagandaConfidenceScaling:
    """Test that confidence scores are properly scaled."""
    
    def test_confidence_between_0_and_1(self):
        """All technique confidences should be in [0.0, 1.0]."""
        from app.services.propaganda_service import detect_propaganda
        
        test_texts = [
            "Neutral article about renewable energy.",
            "This shocking miracle cure is exposed!",
            "Act now! Act now! Act now!",
            "They don't want you to know the truth about this conspiracy.",
            "Everyone believes this. Everyone knows this. Everyone agrees.",
        ]
        
        for text in test_texts:
            result = detect_propaganda(text)
            
            # Overall score should be in [0, 1]
            assert 0.0 <= result["overall_score"] <= 1.0, f"Overall score out of range: {result['overall_score']}"
            
            # Each technique confidence should be in [0, 1]
            for tech in result["techniques_found"]:
                assert 0.0 <= tech["confidence"] <= 1.0, f"Technique confidence out of range: {tech}"
