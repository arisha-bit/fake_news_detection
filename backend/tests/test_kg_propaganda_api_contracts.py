"""
Test API contracts for Knowledge Graph and Propaganda Detection.

Ensures:
1. API responses match schema expectations
2. Frontend data structures are correct
3. Response codes are appropriate
4. Required fields are always present
"""

import uuid
from unittest.mock import MagicMock, patch

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
# KNOWLEDGE GRAPH API CONTRACT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeGraphAPIContract:
    """Ensure API response structure matches frontend expectations."""
    
    SAMPLE_GRAPH = {
        "total_entities": 5,
        "total_nodes": 3,
        "total_edges": 2,
        "entity_counts": {"PERSON": 1, "ORG": 2},
        "nodes": [
            {"id": "elon musk", "label": "Elon Musk", "type": "PERSON", "frequency": 2},
            {"id": "nasa", "label": "NASA", "type": "ORG", "frequency": 1},
            {"id": "spacex", "label": "SpaceX", "type": "ORG", "frequency": 2},
        ],
        "edges": [
            {"source": "elon musk", "target": "spacex", "weight": 2},
            {"source": "spacex", "target": "nasa", "weight": 1},
        ],
    }
    
    def test_response_code_200(self, client, auth_override, db_override):
        """Successful requests should return 200."""
        with patch(
            "app.api.knowledge_graph.extract_knowledge_graph",
            return_value=self.SAMPLE_GRAPH,
        ):
            response = client.post(
                "/knowledge-graph/extract",
                json={"text": "Sample article text"},
            )
        
        assert response.status_code == 200
    
    def test_response_has_all_required_top_level_fields(
        self, client, auth_override, db_override
    ):
        """Response must have all required top-level fields."""
        with patch(
            "app.api.knowledge_graph.extract_knowledge_graph",
            return_value=self.SAMPLE_GRAPH,
        ):
            response = client.post(
                "/knowledge-graph/extract",
                json={"text": "Sample article text"},
            )
        
        data = response.json()
        required_fields = {
            "total_entities",
            "total_nodes",
            "total_edges",
            "entity_counts",
            "nodes",
            "edges",
        }
        
        assert required_fields.issubset(data.keys()), f"Missing fields: {required_fields - set(data.keys())}"
    
    def test_node_structure_matches_frontend_expectations(
        self, client, auth_override, db_override
    ):
        """Each node must have id, label, type, frequency for frontend display."""
        with patch(
            "app.api.knowledge_graph.extract_knowledge_graph",
            return_value=self.SAMPLE_GRAPH,
        ):
            response = client.post(
                "/knowledge-graph/extract",
                json={"text": "Sample article text"},
            )
        
        nodes = response.json()["nodes"]
        
        # Frontend code (ToolsPage.jsx line 178):
        # {result.nodes.map((node) => (
        #   <div ... key={node.id} ...>
        #     <span>{node.label}</span>
        #     <span>×{node.frequency}</span>
        #   </div>
        # ))}
        
        required_node_fields = {"id", "label", "type", "frequency"}
        
        for node in nodes:
            assert required_node_fields.issubset(
                node.keys()
            ), f"Node missing fields: {required_node_fields - set(node.keys())}"
            
            # Type should be one of the known types or OTHER
            valid_types = {"PERSON", "ORG", "GPE", "DATE", "EVENT", "OTHER"}
            assert (
                node["type"] in valid_types
            ), f"Invalid node type: {node['type']}"
            
            # Frequency should be positive integer
            assert (
                isinstance(node["frequency"], int) and node["frequency"] > 0
            ), f"Invalid frequency: {node['frequency']}"
    
    def test_edge_structure_matches_frontend_expectations(
        self, client, auth_override, db_override
    ):
        """Each edge must have source, target, weight for frontend display."""
        with patch(
            "app.api.knowledge_graph.extract_knowledge_graph",
            return_value=self.SAMPLE_GRAPH,
        ):
            response = client.post(
                "/knowledge-graph/extract",
                json={"text": "Sample article text"},
            )
        
        edges = response.json()["edges"]
        
        # Frontend code (ToolsPage.jsx line 189):
        # {result.edges.slice(0, 8).map((edge, i) => (
        #   <div key={i} ...>
        #     <span>{edge.source}</span>
        #     <span>↔</span>
        #     <span>{edge.target}</span>
        #     <span>×{edge.weight}</span>
        #   </div>
        # ))}
        
        required_edge_fields = {"source", "target", "weight"}
        
        for edge in edges:
            assert required_edge_fields.issubset(
                edge.keys()
            ), f"Edge missing fields: {required_edge_fields - set(edge.keys())}"
            
            # Weight should be positive integer
            assert (
                isinstance(edge["weight"], int) and edge["weight"] > 0
            ), f"Invalid weight: {edge['weight']}"
    
    def test_empty_graph_has_correct_structure(
        self, client, auth_override, db_override
    ):
        """Empty graph (no entities) should have correct structure."""
        empty_graph = {
            "total_entities": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "entity_counts": {},
            "nodes": [],
            "edges": [],
        }
        
        with patch(
            "app.api.knowledge_graph.extract_knowledge_graph",
            return_value=empty_graph,
        ):
            response = client.post(
                "/knowledge-graph/extract",
                json={"text": "text with no entities"},
            )
        
        data = response.json()
        assert data["total_nodes"] == 0
        assert data["total_edges"] == 0
        assert data["nodes"] == []
        assert data["edges"] == []
        assert data["entity_counts"] == {}
    
    def test_entity_counts_populated_correctly(
        self, client, auth_override, db_override
    ):
        """entity_counts dict should match node types."""
        with patch(
            "app.api.knowledge_graph.extract_knowledge_graph",
            return_value=self.SAMPLE_GRAPH,
        ):
            response = client.post(
                "/knowledge-graph/extract",
                json={"text": "Sample article text"},
            )
        
        data = response.json()
        
        # Frontend code (ToolsPage.jsx line 172):
        # {Object.entries(result.entity_counts).map(([type, count]) => (
        #   <span key={type}>{type}: {count}</span>
        # ))}
        
        # entity_counts should be a dict
        assert isinstance(data["entity_counts"], dict)
        
        # Counts should match the number of nodes with that type
        for entity_type, count in data["entity_counts"].items():
            type_node_count = sum(1 for n in data["nodes"] if n["type"] == entity_type)
            assert (
                count == type_node_count
            ), f"entity_counts[{entity_type}]={count} but found {type_node_count} nodes"
    
    def test_totals_match_lists(self, client, auth_override, db_override):
        """total_nodes and total_edges should match list lengths."""
        with patch(
            "app.api.knowledge_graph.extract_knowledge_graph",
            return_value=self.SAMPLE_GRAPH,
        ):
            response = client.post(
                "/knowledge-graph/extract",
                json={"text": "Sample article text"},
            )
        
        data = response.json()
        
        assert data["total_nodes"] == len(data["nodes"])
        assert data["total_edges"] == len(data["edges"])


# ─────────────────────────────────────────────────────────────────────────────
# PROPAGANDA DETECTION API CONTRACT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPropagandaAPIContract:
    """Ensure API response structure matches frontend expectations."""
    
    SAMPLE_PROPAGANDA = {
        "propaganda_detected": True,
        "overall_score": 0.65,
        "techniques_found": [
            {
                "technique": "Fear Appeal",
                "confidence": 0.75,
                "matched_phrases": ["danger", "threat", "terrifying"],
                "description": "Uses threatening or alarming language to provoke fear.",
            },
            {
                "technique": "Clickbait",
                "confidence": 0.5,
                "matched_phrases": ["shocking", "exposed"],
                "description": "Uses sensational language to bait readers.",
            },
        ],
        "summary": "Moderate propaganda signals detected. Techniques identified: Fear Appeal, Clickbait.",
    }
    
    def test_response_code_200(self, client, auth_override, db_override):
        """Successful requests should return 200."""
        with patch(
            "app.api.propaganda.detect_propaganda",
            return_value=self.SAMPLE_PROPAGANDA,
        ):
            response = client.post(
                "/propaganda/analyse",
                json={"text": "Sample article text"},
            )
        
        assert response.status_code == 200
    
    def test_response_has_all_required_top_level_fields(
        self, client, auth_override, db_override
    ):
        """Response must have all required top-level fields."""
        with patch(
            "app.api.propaganda.detect_propaganda",
            return_value=self.SAMPLE_PROPAGANDA,
        ):
            response = client.post(
                "/propaganda/analyse",
                json={"text": "Sample article text"},
            )
        
        data = response.json()
        required_fields = {
            "propaganda_detected",
            "overall_score",
            "techniques_found",
            "summary",
        }
        
        assert required_fields.issubset(
            data.keys()
        ), f"Missing fields: {required_fields - set(data.keys())}"
    
    def test_technique_structure_matches_frontend_expectations(
        self, client, auth_override, db_override
    ):
        """Each technique must have correct fields for frontend display."""
        with patch(
            "app.api.propaganda.detect_propaganda",
            return_value=self.SAMPLE_PROPAGANDA,
        ):
            response = client.post(
                "/propaganda/analyse",
                json={"text": "Sample article text"},
            )
        
        techniques = response.json()["techniques_found"]
        
        # Frontend code (ToolsPage.jsx line 115):
        # {result.techniques_found.map((t, i) => (
        #   <div key={i}>
        #     <p>{t.technique}</p>
        #     <span>{Math.round(t.confidence * 100)}%</span>
        #     <p>{t.description}</p>
        #     {t.matched_phrases.map(...)}
        #   </div>
        # ))}
        
        required_technique_fields = {
            "technique",
            "confidence",
            "matched_phrases",
            "description",
        }
        
        for tech in techniques:
            assert required_technique_fields.issubset(
                tech.keys()
            ), f"Technique missing fields: {required_technique_fields - set(tech.keys())}"
            
            # Confidence should be 0.0 - 1.0
            assert (
                isinstance(tech["confidence"], (int, float))
                and 0.0 <= tech["confidence"] <= 1.0
            ), f"Invalid confidence: {tech['confidence']}"
            
            # matched_phrases should be a list
            assert isinstance(
                tech["matched_phrases"], list
            ), f"matched_phrases is not a list: {tech['matched_phrases']}"
            
            # All matched phrases should be strings
            for phrase in tech["matched_phrases"]:
                assert isinstance(phrase, str), f"Matched phrase is not string: {phrase}"
    
    def test_overall_score_in_valid_range(self, client, auth_override, db_override):
        """overall_score should be 0.0 - 1.0."""
        with patch(
            "app.api.propaganda.detect_propaganda",
            return_value=self.SAMPLE_PROPAGANDA,
        ):
            response = client.post(
                "/propaganda/analyse",
                json={"text": "Sample article text"},
            )
        
        score = response.json()["overall_score"]
        assert isinstance(score, (int, float)) and 0.0 <= score <= 1.0, f"Invalid score: {score}"
    
    def test_propaganda_detected_is_boolean(self, client, auth_override, db_override):
        """propaganda_detected should be boolean."""
        with patch(
            "app.api.propaganda.detect_propaganda",
            return_value=self.SAMPLE_PROPAGANDA,
        ):
            response = client.post(
                "/propaganda/analyse",
                json={"text": "Sample article text"},
            )
        
        detected = response.json()["propaganda_detected"]
        assert isinstance(detected, bool), f"propaganda_detected is not boolean: {detected}"
    
    def test_summary_is_always_present(self, client, auth_override, db_override):
        """summary field should always be a non-empty string."""
        with patch(
            "app.api.propaganda.detect_propaganda",
            return_value=self.SAMPLE_PROPAGANDA,
        ):
            response = client.post(
                "/propaganda/analyse",
                json={"text": "Sample article text"},
            )
        
        data = response.json()
        assert isinstance(data["summary"], str) and len(data["summary"]) > 0
    
    def test_empty_propaganda_has_correct_structure(
        self, client, auth_override, db_override
    ):
        """No propaganda detected should have correct structure."""
        empty_propaganda = {
            "propaganda_detected": False,
            "overall_score": 0.0,
            "techniques_found": [],
            "summary": "No significant propaganda techniques detected.",
        }
        
        with patch(
            "app.api.propaganda.detect_propaganda",
            return_value=empty_propaganda,
        ):
            response = client.post(
                "/propaganda/analyse",
                json={"text": "neutral text"},
            )
        
        data = response.json()
        assert data["propaganda_detected"] is False
        assert data["overall_score"] == 0.0
        assert data["techniques_found"] == []
        assert len(data["summary"]) > 0
    
    def test_matched_phrases_capped_but_still_valid(
        self, client, auth_override, db_override
    ):
        """Even if capped at 5, matched_phrases should be valid strings."""
        with patch(
            "app.api.propaganda.detect_propaganda",
            return_value=self.SAMPLE_PROPAGANDA,
        ):
            response = client.post(
                "/propaganda/analyse",
                json={"text": "Sample article text"},
            )
        
        techniques = response.json()["techniques_found"]
        
        for tech in techniques:
            phrases = tech["matched_phrases"]
            # Should have at most 5 phrases (backend cap)
            assert len(phrases) <= 5, f"More than 5 phrases: {len(phrases)}"
            # All should be strings
            assert all(isinstance(p, str) for p in phrases), f"Non-string phrase in {phrases}"
