"""
Pydantic schemas for Knowledge Graph Extraction (Phase 10).
"""

from typing import Optional
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """A single entity node in the knowledge graph."""

    id: str = Field(..., description="Unique node identifier (normalised entity text)")
    label: str = Field(..., description="Display label")
    type: str = Field(
        ...,
        description="Entity type: PERSON | ORG | GPE | DATE | EVENT | OTHER"
    )
    frequency: int = Field(..., description="Number of times this entity appears")


class GraphEdge(BaseModel):
    """A co-occurrence relationship between two nodes."""

    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")
    weight: int = Field(..., description="Number of sentences where both appear together")


class GraphRequest(BaseModel):
    text: str = Field(..., description="Article text to extract entities from")
    min_frequency: int = Field(
        default=1,
        ge=1,
        description="Minimum entity frequency to include as a node"
    )
    max_nodes: int = Field(
        default=50,
        ge=5,
        le=200,
        description="Maximum number of nodes to return"
    )


class KnowledgeGraphResponse(BaseModel):
    """Full knowledge graph response."""

    total_entities: int
    total_nodes: int
    total_edges: int
    entity_counts: dict[str, int] = Field(
        ..., description="Entity type breakdown e.g. {PERSON: 3, ORG: 2}"
    )
    nodes: list[GraphNode]
    edges: list[GraphEdge]
