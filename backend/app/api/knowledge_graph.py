"""
Knowledge Graph API — POST /knowledge-graph/extract

Accepts article text and returns a structured entity graph:
- Nodes: named entities (people, organisations, locations, dates)
- Edges: co-occurrence relationships between entities

The response is ready for frontend visualisation with React Force Graph,
D3.js, or any graph library that accepts nodes/edges format.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.dependencies import get_db
from app.models.user import User
from app.schemas.graph import GraphEdge, GraphNode, GraphRequest, KnowledgeGraphResponse
from app.services.graph_service import extract_knowledge_graph

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/knowledge-graph",
    tags=["Knowledge Graph"],
)


@router.post("/extract", response_model=KnowledgeGraphResponse)
def extract_graph(
    payload: GraphRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Extract a knowledge graph from article text.

    Identifies named entities (PERSON, ORG, GPE, DATE, EVENT) and
    builds co-occurrence edges between entities that appear in the
    same sentence.

    - **text**: Full article text.
    - **min_frequency**: Minimum times an entity must appear to be a node (default 1).
    - **max_nodes**: Cap on returned nodes, sorted by frequency (default 50, max 200).

    Returns nodes + edges in a format compatible with React Force Graph / D3.
    """
    logger.info(
        "Knowledge graph extraction — user=%s, text_len=%d",
        current_user.id,
        len(payload.text),
    )

    result = extract_knowledge_graph(
        text=payload.text,
        min_frequency=payload.min_frequency,
        max_nodes=payload.max_nodes,
    )

    return KnowledgeGraphResponse(
        total_entities=result["total_entities"],
        total_nodes=result["total_nodes"],
        total_edges=result["total_edges"],
        entity_counts=result["entity_counts"],
        nodes=[GraphNode(**n) for n in result["nodes"]],
        edges=[GraphEdge(**e) for e in result["edges"]],
    )
