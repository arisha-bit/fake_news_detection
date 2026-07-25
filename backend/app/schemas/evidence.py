"""
Pydantic schemas for Evidence Retrieval (Phase 4).
"""

from pydantic import BaseModel, Field


class EvidenceSearchRequest(BaseModel):
    """Request body for semantic evidence search."""

    query: str = Field(
        ...,
        description="Claim or article text to search evidence for",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of evidence results to return (1–20)",
    )


class EvidenceItem(BaseModel):
    """A single evidence match from the trusted corpus."""

    rank: int = Field(..., description="1-based rank by similarity")
    title: str
    snippet: str = Field(..., description="First 300 chars of article text")
    label: str = Field(..., description="REAL or FAKE label from training data")
    similarity: float = Field(..., description="Cosine similarity score 0.0–1.0")
    subject: str = Field(default="")
    date: str = Field(default="")


class EvidenceSearchResponse(BaseModel):
    """Response containing ranked evidence items."""

    query: str
    total_results: int
    evidence: list[EvidenceItem]
