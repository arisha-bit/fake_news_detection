"""
Pydantic schemas for Reverse Image Verification (Phase 5).
"""

from pydantic import BaseModel, Field


class ImageMatchItem(BaseModel):
    """A single match from the reverse image search."""

    rank: int
    title: str = Field(..., description="Title of matched article")
    potential_source: str = Field(
        ..., description="Subject/category as proxy for source"
    )
    label: str = Field(..., description="REAL or FAKE")
    similarity: float = Field(..., description="CLIP cosine similarity 0.0–1.0")
    date: str = Field(default="")
    snippet: str = Field(default="", description="First 200 chars of article text")


class ReverseImageResponse(BaseModel):
    """Response for reverse image search."""

    total_results: int
    possible_reuse_detected: bool = Field(
        ...,
        description="True if any match exceeds the reuse similarity threshold",
    )
    matches: list[ImageMatchItem]
