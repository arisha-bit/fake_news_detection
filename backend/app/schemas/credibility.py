"""
Pydantic schemas for Source Credibility Scoring (Phase 6).
"""

from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class CredibilityCheckRequest(BaseModel):
    """Request body — accepts a full URL or bare domain."""

    url: str = Field(
        ...,
        description="Full URL (https://bbc.com/article) or bare domain (bbc.com)",
    )


class CredibilityResponse(BaseModel):
    """
    Full credibility assessment for a news source domain.
    """

    domain: str = Field(..., description="Extracted domain e.g. bbc.com")
    found_in_database: bool = Field(
        ..., description="Whether the domain exists in our trust database"
    )

    # Scores (0–100). None if domain not found.
    trust_score: Optional[float] = Field(
        None, description="Overall trust score 0–100"
    )
    reliability_score: Optional[float] = Field(
        None, description="Factual reporting reliability 0–100"
    )

    # Categorical ratings
    bias_rating: Optional[str] = Field(
        None,
        description=(
            "Political/editorial bias: "
            "LEFT | LEFT-CENTER | CENTER | RIGHT-CENTER | RIGHT | UNKNOWN"
        ),
    )
    category: Optional[str] = Field(
        None,
        description="Source category e.g. Mainstream, Satire, Conspiracy, Government",
    )
    credibility_label: Optional[str] = Field(
        None,
        description="HIGH | MEDIUM | LOW | VERY_LOW",
    )

    # Human-readable summary
    verdict: str = Field(..., description="One-line credibility summary")
    notes: Optional[str] = Field(
        None, description="Additional notes about the source"
    )
