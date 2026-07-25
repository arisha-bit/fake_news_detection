"""
Pydantic schemas for Verification Report Generator (Phase 8).
"""

from typing import Optional
from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    """
    Request body for report generation.
    All sections are optional — only requested sections are included.
    """
    text: str = Field(..., description="Article text to analyse")
    model: str = Field(default="logistic", description="logistic | lstm | bert")
    source_url: Optional[str] = Field(
        default=None,
        description="Optional source URL for credibility scoring"
    )
    include_claims: bool = Field(default=True)
    include_evidence: bool = Field(default=True)
    include_propaganda: bool = Field(default=True)
    include_credibility: bool = Field(default=True)
    top_k_evidence: int = Field(default=3, ge=1, le=10)


class ReportMetadata(BaseModel):
    """Metadata included in every report."""
    generated_at: str
    model_used: str
    text_length: int
    source_url: Optional[str] = None
