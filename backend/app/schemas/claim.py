"""
Pydantic schemas for the Claim Extraction feature (Phase 3).
"""

from pydantic import BaseModel, Field


class ClaimResult(BaseModel):
    """
    A single extracted factual claim with its individual prediction.
    """
    claim_index: int = Field(..., description="0-based position in the article")
    text: str = Field(..., description="The extracted claim sentence")
    prediction: str = Field(..., description="REAL or FAKE")
    confidence: float = Field(..., description="Model confidence 0.0–1.0")


class ClaimExtractionRequest(BaseModel):
    """
    Request body for claim extraction.
    """
    text: str = Field(..., description="Full article text to extract claims from")
    model: str = Field(
        default="logistic",
        description="Inference model: logistic | lstm | bert"
    )


class ClaimExtractionResponse(BaseModel):
    """
    Response containing all extracted claims and an overall verdict.
    """
    total_claims: int
    fake_claims: int
    real_claims: int
    overall_verdict: str = Field(
        ..., description="Majority-vote verdict: REAL or FAKE"
    )
    overall_confidence: float = Field(
        ..., description="Average confidence across all claims"
    )
    claims: list[ClaimResult]
