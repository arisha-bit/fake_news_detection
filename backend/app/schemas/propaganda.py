"""
Pydantic schemas for Propaganda Detection (Phase 7).
"""

from pydantic import BaseModel, Field


class PropagandaTechnique(BaseModel):
    """A single detected propaganda technique with evidence."""

    technique: str = Field(..., description="Name of the technique")
    confidence: float = Field(..., description="Detection confidence 0.0–1.0")
    matched_phrases: list[str] = Field(
        default_factory=list,
        description="Phrases that triggered this detection",
    )
    description: str = Field(
        ..., description="Plain-English explanation of this technique"
    )


class PropagandaRequest(BaseModel):
    text: str = Field(..., description="Article or claim text to analyse")


class PropagandaResponse(BaseModel):
    """Full propaganda analysis result."""

    propaganda_detected: bool = Field(
        ..., description="True if any technique was detected"
    )
    overall_score: float = Field(
        ..., description="Aggregate propaganda score 0.0–1.0"
    )
    techniques_found: list[PropagandaTechnique]
    summary: str = Field(..., description="One-line human-readable summary")
