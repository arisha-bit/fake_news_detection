from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from uuid import UUID


class SHAPFeature(BaseModel):
    word: str
    impact: float
    direction: str  # "REAL" or "FAKE"
    weight: Optional[float] = None  # Signed contribution in explainer's output space


class SHAPExplanation(BaseModel):
    fake_features: list[str]
    real_features: list[str]
    top_features: list[SHAPFeature]


class XAIResult(BaseModel):
    shap: Optional[SHAPExplanation] = None
    lime: Optional[SHAPExplanation] = None
    keywords: list[str] = []
    clickbait: Optional[dict] = None
    explanation: str = ""
    model_explained: str = "logistic"
    method_explained: str = "shap"


class ExplanationRequest(BaseModel):
    text: str
    model: str = "logistic"  # "logistic" | "lstm" | "bert"
    method: str = "shap"     # "shap" | "lime"
    top_k: int = 10


class ExplanationResponse(BaseModel):
    model: str
    method: str
    prediction: str
    confidence: float
    output_space: Optional[str] = None
    explanations: list[SHAPFeature]
    fake_features: list[str] = []
    real_features: list[str] = []



class PredictionRequest(BaseModel):
    text: str
    model: str = "logistic"


class PredictionResponse(BaseModel):
    prediction_id: UUID
    prediction: str
    confidence: float
    keywords: list[str]
    clickbait_score: int
    explanation: str
    xai: Optional[XAIResult] = None  # populated for logistic model


class PredictionHistoryResponse(BaseModel):
    id: UUID
    text: str
    prediction: str
    confidence: float
    model_name: str
    created_at: datetime

    class Config:
        from_attributes = True
