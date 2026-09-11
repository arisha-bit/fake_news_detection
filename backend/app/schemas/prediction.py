from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from uuid import UUID


class SHAPFeature(BaseModel):
    word: str
    impact: float
    direction: str  # "REAL" or "FAKE"


class SHAPExplanation(BaseModel):
    fake_features: list[str]
    real_features: list[str]
    top_features: list[SHAPFeature]


class XAIResult(BaseModel):
    """
    XAI explanation returned alongside prediction.
    Only populated when model='logistic' (SHAP explains Logistic Regression only).
    LSTM and DistilBERT return xai=None.
    """
    shap: Optional[SHAPExplanation] = None
    keywords: list[str] = []
    clickbait: Optional[dict] = None
    explanation: str = ""
    model_explained: str = "logistic"


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
