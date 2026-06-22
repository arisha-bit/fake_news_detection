from datetime import datetime
from pydantic import BaseModel
from uuid import UUID


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


class PredictionHistoryResponse(BaseModel):

    id: UUID

    text: str

    prediction: str

    confidence: float

    model_name: str

    created_at: datetime

    class Config:
        from_attributes = True