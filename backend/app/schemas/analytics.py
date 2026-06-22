from pydantic import BaseModel


class AnalyticsResponse(BaseModel):

    total_predictions: int

    fake_predictions: int

    real_predictions: int

    average_confidence: float
