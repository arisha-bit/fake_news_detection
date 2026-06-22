from pydantic import BaseModel


class EnsembleRequest(BaseModel):
    text: str


class EnsembleResponse(BaseModel):
    prediction: str
    confidence: float
    votes: dict
    