from pydantic import BaseModel

class CompareRequest(BaseModel):
    text: str

class ModelResult(BaseModel):
    prediction: str
    confidence: float

class CompareResponse(BaseModel):
    logistic: ModelResult
    lstm: ModelResult
    bert: ModelResult