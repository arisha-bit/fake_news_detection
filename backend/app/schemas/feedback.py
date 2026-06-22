from uuid import UUID

from pydantic import BaseModel


class FeedbackRequest(BaseModel):

    prediction_id: UUID

    is_correct: bool


class FeedbackResponse(BaseModel):

    message: str