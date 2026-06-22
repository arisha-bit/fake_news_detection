from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from sqlalchemy.orm import Session

from app.db.dependencies import get_db

from app.schemas.feedback import (
    FeedbackRequest,
    FeedbackResponse
)

from app.services.feedback_service import (
    create_feedback
)

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"]
)


@router.post(
    "",
    response_model=FeedbackResponse
)
def submit_feedback(
    payload: FeedbackRequest,
    db: Session = Depends(get_db)
):

    feedback = create_feedback(
        db=db,
        prediction_id=payload.prediction_id,
        is_correct=payload.is_correct
    )

    if feedback is None:

        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    return {
        "message": "Feedback saved successfully"
    }
