from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session
from sqlalchemy import func, case

from app.db.dependencies import get_db
from app.models.feedback import Feedback

router = APIRouter(
    prefix="/metrics",
    tags=["Metrics"]
)


@router.get("")
def get_metrics(
    db: Session = Depends(get_db)
):

    results = (
        db.query(
            Feedback.model_name,
            func.count(
                Feedback.id
            ).label("total"),
            func.sum(
                case(
                    (
                        Feedback.is_correct == True,
                        1
                    ),
                    else_=0
                )
            ).label("correct")
        )
        .group_by(
            Feedback.model_name
        )
        .all()
    )

    metrics = {}

    for row in results:

        total = row.total or 0

        correct = row.correct or 0

        accuracy = (
            round(
                (correct / total) * 100,
                2
            )
            if total > 0
            else 0
        )

        metrics[row.model_name] = {
            "total_feedback": total,
            "correct_predictions": correct,
            "accuracy": accuracy
        }

    return metrics