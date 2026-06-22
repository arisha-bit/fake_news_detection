from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.dependencies import get_db
from app.core.security import get_current_user

from app.models.user import User
from app.models.prediction import Prediction

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"]
)


@router.get("/overview")
def analytics_overview(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    total = (
        db.query(Prediction)
        .filter(
            Prediction.user_id == current_user.id
        )
        .count()
    )

    fake_count = (
        db.query(Prediction)
        .filter(
            Prediction.user_id == current_user.id,
            Prediction.prediction == "FAKE"
        )
        .count()
    )

    real_count = (
        db.query(Prediction)
        .filter(
            Prediction.user_id == current_user.id,
            Prediction.prediction == "REAL"
        )
        .count()
    )

    return {
        "total_predictions": total,
        "fake_predictions": fake_count,
        "real_predictions": real_count
    }
@router.get("/models")
def model_usage(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    results = (
        db.query(
            Prediction.model_name,
            func.count(Prediction.id)
        )
        .filter(
            Prediction.user_id == current_user.id
        )
        .group_by(
            Prediction.model_name
        )
        .all()
    )

    return {
        row[0]: row[1]
        for row in results
    }
