from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from sqlalchemy.orm import Session

from app.db.dependencies import get_db

from app.core.security import get_current_user

from app.models.user import User
from app.models.prediction import Prediction


router = APIRouter(
    prefix="/admin",
    tags=["Admin"]
)


@router.get("/stats")
def get_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    if current_user.role != "admin":

        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )

    users = db.query(User).count()

    predictions = db.query(
        Prediction
    ).count()

    fake_predictions = (
        db.query(Prediction)
        .filter(
            Prediction.prediction == "FAKE"
        )
        .count()
    )

    real_predictions = (
        db.query(Prediction)
        .filter(
            Prediction.prediction == "REAL"
        )
        .count()
    )

    logistic_count = (
        db.query(Prediction)
        .filter(
            Prediction.model_name == "logistic"
        )
        .count()
    )

    lstm_count = (
        db.query(Prediction)
        .filter(
            Prediction.model_name == "lstm"
        )
        .count()
    )

    bert_count = (
        db.query(Prediction)
        .filter(
            Prediction.model_name == "bert"
        )
        .count()
    )

    return {
        "users": users,
        "predictions": predictions,
        "fake_predictions": fake_predictions,
        "real_predictions": real_predictions,
        "models": {
            "logistic": logistic_count,
            "lstm": lstm_count,
            "bert": bert_count
        }
    }