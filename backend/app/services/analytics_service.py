from sqlalchemy import func
from app.models.prediction import Prediction
def get_prediction_analytics(
    db,
    user_id
):
    total = (
    db.query(Prediction)
    .filter(
        Prediction.user_id == user_id
    )
    .count()
)
    fake_count = (
    db.query(Prediction)
    .filter(
        Prediction.user_id == user_id,
        Prediction.prediction == "FAKE"
    )
    .count()
)
    real_count = (
    db.query(Prediction)
    .filter(
        Prediction.user_id == user_id,
        Prediction.prediction == "REAL"
    )
    .count()
)
    average_confidence = (
    db.query(
        func.avg(
            Prediction.confidence
        )
    )
    .filter(
        Prediction.user_id == user_id
    )
    .scalar()
)
    return {
    "total_predictions":
        total,

    "fake_predictions":
        fake_count,

    "real_predictions":
        real_count,

    "average_confidence":
        round(
            average_confidence or 0,
            2
        )
}