from sqlalchemy.orm import Session

from app.models.prediction import Prediction


def save_prediction(
    db: Session,
    user_id,
    text: str,
    prediction: str,
    confidence: float,
    model_name: str
):

    prediction_record = Prediction(
        user_id=user_id,
        text=text,
        prediction=prediction,
        confidence=confidence,
        model_name=model_name
    )

    db.add(prediction_record)

    db.commit()

    db.refresh(prediction_record)

    return prediction_record