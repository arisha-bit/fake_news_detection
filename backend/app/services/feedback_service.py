from app.models.feedback import Feedback
from app.models.prediction import Prediction


def create_feedback(
    db,
    prediction_id,
    is_correct
):

    prediction = (
        db.query(Prediction)
        .filter(
            Prediction.id == prediction_id
        )
        .first()
    )

    if not prediction:
        return None

    feedback = Feedback(
        prediction_id=prediction.id,
        model_name=prediction.model_name,
        is_correct=is_correct
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return feedback
