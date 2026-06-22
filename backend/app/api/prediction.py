from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.dependencies import get_db

from app.models.user import User
from app.models.prediction import Prediction

from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionHistoryResponse
)

from app.services.prediction_service import save_prediction

from app.services.explainability_service import (
    extract_keywords,
    clickbait_score,
    generate_explanation
)
from app.schemas.comparison import (
    CompareRequest,
    CompareResponse
)
from app.schemas.ensemble import (
    EnsembleRequest,
    EnsembleResponse
)


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


def _run_model(model_choice: str, text: str) -> dict:
    try:
        if model_choice == "logistic":
            from app.ml.inference.predict import predict_news
            return predict_news(text)

        elif model_choice == "lstm":
            from app.ml.inference.predict_lstm import predict_news_lstm
            return predict_news_lstm(text)

        elif model_choice == "bert":
            from app.ml.inference.predict_bert import predict_news_bert
            return predict_news_bert(text)

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model. Use logistic, lstm or bert."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed for '{model_choice}': {str(e)}"
        )


@router.post("", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    model_choice = payload.model.lower()
    result = _run_model(model_choice, payload.text)

    keywords = extract_keywords(payload.text)
    score = clickbait_score(payload.text)
    explanation = generate_explanation(result["prediction"], score)

    saved = save_prediction(
        db=db,
        user_id=current_user.id,
        text=payload.text,
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_name=model_choice
    )

    return {
        "prediction_id": str(saved.id),
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "keywords": keywords,
        "clickbait_score": score,
        "explanation": explanation
    }


@router.post("/compare", response_model=CompareResponse)
def compare_models(payload: CompareRequest):
    return {
        "logistic": _run_model("logistic", payload.text),
        "lstm": _run_model("lstm", payload.text),
        "bert": _run_model("bert", payload.text),
    }


@router.get("/history", response_model=list[PredictionHistoryResponse])
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return (
        db.query(Prediction)
        .filter(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )


@router.post("/ensemble", response_model=EnsembleResponse)
def ensemble_prediction(payload: EnsembleRequest):
    logistic = _run_model("logistic", payload.text)
    lstm = _run_model("lstm", payload.text)
    bert = _run_model("bert", payload.text)

    votes = [logistic["prediction"], lstm["prediction"], bert["prediction"]]
    final_prediction = "FAKE" if votes.count("FAKE") > votes.count("REAL") else "REAL"
    avg_confidence = round(
        (logistic["confidence"] + lstm["confidence"] + bert["confidence"]) / 3,
        4
    )

    return {
        "prediction": final_prediction,
        "confidence": avg_confidence,
        "votes": {
            "logistic": logistic,
            "lstm": lstm,
            "bert": bert
        }
    }
