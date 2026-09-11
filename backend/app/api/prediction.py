"""
Prediction API.

XAI (SHAP) is only supported for the Logistic Regression model.
LSTM and DistilBERT return xai=None — SHAP does not explain them.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.dependencies import get_db
from app.models.user import User
from app.models.prediction import Prediction
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionHistoryResponse,
    XAIResult,
    SHAPExplanation,
)
from app.services.prediction_service import save_prediction
from app.services.explainability_service import (
    extract_keywords,
    clickbait_score,
    generate_explanation,
)
from app.schemas.comparison import CompareRequest, CompareResponse
from app.schemas.ensemble import EnsembleRequest, EnsembleResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])


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
                detail="Invalid model. Use logistic, lstm or bert.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model inference failed for '%s': %s", model_choice, e)
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed for '{model_choice}': {str(e)}",
        )


def _run_shap(text: str, model_choice: str) -> XAIResult | None:
    """
    Run SHAP explanation for Logistic Regression only.
    Returns None for LSTM and BERT — SHAP is not applicable.
    Catches all errors silently so XAI failures never break prediction.
    """
    if model_choice != "logistic":
        return None

    try:
        from app.ml.explainability.shap_explainer import explain_prediction

        result = explain_prediction(text, top_k=10)
        explanations = result.get("explanations", [])

        fake_feats = [e["word"] for e in explanations if e["direction"] == "FAKE"]
        real_feats = [e["word"] for e in explanations if e["direction"] == "REAL"]

        # keywords and clickbait
        kws = extract_keywords(text)
        cb = clickbait_score(text)
        exp_text = generate_explanation(result["prediction"], cb)

        return XAIResult(
            shap=SHAPExplanation(
                fake_features=fake_feats,
                real_features=real_feats,
                top_features=explanations,
            ),
            keywords=kws,
            clickbait={"score": cb},
            explanation=exp_text,
            model_explained="logistic",
        )
    except Exception as exc:
        # Never let XAI crash the prediction
        logger.warning("SHAP explanation failed (non-fatal): %s", exc)
        return None


@router.post("", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    model_choice = payload.model.lower()
    result = _run_model(model_choice, payload.text)

    keywords = extract_keywords(payload.text)
    score = clickbait_score(payload.text)
    explanation = generate_explanation(result["prediction"], score)

    # Run SHAP for logistic; returns None for lstm/bert
    xai = _run_shap(payload.text, model_choice)

    saved = save_prediction(
        db=db,
        user_id=current_user.id,
        text=payload.text,
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_name=model_choice,
    )

    logger.info(
        "Prediction — user=%s, model=%s, result=%s, conf=%.4f, xai=%s",
        current_user.id,
        model_choice,
        result["prediction"],
        result["confidence"],
        "yes" if xai else "no",
    )

    return PredictionResponse(
        prediction_id=saved.id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        keywords=keywords,
        clickbait_score=score,
        explanation=explanation,
        xai=xai,
    )


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
    current_user: User = Depends(get_current_user),
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
        4,
    )

    return {
        "prediction": final_prediction,
        "confidence": avg_confidence,
        "votes": {
            "logistic": logistic,
            "lstm": lstm,
            "bert": bert,
        },
    }
