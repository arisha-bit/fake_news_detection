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


from sqlalchemy import extract, case
from app.services.analytics_service import (
    get_monthly_usage,
    get_confidence_distribution,
    get_top_keywords,
    get_verdicts_over_time,
)


@router.get("/monthly")
def monthly_usage(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns prediction counts grouped by year-month for the last 12 months.
    Useful for rendering a time-series bar chart on the dashboard.
    """
    return get_monthly_usage(db, current_user.id)


@router.get("/confidence")
def confidence_distribution(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns confidence score distribution bucketed into ranges:
    0-50%, 50-70%, 70-85%, 85-95%, 95-100%.
    Useful for a histogram view.
    """
    return get_confidence_distribution(db, current_user.id)


@router.get("/keywords")
def top_keywords(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns the most frequently appearing keywords across all predictions
    by scanning prediction texts with YAKE.
    Top 20 keywords with their occurrence counts.
    """
    return get_top_keywords(db, current_user.id)


@router.get("/verdicts-over-time")
def verdicts_over_time(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns monthly FAKE vs REAL counts for trend analysis.
    Each entry: { month, fake_count, real_count }
    """
    return get_verdicts_over_time(db, current_user.id)


@router.get("/summary")
def dashboard_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Full dashboard payload — aggregates all analytics in one call.
    Reduces frontend round-trips.
    """
    total = (
        db.query(Prediction)
        .filter(Prediction.user_id == current_user.id)
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
    real_count = total - fake_count

    avg_confidence = (
        db.query(func.avg(Prediction.confidence))
        .filter(Prediction.user_id == current_user.id)
        .scalar()
    ) or 0.0

    model_usage_rows = (
        db.query(Prediction.model_name, func.count(Prediction.id))
        .filter(Prediction.user_id == current_user.id)
        .group_by(Prediction.model_name)
        .all()
    )

    return {
        "overview": {
            "total_predictions": total,
            "fake_predictions": fake_count,
            "real_predictions": real_count,
            "average_confidence": round(float(avg_confidence), 4),
            "fake_ratio": round(fake_count / total, 4) if total > 0 else 0.0,
        },
        "model_usage": {row[0]: row[1] for row in model_usage_rows},
        "monthly": get_monthly_usage(db, current_user.id),
        "confidence_distribution": get_confidence_distribution(db, current_user.id),
        "top_keywords": get_top_keywords(db, current_user.id),
        "verdicts_over_time": get_verdicts_over_time(db, current_user.id),
    }
