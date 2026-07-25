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


from sqlalchemy import extract, case
import re


def get_monthly_usage(db, user_id) -> list[dict]:
    """
    Returns prediction counts grouped by year-month for the last 12 months.

    Returns:
        [ { "month": "2024-01", "count": 12 }, ... ]
    """
    results = (
        db.query(
            extract("year", Prediction.created_at).label("year"),
            extract("month", Prediction.created_at).label("month"),
            func.count(Prediction.id).label("count"),
        )
        .filter(Prediction.user_id == user_id)
        .group_by("year", "month")
        .order_by("year", "month")
        .all()
    )

    return [
        {
            "month": f"{int(row.year):04d}-{int(row.month):02d}",
            "count": row.count,
        }
        for row in results
    ][-12:]  # last 12 months


def get_confidence_distribution(db, user_id) -> dict:
    """
    Buckets confidence scores into 5 ranges.

    Returns:
        {
            "0-50": 3,
            "50-70": 8,
            "70-85": 22,
            "85-95": 41,
            "95-100": 19
        }
    """
    rows = (
        db.query(Prediction.confidence)
        .filter(Prediction.user_id == user_id)
        .all()
    )

    buckets = {"0-50": 0, "50-70": 0, "70-85": 0, "85-95": 0, "95-100": 0}

    for (conf,) in rows:
        if conf is None:
            continue
        pct = conf * 100
        if pct < 50:
            buckets["0-50"] += 1
        elif pct < 70:
            buckets["50-70"] += 1
        elif pct < 85:
            buckets["70-85"] += 1
        elif pct < 95:
            buckets["85-95"] += 1
        else:
            buckets["95-100"] += 1

    return buckets


def get_top_keywords(db, user_id, top_n: int = 20) -> list[dict]:
    """
    Extracts and counts the most frequent keywords across all prediction texts.
    Uses YAKE for keyword extraction — same library as the rest of the project.

    Returns:
        [ { "keyword": "vaccine mandate", "count": 15 }, ... ]
    """
    import yake  # noqa: PLC0415

    rows = (
        db.query(Prediction.text)
        .filter(
            Prediction.user_id == user_id,
            Prediction.text.isnot(None),
        )
        .limit(500)  # cap to avoid memory issues on large histories
        .all()
    )

    if not rows:
        return []

    extractor = yake.KeywordExtractor(lan="en", n=2, top=5)
    counts: dict[str, int] = {}

    for (text,) in rows:
        if not text:
            continue
        try:
            kws = extractor.extract_keywords(text)
            for kw, _ in kws:
                kw_clean = kw.lower().strip()
                counts[kw_clean] = counts.get(kw_clean, 0) + 1
        except Exception:
            continue

    sorted_kws = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
    return [{"keyword": kw, "count": cnt} for kw, cnt in sorted_kws]


def get_verdicts_over_time(db, user_id) -> list[dict]:
    """
    Monthly breakdown of FAKE vs REAL predictions.

    Returns:
        [
            { "month": "2024-01", "fake_count": 5, "real_count": 12 },
            ...
        ]
    """
    results = (
        db.query(
            extract("year", Prediction.created_at).label("year"),
            extract("month", Prediction.created_at).label("month"),
            func.sum(
                case((Prediction.prediction == "FAKE", 1), else_=0)
            ).label("fake_count"),
            func.sum(
                case((Prediction.prediction == "REAL", 1), else_=0)
            ).label("real_count"),
        )
        .filter(Prediction.user_id == user_id)
        .group_by("year", "month")
        .order_by("year", "month")
        .all()
    )

    return [
        {
            "month": f"{int(row.year):04d}-{int(row.month):02d}",
            "fake_count": int(row.fake_count or 0),
            "real_count": int(row.real_count or 0),
        }
        for row in results
    ][-12:]
