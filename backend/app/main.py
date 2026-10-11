import os
# These must be set before torch/transformers/tensorflow are imported anywhere.
# CUDA_VISIBLE_DEVICES=-1 forces CPU-only mode for both torch and tf.
# Transformers 5.x calls torch.accelerator.current_accelerator() during
# from_pretrained(); with no visible CUDA devices this returns cpu cleanly.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import logging
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.logging_config import configure_logging

from app.db.database import engine, Base

from app.models.user import User
from app.models.prediction import Prediction
from app.models.keyword import Keyword
from app.models.model_metric import ModelMetric
from app.models.feedback import Feedback
from app.models.uploaded_file import UploadedFile
from app.models.image_verification import ImageVerification

from app.api.auth import router as auth_router
from app.api.prediction import router as prediction_router
from app.api.analytics import router as analytics_router
from app.api.admin import router as admin_router
from app.api.feedback import router as feedback_router
from app.api.metrics import router as metrics_router
from app.api.upload.routes import router as upload_router
from app.api.claims import router as claims_router
from app.api.evidence import router as evidence_router
from app.api.images import router as images_router
from app.api.propaganda import router as propaganda_router
from app.api.knowledge_graph import router as knowledge_graph_router
from app.api.image_predict import router as image_predict_router
from app.routers.health import router as health_router

Base.metadata.create_all(bind=engine)

configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fake News Detection API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:80",     # Docker frontend
        "http://localhost",        # Docker frontend (no port)
        "http://127.0.0.1",        # Local IP
        "http://127.0.0.1:5173",   # Vite on local IP
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static file serving — Grad-CAM heatmaps and any other generated images
# Heatmaps are saved to static/heatmaps/ and served at /static/heatmaps/<file>
# ---------------------------------------------------------------------------
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "heatmaps").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(auth_router)
app.include_router(prediction_router)
app.include_router(analytics_router)
app.include_router(admin_router)
app.include_router(feedback_router)
app.include_router(metrics_router)
app.include_router(upload_router)
app.include_router(claims_router)
app.include_router(evidence_router)
app.include_router(images_router)
app.include_router(propaganda_router)
app.include_router(knowledge_graph_router)
app.include_router(image_predict_router)
app.include_router(health_router)


# ---------------------------------------------------------------------------
# Startup: pre-warm the logistic regression model + SHAP explainer.
# This is the primary model and its SHAP explainer are cheap to load (~1s)
# and used on every text prediction — worth pre-loading to avoid a slow
# first request.
#
# ResNet18, FAISS, CLIP, and Sentence Transformer are kept lazy because:
#   - They are large (100 MB+) and not always needed (depends on user action)
#   - They will be loaded on first use and cached for the process lifetime
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _warm_logistic_model() -> None:
    """Pre-warm the logistic + SHAP components at startup."""
    t0 = time.perf_counter()
    try:
        from app.ml.inference.predict import predict_news
        predict_news("startup warmup text")
        logger.info("Logistic model warmed up in %.2fs", time.perf_counter() - t0)
    except Exception as exc:
        logger.warning("Logistic model warmup failed (non-fatal): %s", exc)

    try:
        from app.ml.explainability.shap_explainer import _get_components
        _get_components()
        logger.info("SHAP explainer warmed up in %.2fs", time.perf_counter() - t0)
    except Exception as exc:
        logger.warning("SHAP explainer warmup failed (non-fatal): %s", exc)
