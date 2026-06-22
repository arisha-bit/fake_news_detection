import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import engine, Base

from app.models.user import User
from app.models.prediction import Prediction
from app.models.keyword import Keyword
from app.models.model_metric import ModelMetric
from app.models.feedback import Feedback

from app.api.auth import router as auth_router
from app.api.prediction import router as prediction_router
from app.api.analytics import router as analytics_router
from app.api.admin import router as admin_router
from app.api.feedback import router as feedback_router
from app.api.metrics import router as metrics_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Fake News Detection API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(prediction_router)
app.include_router(analytics_router)
app.include_router(admin_router)
app.include_router(feedback_router)
app.include_router(metrics_router)