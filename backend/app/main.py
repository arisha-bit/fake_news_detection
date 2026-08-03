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
from app.models.uploaded_file import UploadedFile

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
from app.api.credibility import router as credibility_router
from app.api.propaganda import router as propaganda_router
from app.api.report import router as report_router
from app.api.knowledge_graph import router as knowledge_graph_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Fake News Detection API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:80",     # Docker frontend
        "http://localhost",        # Docker frontend (no port)
    ],
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
app.include_router(upload_router)
app.include_router(claims_router)
app.include_router(evidence_router)
app.include_router(images_router)
app.include_router(credibility_router)
app.include_router(propaganda_router)
app.include_router(report_router)
app.include_router(knowledge_graph_router)