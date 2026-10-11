# AI-Powered Multimodal Fake News Detection Platform

A production-grade fake-news verification system supporting both **text** and **image** input. Combines NLP, computer vision, OCR, explainable AI, semantic search, knowledge graphs, and source credibility scoring into a single full-stack application.

---

## Features

| Module | Technology |
|--------|------------|
| Text classification | Logistic Regression, LSTM, DistilBERT |
| Model comparison | Side-by-side results for all three models |
| Ensemble prediction | Majority-vote across all three models |
| Image classification | ResNet18 (FAKE / REAL + per-class probabilities) |
| OCR text extraction | EasyOCR — feeds extracted text into the text pipeline |
| Reverse image search | CLIP (ViT-B/32) + FAISS |
| Text evidence retrieval | Sentence Transformers (all-MiniLM-L6-v2) + FAISS |
| SHAP explanations | All 3 text models — word-level feature contributions |
| LIME explanations | All 3 text models — local perturbation-based explanations |
| Grad-CAM heatmap | ResNet18 `layer4[-1]` — visual explanation of image prediction |
| Claim extraction | spaCy NER + sentence segmentation → per-claim verification |
| Knowledge graph | spaCy NER + co-occurrence edges → entity relationship map |
| Propaganda detection | Rule-based, 10 recognised techniques |
| Analytics dashboard | Per-user statistics, monthly trends, keyword frequency |
| Authentication | JWT-based login / registration |
| Prediction history | Full log with model, confidence, and timestamp |

---

## Architecture

```
Browser
  │
  ├── React (Vite) SPA
  │     ├── Dashboard         — text prediction + SHAP
  │     ├── Upload            — image → OCR → text prediction
  │     ├── Image Verify      — full multimodal (OCR + ResNet18 + CLIP + Grad-CAM)
  │     ├── Claims            — claim extraction + per-claim verification
  │     ├── Tools             — Evidence / Credibility / Propaganda / Knowledge Graph
  │     ├── Compare           — side-by-side model comparison
  │     ├── History           — prediction log
  │     └── Analytics         — charts and statistics
  │
  └── Nginx (Docker) → FastAPI (uvicorn)
                          │
                          ├── Text Models (logistic / LSTM / DistilBERT)
                          ├── ResNet18 image classifier
                          ├── OCR (EasyOCR)
                          ├── Grad-CAM (pytorch-grad-cam)
                          ├── SHAP / LIME explainability
                          ├── CLIP reverse image search
                          ├── FAISS — text evidence + image similarity
                          ├── Sentence Transformers
                          ├── Claim extraction (spaCy)
                          ├── Knowledge graph (spaCy)
                          ├── Propaganda detection
                          ├── Source credibility
                          └── PostgreSQL (SQLAlchemy + Alembic)
```

---

## Complete Verification Pipelines

### Text Pipeline

```
User input text
  → POST /predict
  → Preprocessing (TF-IDF / tokenisation)
  → Model inference (logistic / LSTM / DistilBERT)
  → Prediction + confidence
  → SHAP (logistic) or LIME (LSTM / BERT) explanation
  → Keywords (YAKE) + clickbait score
  → Frontend: ResultPage
```

Optional deeper tools (standalone, via ToolsPage):
```
Text → POST /claims/extract     → per-claim predictions
Text → POST /evidence/search    → FAISS semantic evidence
Text → POST /knowledge-graph/extract → entity graph
Text → POST /propaganda/analyse → propaganda technique detection
```

### Image Pipeline

```
Image upload
  → POST /verify/image
  ├── OCR (EasyOCR)
  │     ├── < 20 chars  → skip text pipeline
  │     └── ≥ 20 chars  → text model prediction (logistic / LSTM / BERT)
  ├── ResNet18 classification
  │     └── FAKE / REAL + probabilities + Grad-CAM heatmap
  ├── CLIP → FAISS reverse image search
  └── Combined verdict (text + image + CLIP signals)
     → Frontend: ImageVerifyPage (heatmap displayed inline)
```

OCR-only pipeline (simpler, via Upload page):
```
Image → POST /upload/image → OCR → text prediction → ResultPage
```

### XAI

| Model | Method | What it produces |
|-------|--------|-----------------|
| Logistic Regression | SHAP LinearExplainer | Word → FAKE/REAL contribution |
| Logistic Regression | LIME LimeTextExplainer | Top influential words |
| LSTM | SHAP PartitionExplainer | Word → FAKE/REAL contribution |
| LSTM | LIME LimeTextExplainer | Top influential words |
| DistilBERT | SHAP Explainer | Word → FAKE/REAL contribution |
| DistilBERT | LIME LimeTextExplainer | Top influential words |
| ResNet18 | Grad-CAM (layer4[-1]) | Heatmap PNG at `/static/heatmaps/` |

XAI failures are non-fatal — the prediction is always returned even if explanation fails.

---

## Environment Variables

Copy `backend/.env.example` to `backend/.env`:

```env
DATABASE_URL=postgresql+psycopg2://postgres:CHANGE_ME@db:5432/fakenews
JWT_SECRET_KEY=CHANGE_ME_TO_A_LONG_RANDOM_STRING
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

`DATABASE_URL` must use the Docker service name `db` (not `localhost`) when running via Docker Compose.

---

## Required Model Files

Place these under `backend/app/ml/saved_models/`:

| File | Purpose |
|------|---------|
| `best_image_model.pth` | ResNet18 weights (binary FAKE/REAL) |
| `logistic.pkl` | Logistic Regression model |
| `tfidf.pkl` | TF-IDF vectoriser (used with logistic + SHAP) |
| `lstm.keras` | LSTM model |
| `lstm_tokenizer.pkl` | LSTM tokeniser |
| `distilbert/` | DistilBERT fine-tuned weights directory |

FAISS indexes are **not committed** — they are built automatically at container startup from `backend/datasets/`.

---

## Local Development Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Copy and edit env
cp .env.example .env

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev          # Vite dev server on http://localhost:5173
```

The Vite dev proxy routes `/api/*` → `http://localhost:8000` and `/static/*` → `http://localhost:8000/static` automatically.

---

## Docker Setup

```bash
# Copy env file first
cp backend/.env.example backend/.env
# Edit backend/.env with your values

docker compose up --build
```

- Frontend: http://localhost
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

**Volumes:**

| Volume | Purpose |
|--------|---------|
| `postgres_data` | PostgreSQL data |
| `uploads_data` | Temporary image uploads |
| `faiss_index` | Text FAISS index (built at startup) |
| `clip_index` | CLIP FAISS index (built at startup) |
| `static_data` | Grad-CAM heatmap PNGs |

FAISS indexes are built automatically on first container start if `backend/datasets/` is present. Subsequent starts reuse the persisted volumes.

---

## API Reference

All endpoints (except `/auth/*` and `/health`) require `Authorization: Bearer <token>`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/auth/register` | Register new user |
| `POST` | `/auth/login` | Login, returns JWT |
| `GET` | `/auth/me` | Current user info |
| `POST` | `/predict` | Text fake-news prediction |
| `POST` | `/predict/compare` | Compare all 3 models |
| `POST` | `/predict/ensemble` | Majority-vote ensemble |
| `GET` | `/predict/history` | Prediction history |
| `POST` | `/upload/image` | Image → OCR → text prediction |
| `POST` | `/verify/image` | Full multimodal verification |
| `POST` | `/predict/image` | Standalone ResNet18 classification |
| `POST` | `/claims/extract` | Claim extraction + per-claim verification |
| `POST` | `/evidence/search` | FAISS semantic evidence search |
| `POST` | `/images/reverse-search` | CLIP reverse image search |
| `POST` | `/credibility/check` | Source domain credibility |
| `POST` | `/propaganda/analyse` | Propaganda technique detection |
| `POST` | `/knowledge-graph/extract` | Entity knowledge graph |
| `GET` | `/analytics/overview` | Prediction stats |
| `GET` | `/analytics/models` | Usage per model |
| `GET` | `/analytics/monthly` | Monthly usage chart |
| `GET` | `/analytics/keywords` | Top keywords |
| `GET` | `/analytics/verdicts-over-time` | Monthly FAKE/REAL trend |
| `GET` | `/metrics` | Per-model accuracy from feedback |
| `POST` | `/feedback` | Submit prediction correctness feedback |
| `GET` | `/health` | Health check |

Interactive docs: **http://localhost:8000/docs**
