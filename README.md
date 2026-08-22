# AI-Powered Multimodal News Verification Platform

A production-grade fake news detection system supporting text, image, and multimodal verification using NLP, OCR, Computer Vision, Explainable AI, and Semantic Search.

---

## Features

| Module | Technology |
|--------|-----------|
| Text classification | Logistic Regression, LSTM, DistilBERT |
| Image classification | ResNet18 + Grad-CAM |
| OCR text extraction | EasyOCR |
| Reverse image search | CLIP (ViT-B/32) + FAISS |
| Text evidence retrieval | Sentence Transformers + FAISS |
| Explainability | SHAP (logistic), LIME (LSTM, BERT), Grad-CAM (ResNet18) |
| Keyword extraction | YAKE |
| Clickbait detection | Rule-based |
| Claim extraction | spaCy NER + sentence segmentation |
| Knowledge graph | spaCy NER + co-occurrence edges |
| Propaganda detection | Rule-based, 10 techniques |
| Source credibility | Curated trust database (35+ domains) |
| Verification report | PDF via reportlab |
| Analytics dashboard | Per-user statistics, monthly trends, keyword frequency |

---

## Architecture

```
User
 │
 ├── Text Input
 │     └── Logistic / LSTM / DistilBERT
 │           └── SHAP / LIME / YAKE / Clickbait
 │                 └── Claims → Evidence → Knowledge Graph → Propaganda
 │
 └── Image Upload
       ├── ResNet18 → FAKE/REAL → Grad-CAM
       ├── OCR → (if text found) → Text Pipeline above
       └── CLIP → FAISS → Similar indexed content
```

---

## Quick Start (Docker)

```bash
git clone https://github.com/arisha-bit/fake_news_detection.git
cd fake_news_detection

# Copy and configure environment
cp backend/.env.example backend/.env
# Edit backend/.env with your values

docker compose up --build
```

- Frontend: http://localhost
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

---

## Environment Variables

Copy `backend/.env.example` to `backend/.env`:

```env
DATABASE_URL=postgresql://postgres:CHANGE_ME@db:5432/fakenews
JWT_SECRET_KEY=CHANGE_ME_TO_A_LONG_RANDOM_STRING
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

---

## Required Model Files

Place these in `backend/app/ml/saved_models/`:

| File | Description |
|------|-------------|
| `logistic.pkl` | Logistic Regression classifier |
| `tfidf.pkl` | TF-IDF vectorizer for logistic |
| `lstm.keras` | LSTM model |
| `lstm_tokenizer.pkl` | LSTM tokenizer |
| `distilbert/` | DistilBERT fine-tuned model directory |
| `best_image_model.pth` | ResNet18 image classifier |

---

## FAISS Indexes

Indexes are built automatically on first startup if missing.

| Index | Path | Builder |
|-------|------|---------|
| Text evidence | `app/ml/embeddings/index/` | `python -m app.ml.embeddings.build_index` |
| CLIP image | `app/ml/clip/index/` | `python -m app.ml.clip.build_image_index` |

Both require `datasets/processed/news.csv` to be present.

---

## Local Development (without Docker)

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
alembic upgrade head
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Point `DATABASE_URL` at `localhost` instead of `db` for local dev.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Text fake-news prediction (logistic/lstm/bert) |
| POST | `/predict/compare` | Compare all 3 text models |
| POST | `/predict/ensemble` | Ensemble majority-vote prediction |
| POST | `/upload/image` | OCR image → text prediction |
| POST | `/predict/image` | ResNet18 image classification |
| POST | `/verify/image` | Full multimodal: ResNet18 + OCR + CLIP |
| POST | `/images/reverse-search` | CLIP FAISS reverse image search |
| POST | `/claims/extract` | Claim extraction + per-claim verification |
| POST | `/evidence/search` | Semantic evidence retrieval |
| POST | `/knowledge-graph/extract` | Named entity knowledge graph |
| POST | `/propaganda/analyse` | Propaganda technique detection |
| POST | `/credibility/check` | Source domain credibility scoring |
| POST | `/report/generate` | Download PDF verification report |
| GET | `/analytics/summary` | Full analytics dashboard payload |
| GET | `/analytics/monthly` | Monthly prediction counts |
| GET | `/analytics/keywords` | Top keywords from prediction history |
| GET | `/analytics/verdicts-over-time` | FAKE/REAL trend over time |
| GET | `/health` | Health check |

---

## XAI — Explainability

- **Logistic Regression**: SHAP `LinearExplainer` — word-level FAKE/REAL contributions
- **LSTM**: LIME `LimeTextExplainer` — local token-level explanation
- **DistilBERT**: LIME — token-level explanation for transformer predictions
- **ResNet18**: Grad-CAM — pixel-level heatmap overlay showing which image regions influenced classification

---

## Image Verification Pipeline

```
Uploaded Image
├── ResNet18 → FAKE/REAL + confidence
│              └── Grad-CAM heatmap
├── OCR (EasyOCR)
│     └── if meaningful text found (>20 chars):
│           └── text models + claims + evidence + knowledge graph + propaganda + XAI
└── CLIP (ViT-B/32) → FAISS index
      └── top-k similar articles (similarity evidence only)
```

CLIP results indicate semantic similarity to indexed content. They are not proof of image manipulation or misinformation on their own.

---

## Propaganda Detection Techniques

Fear Appeal, Clickbait, Loaded Language, Conspiracy Framing, Emotional Manipulation, Bandwagon, False Dilemma, Name Calling, Glittering Generalities, Repetition.

---

## Testing

```bash
cd backend
pytest tests/ -v
```

---

## Limitations

- Text models were trained on English-language news corpora.
- ResNet18 was trained on a binary FAKE/REAL image dataset — performance varies by image type.
- CLIP similarity is retrieval evidence, not a direct misinformation classifier.
- FAISS evidence retrieval searches 5,000 indexed articles — extend `MAX_ARTICLES` in `build_index.py` for broader coverage.
- Propaganda detection is rule-based and may miss subtle manipulation.
- Source credibility database covers 35+ major domains — unknown domains return `found_in_database: false`.
