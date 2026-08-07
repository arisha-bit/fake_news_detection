# fake_news_detection
AI-powered fake news detection platform built with FastAPI, PostgreSQL, and Docker. Integrates Logistic Regression, LSTM, and DistilBERT models with ensemble learning, explainability, analytics, prediction history, feedback collection, and model performance monitoring.

## v2.0 — Image Upload & OCR Feature

Users can now upload an image (JPG, JPEG, PNG) containing news text. The backend extracts text via OCR and runs it through the existing fake-news detection pipeline.

### API Endpoint

```
POST /upload/image
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | Image file (JPG/JPEG/PNG, max 10 MB) |
| `model` | string | `logistic` (default), `lstm`, or `bert` |

### Example Request

```bash
curl -X POST http://localhost:8000/upload/image \
  -H "Authorization: Bearer <token>" \
  -F "file=@article.jpg" \
  -F "model=logistic"
```

### Example Response

```json
{
  "prediction": "FAKE",
  "confidence": 0.91,
  "extracted_text": "Scientists say miracle cure discovered overnight...",
  "keywords": ["miracle cure", "scientists say"],
  "clickbait_score": 40,
  "explanation": "The article contains sensational language...",
  "prediction_id": "uuid-...",
  "uploaded_file_id": "uuid-..."
}
```

### OCR Dependencies

EasyOCR is used for text extraction. It is CPU-only (no CUDA required).

```
easyocr
Pillow
python-multipart
```

### Installation (local dev)

```bash
cd backend
pip install easyocr Pillow
```

Docker: dependencies are included in `requirements.txt` and installed automatically on build.

### Running Tests

```bash
cd backend
pytest tests/test_upload.py -v
```

## v2.0 — PDF Verification

Upload a PDF news article for text extraction and fake-news detection.

### API Endpoint

```
POST /upload/pdf
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | PDF file (max 20 MB) |
| `model` | string | `logistic` (default), `lstm`, or `bert` |

### Example Request

```bash
curl -X POST http://localhost:8000/upload/pdf \
  -H "Authorization: Bearer <token>" \
  -F "file=@article.pdf" \
  -F "model=logistic"
```

### Example Response

```json
{
  "prediction": "FAKE",
  "confidence": 0.87,
  "extracted_text": "Miracle cure discovered by local doctor...",
  "keywords": ["miracle cure", "local doctor"],
  "clickbait_score": 60,
  "explanation": "The article contains sensational language...",
  "prediction_id": "uuid-...",
  "uploaded_file_id": "uuid-..."
}
```

### PDF Dependencies

PyMuPDF is used for text extraction.

```
pymupdf
```

## v2.0 — Claim Extraction & Verification

Splits an article into individual factual claims and verifies each one independently using the existing prediction pipeline.

### API Endpoint

```
POST /claims/extract
Content-Type: application/json
Authorization: Bearer <token>
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Full article text |
| `model` | string | `logistic` (default), `lstm`, or `bert` |

### Example Request

```bash
curl -X POST http://localhost:8000/claims/extract \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA confirmed water on Mars. Government declares holiday.", "model": "logistic"}'
```

### Example Response

```json
{
  "total_claims": 2,
  "fake_claims": 1,
  "real_claims": 1,
  "overall_verdict": "FAKE",
  "overall_confidence": 0.87,
  "claims": [
    { "claim_index": 0, "text": "NASA confirmed water on Mars.", "prediction": "REAL", "confidence": 0.91 },
    { "claim_index": 1, "text": "Government declares holiday.", "prediction": "FAKE", "confidence": 0.83 }
  ]
}
```

## v2.0 — Evidence Retrieval

Semantic search over the trusted news corpus. Returns top-k similar articles with similarity scores, labels, and snippets.

### Build the Index (once)

```bash
cd backend
python -m app.ml.embeddings.build_index
```

In Docker this runs automatically at image build time.

### API Endpoint

```
POST /evidence/search
Content-Type: application/json
Authorization: Bearer <token>
```

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Claim or article text |
| `top_k` | int | Results to return (default 5, max 20) |

### Example Request

```bash
curl -X POST http://localhost:8000/evidence/search \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "NASA confirmed water was found on Mars.", "top_k": 3}'
```

### Example Response

```json
{
  "query": "NASA confirmed water was found on Mars.",
  "total_results": 3,
  "evidence": [
    {
      "rank": 1,
      "title": "NASA water discovery announced",
      "snippet": "Scientists at NASA have confirmed...",
      "label": "REAL",
      "similarity": 0.91,
      "subject": "science",
      "date": "2020-09-28"
    }
  ]
}
```

### Dependencies

```
sentence-transformers
faiss-cpu
```

## v2.0 — Reverse Image Verification

Upload an image to find semantically similar known news articles using OpenAI CLIP embeddings + FAISS.

### Build the Index (once)

```bash
cd backend
python -m app.ml.clip.build_image_index
```

In Docker this runs automatically at image build time.

### API Endpoint

```
POST /images/reverse-search
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | Image (JPG/JPEG/PNG, max 10 MB) |
| `top_k` | query param | Results to return (default 5, max 20) |

### Example Request

```bash
curl -X POST "http://localhost:8000/images/reverse-search?top_k=5" \
  -H "Authorization: Bearer <token>" \
  -F "file=@screenshot.jpg"
```

### Example Response

```json
{
  "total_results": 3,
  "possible_reuse_detected": true,
  "matches": [
    {
      "rank": 1,
      "title": "Flood devastates coastal town",
      "potential_source": "news",
      "label": "REAL",
      "similarity": 0.91,
      "date": "2020-08-15",
      "snippet": "Thousands displaced after..."
    }
  ]
}
```

### How It Works

CLIP encodes images and text in the same 512-dim embedding space. Uploading an image returns articles whose titles are semantically close to the image content. A `possible_reuse_detected: true` flag is set when any match exceeds a 0.75 similarity threshold.

### Dependencies

```
transformers (CLIPModel, CLIPProcessor)
faiss-cpu
Pillow
```

## v2.0 — Source Credibility Scoring

Score any news source domain against a curated trust database with bias ratings and reliability scores.

### API Endpoint

```
POST /credibility/check
Content-Type: application/json
Authorization: Bearer <token>
```

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Full URL or bare domain (e.g. `bbc.com`) |

### Example Request

```bash
curl -X POST http://localhost:8000/credibility/check \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.reuters.com/article/123"}'
```

### Example Response

```json
{
  "domain": "reuters.com",
  "found_in_database": true,
  "trust_score": 95.0,
  "reliability_score": 94.0,
  "bias_rating": "CENTER",
  "category": "Mainstream",
  "credibility_label": "HIGH",
  "verdict": "'reuters.com' is a generally reliable source (Mainstream, bias: CENTER).",
  "notes": "International news agency. Highly factual reporting."
}
```

### Credibility Labels

| Label | Trust Score Range |
|-------|------------------|
| HIGH | 70–100 |
| MEDIUM | 45–69 |
| LOW | 20–44 |
| VERY_LOW | 0–19 |

Unknown domains return `found_in_database: false` with a caution verdict. The trust database (`app/data/trust_database.json`) covers 35+ major global sources and is easily extensible.

## v2.0 — Propaganda Detection

Detects manipulation techniques in news articles including fear appeals, clickbait, loaded language, conspiracy framing, and more.

### API Endpoint

```
POST /propaganda/analyse
Content-Type: application/json
Authorization: Bearer <token>
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Article or claim text to analyse |

### Example Request

```bash
curl -X POST http://localhost:8000/propaganda/analyse \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"text": "You wont believe this shocking cover-up! They dont want you to know the catastrophic truth!"}'
```

### Example Response

```json
{
  "propaganda_detected": true,
  "overall_score": 0.82,
  "techniques_found": [
    {
      "technique": "Clickbait",
      "confidence": 0.9,
      "matched_phrases": ["you wont believe", "shocking"],
      "description": "Uses sensational language to bait readers."
    },
    {
      "technique": "Conspiracy Framing",
      "confidence": 0.75,
      "matched_phrases": ["they dont want you to know", "cover-up"],
      "description": "Implies hidden plots or suppressed information."
    }
  ],
  "summary": "High propaganda signals detected. Techniques identified: Clickbait, Conspiracy Framing."
}
```

### Detected Techniques

| Technique | Description |
|-----------|-------------|
| Fear Appeal | Threatening language to provoke fear |
| Clickbait | Sensational hooks to bait clicks |
| Loaded Language | Emotionally charged words |
| Conspiracy Framing | Hidden agenda / cover-up implications |
| Emotional Manipulation | Direct appeals to anger or disgust |
| Bandwagon | "Everyone believes" appeals |
| False Dilemma | Binary either/or framing |
| Name Calling | Personal attacks instead of arguments |
| Glittering Generalities | Vague positive buzzwords |
| Repetition | Key phrase repeated for emphasis |

## v2.0 — Verification Report Generator

Generate a downloadable PDF report combining all analysis results: prediction, claims, evidence, propaganda detection, and source credibility.

### API Endpoint

```
POST /report/generate
Content-Type: application/json
Authorization: Bearer <token>
Returns: application/pdf (file download)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Article text to analyse |
| `model` | string | `logistic` | Prediction model |
| `source_url` | string | null | Enables credibility section |
| `include_claims` | bool | true | Include claim analysis |
| `include_evidence` | bool | true | Include evidence section |
| `include_propaganda` | bool | true | Include propaganda section |
| `include_credibility` | bool | true | Include credibility section |
| `top_k_evidence` | int | 3 | Evidence items to show |

### Example Request

```bash
curl -X POST http://localhost:8000/report/generate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "NASA confirmed water on Mars...",
    "model": "logistic",
    "source_url": "https://reuters.com/article/123"
  }' \
  --output report.pdf
```

### Report Sections

1. Fake News Prediction (verdict + confidence + keywords)
2. Claim-Level Analysis (per-claim verdicts)
3. Similar Evidence Found (top-k semantic matches)
4. Propaganda Analysis (technique scores)
5. Source Credibility (trust score + bias rating)

### Dependencies

```
reportlab
```

## v2.0 — Enhanced Analytics Dashboard

Comprehensive analytics endpoints for dashboard visualisation.

### New Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /analytics/monthly` | Prediction counts per month (last 12 months) |
| `GET /analytics/confidence` | Confidence score distribution in 5 buckets |
| `GET /analytics/keywords` | Top 20 most frequent keywords |
| `GET /analytics/verdicts-over-time` | Monthly FAKE vs REAL trend |
| `GET /analytics/summary` | All analytics in one request |

### Example — Dashboard Summary

```bash
curl http://localhost:8000/analytics/summary \
  -H "Authorization: Bearer <token>"
```

```json
{
  "overview": {
    "total_predictions": 142,
    "fake_predictions": 61,
    "real_predictions": 81,
    "average_confidence": 0.8742,
    "fake_ratio": 0.4296
  },
  "model_usage": { "logistic": 98, "lstm": 30, "bert": 14 },
  "monthly": [
    { "month": "2024-06", "count": 22 },
    { "month": "2024-07", "count": 31 }
  ],
  "confidence_distribution": {
    "0-50": 3, "50-70": 8, "70-85": 19, "85-95": 74, "95-100": 38
  },
  "top_keywords": [
    { "keyword": "fake news", "count": 18 },
    { "keyword": "climate change", "count": 11 }
  ],
  "verdicts_over_time": [
    { "month": "2024-06", "fake_count": 9, "real_count": 13 }
  ]
}
```

## v2.0 — Knowledge Graph Extraction

Extracts named entities and their relationships from article text and returns a structured graph for interactive visualisation.

### API Endpoint

```
POST /knowledge-graph/extract
Content-Type: application/json
Authorization: Bearer <token>
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Article text |
| `min_frequency` | int | 1 | Min occurrences to include a node |
| `max_nodes` | int | 50 | Max nodes returned (cap 200) |

### Example Request

```bash
curl -X POST http://localhost:8000/knowledge-graph/extract \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"text": "Elon Musk announced a NASA partnership. Biden praised the deal."}'
```

### Example Response

```json
{
  "total_entities": 6,
  "total_nodes": 3,
  "total_edges": 2,
  "entity_counts": { "PERSON": 2, "ORG": 1 },
  "nodes": [
    { "id": "elon musk", "label": "Elon Musk", "type": "PERSON", "frequency": 2 },
    { "id": "nasa", "label": "NASA", "type": "ORG", "frequency": 2 },
    { "id": "biden", "label": "Biden", "type": "PERSON", "frequency": 1 }
  ],
  "edges": [
    { "source": "elon musk", "target": "nasa", "weight": 2 },
    { "source": "elon musk", "target": "biden", "weight": 1 }
  ]
}
```

### Frontend Integration

The nodes/edges format is directly compatible with:
- [React Force Graph](https://github.com/vasturiano/react-force-graph)
- [D3.js force simulation](https://d3js.org)
- [Vis.js Network](https://visjs.github.io/vis-network/)

Node `type` field can be used to colour-code nodes by entity category.
Edge `weight` can control link thickness/opacity.

---

## Complete API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register user |
| POST | `/auth/login` | Login + get JWT |
| GET | `/auth/me` | Current user info |
| POST | `/predict` | Fake news prediction |
| POST | `/predict/compare` | Compare all 3 models |
| POST | `/predict/ensemble` | Ensemble prediction |
| GET | `/predict/history` | Prediction history |
| POST | `/upload/image` | OCR image → prediction |
| POST | `/upload/pdf` | PDF → prediction |
| POST | `/claims/extract` | Per-claim verification |
| POST | `/evidence/search` | Semantic evidence search |
| POST | `/images/reverse-search` | CLIP reverse image search |
| POST | `/credibility/check` | Source credibility score |
| POST | `/propaganda/analyse` | Propaganda detection |
| POST | `/report/generate` | Download PDF report |
| GET | `/analytics/overview` | Prediction overview |
| GET | `/analytics/models` | Model usage stats |
| GET | `/analytics/monthly` | Monthly usage trend |
| GET | `/analytics/confidence` | Confidence distribution |
| GET | `/analytics/keywords` | Top keywords |
| GET | `/analytics/verdicts-over-time` | Fake/Real trend |
| GET | `/analytics/summary` | Full dashboard payload |
| POST | `/knowledge-graph/extract` | Entity relationship graph |
| GET | `/feedback` | Submit prediction feedback |
| GET | `/metrics` | Model accuracy metrics |
| GET | `/admin/stats` | Admin statistics |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, SQLAlchemy, PostgreSQL, Alembic |
| Auth | JWT, bcrypt, RBAC |
| ML Models | Logistic Regression, LSTM (TensorFlow), DistilBERT (HuggingFace) |
| OCR | EasyOCR |
| PDF | PyMuPDF, ReportLab |
| NLP | spaCy (NER, sentence splitting), YAKE (keywords) |
| Semantic Search | Sentence Transformers, FAISS |
| Image Search | OpenAI CLIP, FAISS |
| Infrastructure | Docker, Docker Compose, Nginx |
| Frontend | React, Tailwind CSS, Vite |

## Docker Setup

```bash
# Build and start all services
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:80
# API docs: http://localhost:8000/docs
```

## Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## v3.0 — Multimodal Image Verification Platform

### Architecture

```
Uploaded Image
      ↓
  ┌───────────────────────────────────┐
  │  Branch 1: OCR → Text Prediction │  EasyOCR + DistilBERT/Logistic/LSTM
  │  Branch 2: Image Classification  │  EfficientNetB0 (trained)
  │  Branch 3: Reverse Image Search  │  CLIP + FAISS
  └───────────────────────────────────┘
                    ↓
         Combined Verdict Engine
                    ↓
       LIKELY FAKE | LIKELY REAL |
     LIKELY MISLEADING | UNCERTAIN
```

### New API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/image` | Standalone image classification (EfficientNetB0) |
| POST | `/verify/image` | Full multimodal verification report |

### Training the Image Classifier

1. Prepare dataset with `fake/` and `real/` subfolders:
```
datasets/image_dataset/
├── fake/   ← manipulated/fake news images
└── real/   ← authentic news images
```

2. Run training:
```bash
cd backend
python -m app.ml.image_classifier.train \
    --data_dir datasets/image_dataset \
    --epochs 10 \
    --batch_size 32
```

Or use the notebook: `notebooks/06_image_classifier.ipynb`

Model saved to: `app/ml/saved_models/image_classifier.pt`

### POST /verify/image — Example Response

```json
{
  "ocr_text": "Shocking discovery: scientists confirm...",
  "text_prediction": "FAKE",
  "text_confidence": 0.88,
  "image_prediction": "FAKE",
  "image_confidence": 0.91,
  "image_class_probabilities": { "FAKE": 0.91, "REAL": 0.09 },
  "similar_articles": [
    {
      "rank": 1,
      "title": "Moon conspiracy theory debunked",
      "label": "FAKE",
      "similarity": 0.88,
      "date": "2023-01-01"
    }
  ],
  "clip_reuse_detected": true,
  "overall_verdict": "LIKELY FAKE",
  "reasoning": [
    "Both OCR text analysis and image classification detected fake signals.",
    "Image appears in known misleading news contexts."
  ]
}
```

### Graceful Degradation

If the image classifier model hasn't been trained yet, `/verify/image` still works — it returns `image_prediction: "UNAVAILABLE"` and computes the verdict from OCR text and CLIP results only. The endpoint never crashes due to a missing model.
