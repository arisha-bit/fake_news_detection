# DistilBERT Diagnostic & Testing Guide

## Quick Start

### Verify DistilBERT is Working

Run the diagnostic script inside the Docker container:

```bash
docker exec fake-news-detector-backend python test_distilbert_diagnostic.py
```

This will run 6 tests and report:
1. Import verification
2. Model file existence
3. Model loading
4. Inference on sample texts
5. Prediction API integration
6. Ensemble prediction

Expected output:
```
============================================================
DISTILBERT DIAGNOSTIC REPORT
============================================================
✓ DistilBERT is fully operational!
```

---

## Testing

### Test Suite
Comprehensive tests are available in `tests/test_distilbert.py`

Tests are organized into:
- **TestDistilBERTLoad** — Model loading
- **TestDistilBERTPrediction** — Inference
- **TestPredictionAPIWithBERT** — API integration
- **TestBERTExplanability** — LIME/SHAP
- **TestBERTErrorHandling** — Error cases
- **TestImageVerificationWithBERT** — Image OCR path
- **TestBERTRegression** — Regression tests

Run tests (in container):
```bash
docker exec fake-news-detector-backend python -m pytest tests/test_distilbert.py -v
```

---

## API Usage

### Single Model Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This article claims the earth is flat",
    "model": "bert"
  }'
```

### Compare All Models
```bash
curl -X POST http://localhost:8000/predict/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Article text here..."
  }'
```

Response includes logistic, lstm, and bert predictions.

### Ensemble Prediction
```bash
curl -X POST http://localhost:8000/predict/ensemble \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Article text here..."
  }'
```

Combines predictions from all three models (majority vote).

### LIME Explanation
```bash
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Article text here...",
    "model": "bert",
    "method": "lime",
    "top_k": 10
  }'
```

### SHAP Explanation
```bash
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Article text here...",
    "model": "bert",
    "method": "shap",
    "top_k": 10
  }'
```

---

## Model Status

✓ Model files present: `backend/app/ml/saved_models/distilbert/`
✓ Inference module: `backend/app/ml/inference/predict_bert.py`
✓ LIME explainability: `backend/app/ml/explainability/lime_bert.py`
✓ SHAP explainability: `backend/app/ml/explainability/shap_bert.py`
✓ API integration: `backend/app/api/prediction.py`

---

## Troubleshooting

### DistilBERT model not found
Check files exist:
```bash
ls -la backend/app/ml/saved_models/distilbert/
```

Should show:
- config.json
- model.safetensors
- tokenizer.json
- tokenizer_config.json

### ModuleNotFoundError: No module named 'torch'
Torch is installed in Docker via a special PyTorch wheel. Run tests inside container:
```bash
docker exec fake-news-detector-backend python test_distilbert_diagnostic.py
```

### Transformer version mismatch
The Dockerfile verifies imports at build time. If it passes build, you're good.

### Model loading timeout
DistilBERT downloads models on first load. Increase timeout or run in background.

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/app/ml/inference/predict_bert.py` | Model inference |
| `backend/app/ml/explainability/lime_bert.py` | LIME explanations |
| `backend/app/ml/explainability/shap_bert.py` | SHAP explanations |
| `backend/app/api/prediction.py` | Prediction API endpoints |
| `backend/tests/test_distilbert.py` | Test suite |
| `backend/test_distilbert_diagnostic.py` | Diagnostic script |
| `backend/app/ml/saved_models/distilbert/` | Model weights |

---

## Performance

- **First load**: ~2-3 seconds (loads tokenizer + model)
- **Subsequent loads**: <100ms (uses cached singleton)
- **Inference time**: 50-200ms per text (CPU, depends on text length)
- **Memory usage**: ~500MB for model + inference

---

## Common Issues From Previous Sessions

**Issue:** torch.accelerator dispatch on CPU-only machines
**Fix:** Already handled in `predict_bert.py` via:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# And NOT passing device_map parameter
```

**Issue:** Version mismatch between transformers and tokenizers
**Fix:** Already pinned in `requirements.txt`:
```
transformers==5.12.1
tokenizers>=0.21,<0.22
```

---

## Next Steps

1. Run diagnostic script: `docker exec fake-news-detector-backend python test_distilbert_diagnostic.py`
2. Check API endpoints: Use curl examples above
3. Run full test suite: `docker exec fake-news-detector-backend python -m pytest tests/test_distilbert.py -v`
4. Monitor logs: `docker logs -f fake-news-detector-backend`
