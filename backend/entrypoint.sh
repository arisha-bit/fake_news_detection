#!/bin/sh
set -e

echo "Running Alembic migrations..."
alembic upgrade head

# Build FAISS evidence index only if it doesn't exist yet
if [ ! -f "app/ml/embeddings/index/faiss.index" ]; then
    echo "Building evidence retrieval index..."
    python -m app.ml.embeddings.build_index || echo "WARNING: Evidence index build failed (dataset may be missing). Skipping."
else
    echo "Evidence index already exists. Skipping build."
fi

# Build CLIP image index only if it doesn't exist yet
if [ ! -f "app/ml/clip/index/clip_faiss.index" ]; then
    echo "Building CLIP reverse image index..."
    python -m app.ml.clip.build_image_index || echo "WARNING: CLIP index build failed (dataset may be missing). Skipping."
else
    echo "CLIP index already exists. Skipping build."
fi

echo "Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
