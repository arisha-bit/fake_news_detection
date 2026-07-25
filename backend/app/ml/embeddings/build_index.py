"""
Offline index builder — run once to generate the FAISS evidence index.

Usage:
    python -m app.ml.embeddings.build_index

Output files written to app/ml/embeddings/index/:
    faiss.index      — FAISS flat inner-product index
    metadata.pkl     — list of dicts with title, snippet, label, subject, date

The index is built from datasets/processed/news.csv.
Only the first MAX_ARTICLES articles are embedded to keep memory manageable.
Increase MAX_ARTICLES for production if hardware allows.

Model: all-MiniLM-L6-v2 (384-dim, fast, high quality for semantic search)
"""

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent.parent.parent / "datasets" / "processed" / "news.csv"
INDEX_DIR = BASE_DIR / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_ARTICLES = 5000   # increase to full dataset in production
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build():
    import faiss
    from sentence_transformers import SentenceTransformer

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    logger.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["title", "content"])
    df = df.head(MAX_ARTICLES).reset_index(drop=True)
    logger.info("Loaded %d articles", len(df))

    # 2. Build metadata store
    metadata = []
    texts_to_embed = []

    for _, row in df.iterrows():
        snippet = str(row["content"])[:300].strip()
        title = str(row.get("title", "")).strip()
        label = "REAL" if int(row["label"]) == 1 else "FAKE"

        metadata.append({
            "title": title,
            "snippet": snippet,
            "label": label,
            "subject": str(row.get("subject", "")),
            "date": str(row.get("date", "")),
        })

        # Embed title + snippet for richer representation
        texts_to_embed.append(f"{title}. {snippet}")

    # 3. Generate embeddings
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info("Generating embeddings for %d articles...", len(texts_to_embed))
    embeddings = model.encode(
        texts_to_embed,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # enables cosine similarity via inner product
    )

    embeddings = embeddings.astype(np.float32)

    # 4. Build FAISS index (flat inner-product = cosine similarity on normalised vecs)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built — %d vectors, dim=%d", index.ntotal, dim)

    # 5. Persist
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    logger.info("Index saved to %s", INDEX_DIR)
    logger.info("Metadata saved to %s", METADATA_PATH)


if __name__ == "__main__":
    build()
