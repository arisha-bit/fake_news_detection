"""
Offline CLIP index builder — run once to generate the reverse image search index.

Usage:
    python -m app.ml.clip.build_image_index

How it works:
    CLIP encodes both images and text into the same 512-dim embedding space.
    Since we have a text corpus (not an image corpus), we embed article titles
    using CLIP's text encoder. When a user uploads an image, its CLIP visual
    embedding is compared against these text embeddings.

    This is semantically valid — CLIP is trained to align visual and textual
    representations, so "a photo of a flood disaster" will be near articles
    about floods in CLIP space.

    In production, replace text embeddings with actual news image embeddings
    when an image corpus is available.

Output (app/ml/clip/index/):
    clip_faiss.index   — FAISS flat inner-product index (512-dim)
    clip_metadata.pkl  — list of dicts with title, label, source, date, snippet
"""

import logging
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
DATASETS_DIR = BASE_DIR.parent.parent.parent / "datasets"
INDEX_DIR = BASE_DIR / "index"
FAISS_INDEX_PATH = INDEX_DIR / "clip_faiss.index"
METADATA_PATH = INDEX_DIR / "clip_metadata.pkl"

CLIP_MODEL = "openai/clip-vit-base-patch32"
MAX_ARTICLES = 3000   # text-embed this many titles
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build():
    import faiss
    from transformers import CLIPModel, CLIPTokenizer

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load both datasets
    logger.info("Loading Real.csv and Fake.csv...")
    real_df = pd.read_csv(DATASETS_DIR / "Real.csv").head(MAX_ARTICLES // 2)
    fake_df = pd.read_csv(DATASETS_DIR / "Fake.csv").head(MAX_ARTICLES // 2)

    real_df["label"] = "REAL"
    fake_df["label"] = "FAKE"

    df = pd.concat([real_df, fake_df], ignore_index=True)
    df = df.dropna(subset=["title"]).reset_index(drop=True)
    logger.info("Total articles for index: %d", len(df))

    # 2. Build metadata
    metadata = []
    titles = []

    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        snippet = str(row.get("text", ""))[:200].strip()

        metadata.append({
            "title": title,
            "label": row["label"],
            "potential_source": str(row.get("subject", "news")),
            "date": str(row.get("date", "")),
            "snippet": snippet,
        })
        titles.append(title)

    # 3. Load CLIP text encoder
    logger.info("Loading CLIP model: %s", CLIP_MODEL)
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL)
    model.eval()

    import torch

    # 4. Embed titles in batches
    logger.info("Embedding %d article titles with CLIP...", len(titles))
    all_embeddings = []

    for i in range(0, len(titles), BATCH_SIZE):
        batch = titles[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,  # CLIP text limit
        )
        with torch.no_grad():
            text_feats = model.get_text_features(**inputs)
            # L2-normalise for cosine similarity via inner product
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        all_embeddings.append(text_feats.cpu().numpy())
        logger.info("  Embedded %d / %d", min(i + BATCH_SIZE, len(titles)), len(titles))

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    # 5. Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("CLIP FAISS index built — %d vectors, dim=%d", index.ntotal, dim)

    # 6. Persist
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    logger.info("CLIP index saved to %s", INDEX_DIR)


if __name__ == "__main__":
    build()
