"""
Evidence Retrieval Service — Phase 4.

Provides semantic search over a pre-built FAISS index of trusted/known news articles.
Given a query (claim or full article), returns the top-k most similar articles
with their similarity scores, labels, and text snippets.

Design:
- FAISS index + metadata loaded once as module-level singletons.
- Sentence Transformer model loaded once as a singleton.
- If the index does not exist, a clear HTTP 503 is raised with build instructions.
- Cosine similarity is used (inner product on normalised embeddings).
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE = Path(__file__).resolve().parent.parent
INDEX_DIR = _BASE / "ml" / "embeddings" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_faiss_index = None
_metadata: Optional[list[dict]] = None
_embedder = None


def _get_index():
    global _faiss_index, _metadata

    if _faiss_index is None:
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Evidence index not found. "
                    "Run: python -m app.ml.embeddings.build_index "
                    "to build the index before using evidence retrieval."
                ),
            )
        try:
            import faiss  # noqa: PLC0415

            logger.info("Loading FAISS index from %s", FAISS_INDEX_PATH)
            _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

            with open(METADATA_PATH, "rb") as f:
                _metadata = pickle.load(f)

            logger.info(
                "FAISS index loaded — %d vectors", _faiss_index.ntotal
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Evidence index failed to load: {str(exc)}",
            ) from exc

    return _faiss_index, _metadata


def _get_embedder():
    global _embedder

    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model ready.")
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Embedding model failed to load: {str(exc)}",
            ) from exc

    return _embedder


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_evidence(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the FAISS index for the top-k most semantically similar articles.

    Args:
        query:  The claim or article text to search for.
        top_k:  Number of results to return.

    Returns:
        List of dicts with keys:
            rank, title, snippet, label, similarity, subject, date

    Raises:
        HTTP 422 — empty query.
        HTTP 503 — index not built yet.
        HTTP 500 — unexpected failure.
    """
    if not query or not query.strip():
        raise HTTPException(
            status_code=422,
            detail="Query text cannot be empty.",
        )

    index, metadata = _get_index()
    embedder = _get_embedder()

    logger.info("Searching evidence for query (%d chars)...", len(query))

    # Embed + normalise query
    query_vec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # FAISS inner product search (= cosine similarity on normalised vecs)
    k = min(top_k, index.ntotal)
    similarities, indices = index.search(query_vec, k)

    results = []
    for rank, (sim, idx) in enumerate(
        zip(similarities[0], indices[0]), start=1
    ):
        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]
        results.append({
            "rank": rank,
            "title": meta.get("title", ""),
            "snippet": meta.get("snippet", ""),
            "label": meta.get("label", "UNKNOWN"),
            "similarity": round(float(sim), 4),
            "subject": meta.get("subject", ""),
            "date": meta.get("date", ""),
        })

    logger.info("Evidence search complete — %d results returned.", len(results))
    return results
