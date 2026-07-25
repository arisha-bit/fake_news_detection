"""
Reverse Image Verification Service — Phase 5.

Uses OpenAI CLIP to embed an uploaded image and search a pre-built FAISS
index of article title embeddings (in the same CLIP embedding space).

Why CLIP works here:
    CLIP is trained to align image and text in the same embedding space.
    An image of a flooded city will be close to articles about floods.
    An image of a politician will be close to political articles.
    This enables cross-modal similarity search without a real image corpus.

Singletons:
    - CLIP model + processor loaded once on first request.
    - FAISS index + metadata loaded once on first request.

Reuse detection:
    Any match with similarity >= REUSE_THRESHOLD is flagged as
    possible_reuse_detected=True in the response.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------

_BASE = Path(__file__).resolve().parent.parent
INDEX_DIR = _BASE / "ml" / "clip" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "clip_faiss.index"
METADATA_PATH = INDEX_DIR / "clip_metadata.pkl"

CLIP_MODEL = "openai/clip-vit-base-patch32"
REUSE_THRESHOLD = 0.75  # similarity above this → flag as possible reuse

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_clip_model = None
_clip_processor = None
_faiss_index = None
_metadata: Optional[list[dict]] = None


def _get_clip():
    global _clip_model, _clip_processor

    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

            logger.info("Loading CLIP model: %s", CLIP_MODEL)
            _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            _clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
            _clip_model.eval()
            logger.info("CLIP model ready.")
        except Exception as exc:
            logger.error("Failed to load CLIP model: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"CLIP model failed to load: {str(exc)}",
            ) from exc

    return _clip_model, _clip_processor


def _get_index():
    global _faiss_index, _metadata

    if _faiss_index is None:
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "CLIP image index not found. "
                    "Run: python -m app.ml.clip.build_image_index "
                    "to build the index before using reverse image search."
                ),
            )
        try:
            import faiss  # noqa: PLC0415

            logger.info("Loading CLIP FAISS index from %s", FAISS_INDEX_PATH)
            _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

            with open(METADATA_PATH, "rb") as f:
                _metadata = pickle.load(f)

            logger.info(
                "CLIP FAISS index loaded — %d vectors", _faiss_index.ntotal
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to load CLIP index: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"CLIP index failed to load: {str(exc)}",
            ) from exc

    return _faiss_index, _metadata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reverse_image_search(image_path: str, top_k: int = 5) -> tuple[list[dict], bool]:
    """
    Embed the image at *image_path* with CLIP and return the top-k most
    similar articles from the index.

    Args:
        image_path: Path to the image file on disk.
        top_k:      Number of results to return.

    Returns:
        (results, possible_reuse_detected)
        results: list of dicts with rank, title, label, similarity, etc.
        possible_reuse_detected: True if any similarity >= REUSE_THRESHOLD.

    Raises:
        HTTP 404 — image file not found.
        HTTP 422 — image cannot be opened.
        HTTP 503 — index not built.
        HTTP 500 — unexpected failure.
    """
    path = Path(image_path)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Image file not found: {image_path}",
        )

    # Load and embed image
    try:
        from PIL import Image  # noqa: PLC0415
        import torch  # noqa: PLC0415

        image = Image.open(path).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot open image file. It may be corrupted: {str(exc)}",
        ) from exc

    logger.info("Running CLIP reverse image search on: %s", image_path)

    model, processor = _get_clip()
    index, metadata = _get_index()

    try:
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_feats = model.get_image_features(**inputs)
            # L2-normalise for cosine via inner product
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

        query_vec = image_feats.cpu().numpy().astype(np.float32)

    except Exception as exc:
        logger.error("CLIP inference error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"CLIP embedding failed: {str(exc)}",
        ) from exc

    # FAISS search
    k = min(top_k, index.ntotal)
    similarities, indices = index.search(query_vec, k)

    results = []
    possible_reuse = False

    for rank, (sim, idx) in enumerate(
        zip(similarities[0], indices[0]), start=1
    ):
        if idx < 0 or idx >= len(metadata):
            continue

        sim_float = round(float(sim), 4)
        meta = metadata[idx]

        if sim_float >= REUSE_THRESHOLD:
            possible_reuse = True

        results.append({
            "rank": rank,
            "title": meta.get("title", ""),
            "potential_source": meta.get("potential_source", "unknown"),
            "label": meta.get("label", "UNKNOWN"),
            "similarity": sim_float,
            "date": meta.get("date", ""),
            "snippet": meta.get("snippet", ""),
        })

    logger.info(
        "Reverse image search complete — %d results, reuse_detected=%s",
        len(results),
        possible_reuse,
    )

    return results, possible_reuse
