"""
Reverse Image Verification Service.

Uses CLIP embeddings and a FAISS index for semantic reverse-image search.
If the index or model is unavailable, the service degrades gracefully and
returns an empty result set instead of breaking verification flows.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import HTTPException

logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent.parent
INDEX_DIR = _BASE / "ml" / "clip" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "clip_faiss.index"
METADATA_PATH = INDEX_DIR / "clip_metadata.pkl"

CLIP_MODEL = "openai/clip-vit-base-patch32"
REUSE_THRESHOLD = 0.75

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
            logger.info("CLIP FAISS index loaded — %d vectors", _faiss_index.ntotal)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to load CLIP index: %s", exc)
            raise HTTPException(status_code=500, detail=f"CLIP index failed to load: {str(exc)}") from exc

    return _faiss_index, _metadata


def reverse_image_search(image_path: str, top_k: int = 5) -> tuple[list[dict], bool]:
    path = Path(image_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")

    try:
        from PIL import Image  # noqa: PLC0415
        import torch  # noqa: PLC0415

        image = Image.open(path).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open image file. It may be corrupted: {str(exc)}") from exc

    logger.info("Running CLIP reverse image search on: %s", image_path)

    try:
        model, processor = _get_clip()
        index, metadata = _get_index()
    except HTTPException as exc:
        logger.warning("Reverse image search unavailable: %s", exc.detail)
        return [], False

    try:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = model.get_image_features(**inputs)
            image_feats = output.last_hidden_state[:, 0, :] if hasattr(output, "last_hidden_state") else output
            image_feats = torch.nn.functional.normalize(image_feats, dim=-1)
        query_vec = image_feats.cpu().numpy().astype(np.float32)
    except Exception as exc:
        logger.warning("CLIP embedding failed: %s", exc)
        return [], False

    if index.ntotal == 0:
        return [], False

    k = min(top_k, index.ntotal)
    similarities, indices = index.search(query_vec, k)

    results = []
    possible_reuse = False

    for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), start=1):
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

    logger.info("Reverse image search complete — %d results, reuse_detected=%s", len(results), possible_reuse)
    return results, possible_reuse
