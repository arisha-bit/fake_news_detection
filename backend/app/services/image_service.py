"""
Reverse Image Verification Service.

Uses CLIP ViT-B/32 embeddings (512-dim) and a FAISS index for semantic
reverse-image search. If the index or model is unavailable, the service
degrades gracefully and returns an empty result set.

IMPORTANT — embedding extraction:
  CLIPModel.get_image_features() returns the projected 512-dim visual
  embedding directly as a plain tensor (not a dataclass). Do NOT index
  into .last_hidden_state (768-dim encoder output) — that causes a FAISS
  dimension mismatch (768 != 512).
"""

import logging
import pickle
import threading
import time
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
_lock = threading.Lock()


def _get_clip():
    global _clip_model, _clip_processor

    if _clip_model is None:
        with _lock:
            # Check again after acquiring lock
            if _clip_model is None:
                try:
                    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

                    t0 = time.perf_counter()
                    logger.info("Loading CLIP model: %s", CLIP_MODEL)
                    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
                    _clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
                    _clip_model.eval()
                    logger.info("CLIP model ready in %.2fs.", time.perf_counter() - t0)
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
        with _lock:
            # Check again after acquiring lock
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

                    t0 = time.perf_counter()
                    logger.info("Loading CLIP FAISS index from %s", FAISS_INDEX_PATH)
                    _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
                    with open(METADATA_PATH, "rb") as f:
                        _metadata = pickle.load(f)
                    logger.info(
                        "CLIP FAISS index loaded — %d vectors, dim=%d in %.2fs",
                        _faiss_index.ntotal,
                        _faiss_index.d,
                        time.perf_counter() - t0,
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


def _extract_clip_embedding(image) -> "np.ndarray":
    """
    Extract a 512-dim L2-normalised CLIP image embedding.

    CLIPModel.get_image_features() returns the projected 512-dim embedding
    directly as a tensor — NOT a BaseModelOutput dataclass. We must not
    attempt to access .pooler_output or .last_hidden_state on it.
    
    Handles both plain tensor returns and potential dataclass wrappers.
    """
    import torch  # noqa: PLC0415

    model, processor = _get_clip()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        # get_image_features usually returns a plain (batch, 512) tensor
        output = model.get_image_features(**inputs)
        
        # Handle both plain tensor and potential dataclass/wrapper
        if hasattr(output, 'last_hidden_state'):
            # Unexpected dataclass wrapper — take the embedding attribute if present
            if hasattr(output, 'pooler_output'):
                image_feats = output.pooler_output
            else:
                image_feats = output.last_hidden_state[:, 0, :]  # Take CLS token
            logger.warning(
                "CLIP output was a dataclass; extracted pooler or last_hidden_state. "
                "Shape: %s", image_feats.shape
            )
        else:
            # Normal case: plain tensor
            image_feats = output

    # Validate shape before normalizing
    if image_feats.shape[-1] != 512:
        logger.warning(
            "Unexpected CLIP embedding dimension: %d (expected 512)",
            image_feats.shape[-1],
        )
    
    # L2-normalise — the FAISS index uses inner product (= cosine on unit vecs)
    image_feats = torch.nn.functional.normalize(image_feats, dim=-1)
    result = image_feats.cpu().numpy().astype(np.float32)
    
    # Validate result
    if result.shape[-1] != 512:
        raise ValueError(
            f"CLIP embedding has wrong dimension: {result.shape[-1]} (expected 512)"
        )
    
    return result


def reverse_image_search(image_path: str, top_k: int = 5) -> tuple[list[dict], bool]:
    path = Path(image_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")

    try:
        from PIL import Image  # noqa: PLC0415
        image = Image.open(path).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot open image file. It may be corrupted: {str(exc)}",
        ) from exc

    logger.info("Running CLIP reverse image search on: %s", image_path)

    try:
        index, metadata = _get_index()
    except HTTPException as exc:
        logger.warning("Reverse image search unavailable: %s", exc.detail)
        return [], False

    try:
        query_vec = _extract_clip_embedding(image)
    except Exception as exc:
        logger.warning("CLIP embedding failed: %s", exc)
        return [], False

    # Sanity-check dimension
    if query_vec.shape[-1] != index.d:
        logger.error(
            "Embedding dim mismatch: query=%d, index=%d; cannot search",
            query_vec.shape[-1],
            index.d,
        )
        raise HTTPException(
            status_code=500,
            detail=f"CLIP embedding dimension mismatch: {query_vec.shape[-1]} vs index {index.d}",
        )

    if index.ntotal == 0:
        logger.warning("CLIP FAISS index is empty — no vectors indexed.")
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

    logger.info(
        "Reverse image search complete — %d results, reuse_detected=%s",
        len(results),
        possible_reuse,
    )
    return results, possible_reuse
