"""
OCR service using EasyOCR.
Singleton reader to avoid reloading the model on every request.

IMPORTANT: extract_text_from_image() returns an EMPTY STRING when no text
is found — it does NOT raise an exception. This allows the multimodal
pipeline to continue with image-only analysis when OCR finds nothing.

The caller is responsible for deciding whether empty text is an error.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

_reader: Optional[object] = None


def _get_reader():
    """
    Return shared EasyOCR Reader (lazy singleton, CPU-only).
    EasyOCR may download models on first call — this is expected behaviour.
    """
    global _reader
    if _reader is None:
        try:
            import easyocr  # noqa: PLC0415
            logger.info("Initialising EasyOCR reader (CPU)...")
            _reader = easyocr.Reader(["en"], gpu=False)
            logger.info("EasyOCR reader ready.")
        except Exception as exc:
            logger.error("Failed to initialise EasyOCR: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"OCR engine failed to initialise: {str(exc)}",
            ) from exc
    return _reader


def extract_text_from_image(file_path: str) -> str:
    """
    Run OCR on the image at *file_path*.

    Returns:
        Clean extracted text string.
        Returns EMPTY STRING if no text is detected — does NOT raise.

    Raises:
        HTTP 404 — file not found.
        HTTP 500 — OCR engine failure (crash, not just empty result).
    """
    path = Path(file_path)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Image file not found at path: {file_path}",
        )

    logger.info("Starting OCR on: %s", file_path)

    try:
        reader = _get_reader()
        results = reader.readtext(str(path), detail=0, paragraph=True)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("OCR engine error on %s: %s", file_path, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(exc)}",
        ) from exc

    if not results:
        logger.info("OCR found no text in: %s", file_path)
        return ""  # empty — caller decides what to do

    raw_text = " ".join(results)
    clean_text = _clean_text(raw_text)

    logger.info("OCR complete — extracted %d characters from %s", len(clean_text), file_path)
    return clean_text


def _clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
