"""
OCR service using EasyOCR.
Implements a singleton reader to avoid reloading the model on every request.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton reader — loaded once at first use
# ---------------------------------------------------------------------------

_reader: Optional[object] = None


def _get_reader():
    """
    Return the shared EasyOCR Reader instance.
    Initialises it on first call (lazy singleton).
    CPU-only: gpu=False to stay consistent with the rest of the project.
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
                detail="OCR engine failed to initialise. Check EasyOCR installation."
            ) from exc
    return _reader


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_image(file_path: str) -> str:
    """
    Run OCR on the image at *file_path* and return cleaned extracted text.

    Steps:
        1. Verify the file exists.
        2. Run EasyOCR on the image.
        3. Join detected text blocks into a single string.
        4. Collapse excessive whitespace and blank lines.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        A clean, single-string representation of all detected text.

    Raises:
        HTTP 404  — file not found.
        HTTP 422  — OCR produced no readable text.
        HTTP 500  — unexpected OCR engine failure.
    """
    path = Path(file_path)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Image file not found at path: {file_path}"
        )

    logger.info("Starting OCR on: %s", file_path)

    try:
        reader = _get_reader()
        results = reader.readtext(str(path), detail=0, paragraph=True)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("OCR engine error on %s: %s", file_path, exc)
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(exc)}"
        ) from exc

    if not results:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the image. The image may be blank or corrupted."
        )

    # Join paragraphs, collapse internal whitespace
    raw_text = " ".join(results)
    clean_text = _clean_text(raw_text)

    logger.info("OCR complete. Extracted %d characters.", len(clean_text))
    return clean_text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Normalise whitespace in OCR output:
      - Replace newlines/tabs with spaces
      - Collapse multiple spaces into one
      - Strip leading/trailing whitespace
    """
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
