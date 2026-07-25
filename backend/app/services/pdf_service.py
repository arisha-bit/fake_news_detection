"""
PDF text extraction service using PyMuPDF (fitz).

Implements a clean extraction pipeline:
  1. Open PDF from disk.
  2. Iterate all pages.
  3. Extract text blocks per page.
  4. Join and normalise whitespace.
  5. Return a single clean string.

PyMuPDF is chosen over pdfplumber for speed and lower memory usage.
pdfplumber is a drop-in alternative if table-aware extraction is needed later.
"""

import logging
import re
from pathlib import Path

from fastapi import HTTPException

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all readable text from a PDF file.

    Args:
        file_path: Path to the saved PDF on disk.

    Returns:
        Clean, single-string representation of all page text.

    Raises:
        HTTP 404 — file not found.
        HTTP 422 — PDF contains no extractable text (scanned/image-only PDF).
        HTTP 500 — unexpected PyMuPDF failure.
    """
    path = Path(file_path)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found at path: {file_path}",
        )

    logger.info("Starting PDF extraction: %s", file_path)

    try:
        import fitz  # PyMuPDF — imported lazily to avoid load-time overhead

        doc = fitz.open(str(path))
        pages_text: list[str] = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")  # plain text mode
            if text.strip():
                pages_text.append(text)

        doc.close()

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("PyMuPDF error on %s: %s", file_path, exc)
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(exc)}",
        ) from exc

    if not pages_text:
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be extracted from the PDF. "
                "The file may be image-only or corrupted. "
                "Try uploading a scanned image instead."
            ),
        )

    raw_text = "\n".join(pages_text)
    clean = _clean_text(raw_text)

    logger.info(
        "PDF extraction complete — %d pages, %d characters.",
        len(pages_text),
        len(clean),
    )
    return clean


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Normalise extracted PDF text:
      - Replace form-feed and tab characters with spaces.
      - Collapse runs of 3+ newlines into two (preserve paragraph breaks).
      - Collapse multiple spaces into one.
      - Strip leading/trailing whitespace.
    """
    text = text.replace("\f", "\n").replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
