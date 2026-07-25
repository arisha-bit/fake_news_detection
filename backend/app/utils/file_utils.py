"""
Reusable file utility helpers for the upload pipeline.
All file I/O logic lives here — never in route handlers.

Supports images (JPG/JPEG/PNG) and PDFs.
Future types (DOCX, URL) can be added by extending the validator constants.
"""

import logging
import os
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALLOWED_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
ALLOWED_IMAGE_MIME_TYPES: set[str] = {"image/jpeg", "image/png"}

ALLOWED_PDF_EXTENSIONS: set[str] = {".pdf"}
ALLOWED_PDF_MIME_TYPES: set[str] = {"application/pdf"}

# 10 MB for images, 20 MB for PDFs
MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024
MAX_PDF_SIZE_BYTES: int = 20 * 1024 * 1024

IMAGE_UPLOAD_DIR = Path("uploads/images")
PDF_UPLOAD_DIR = Path("uploads/pdfs")


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------

def ensure_dir(directory: Path) -> Path:
    """
    Create *directory* (and any parents) if it does not already exist.
    Returns the resolved Path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug("Upload directory ensured: %s", directory.resolve())
    return directory


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_image(file: UploadFile) -> None:
    """
    Validate that *file* is an accepted image format.

    Raises:
        HTTP 400 — unsupported extension.
        HTTP 415 — unsupported MIME type.
    """
    _validate_file(
        file,
        allowed_extensions=ALLOWED_IMAGE_EXTENSIONS,
        allowed_mime_types=ALLOWED_IMAGE_MIME_TYPES,
    )


def validate_pdf(file: UploadFile) -> None:
    """
    Validate that *file* is an accepted PDF.

    Raises:
        HTTP 400 — unsupported extension.
        HTTP 415 — unsupported MIME type.
    """
    _validate_file(
        file,
        allowed_extensions=ALLOWED_PDF_EXTENSIONS,
        allowed_mime_types=ALLOWED_PDF_MIME_TYPES,
    )


def _validate_file(
    file: UploadFile,
    allowed_extensions: set[str],
    allowed_mime_types: set[str],
) -> None:
    """Internal generic file validator."""
    original_name = file.filename or ""
    ext = Path(original_name).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Allowed: {allowed_extensions}"
            ),
        )

    if file.content_type not in allowed_mime_types:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported MIME type '{file.content_type}'. "
                f"Allowed: {allowed_mime_types}"
            ),
        )


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def generate_safe_filename(original_filename: str) -> str:
    """
    Return a UUID-based filename preserving the original extension.
    Prevents path traversal and filename collisions.
    """
    ext = Path(original_filename).suffix.lower()
    return f"{uuid.uuid4().hex}{ext}"


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

async def save_upload(
    file: UploadFile,
    upload_dir: Path,
    max_size_bytes: int,
) -> tuple[str, str, str]:
    """
    Read *file* into memory, enforce *max_size_bytes*, then persist to *upload_dir*.

    Args:
        file:            The incoming UploadFile.
        upload_dir:      Destination directory (created if missing).
        max_size_bytes:  Maximum allowed file size in bytes.

    Returns:
        (safe_filename, original_filename, file_path_str)

    Raises:
        HTTP 422 — empty file.
        HTTP 413 — file exceeds max_size_bytes.
    """
    ensure_dir(upload_dir)

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    if len(contents) > max_size_bytes:
        limit_mb = max_size_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {limit_mb} MB.",
        )

    original_filename = file.filename or "unknown"
    safe_filename = generate_safe_filename(original_filename)
    file_path = upload_dir / safe_filename

    with open(file_path, "wb") as f:
        f.write(contents)

    logger.info("File saved: %s (original: %s)", file_path, original_filename)
    return safe_filename, original_filename, str(file_path)


async def save_image_upload(file: UploadFile) -> tuple[str, str, str]:
    """Convenience wrapper — save an image to the images upload directory."""
    return await save_upload(file, IMAGE_UPLOAD_DIR, MAX_IMAGE_SIZE_BYTES)


async def save_pdf_upload(file: UploadFile) -> tuple[str, str, str]:
    """Convenience wrapper — save a PDF to the pdfs upload directory."""
    return await save_upload(file, PDF_UPLOAD_DIR, MAX_PDF_SIZE_BYTES)


def delete_temp_file(file_path: str) -> None:
    """
    Silently remove a temporary file from disk.
    Logs a warning on failure but does not raise.
    """
    try:
        path = Path(file_path)
        if path.exists():
            os.remove(path)
            logger.info("Temporary file deleted: %s", file_path)
    except OSError as exc:
        logger.warning("Could not delete temporary file %s: %s", file_path, exc)
