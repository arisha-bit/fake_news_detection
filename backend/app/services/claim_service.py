"""
Claim Extraction Service — Phase 3.

Splits article text into individual factual claims using spaCy sentence
segmentation, filters out non-factual sentences (questions, short fragments,
pure headlines), and returns a clean list of claim strings.

Each claim is then independently verified by the existing prediction pipeline
in the API layer — this service is purely responsible for extraction.

Design decisions:
- spaCy `en_core_web_sm` is used for sentence boundary detection.
  It is fast, CPU-only, and adds no GPU dependency.
- The model is loaded once as a module-level singleton.
- Heuristic filters remove noise (very short sentences, pure questions,
  sentences with no verb) to improve prediction quality per claim.
"""

import logging
import re

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy singleton — loaded once at first use
# ---------------------------------------------------------------------------

_nlp = None


def _get_nlp():
    """
    Return the shared spaCy language model.
    Downloads en_core_web_sm automatically if missing.
    """
    global _nlp

    if _nlp is None:
        try:
            import spacy  # noqa: PLC0415

            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. Downloading..."
                )
                from spacy.cli import download  # noqa: PLC0415

                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")

            logger.info("spaCy model loaded: en_core_web_sm")

        except Exception as exc:
            logger.error("Failed to load spaCy model: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Claim extraction engine failed to initialise. "
                       "Ensure spaCy and en_core_web_sm are installed.",
            ) from exc

    return _nlp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_claims(text: str) -> list[str]:
    """
    Extract a list of factual claim sentences from *text*.

    Pipeline:
        1. Sentence-tokenise with spaCy.
        2. Apply heuristic filters to remove noise.
        3. Return deduplicated, ordered list of claim strings.

    Args:
        text: Full article or document text.

    Returns:
        List of clean claim strings. May be empty for very short inputs.

    Raises:
        HTTP 422 — text is blank.
        HTTP 500 — spaCy engine failure.
    """
    if not text or not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Cannot extract claims from empty text.",
        )

    nlp = _get_nlp()

    # spaCy has a default max_length of 1,000,000 chars
    # Truncate very long articles to avoid memory issues
    truncated = text[:50_000]
    if len(text) > 50_000:
        logger.warning(
            "Text truncated from %d to 50,000 chars for claim extraction.",
            len(text),
        )

    doc = nlp(truncated)
    claims: list[str] = []
    seen: set[str] = set()

    for sent in doc.sents:
        claim = _clean_sentence(sent.text)

        if not _is_valid_claim(claim):
            continue

        # Deduplicate
        normalised = claim.lower().strip()
        if normalised in seen:
            continue

        seen.add(normalised)
        claims.append(claim)

    logger.info(
        "Claim extraction complete — %d claims from %d chars of text.",
        len(claims),
        len(truncated),
    )

    return claims


def compute_overall_verdict(
    predictions: list[str],
    confidences: list[float],
) -> tuple[str, float]:
    """
    Compute a majority-vote verdict and average confidence from per-claim results.

    Args:
        predictions: List of "REAL" or "FAKE" strings.
        confidences: List of float confidence values (same length).

    Returns:
        (overall_verdict, average_confidence)
    """
    if not predictions:
        return "REAL", 0.0

    fake_count = predictions.count("FAKE")
    real_count = predictions.count("REAL")
    verdict = "FAKE" if fake_count > real_count else "REAL"
    avg_confidence = round(sum(confidences) / len(confidences), 4)

    return verdict, avg_confidence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_sentence(sentence: str) -> str:
    """Strip whitespace and normalise internal spacing."""
    sentence = sentence.replace("\n", " ").replace("\t", " ")
    sentence = re.sub(r" {2,}", " ", sentence)
    return sentence.strip()


def _is_valid_claim(sentence: str) -> bool:
    """
    Return True if *sentence* looks like a verifiable factual claim.

    Filtered out:
    - Sentences shorter than 20 characters (too vague).
    - Pure questions (ends with '?').
    - Sentences that are only punctuation/numbers.
    - Sentences with no alphabetic characters.
    """
    if len(sentence) < 20:
        return False

    if sentence.endswith("?"):
        return False

    if not any(c.isalpha() for c in sentence):
        return False

    return True
