"""
Source Credibility Service — Phase 6.

Scores a URL or domain against a curated trust database.

Design decisions:
- Trust database is a local JSON file (app/data/trust_database.json).
  This makes it fast, offline, version-controllable, and easy to extend.
  Swap for an external API (MBFC, NewsGuard) by replacing _lookup().
- Domain extraction handles full URLs, subdomains, and bare domains.
- Database loaded once as a module-level singleton.
- Unknown domains return a neutral "not found" response — never a crash.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "trust_database.json"

# Thresholds for credibility_label override consistency check
_LABEL_MAP = {
    "HIGH": (70, 100),
    "MEDIUM": (45, 69),
    "LOW": (20, 44),
    "VERY_LOW": (0, 19),
}

# ---------------------------------------------------------------------------
# Singleton database loader
# ---------------------------------------------------------------------------

_trust_db: Optional[dict] = None


def _get_db() -> dict:
    global _trust_db

    if _trust_db is None:
        if not DB_PATH.exists():
            logger.error("Trust database not found at %s", DB_PATH)
            raise HTTPException(
                status_code=500,
                detail="Trust database file is missing. Check app/data/trust_database.json.",
            )
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                _trust_db = json.load(f)
            logger.info(
                "Trust database loaded — %d domains", len(_trust_db)
            )
        except Exception as exc:
            logger.error("Failed to load trust database: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Trust database failed to load: {str(exc)}",
            ) from exc

    return _trust_db


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_credibility(url_or_domain: str) -> dict:
    """
    Score the credibility of a news source by URL or domain.

    Args:
        url_or_domain: Full URL or bare domain string.

    Returns:
        Dict matching CredibilityResponse schema fields.

    Raises:
        HTTP 400 — input is blank or unparseable.
        HTTP 500 — database load failure.
    """
    if not url_or_domain or not url_or_domain.strip():
        raise HTTPException(
            status_code=400,
            detail="URL or domain cannot be empty.",
        )

    domain = _extract_domain(url_or_domain.strip())

    if not domain:
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract a valid domain from: '{url_or_domain}'",
        )

    logger.info("Checking credibility for domain: %s", domain)

    db = _get_db()
    entry = _lookup(domain, db)

    if entry is None:
        # Unknown domain — return neutral not-found response
        return {
            "domain": domain,
            "found_in_database": False,
            "trust_score": None,
            "reliability_score": None,
            "bias_rating": None,
            "category": None,
            "credibility_label": None,
            "verdict": (
                f"'{domain}' was not found in our credibility database. "
                "Treat information from this source with caution until verified."
            ),
            "notes": None,
        }

    trust_score = entry["trust_score"]
    credibility_label = entry.get("credibility_label", _score_to_label(trust_score))
    verdict = _build_verdict(domain, credibility_label, entry)

    return {
        "domain": domain,
        "found_in_database": True,
        "trust_score": trust_score,
        "reliability_score": entry.get("reliability_score"),
        "bias_rating": entry.get("bias_rating"),
        "category": entry.get("category"),
        "credibility_label": credibility_label,
        "verdict": verdict,
        "notes": entry.get("notes"),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_domain(url_or_domain: str) -> str:
    """
    Extract the root domain from a full URL or bare domain string.

    Examples:
        https://www.bbc.com/news/article  →  bbc.com
        www.reuters.com                   →  reuters.com
        reuters.com                       →  reuters.com
        bbc.co.uk                         →  bbc.co.uk
    """
    raw = url_or_domain.strip()

    # Add scheme if missing so urlparse works correctly
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw

    try:
        parsed = urlparse(raw)
        hostname = parsed.hostname or ""
    except Exception:
        return ""

    # Strip leading www.
    domain = re.sub(r"^www\.", "", hostname).lower()
    return domain


def _lookup(domain: str, db: dict) -> Optional[dict]:
    """
    Look up *domain* in the trust database.

    Tries:
    1. Exact match (e.g. bbc.com)
    2. Subdomain stripping (e.g. news.bbc.com → bbc.com)
    """
    if domain in db:
        return db[domain]

    # Try stripping one subdomain level
    parts = domain.split(".")
    if len(parts) > 2:
        parent = ".".join(parts[-2:])
        if parent in db:
            return db[parent]

    # Try two-level TLD (e.g. bbc.co.uk)
    if len(parts) > 3:
        parent = ".".join(parts[-3:])
        if parent in db:
            return db[parent]

    return None


def _score_to_label(score: float) -> str:
    """Convert a numeric trust score to a credibility label."""
    for label, (low, high) in _LABEL_MAP.items():
        if low <= score <= high:
            return label
    return "UNKNOWN"


def _build_verdict(domain: str, label: str, entry: dict) -> str:
    """Generate a one-line human-readable credibility verdict."""
    category = entry.get("category", "Unknown")
    bias = entry.get("bias_rating", "UNKNOWN")

    verdicts = {
        "HIGH": f"'{domain}' is a generally reliable source ({category}, bias: {bias}).",
        "MEDIUM": f"'{domain}' has mixed reliability. Verify claims independently ({category}, bias: {bias}).",
        "LOW": f"'{domain}' has a poor reliability record. Treat content with scepticism ({category}).",
        "VERY_LOW": f"'{domain}' is rated very low credibility. Known for misinformation or satire ({category}).",
    }
    return verdicts.get(label, f"Credibility of '{domain}' is uncertain.")
