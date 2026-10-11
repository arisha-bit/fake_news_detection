"""
Knowledge Graph Extraction Service.

Extracts named entities and co-occurrence relationships from article text
using spaCy's en_core_web_sm model.

The spaCy singleton is loaded locally in this module (not imported from
claim_service) to keep the two services independent and avoid circular imports.
"""

import logging
import re
from collections import Counter, defaultdict

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Entity types to keep — everything else is noise
_KEPT_TYPES = {"PERSON", "ORG", "GPE", "DATE", "EVENT", "NORP", "FAC", "LOC"}

# Map spaCy labels to our cleaner display types
_TYPE_MAP = {
    "PERSON": "PERSON",
    "ORG":    "ORG",
    "GPE":    "GPE",
    "LOC":    "GPE",    # merge location types
    "FAC":    "ORG",    # facilities → organisation
    "NORP":   "ORG",    # nationalities / groups → organisation
    "DATE":   "DATE",
    "EVENT":  "EVENT",
}

# ---------------------------------------------------------------------------
# spaCy singleton — independent of claim_service
# ---------------------------------------------------------------------------

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy  # noqa: PLC0415

            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found — downloading...")
                from spacy.cli import download  # noqa: PLC0415
                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")

            logger.info("spaCy model loaded for graph_service.")
        except Exception as exc:
            logger.error("Failed to load spaCy model: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Knowledge graph engine failed to initialise. "
                       "Ensure spaCy and en_core_web_sm are installed.",
            ) from exc
    return _nlp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_knowledge_graph(
    text: str,
    min_frequency: int = 1,
    max_nodes: int = 50,
) -> dict:
    """
    Extract a knowledge graph from *text*.

    Args:
        text:          Article or document text.
        min_frequency: Minimum occurrence count for a node to be included.
        max_nodes:     Maximum nodes to return (sorted by frequency desc).

    Returns:
        Dict matching KnowledgeGraphResponse schema.

    Raises:
        HTTP 422 — blank text.
        HTTP 500 — spaCy failure.
    """
    if not text or not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Cannot extract a knowledge graph from empty text.",
        )

    nlp = _get_nlp()

    # Truncate very long texts to avoid memory issues
    is_truncated = False
    truncated = text
    if len(text) > 50_000:
        is_truncated = True
        truncated = text[:50_000]
        logger.warning("Text truncated to 50,000 chars for graph extraction (original: %d chars).", len(text))

    logger.info("Extracting knowledge graph from %d chars of text.", len(truncated))

    doc = nlp(truncated)

    # ------------------------------------------------------------------
    # 1. Collect entity occurrences with sentence context
    # ------------------------------------------------------------------
    entity_labels: dict[str, str] = {}
    entity_types:  dict[str, str] = {}
    entity_freq:   Counter        = Counter()
    sent_entities: dict[int, list[str]] = defaultdict(list)

    for sent_idx, sent in enumerate(doc.sents):
        for ent in sent.ents:
            if ent.label_ not in _KEPT_TYPES:
                continue
            raw = ent.text.strip()
            if not raw or len(raw) < 2:
                continue

            eid = _normalise_id(raw)
            if not eid:
                continue

            entity_labels[eid] = _choose_label(entity_labels.get(eid), raw)
            entity_types[eid]  = _TYPE_MAP.get(ent.label_, "OTHER")
            entity_freq[eid]  += 1
            sent_entities[sent_idx].append(eid)

    # ------------------------------------------------------------------
    # 2. Filter by min_frequency and cap at max_nodes
    # ------------------------------------------------------------------
    filtered_ids = {
        eid for eid, freq in entity_freq.items() if freq >= min_frequency
    }
    top_ids = sorted(filtered_ids, key=lambda e: -entity_freq[e])[:max_nodes]
    top_set = set(top_ids)

    # ------------------------------------------------------------------
    # 3. Build nodes
    # ------------------------------------------------------------------
    nodes = [
        {
            "id":        eid,
            "label":     entity_labels[eid],
            "type":      entity_types[eid],
            "frequency": entity_freq[eid],
        }
        for eid in top_ids
    ]

    # ------------------------------------------------------------------
    # 4. Build edges (co-occurrence within same sentence)
    # ------------------------------------------------------------------
    edge_weights: Counter = Counter()

    for sent_ents in sent_entities.values():
        present = [e for e in sent_ents if e in top_set]
        present = list(dict.fromkeys(present))  # deduplicate, preserve order

        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                key = (min(a, b), max(a, b))
                edge_weights[key] += 1

    edges = [
        {"source": src, "target": tgt, "weight": w}
        for (src, tgt), w in edge_weights.items()
    ]
    edges.sort(key=lambda e: -e["weight"])

    # ------------------------------------------------------------------
    # 5. Entity type counts
    # ------------------------------------------------------------------
    type_counts: Counter = Counter()
    for eid in top_ids:
        type_counts[entity_types[eid]] += 1

    total_entities = sum(entity_freq[eid] for eid in top_ids)

    logger.info(
        "Knowledge graph extracted — %d nodes, %d edges, %d entity occurrences.",
        len(nodes), len(edges), total_entities,
    )

    return {
        "total_entities": total_entities,
        "total_nodes":    len(nodes),
        "total_edges":    len(edges),
        "entity_counts":  dict(type_counts),
        "nodes":          nodes,
        "edges":          edges,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_id(text: str) -> str:
    """Create a stable lowercase node ID from an entity string.
    
    Normalization steps:
    1. Collapse multiple spaces to single space
    2. Convert to lowercase
    3. Remove punctuation, keeping alphanumerics and spaces
    4. Strip leading/trailing spaces
    """
    # Step 1-2: Collapse spaces and lowercase
    normalised = re.sub(r"\s+", " ", text).strip().lower()
    
    # Step 3: Remove punctuation but keep alphanumerics and spaces
    # This replaces all non-alphanumeric, non-space characters with nothing
    normalised = re.sub(r"[^a-z0-9\s]+", "", normalised)
    
    # Step 4: Collapse any spaces created by punctuation removal
    normalised = re.sub(r"\s+", " ", normalised).strip()
    
    return normalised


def _choose_label(existing: str | None, new: str) -> str:
    """Prefer the longer, more specific label when the same entity appears multiple times."""
    if existing is None:
        return new
    return new if len(new) > len(existing) else existing
