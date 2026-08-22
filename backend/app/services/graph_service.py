
import logging
from collections import Counter, defaultdict

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Entity types to keep — everything else is noise for this use case
_KEPT_TYPES = {"PERSON", "ORG", "GPE", "DATE", "EVENT", "NORP", "FAC", "LOC"}

# Map spaCy labels to our cleaner display types
_TYPE_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "GPE",
    "LOC": "GPE",        # merge location types
    "FAC": "ORG",        # facilities → organisation
    "NORP": "ORG",       # nationalities/groups → organisation
    "DATE": "DATE",
    "EVENT": "EVENT",
}


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

    # Reuse the singleton from claim_service
    from app.services.claim_service import _get_nlp  # noqa: PLC0415

    nlp = _get_nlp()

    # Truncate very long texts
    truncated = text[:50_000]
    if len(text) > 50_000:
        logger.warning("Text truncated to 50,000 chars for graph extraction.")

    logger.info("Extracting knowledge graph from %d chars of text.", len(truncated))

    doc = nlp(truncated)

    # ------------------------------------------------------------------
    # 1. Collect all entity occurrences with their sentence context
    # ------------------------------------------------------------------
    # entity_id → normalised display label
    entity_labels: dict[str, str] = {}
    # entity_id → spaCy type mapped to our clean type
    entity_types: dict[str, str] = {}
    # entity_id → occurrence count
    entity_freq: Counter = Counter()
    # sentence_index → list of entity_ids in that sentence
    sent_entities: dict[int, list[str]] = defaultdict(list)

    for sent_idx, sent in enumerate(doc.sents):
        for ent in sent.ents:
            if ent.label_ not in _KEPT_TYPES:
                continue

            raw = ent.text.strip()
            if not raw or len(raw) < 2:
                continue

            # Normalise: lowercase for ID, title-case for display
            eid = _normalise_id(raw)
            if not eid:
                continue

            entity_labels[eid] = _choose_label(entity_labels.get(eid), raw)
            entity_types[eid] = _TYPE_MAP.get(ent.label_, "OTHER")
            entity_freq[eid] += 1
            sent_entities[sent_idx].append(eid)

    # ------------------------------------------------------------------
    # 2. Filter by min_frequency and cap at max_nodes
    # ------------------------------------------------------------------
    filtered_ids = {
        eid for eid, freq in entity_freq.items()
        if freq >= min_frequency
    }
    # Sort by frequency descending, cap at max_nodes
    top_ids = sorted(filtered_ids, key=lambda e: -entity_freq[e])[:max_nodes]
    top_set = set(top_ids)

    # ------------------------------------------------------------------
    # 3. Build nodes
    # ------------------------------------------------------------------
    nodes = [
        {
            "id": eid,
            "label": entity_labels[eid],
            "type": entity_types[eid],
            "frequency": entity_freq[eid],
        }
        for eid in top_ids
    ]

    # ------------------------------------------------------------------
    # 4. Build edges (co-occurrence)
    # ------------------------------------------------------------------
    edge_weights: Counter = Counter()

    for sent_ents in sent_entities.values():
        # Only entities that passed the filter
        present = [e for e in sent_ents if e in top_set]
        # Deduplicate within sentence
        present = list(dict.fromkeys(present))

        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                key = (min(a, b), max(a, b))
                edge_weights[key] += 1

    edges = [
        {"source": src, "target": tgt, "weight": w}
        for (src, tgt), w in edge_weights.items()
    ]

    # Sort edges by weight descending
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
        len(nodes),
        len(edges),
        total_entities,
    )

    return {
        "total_entities": total_entities,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "entity_counts": dict(type_counts),
        "nodes": nodes,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_id(text: str) -> str:
    """
    Create a stable, lowercase node ID from an entity string.
    Strips punctuation at boundaries and collapses whitespace.
    """
    import re  # noqa: PLC0415

    normalised = re.sub(r"\s+", " ", text).strip().lower()
    normalised = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", normalised)
    return normalised


def _choose_label(existing: str | None, new: str) -> str:
    """
    When the same entity appears multiple times with different capitalisation,
    prefer the title-cased or longer version as the display label.
    """
    if existing is None:
        return new
    # Prefer the longer, more specific version
    if len(new) > len(existing):
        return new
    return existing
