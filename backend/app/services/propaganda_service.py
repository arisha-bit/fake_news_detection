"""
Propaganda Detection Service — Phase 7.

Detects persuasion and manipulation techniques commonly found in
propaganda, misinformation, and biased news articles.

Techniques detected:
    1.  Fear Appeal          — language designed to provoke fear
    2.  Clickbait            — sensational/exaggerated hooks
    3.  Loaded Language      — emotionally charged / value-laden words
    4.  Conspiracy Framing   — "they don't want you to know" patterns
    5.  Emotional Manipulation — direct appeals to anger, disgust, pride
    6.  Bandwagon            — "everyone believes / is doing" appeals
    7.  False Dilemma        — binary either/or framing
    8.  Name Calling         — personal attacks instead of arguments
    9.  Glittering Generalities — vague positive buzzwords
    10. Repetition           — same key phrase repeated for emphasis

Design:
    - Entirely rule-based: pattern dictionaries + regex matching.
    - No external ML model — fast, explainable, offline.
    - Each technique has a phrase list, a regex set, and a description.
    - Confidence scales with the number of matched phrases relative to
      a saturation threshold (more matches = higher confidence, up to 1.0).
    - overall_score = weighted average of per-technique confidences.
    - Easy to extend: add a new entry to _TECHNIQUES.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Technique definitions
# ---------------------------------------------------------------------------

@dataclass
class _TechniqueSpec:
    name: str
    description: str
    patterns: list[str]          # regex patterns (case-insensitive)
    saturation: int = 3          # matches needed to reach confidence=1.0
    weight: float = 1.0          # contribution to overall_score


_TECHNIQUES: list[_TechniqueSpec] = [
    _TechniqueSpec(
        name="Fear Appeal",
        description=(
            "Uses threatening or alarming language to provoke fear "
            "and override rational thinking."
        ),
        patterns=[
            r"\b(danger|dangerous|threat|threaten|terrif|terror|horrify|horrific|"
            r"catastroph|devastat|deadly|lethal|kill|destroy|collapse|crisis|"
            r"emergency|alarm|panic|chaos|doomed|extinction|apocalyp)\w*\b",
            r"\byou (will|could|might|may) (lose|die|suffer|face|be)\b",
            r"\bif (we|you|they) don'?t\b",
        ],
        saturation=4,
        weight=1.2,
    ),
    _TechniqueSpec(
        name="Clickbait",
        description=(
            "Uses sensational, exaggerated, or curiosity-gap language "
            "to bait readers into clicking."
        ),
        patterns=[
            r"\byou won'?t believe\b",
            r"\bshocking(ly)?\b",
            r"\bmiracle\b",
            r"\bsecret(s)?\b",
            r"\bexposed?\b",
            r"\btruth revealed\b",
            r"\bbreaking\b",
            r"\bunbelievable\b",
            r"\binstant(ly)?\b",
            r"\bguaranteed?\b",
            r"\bwhat (doctors|experts|scientists|they) (don'?t|won'?t) tell\b",
            r"\bthis (one|simple) trick\b",
        ],
        saturation=3,
        weight=1.0,
    ),
    _TechniqueSpec(
        name="Loaded Language",
        description=(
            "Uses emotionally charged words that carry strong "
            "positive or negative connotations to influence opinion."
        ),
        patterns=[
            r"\b(radical|extremist|regime|thug|elite|globalist|puppet|traitor|"
            r"criminal|corrupt|evil|wicked|vile|disgusting|outrage|shameful|"
            r"heroic|patriot|freedom fighter|savior|champion)\w*\b",
            r"\b(deep state|fake news|mainstream media|lamestream|the establishment)\b",
            r"\b(invasion|infestation|plague|vermin|parasite)\b",
        ],
        saturation=4,
        weight=1.1,
    ),
    _TechniqueSpec(
        name="Conspiracy Framing",
        description=(
            "Implies hidden plots, secret agendas, or suppressed information "
            "to undermine trust in official sources."
        ),
        patterns=[
            r"\b(cover.?up|coverup)\b",
            r"\bthey (don'?t|won'?t) want you to (know|see|hear)\b",
            r"\bhidden agenda\b",
            r"\b(wake up|wake.?up|sheeple|sheep)\b",
            r"\bthe (truth|real story) (is|they'?re|about)\b",
            r"\b(planned|orchestrated|staged|false.?flag)\b",
            r"\bnew world order\b",
            r"\billuminati\b",
            r"\bpharma (conspiracy|agenda|profit)\b",
            r"\bsuppressed (by|cure|evidence|data)\b",
        ],
        saturation=2,
        weight=1.5,
    ),
    _TechniqueSpec(
        name="Emotional Manipulation",
        description=(
            "Directly targets the reader's emotions (anger, disgust, pride, pity) "
            "rather than presenting logical arguments."
        ),
        patterns=[
            r"\b(outrage|outrageou|infuriat|enrag|appall|disgust|shame|"
            r"betray|heartbreak|devastating|furious|angry|anger)\w*\b",
            r"\b(we must|we have to|you must|you have to) (act|fight|resist|stop)\b",
            r"\b(how (dare|could) they)\b",
            r"\b(blood on (their|his|her) hands)\b",
            r"\b(our (children|future|nation|country) (is|are) at (stake|risk))\b",
        ],
        saturation=3,
        weight=1.1,
    ),
    _TechniqueSpec(
        name="Bandwagon",
        description=(
            "Encourages following the crowd by implying that 'everyone' "
            "holds a certain belief or is taking a certain action."
        ),
        patterns=[
            r"\beveryone (knows|believes|agrees|is saying|can see)\b",
            r"\bmost (people|americans|experts|scientists) (believe|agree|know|say)\b",
            r"\bthe whole (world|country|nation)\b",
            r"\ball (patriots|true|real|honest) (americans|people|citizens)\b",
            r"\bnobody (believes|trusts|supports)\b",
            r"\bjoin (the|millions of|thousands of)\b",
        ],
        saturation=2,
        weight=0.9,
    ),
    _TechniqueSpec(
        name="False Dilemma",
        description=(
            "Presents only two options when more exist, "
            "forcing an artificial binary choice."
        ),
        patterns=[
            r"\b(either|you'?re either)\b.{0,50}\b(or|or you'?re)\b",
            r"\b(with us or against us)\b",
            r"\b(if you'?re not .{0,30} you'?re)\b",
            r"\b(only two (choices|options|paths|ways))\b",
            r"\b(no other (choice|option|way|alternative))\b",
            r"\b(you (must|have to) choose (between|either))\b",
        ],
        saturation=2,
        weight=1.0,
    ),
    _TechniqueSpec(
        name="Name Calling",
        description=(
            "Attacks a person or group with negative labels "
            "rather than addressing the substance of their argument."
        ),
        patterns=[
            r"\b(idiot|moron|stupid|fool|liar|lunatic|nutjob|nut.?case|"
            r"loser|clown|buffoon|hypocrite|coward|weakling|hack)\w*\b",
            r"\b(so.?called (expert|scientist|journalist|leader|president))\b",
            r"\bself.?proclaimed\b",
        ],
        saturation=2,
        weight=0.8,
    ),
    _TechniqueSpec(
        name="Glittering Generalities",
        description=(
            "Uses vague, highly positive buzzwords that sound appealing "
            "but carry no specific meaning."
        ),
        patterns=[
            r"\b(freedom|liberty|justice|democracy|values|tradition|"
            r"family values|god.?given|natural|organic|pure|wholesome|"
            r"patriot|constitutional|god and country)\b",
            r"\b(the (american|true|real|original) (way|values|spirit|dream))\b",
            r"\b(common sense (tells|shows|proves|says))\b",
        ],
        saturation=4,
        weight=0.7,
    ),
    _TechniqueSpec(
        name="Repetition",
        description=(
            "Repeats a key phrase or claim multiple times to reinforce "
            "it through sheer frequency rather than evidence."
        ),
        patterns=[],   # handled separately via frequency analysis
        saturation=3,
        weight=0.8,
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_propaganda(text: str) -> dict:
    """
    Analyse *text* for propaganda techniques.

    Returns a dict matching PropagandaResponse fields.
    """
    if not text or not text.strip():
        return _empty_result()

    normalised = text.lower()
    results = []

    for spec in _TECHNIQUES:
        if spec.name == "Repetition":
            confidence, matched = _detect_repetition(normalised)
        else:
            confidence, matched = _match_patterns(normalised, spec)

        if confidence > 0:
            results.append({
                "technique": spec.name,
                "confidence": confidence,
                "matched_phrases": matched[:5],  # cap at 5 examples
                "description": spec.description,
                "_weight": spec.weight,
            })

    # Overall score: weighted average of detected techniques
    if results:
        total_weight = sum(r["_weight"] for r in results)
        weighted_sum = sum(r["confidence"] * r["_weight"] for r in results)
        overall = round(min(weighted_sum / total_weight, 1.0), 4)
    else:
        overall = 0.0

    # Strip internal weight key from response
    clean_results = [
        {k: v for k, v in r.items() if k != "_weight"}
        for r in results
    ]

    detected = len(clean_results) > 0
    summary = _build_summary(detected, overall, clean_results)

    logger.info(
        "Propaganda analysis — %d techniques found, overall_score=%.3f",
        len(clean_results),
        overall,
    )

    return {
        "propaganda_detected": detected,
        "overall_score": overall,
        "techniques_found": clean_results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _match_patterns(
    text: str, spec: _TechniqueSpec
) -> tuple[float, list[str]]:
    """
    Run all regex patterns for *spec* against *text*.
    Returns (confidence, matched_phrase_list).
    """
    matched: list[str] = []

    for pattern in spec.patterns:
        try:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                phrase = m.group(0).strip()
                if phrase and phrase not in matched:
                    matched.append(phrase)
        except re.error:
            continue

    if not matched:
        return 0.0, []

    confidence = round(min(len(matched) / spec.saturation, 1.0), 4)
    return confidence, matched


def _detect_repetition(text: str) -> tuple[float, list[str]]:
    """
    Detect significant phrase repetition in *text*.
    Finds 3–6 word n-grams repeated 3+ times.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    n = 4  # 4-gram window

    counts: dict[str, int] = {}
    for i in range(len(words) - n + 1):
        gram = " ".join(words[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1

    # Filter stop-gram noise
    _STOP_GRAMS = {
        "the", "and", "of", "in", "to", "a", "that", "is", "it", "for"
    }
    repeated = [
        (gram, count)
        for gram, count in counts.items()
        if count >= 3 and not any(w in _STOP_GRAMS for w in gram.split())
    ]
    repeated.sort(key=lambda x: -x[1])

    if not repeated:
        return 0.0, []

    matched = [f'"{gram}" (×{count})' for gram, count in repeated[:5]]
    confidence = round(min(len(repeated) / 3, 1.0), 4)
    return confidence, matched


def _build_summary(
    detected: bool, score: float, results: list[dict]
) -> str:
    if not detected:
        return "No significant propaganda techniques detected in this text."

    names = ", ".join(r["technique"] for r in results[:3])
    more = f" and {len(results) - 3} more" if len(results) > 3 else ""
    level = (
        "High" if score >= 0.7
        else "Moderate" if score >= 0.4
        else "Low-level"
    )
    return (
        f"{level} propaganda signals detected. "
        f"Techniques identified: {names}{more}."
    )


def _empty_result() -> dict:
    return {
        "propaganda_detected": False,
        "overall_score": 0.0,
        "techniques_found": [],
        "summary": "No text provided for analysis.",
    }
