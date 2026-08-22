import yake


# ============================================================
# YAKE KEYWORD EXTRACTION
# ============================================================

keyword_extractor = yake.KeywordExtractor(
    lan="en",
    n=2,
    top=5
)


def extract_keywords(text: str) -> list[str]:
    """
    Extract important keywords/keyphrases using YAKE.
    """
    keywords = keyword_extractor.extract_keywords(text)

    return [
        keyword
        for keyword, score in keywords
    ]


# ============================================================
# CLICKBAIT DETECTION
# ============================================================

CLICKBAIT_PHRASES = [
    "shocking",
    "miracle",
    "secret",
    "you won't believe",
    "breaking",
    "unbelievable",
    "instant",
    "guaranteed",
    "exposed",
    "truth revealed"
]


def clickbait_score(text: str) -> int:
    """
    Calculates a simple 0-100 clickbait score.
    Each matched phrase contributes 20 points.
    """

    text = text.lower()

    matches = 0

    for phrase in CLICKBAIT_PHRASES:

        if phrase in text:
            matches += 1

    return min(
        100,
        matches * 20
    )


# ============================================================
# COMBINED TEXT EXPLANATION
# ============================================================

def generate_text_explanation(
    text: str,
    prediction: str,
    shap_explanations: list
):
    """
    Combines:
    - SHAP model explanation
    - YAKE keywords
    - Clickbait analysis
    """

    # --------------------------------------------
    # YAKE
    # --------------------------------------------

    keywords = extract_keywords(text)

    # --------------------------------------------
    # Clickbait
    # --------------------------------------------

    clickbait = clickbait_score(text)

    # --------------------------------------------
    # Human-readable explanation
    # --------------------------------------------

    fake_words = [
        item["word"]
        for item in shap_explanations
        if item["direction"] == "FAKE"
    ]

    real_words = [
        item["word"]
        for item in shap_explanations
        if item["direction"] == "REAL"
    ]

    if prediction == "FAKE":

        if clickbait > 50:
            explanation = (
                "The model detected features associated "
                "with misleading news, and the article "
                "also contains significant sensational "
                "or clickbait language."
            )

        elif fake_words:
            explanation = (
                "The model identified linguistic features "
                "that contributed toward the FAKE prediction."
            )

        else:
            explanation = (
                "The model detected linguistic patterns "
                "commonly associated with misleading news."
            )

    else:

        explanation = (
            "The model detected linguistic patterns "
            "that are more consistent with authentic "
            "news reporting."
        )

    return {
        "prediction": prediction,

        "shap": {
            "fake_features": fake_words,
            "real_features": real_words,
            "top_features": shap_explanations
        },

        "keywords": keywords,

        "clickbait": {
            "score": clickbait
        },

        "explanation": explanation
    }