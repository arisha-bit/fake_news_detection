import yake

# Initialize the YAKE keyword extractor
keyword_extractor = yake.KeywordExtractor(
    lan="en",
    n=2,
    top=5
)

def extract_keywords(text: str) -> list[str]:
    """Extracts top keywords from the text using YAKE."""
    keywords = keyword_extractor.extract_keywords(text)
    return [keyword for keyword, score in keywords]


# Predefined list of sensational/clickbait phrases
CLICKBAIT_PHRASES = [
    "shocking", "miracle", "secret", "you won't believe", 
    "breaking", "unbelievable", "instant", "guaranteed", 
    "exposed", "truth revealed"
]

def clickbait_score(text: str) -> int:
    """Calculates a score (0-100) based on the presence of clickbait phrases."""
    text = text.lower()
    matches = 0

    for phrase in CLICKBAIT_PHRASES:
        if phrase in text:
            matches += 1

    return min(100, matches * 20)


def generate_explanation(prediction: str, score: int) -> str:
    """Generates a user-friendly explanation based on model prediction and clickbait score."""
    if prediction == "FAKE":
        if score > 50:
            return (
                "The article contains sensational language and "
                "clickbait patterns often associated with fake news."
            )
        else:
            return (
                "The model detected linguistic patterns commonly "
                "found in misleading news articles."
            )
    else:
        # Handles "REAL" news or any other prediction outcome
        return (
            "The article uses neutral language "
            "and resembles authentic news reporting."
        )