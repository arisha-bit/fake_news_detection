from app.ml.explainability.shap_explainer import (
    explain_prediction
)

from app.ml.explainability.text_explainer import (
    generate_text_explanation
)


def explain_text(
    text: str,
    top_k: int = 10
):
    """
    Generate the complete text explanation.

    Uses:
    - Logistic Regression prediction explanation with SHAP
    - YAKE keywords
    - Clickbait analysis
    """

    # --------------------------------------------
    # SHAP
    # --------------------------------------------

    shap_result = explain_prediction(
        text,
        top_k=top_k
    )

    prediction = shap_result[
        "prediction"
    ]

    shap_explanations = shap_result[
        "explanations"
    ]

    # --------------------------------------------
    # Combine everything
    # --------------------------------------------

    explanation = generate_text_explanation(
        text=text,
        prediction=prediction,
        shap_explanations=shap_explanations
    )

    return explanation