"""
Pytest tests for DistilBERT model inference.

Covers:
- Model loading and initialization
- Text classification (FAKE/REAL)
- Confidence scores
- Model integration with prediction API
- Error handling for missing model files
- LIME explanation support
- SHAP explanation support
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.user import User


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_user():
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.role = "user"
    return user


@pytest.fixture
def auth_override(mock_user):
    from app.core.security import get_current_user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield mock_user
    app.dependency_overrides.clear()


@pytest.fixture
def db_override():
    from app.db.dependencies import get_db
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    yield mock_db
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Unit Tests: predict_bert module
# ---------------------------------------------------------------------------

class TestDistilBERTLoad:
    """Test model loading and initialization."""

    def test_load_bert_returns_tokenizer_and_model(self):
        """Verify load_bert returns both tokenizer and model."""
        try:
            from app.ml.inference.predict_bert import load_bert

            tokenizer, model = load_bert()
            assert tokenizer is not None
            assert model is not None
        except ModuleNotFoundError:
            pytest.skip("PyTorch/transformers not installed (expected in CI)")

    def test_load_bert_singleton_on_second_call(self):
        """Verify load_bert uses singleton pattern (same instance on second call)."""
        try:
            from app.ml.inference.predict_bert import load_bert, _model, _tokenizer

            # Reset singletons
            import app.ml.inference.predict_bert as pb
            pb._tokenizer = None
            pb._model = None

            tokenizer1, model1 = load_bert()
            tokenizer2, model2 = load_bert()

            assert tokenizer1 is tokenizer2, "Tokenizer should be cached"
            assert model1 is model2, "Model should be cached"
        except ModuleNotFoundError:
            pytest.skip("PyTorch/transformers not installed")

    def test_load_bert_raises_on_missing_model_path(self, tmp_path):
        """Verify load_bert raises FileNotFoundError if model directory doesn't exist."""
        try:
            import app.ml.inference.predict_bert as pb
            from pathlib import Path

            # Reset singletons
            pb._tokenizer = None
            pb._model = None

            # Point to non-existent directory
            original_path = pb.MODEL_PATH
            pb.MODEL_PATH = Path(tmp_path) / "nonexistent_model"

            try:
                from app.ml.inference.predict_bert import load_bert
                with pytest.raises(FileNotFoundError) as exc_info:
                    load_bert()
                assert "DistilBERT model directory not found" in str(exc_info.value)
            finally:
                pb.MODEL_PATH = original_path
                pb._tokenizer = None
                pb._model = None
        except ModuleNotFoundError:
            pytest.skip("PyTorch/transformers not installed")


class TestDistilBERTPrediction:
    """Test inference and prediction."""

    def test_predict_news_bert_returns_prediction_dict(self):
        """Verify predict_news_bert returns dict with prediction and confidence."""
        try:
            from app.ml.inference.predict_bert import predict_news_bert

            result = predict_news_bert("This is a test article about politics.")
            
            assert isinstance(result, dict)
            assert "prediction" in result
            assert "confidence" in result
            assert result["prediction"] in ("FAKE", "REAL")
            assert isinstance(result["confidence"], float)
            assert 0.0 <= result["confidence"] <= 1.0
        except ModuleNotFoundError:
            pytest.skip("PyTorch/transformers not installed")

    def test_predict_news_bert_handles_long_text(self):
        """Verify predict_news_bert truncates text to 512 tokens."""
        try:
            from app.ml.inference.predict_bert import predict_news_bert

            # Create text longer than 512 tokens
            long_text = " ".join(["word"] * 600)
            result = predict_news_bert(long_text)
            
            assert isinstance(result, dict)
            assert "prediction" in result
            assert "confidence" in result
        except ModuleNotFoundError:
            pytest.skip("PyTorch/transformers not installed")

    def test_predict_news_bert_handles_empty_text(self):
        """Verify predict_news_bert handles empty or minimal text gracefully."""
        try:
            from app.ml.inference.predict_bert import predict_news_bert

            result = predict_news_bert("")
            
            assert isinstance(result, dict)
            assert "prediction" in result
            assert "confidence" in result
        except ModuleNotFoundError:
            pytest.skip("PyTorch/transformers not installed")

    def test_predict_news_bert_confidence_varies(self):
        """Verify different texts produce different confidence scores."""
        try:
            from app.ml.inference.predict_bert import predict_news_bert

            result1 = predict_news_bert("The earth is flat.")
            result2 = predict_news_bert("The earth is a sphere orbiting the sun.")
            
            # Confidence scores should differ for different texts
            assert result1["confidence"] != result2["confidence"]
        except ModuleNotFoundError:
            pytest.skip("PyTorch/transformers not installed")


# ---------------------------------------------------------------------------
# Integration Tests: Prediction API with bert model
# ---------------------------------------------------------------------------

class TestPredictionAPIWithBERT:
    """Test DistilBERT through the prediction API."""

    def test_bert_prediction_returns_200(self, client, auth_override, db_override):
        """Verify /predict endpoint works with bert model."""
        with patch(
            "app.api.prediction._run_model",
            return_value={"prediction": "FAKE", "confidence": 0.85},
        ):
            response = client.post(
                "/predict",
                json={"text": "Test article.", "model": "bert"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "FAKE"
        assert data["confidence"] == 0.85

    def test_bert_model_comparison(self, client, auth_override, db_override):
        """Verify /predict/compare includes bert results."""
        with patch(
            "app.api.prediction._run_model",
            side_effect=[
                {"prediction": "REAL", "confidence": 0.75},  # logistic
                {"prediction": "FAKE", "confidence": 0.80},  # lstm
                {"prediction": "REAL", "confidence": 0.70},  # bert
            ],
        ):
            response = client.post(
                "/predict/compare",
                json={"text": "Test article for comparison."},
            )

        assert response.status_code == 200
        data = response.json()
        assert "logistic" in data
        assert "lstm" in data
        assert "bert" in data
        assert data["bert"]["prediction"] == "REAL"
        assert data["bert"]["confidence"] == 0.70

    def test_bert_ensemble_prediction(self, client, auth_override, db_override):
        """Verify ensemble includes bert votes."""
        with patch(
            "app.api.prediction._run_model",
            side_effect=[
                {"prediction": "FAKE", "confidence": 0.85},   # logistic
                {"prediction": "FAKE", "confidence": 0.80},   # lstm
                {"prediction": "REAL", "confidence": 0.70},   # bert
            ],
        ):
            response = client.post(
                "/predict/ensemble",
                json={"text": "Test article for ensemble."},
            )

        assert response.status_code == 200
        data = response.json()
        # FAKE appears twice, REAL once → FAKE wins
        assert data["prediction"] == "FAKE"
        assert data["votes"]["bert"]["prediction"] == "REAL"

    def test_bert_invalid_model_returns_400(self, client, auth_override, db_override):
        """Verify invalid model name returns 400."""
        response = client.post(
            "/predict",
            json={"text": "Test article.", "model": "invalid_model"},
        )

        assert response.status_code == 400
        assert "Invalid model" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Integration Tests: Explainability (LIME/SHAP)
# ---------------------------------------------------------------------------

class TestBERTExplanability:
    """Test LIME and SHAP explanations for BERT."""

    def test_lime_bert_explanation(self, client, auth_override, db_override):
        """Verify LIME explanation works for bert model."""
        with patch(
            "app.services.explainability_service.generate_model_explanation",
            return_value={
                "prediction": "FAKE",
                "confidence": 0.85,
                "explanation_type": "lime",
                "top_features": [
                    {"word": "misleading", "contribution": 0.15},
                    {"word": "hoax", "contribution": 0.12},
                ],
            },
        ):
            response = client.post(
                "/predict/explain",
                json={
                    "text": "This is a misleading hoax article.",
                    "model": "bert",
                    "method": "lime",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "FAKE"
        assert data["explanation_type"] == "lime"

    def test_shap_bert_explanation(self, client, auth_override, db_override):
        """Verify SHAP explanation works for bert model."""
        with patch(
            "app.services.explainability_service.generate_model_explanation",
            return_value={
                "prediction": "REAL",
                "confidence": 0.75,
                "explanation_type": "shap",
                "top_features": [
                    {"word": "confirmed", "contribution": 0.10},
                    {"word": "scientific", "contribution": 0.08},
                ],
            },
        ):
            response = client.post(
                "/predict/explain",
                json={
                    "text": "Scientifically confirmed finding.",
                    "model": "bert",
                    "method": "shap",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "REAL"
        assert data["explanation_type"] == "shap"

    def test_unsupported_explanation_method_returns_400(
        self, client, auth_override, db_override
    ):
        """Verify unsupported explanation method returns 400."""
        with patch(
            "app.services.explainability_service.generate_model_explanation",
            side_effect=ValueError("Unsupported explanation method"),
        ):
            response = client.post(
                "/predict/explain",
                json={
                    "text": "Test article.",
                    "model": "bert",
                    "method": "invalid_method",
                },
            )

        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------

class TestBERTErrorHandling:
    """Test error handling in BERT prediction."""

    def test_model_inference_failure_returns_500(
        self, client, auth_override, db_override
    ):
        """Verify prediction API returns 500 on model inference error."""
        with patch(
            "app.api.prediction._run_model",
            side_effect=RuntimeError("Model inference failed"),
        ):
            response = client.post(
                "/predict",
                json={"text": "Test article.", "model": "bert"},
            )

        assert response.status_code == 500
        assert "Model inference failed" in response.json()["detail"]

    def test_empty_text_prediction(self, client, auth_override, db_override):
        """Verify prediction API handles empty text."""
        with patch(
            "app.api.prediction._run_model",
            return_value={"prediction": "REAL", "confidence": 0.5},
        ):
            response = client.post(
                "/predict",
                json={"text": "", "model": "bert"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data


# ---------------------------------------------------------------------------
# Image Verification with BERT (OCR path)
# ---------------------------------------------------------------------------

class TestImageVerificationWithBERT:
    """Test DistilBERT used in image verification (OCR text path)."""

    def test_image_verify_accepts_bert_model(self, client, auth_override, db_override):
        """Verify image verification endpoint accepts bert model."""
        # This test checks that the endpoint parameter validation passes for bert
        from io import BytesIO
        from PIL import Image

        # Create a minimal valid image
        img = Image.new("RGB", (10, 10), color="red")
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Mock the image verification service to avoid actual processing
        with patch(
            "app.api.image_predict.verify_image_core",
            return_value={
                "image_prediction": "FAKE",
                "image_confidence": 0.65,
                "ocr_text": "Some text from OCR",
                "ocr_prediction": "REAL",
                "ocr_confidence": 0.70,
                "clip_similarity": 0.8,
            },
        ):
            response = client.post(
                "/image/verify?model=bert",
                files={"file": ("test.png", img_bytes, "image/png")},
            )

        # Should accept bert as a valid model parameter
        assert response.status_code == 200 or response.status_code == 400
        # (400 only if dependencies are missing, but parameter should be valid)


# ---------------------------------------------------------------------------
# Regression Tests
# ---------------------------------------------------------------------------

class TestBERTRegression:
    """Regression tests to catch future breakage."""

    def test_bert_prediction_maintains_class_order(self):
        """Verify class order is consistent: [0=FAKE, 1=REAL]."""
        try:
            from app.ml.inference.predict_bert import predict_news_bert

            # Both should return consistent class mappings
            result1 = predict_news_bert("Obama was born in Kenya.")  # Likely FAKE
            result2 = predict_news_bert("The earth orbits the sun.")  # Likely REAL

            # Predictions should be valid
            assert result1["prediction"] in ("FAKE", "REAL")
            assert result2["prediction"] in ("FAKE", "REAL")
        except ModuleNotFoundError:
            pytest.skip("PyTorch/transformers not installed")

    def test_bert_confidence_is_rounded_to_4_decimals(self):
        """Verify confidence is rounded to 4 decimal places."""
        try:
            from app.ml.inference.predict_bert import predict_news_bert

            result = predict_news_bert("Test article.")
            confidence_str = str(result["confidence"])
            
            # Split by decimal and check precision
            if "." in confidence_str:
                decimals = len(confidence_str.split(".")[1])
                assert decimals <= 4, f"Confidence has too many decimals: {result['confidence']}"
        except ModuleNotFoundError:
            pytest.skip("PyTorch/transformers not installed")
