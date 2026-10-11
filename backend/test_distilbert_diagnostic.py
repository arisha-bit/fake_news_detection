#!/usr/bin/env python
"""
DistilBERT diagnostic script.
Run this inside the Docker container to verify DistilBERT is working.
"""

import os
import sys
import logging

# Force CPU mode before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required imports work."""
    logger.info("=" * 60)
    logger.info("Testing imports...")
    logger.info("=" * 60)
    
    try:
        import torch
        logger.info("✓ torch imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import torch: {e}")
        return False
    
    try:
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        logger.info("✓ transformers (DistilBERT classes) imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import from transformers: {e}")
        return False
    
    return True


def test_model_files():
    """Verify model files exist."""
    logger.info("\n" + "=" * 60)
    logger.info("Checking model files...")
    logger.info("=" * 60)
    
    from pathlib import Path
    
    model_dir = Path(__file__).parent / "app" / "ml" / "saved_models" / "distilbert"
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    
    if not model_dir.exists():
        logger.error(f"✗ Model directory does not exist: {model_dir}")
        return False
    
    logger.info(f"Model directory: {model_dir}")
    
    all_exist = True
    for filename in required_files:
        filepath = model_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            logger.info(f"✓ {filename} ({size_kb:.1f} KB)")
        else:
            logger.error(f"✗ {filename} MISSING")
            all_exist = False
    
    return all_exist


def test_model_loading():
    """Test loading the DistilBERT model."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing model loading...")
    logger.info("=" * 60)
    
    try:
        from app.ml.inference.predict_bert import load_bert
        
        logger.info("Loading tokenizer and model...")
        tokenizer, model = load_bert()
        
        logger.info(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")
        logger.info(f"✓ Model loaded: {type(model).__name__}")
        logger.info(f"✓ Model device: {next(model.parameters()).device}")
        logger.info(f"✓ Model training mode: {model.training}")
        
        return tokenizer, model
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}", exc_info=True)
        return None, None


def test_inference(tokenizer, model):
    """Test inference on sample texts."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing inference...")
    logger.info("=" * 60)
    
    if tokenizer is None or model is None:
        logger.error("✗ Cannot test inference: model not loaded")
        return False
    
    try:
        from app.ml.inference.predict_bert import predict_news_bert
        
        test_texts = [
            ("This article contains misleading information and is completely false.", "FAKE"),
            ("Scientists have confirmed the earth is round based on overwhelming evidence.", "REAL"),
            ("Obama was born in Kenya and is not a citizen.", "FAKE"),
            ("The sun rises in the east every day.", "REAL"),
        ]
        
        logger.info(f"Running inference on {len(test_texts)} test texts...")
        for text, expected in test_texts:
            result = predict_news_bert(text)
            match = "✓" if result["prediction"] == expected else "?"
            logger.info(
                f"{match} Text: '{text[:50]}...' "
                f"→ {result['prediction']} (confidence: {result['confidence']:.4f})"
            )
        
        logger.info("✓ Inference completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Inference failed: {e}", exc_info=True)
        return False


def test_prediction_api(text):
    """Test the prediction API endpoint."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing prediction API integration...")
    logger.info("=" * 60)
    
    try:
        from app.api.prediction import _run_model
        
        logger.info(f"Running bert model prediction on: '{text[:60]}...'")
        result = _run_model("bert", text)
        
        logger.info(f"✓ Prediction: {result['prediction']}")
        logger.info(f"✓ Confidence: {result['confidence']:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Prediction API test failed: {e}", exc_info=True)
        return False


def test_ensemble():
    """Test ensemble prediction with all three models."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing ensemble prediction...")
    logger.info("=" * 60)
    
    text = "This is a test article for ensemble prediction."
    
    try:
        from app.api.prediction import _run_model
        
        logger.info(f"Running all three models on: '{text}'")
        
        logistic = _run_model("logistic", text)
        logger.info(f"✓ Logistic: {logistic['prediction']} ({logistic['confidence']:.4f})")
        
        lstm = _run_model("lstm", text)
        logger.info(f"✓ LSTM: {lstm['prediction']} ({lstm['confidence']:.4f})")
        
        bert = _run_model("bert", text)
        logger.info(f"✓ DistilBERT: {bert['prediction']} ({bert['confidence']:.4f})")
        
        votes = [logistic["prediction"], lstm["prediction"], bert["prediction"]]
        final = "FAKE" if votes.count("FAKE") > votes.count("REAL") else "REAL"
        avg_conf = (logistic["confidence"] + lstm["confidence"] + bert["confidence"]) / 3
        
        logger.info(f"✓ Ensemble votes: {votes}")
        logger.info(f"✓ Ensemble prediction: {final} ({avg_conf:.4f})")
        
        return True
    except Exception as e:
        logger.error(f"✗ Ensemble test failed: {e}", exc_info=True)
        return False


def main():
    """Run all diagnostics."""
    logger.info("\n" + "=" * 60)
    logger.info("DISTILBERT DIAGNOSTIC REPORT")
    logger.info("=" * 60)
    
    success_count = 0
    
    # Test 1: Imports
    if test_imports():
        success_count += 1
    
    # Test 2: Model files
    if test_model_files():
        success_count += 1
    
    # Test 3: Model loading
    tokenizer, model = test_model_loading()
    if tokenizer is not None and model is not None:
        success_count += 1
    
    # Test 4: Inference
    if test_inference(tokenizer, model):
        success_count += 1
    
    # Test 5: Prediction API
    if test_prediction_api("Test article about politics."):
        success_count += 1
    
    # Test 6: Ensemble
    if test_ensemble():
        success_count += 1
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"SUMMARY: {success_count}/6 tests passed")
    logger.info("=" * 60)
    
    if success_count == 6:
        logger.info("✓ DistilBERT is fully operational!")
        return 0
    else:
        logger.error(f"✗ DistilBERT diagnostic failed ({6 - success_count} failures)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
