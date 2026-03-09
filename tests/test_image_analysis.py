"""Tests for image analysis module."""

import pytest
import numpy as np
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestImagePreprocessing:
    """Test image preprocessing utilities."""

    def test_preprocess_image(self):
        from app.models.image_analysis.preprocess import preprocess_image

        img = Image.new("RGB", (400, 300), color=(255, 255, 245))
        result = preprocess_image(img)
        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

    def test_base64_roundtrip(self):
        from app.models.image_analysis.preprocess import (
            image_to_base64, load_image_from_base64,
        )

        original = Image.new("RGB", (100, 100), color=(128, 128, 128))
        b64 = image_to_base64(original)
        assert isinstance(b64, str)
        assert len(b64) > 0

        restored = load_image_from_base64(b64)
        assert restored is not None
        assert restored.size == (100, 100)

    def test_generate_synthetic_tampered(self):
        from app.models.image_analysis.preprocess import generate_synthetic_tampered_images

        samples = generate_synthetic_tampered_images(n_samples=5)
        assert len(samples) == 5
        for s in samples:
            assert s["authentic"].shape == (3, 224, 224)
            assert s["tampered"].shape == (3, 224, 224)
            assert s["tampering_type"] in [
                "pixel_alter", "text_overlay", "compression", "blur", "crop_paste"
            ]


class TestCNNModel:
    """Test CNN model structure."""

    def test_model_forward_pass(self):
        import torch
        from app.models.image_analysis.cnn_model import TamperingDetectorCNN

        model = TamperingDetectorCNN(num_classes=2)
        model.eval()

        # Random input
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 2)

    def test_analyze_image_function(self):
        from app.models.image_analysis.cnn_model import analyze_image

        img = Image.new("RGB", (300, 200), color=(255, 255, 245))
        result = analyze_image(img, certificate_id="test-123")

        assert "is_tampered" in result
        assert "tamper_probability" in result
        assert "confidence" in result
        assert result["certificate_id"] == "test-123"


class TestAnomalyDetection:
    """Test anomaly detection."""

    def test_detect_anomaly_structure(self):
        from app.models.anomaly.isolation_forest import detect_anomaly

        cert_data = {
            "issuer_reputation_score": 0.1,
            "certificate_age_days": 10,
            "metadata_completeness_score": 0.2,
            "ocr_confidence_score": 0.5,
            "template_match_score": 0.1,
            "domain_verification_status": 0,
            "previous_verification_count": 0,
            "time_since_last_verification_days": 300,
        }

        result = detect_anomaly(cert_data)
        if "error" not in result:
            assert "is_anomaly" in result
            assert "anomaly_score" in result
            assert "anomaly_rank" in result
            assert result["anomaly_rank"] in ["LOW", "MEDIUM", "HIGH"]
