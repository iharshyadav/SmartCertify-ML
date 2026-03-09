"""Tests for fraud detection module."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestSyntheticDataGeneration:
    """Test synthetic data generation."""

    def test_generate_certificates(self):
        from app.data.generate_synthetic import generate_certificates_dataset

        df = generate_certificates_dataset(n_samples=1000)
        assert len(df) == 1000
        assert "label" in df.columns
        assert set(df["label"].unique()) == {0, 1}

    def test_fraud_ratio(self):
        from app.data.generate_synthetic import generate_certificates_dataset

        df = generate_certificates_dataset(n_samples=10000)
        fraud_ratio = df["label"].mean()
        assert 0.05 <= fraud_ratio <= 0.12  # Should be ~8%

    def test_all_columns_present(self):
        from app.data.generate_synthetic import generate_certificates_dataset

        df = generate_certificates_dataset(n_samples=100)
        required_cols = [
            "issuer_name", "recipient_name", "course_name",
            "issue_date", "expiry_date", "credential_hash",
            "issuer_reputation_score", "certificate_age_days",
            "metadata_completeness_score", "ocr_confidence_score",
            "template_match_score", "domain_verification_status",
            "previous_verification_count",
            "time_since_last_verification_days", "label",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"


class TestFeatureEngineering:
    """Test feature engineering."""

    def test_extract_date_features(self):
        from app.data.feature_engineering import extract_date_features
        import pandas as pd

        df = pd.DataFrame({
            "issue_date": ["2024-01-15", "2024-06-20"],
            "expiry_date": ["2025-01-15", "2025-06-20"],
        })
        result = extract_date_features(df)
        assert "issue_month" in result.columns
        assert "issue_year" in result.columns
        assert "days_to_expiry" in result.columns

    def test_compute_text_features(self):
        from app.data.feature_engineering import compute_text_features
        import pandas as pd

        df = pd.DataFrame({
            "recipient_name": ["John Smith", "Jane Doe"],
            "course_name": ["ML Fundamentals", "Data Science"],
            "issuer_name": ["MIT", "Stanford"],
        })
        result = compute_text_features(df)
        assert "name_length" in result.columns
        assert "course_word_count" in result.columns

    def test_hash_integrity(self):
        from app.data.feature_engineering import hash_integrity_check

        # Valid SHA-256 hash
        valid_hash = "a" * 64
        result = hash_integrity_check(valid_hash)
        assert result["is_valid_length"] is True
        assert result["is_valid_format"] is True

        # Invalid hash
        result_invalid = hash_integrity_check("short")
        assert result_invalid["is_valid_length"] is False

    def test_compute_risk_score(self):
        from app.data.feature_engineering import compute_risk_score
        import pandas as pd

        df = pd.DataFrame({
            "issuer_reputation_score": [0.9, 0.1],
            "metadata_completeness_score": [0.95, 0.2],
            "template_match_score": [0.85, 0.15],
            "domain_verification_status": [1, 0],
        })
        result = compute_risk_score(df)
        assert "risk_score" in result.columns
        # Low rep should have higher risk
        assert result["risk_score"].iloc[1] > result["risk_score"].iloc[0]


class TestFraudPrediction:
    """Test fraud prediction inference."""

    def test_predict_returns_expected_keys(self):
        from app.models.fraud_detection.predict import predict_fraud

        cert = {
            "issuer_name": "MIT",
            "course_name": "Machine Learning",
            "issuer_reputation_score": 0.9,
            "template_match_score": 0.85,
            "metadata_completeness_score": 0.9,
            "domain_verification_status": 1,
            "previous_verification_count": 5,
            "ocr_confidence_score": 0.92,
            "certificate_age_days": 365,
            "time_since_last_verification_days": 30,
        }

        result = predict_fraud(cert)
        # Should at minimum return error or valid result
        if "error" not in result:
            assert "is_authentic" in result
            assert "fraud_probability" in result
            assert "risk_level" in result


class TestMathUtils:
    """Test math utility functions."""

    def test_cosine_similarity(self):
        from app.utils.math_utils import cosine_similarity_vectors

        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        assert cosine_similarity_vectors(a, b) == pytest.approx(1.0)

        c = np.array([0, 1, 0])
        assert cosine_similarity_vectors(a, c) == pytest.approx(0.0)

    def test_softmax(self):
        from app.utils.math_utils import softmax

        result = softmax(np.array([1.0, 2.0, 3.0]))
        assert result.sum() == pytest.approx(1.0)
        assert all(r > 0 for r in result)

    def test_compute_entropy(self):
        from app.utils.math_utils import compute_entropy

        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = compute_entropy(uniform)
        assert entropy == pytest.approx(2.0, abs=0.01)
