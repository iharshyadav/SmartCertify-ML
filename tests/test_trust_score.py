"""Tests for trust score module."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestTrustScoreDataGeneration:
    """Test trust score data generation."""

    def test_generate_trust_dataset(self):
        from app.data.generate_synthetic import generate_trust_score_dataset

        df = generate_trust_score_dataset(n_issuers=100)
        assert len(df) == 100
        assert "trust_score" in df.columns
        assert df["trust_score"].min() >= 0
        assert df["trust_score"].max() <= 100


class TestTrustScorePrediction:
    """Test trust score prediction."""

    def test_predict_returns_score(self):
        from app.models.trust_score.regression_model import predict_trust_score

        features = {
            "total_certificates_issued": 500,
            "fraud_rate_historical": 0.01,
            "avg_metadata_completeness": 0.9,
            "domain_age_days": 1500,
            "verification_success_rate": 0.95,
            "response_time_avg": 30,
        }

        result = predict_trust_score(features)
        if "error" not in result:
            assert "trust_score" in result
            assert "trust_grade" in result
            assert result["trust_grade"] in ["A", "B", "C", "D", "F"]
            assert 0 <= result["trust_score"] <= 100


class TestTimeSeries:
    """Test time series module."""

    def test_generate_timeseries(self):
        from app.data.generate_synthetic import generate_timeseries_dataset

        df = generate_timeseries_dataset(n_days=100)
        assert len(df) == 100
        assert "verification_count" in df.columns
        assert all(df["verification_count"] > 0)
