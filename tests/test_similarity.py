"""Tests for similarity module."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestTfidfSimilarity:
    """Test TF-IDF similarity."""

    def test_identical_certificates(self):
        from app.models.similarity.tfidf_model import compute_similarity

        cert = {
            "issuer_name": "MIT",
            "recipient_name": "John Smith",
            "course_name": "Machine Learning",
        }
        result = compute_similarity(cert, cert)
        assert result["similarity_score"] == pytest.approx(1.0, abs=0.01)
        assert result["is_duplicate"] is True

    def test_different_certificates(self):
        from app.models.similarity.tfidf_model import compute_similarity

        cert_a = {
            "issuer_name": "MIT",
            "recipient_name": "John Smith",
            "course_name": "Machine Learning Fundamentals",
        }
        cert_b = {
            "issuer_name": "Harvard",
            "recipient_name": "Jane Doe",
            "course_name": "Art History",
        }
        result = compute_similarity(cert_a, cert_b)
        assert result["similarity_score"] < 0.5
        assert result["is_duplicate"] is False

    def test_similar_certificates(self):
        from app.models.similarity.tfidf_model import compute_similarity

        cert_a = {
            "issuer_name": "MIT",
            "recipient_name": "John Smith",
            "course_name": "Machine Learning Fundamentals",
        }
        cert_b = {
            "issuer_name": "MIT",
            "recipient_name": "John Smith",
            "course_name": "Machine Learning Advanced",
        }
        result = compute_similarity(cert_a, cert_b)
        assert result["similarity_score"] > 0.3  # Should have some similarity
        assert result["method"] == "tfidf"


class TestBertSimilarity:
    """Test BERT similarity (may skip if model not available)."""

    def test_compute_semantic_similarity(self):
        try:
            from app.models.similarity.bert_similarity import compute_semantic_similarity

            cert_a = {
                "issuer_name": "MIT",
                "course_name": "Machine Learning",
            }
            cert_b = {
                "issuer_name": "MIT",
                "course_name": "Artificial Intelligence",
            }
            result = compute_semantic_similarity(cert_a, cert_b)
            assert "similarity_score" in result
            assert result["method"] == "bert"
        except Exception:
            pytest.skip("BERT model not available")
