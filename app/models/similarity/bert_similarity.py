"""
SmartCertify ML — BERT Semantic Similarity
Semantic similarity detection using sentence-transformers.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import BERT_MODEL_NAME, SIMILARITY_DUPLICATE_THRESHOLD

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Load and cache the sentence-transformers model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(BERT_MODEL_NAME)
            logger.info(f"Loaded sentence-transformers model: {BERT_MODEL_NAME}")
        except ImportError:
            logger.warning("sentence-transformers not installed")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
    return _model


def _combine_certificate_text(cert: Dict[str, Any]) -> str:
    """Combine certificate fields into text for embedding."""
    parts = []
    for field in ["issuer_name", "recipient_name", "course_name"]:
        if field in cert and cert[field]:
            parts.append(str(cert[field]))
    return " ".join(parts)


def compute_semantic_similarity(
    cert_a: Dict[str, Any],
    cert_b: Dict[str, Any],
    threshold: float = SIMILARITY_DUPLICATE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compute BERT semantic similarity between two certificates.

    Args:
        cert_a: First certificate dict
        cert_b: Second certificate dict
        threshold: Similarity threshold for flagging

    Returns:
        Dictionary with semantic similarity score
    """
    model = _get_model()

    if model is None:
        return {
            "similarity_score": 0.0,
            "is_duplicate": False,
            "method": "bert",
            "error": "BERT model not available",
        }

    text_a = _combine_certificate_text(cert_a)
    text_b = _combine_certificate_text(cert_b)

    # Encode texts to embeddings
    embeddings = model.encode([text_a, text_b], convert_to_numpy=True)

    # Compute cosine similarity
    from numpy.linalg import norm
    similarity = float(
        np.dot(embeddings[0], embeddings[1])
        / (norm(embeddings[0]) * norm(embeddings[1]) + 1e-8)
    )

    return {
        "similarity_score": round(similarity, 4),
        "is_duplicate": similarity > threshold,
        "method": "bert",
        "threshold_used": threshold,
        "text_a_preview": text_a[:100],
        "text_b_preview": text_b[:100],
    }


def compute_batch_embeddings(certificates: list) -> np.ndarray:
    """Compute BERT embeddings for a batch of certificates."""
    model = _get_model()
    if model is None:
        return np.array([])

    texts = [_combine_certificate_text(cert) for cert in certificates]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def find_semantic_duplicates(
    certificates: list,
    threshold: float = SIMILARITY_DUPLICATE_THRESHOLD,
) -> list:
    """Find semantically similar certificates using BERT embeddings."""
    embeddings = compute_batch_embeddings(certificates)
    if len(embeddings) == 0:
        return []

    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)

    duplicates = []
    for i in range(len(certificates)):
        for j in range(i + 1, len(certificates)):
            if sim_matrix[i, j] > threshold:
                duplicates.append({
                    "cert_a_index": i,
                    "cert_b_index": j,
                    "similarity_score": round(float(sim_matrix[i, j]), 4),
                })

    return sorted(duplicates, key=lambda x: x["similarity_score"], reverse=True)
