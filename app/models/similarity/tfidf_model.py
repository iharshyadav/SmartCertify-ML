"""
SmartCertify ML — TF-IDF Similarity Model
Certificate similarity detection using TF-IDF vectorization and cosine similarity.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import MODEL_DIR, SIMILARITY_DUPLICATE_THRESHOLD
from app.utils.model_io import save_sklearn_model, load_sklearn_model

logger = logging.getLogger(__name__)

_vectorizer = None


def _get_vectorizer() -> TfidfVectorizer:
    """Load or create the TF-IDF vectorizer."""
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = load_sklearn_model("tfidf_vectorizer.joblib")
        if _vectorizer is None:
            _vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words="english",
                lowercase=True,
            )
    return _vectorizer


def _combine_certificate_fields(cert: Dict[str, Any]) -> str:
    """Combine certificate fields into a single text string."""
    parts = []
    for field in ["issuer_name", "recipient_name", "course_name"]:
        if field in cert and cert[field]:
            parts.append(str(cert[field]))
    return " ".join(parts)


def fit_vectorizer(certificates: list) -> TfidfVectorizer:
    """Fit the TF-IDF vectorizer on a collection of certificates."""
    texts = [_combine_certificate_fields(cert) for cert in certificates]
    vectorizer = _get_vectorizer()
    vectorizer.fit(texts)
    save_sklearn_model(vectorizer, "tfidf_vectorizer.joblib")
    logger.info(f"Fitted TF-IDF vectorizer on {len(texts)} certificates")
    return vectorizer


def compute_similarity(
    cert_a: Dict[str, Any],
    cert_b: Dict[str, Any],
    threshold: float = SIMILARITY_DUPLICATE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compute TF-IDF cosine similarity between two certificates.

    Args:
        cert_a: First certificate dict
        cert_b: Second certificate dict
        threshold: Similarity threshold for duplicate flagging

    Returns:
        Dictionary with similarity score and duplicate flag
    """
    text_a = _combine_certificate_fields(cert_a)
    text_b = _combine_certificate_fields(cert_b)

    vectorizer = _get_vectorizer()

    # If vectorizer isn't fitted, fit on these texts
    try:
        vectors = vectorizer.transform([text_a, text_b])
    except Exception:
        vectorizer.fit([text_a, text_b])
        vectors = vectorizer.transform([text_a, text_b])

    similarity_score = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])

    return {
        "similarity_score": round(similarity_score, 4),
        "is_duplicate": similarity_score > threshold,
        "method": "tfidf",
        "threshold_used": threshold,
        "text_a_preview": text_a[:100],
        "text_b_preview": text_b[:100],
    }


def find_duplicates(
    certificates: list,
    threshold: float = SIMILARITY_DUPLICATE_THRESHOLD,
) -> list:
    """Find potential duplicate certificates in a collection."""
    texts = [_combine_certificate_fields(cert) for cert in certificates]
    vectorizer = _get_vectorizer()

    try:
        vectors = vectorizer.transform(texts)
    except Exception:
        vectorizer.fit(texts)
        vectors = vectorizer.transform(texts)

    sim_matrix = cosine_similarity(vectors)
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
