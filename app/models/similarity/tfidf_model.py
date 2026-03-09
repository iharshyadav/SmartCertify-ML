"""
SmartCertify ML — Similarity Analysis (Lightweight)
TF-IDF based certificate similarity detection (no BERT).
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_tfidf_vectorizer = None
_tfidf_matrix = None
_corpus = []


def _build_text(cert: Dict[str, Any]) -> str:
    """Build a text representation from certificate data."""
    parts = []
    for key in ["issuer_name", "course_name", "recipient_name", "credential_hash"]:
        if key in cert and cert[key]:
            parts.append(str(cert[key]))
    return " ".join(parts) if parts else "unknown"


def find_similar(
    certificate: Dict[str, Any],
    corpus: List[Dict[str, Any]],
    top_n: int = 5,
    threshold: float = 0.5,
    method: str = "tfidf",
) -> Dict[str, Any]:
    """Find similar certificates using TF-IDF cosine similarity."""

    cert_text = _build_text(certificate)
    corpus_texts = [_build_text(c) for c in corpus]

    if not corpus_texts:
        return {"similar_certificates": [], "method": "tfidf"}

    all_texts = [cert_text] + corpus_texts

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    results = []
    indices = np.argsort(similarities)[::-1][:top_n]

    for idx in indices:
        score = float(similarities[idx])
        if score >= threshold:
            results.append({
                "index": int(idx),
                "similarity_score": round(score, 4),
                "is_duplicate": score > 0.9,
                "certificate": corpus[idx] if idx < len(corpus) else {},
            })

    return {
        "query_certificate": certificate,
        "similar_certificates": results,
        "total_compared": len(corpus),
        "method": "tfidf",
    }
