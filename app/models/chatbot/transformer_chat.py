"""
SmartCertify ML — Chatbot (Lightweight)
Keyword-matching knowledge base for certificate-related Q&A.
No transformer model — saves ~300MB memory.
"""

import logging
from typing import Dict, Any, List
from difflib import get_close_matches

logger = logging.getLogger(__name__)

# ─── Knowledge Base ──────────────────────────────────────────
KNOWLEDGE_BASE = {
    "verify": "To verify a certificate, submit it through our verification portal or use the /api/ml/verify endpoint with the certificate details.",
    "fraud": "Our fraud detection system uses machine learning to analyze issuer reputation, metadata completeness, template matching, and other signals to detect fraudulent certificates.",
    "blockchain": "SmartCertify stores certificate hashes on the blockchain, providing immutable proof of authenticity that cannot be altered.",
    "trust score": "Trust scores are calculated based on the issuer's history, verification rates, and credential patterns. Scores range from 0 to 100.",
    "tamper": "Certificate tampering is detected by analyzing image properties and comparing against verified templates.",
    "issue": "Certificates are issued by verified institutions through the SmartCertify platform and recorded on the blockchain.",
    "expire": "Certificate expiry depends on the issuing institution's policy. Check the certificate details for the expiration date.",
    "revoke": "A certificate can be revoked by the issuing institution if it was issued in error or the holder no longer meets the requirements.",
    "api": "The SmartCertify ML API provides endpoints for fraud detection, similarity analysis, trust scoring, image analysis, anomaly detection, recommendations, and chat.",
    "help": "I can help with: certificate verification, fraud detection, trust scores, blockchain records, tampering detection, and general platform questions.",
    "status": "You can check a certificate's status by its credential hash or certificate ID through the verification portal.",
    "similarity": "Our similarity engine detects duplicate or near-duplicate certificates using text analysis techniques.",
    "anomaly": "Anomaly detection identifies unusual patterns in certificates that may indicate systematic fraud or errors.",
    "recommend": "Our recommendation system suggests courses and certifications based on your skills and completed courses.",
}

GREETINGS = ["hello", "hi", "hey", "greetings", "good morning", "good evening"]
FAREWELLS = ["bye", "goodbye", "see you", "thanks", "thank you"]


def chat(query: str, session_id: str = "default") -> Dict[str, Any]:
    """Process a chat query and return a response."""
    query_lower = query.lower().strip()

    # Greeting
    if any(g in query_lower for g in GREETINGS):
        return {
            "response": "Hello! I'm the SmartCertify AI assistant. How can I help you with certificate verification today?",
            "confidence": 1.0,
            "source": "greeting",
        }

    # Farewell
    if any(f in query_lower for f in FAREWELLS):
        return {
            "response": "Thank you for using SmartCertify! Feel free to ask anytime.",
            "confidence": 1.0,
            "source": "farewell",
        }

    # Keyword matching
    best_match = None
    best_score = 0

    for keyword, answer in KNOWLEDGE_BASE.items():
        if keyword in query_lower:
            score = len(keyword) / len(query_lower)
            if score > best_score:
                best_score = score
                best_match = answer

    if best_match and best_score > 0.05:
        return {
            "response": best_match,
            "confidence": round(min(best_score * 2, 0.95), 2),
            "source": "knowledge_base",
        }

    # Fuzzy matching
    keywords = list(KNOWLEDGE_BASE.keys())
    words = query_lower.split()
    for word in words:
        matches = get_close_matches(word, keywords, n=1, cutoff=0.6)
        if matches:
            return {
                "response": KNOWLEDGE_BASE[matches[0]],
                "confidence": 0.6,
                "source": "fuzzy_match",
            }

    return {
        "response": "I'm not sure I understand your question. I can help with certificate verification, fraud detection, trust scores, blockchain records, and general SmartCertify queries. Could you rephrase?",
        "confidence": 0.1,
        "source": "default",
    }
