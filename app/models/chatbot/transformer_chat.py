"""
SmartCertify ML — Transformer-based AI Chatbot
Question-answering chatbot for certificate-related queries.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

# ─── Knowledge Base ───────────────────────────────────────────

KNOWLEDGE_BASE = {
    "verification": {
        "keywords": ["verify", "verification", "check", "validate", "authentic", "real", "legit"],
        "responses": [
            "To verify a certificate, you can use our ML-powered verification endpoint. "
            "It analyzes multiple features including issuer reputation, template matching, "
            "metadata completeness, and blockchain hash verification to determine authenticity.",
            "Our system uses an ensemble of 7 machine learning models including Random Forest, "
            "XGBoost, and Neural Networks to detect fraudulent certificates with over 90% accuracy.",
        ],
    },
    "fraud": {
        "keywords": ["fraud", "fake", "fraudulent", "scam", "counterfeit", "forged"],
        "responses": [
            "Our fraud detection system checks for multiple red flags: low issuer reputation, "
            "mismatched templates, incomplete metadata, unverified domains, and suspicious hash patterns. "
            "The system provides a risk score and detailed explanation for each analysis.",
            "Common fraud indicators include: future issue dates, zero verification history, "
            "hash integrity failures, and low template match scores.",
        ],
    },
    "trust_score": {
        "keywords": ["trust", "score", "reputation", "issuer", "institution", "reliable"],
        "responses": [
            "The issuer trust score is calculated based on: historical fraud rate, metadata completeness, "
            "domain age, verification success rate, and response time. Scores range from 0-100 with "
            "grades A through F.",
            "Trust scores are updated continuously as new verification data comes in. "
            "Grade A (90-100) indicates highly trustworthy issuers with excellent track records.",
        ],
    },
    "similarity": {
        "keywords": ["similar", "duplicate", "copy", "match", "identical", "same"],
        "responses": [
            "We use both TF-IDF and BERT semantic similarity to detect duplicate certificates. "
            "Certificates with similarity scores above 0.85 are flagged as potential duplicates.",
            "Our BERT model can detect semantically similar content even when wording is different, "
            "catching rephrased or paraphrased certificate fields.",
        ],
    },
    "blockchain": {
        "keywords": ["blockchain", "hash", "chain", "block", "crypto", "decentralized"],
        "responses": [
            "SmartCertify uses blockchain technology to create immutable records of certificates. "
            "Each certificate's hash is stored on-chain, making tampering detectable.",
            "The credential hash is a SHA-256 digest of the certificate data. We validate "
            "hash format, length (64 hex characters), and entropy to detect alterations.",
        ],
    },
    "general": {
        "keywords": [],
        "responses": [
            "I'm the SmartCertify AI assistant. I can help with certificate verification, "
            "fraud detection, trust scoring, and general questions about the platform. "
            "How can I assist you?",
        ],
    },
}


def _keyword_match(message: str) -> str:
    """Find the best matching topic based on keywords."""
    message_lower = message.lower()
    best_topic = "general"
    best_score = 0

    for topic, data in KNOWLEDGE_BASE.items():
        score = sum(1 for kw in data["keywords"] if kw in message_lower)
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic


def _get_transformer_model():
    """Load transformer model (lazy loaded)."""
    global _model, _tokenizer
    if _model is None:
        try:
            from transformers import pipeline
            _model = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_length=256,
            )
            logger.info("Loaded Flan-T5 model for chatbot")
        except ImportError:
            logger.warning("transformers not installed, using keyword-based responses")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
    return _model


def generate_response(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    use_transformer: bool = False,
) -> Dict[str, Any]:
    """
    Generate a chatbot response.

    Args:
        message: User message
        context: Optional certificate context data
        use_transformer: Whether to use transformer model (slow) or keyword matching (fast)

    Returns:
        Dictionary with response text and sources
    """
    sources = []

    # Try transformer-based response
    if use_transformer:
        model = _get_transformer_model()
        if model is not None:
            try:
                # Build prompt with context
                prompt_parts = [f"Answer the following question about certificate verification: {message}"]

                if context and "certificate_data" in context:
                    cert = context["certificate_data"]
                    prompt_parts.append(f"\nCertificate info: Issuer: {cert.get('issuer_name', 'unknown')}, "
                                       f"Course: {cert.get('course_name', 'unknown')}, "
                                       f"Reputation: {cert.get('issuer_reputation_score', 'unknown')}")

                prompt = " ".join(prompt_parts)
                result = model(prompt)[0]["generated_text"]

                return {
                    "response": result,
                    "sources": ["transformer_model"],
                    "method": "transformer",
                }
            except Exception as e:
                logger.warning(f"Transformer generation failed: {e}")

    # Keyword-based response (fast, reliable)
    topic = _keyword_match(message)
    responses = KNOWLEDGE_BASE[topic]["responses"]

    import numpy as np
    response = np.random.choice(responses)

    # Add context-specific information if available
    if context and "certificate_data" in context:
        cert = context["certificate_data"]
        context_info = []

        if "issuer_name" in cert:
            context_info.append(f"The certificate is from {cert['issuer_name']}.")

        rep = cert.get("issuer_reputation_score")
        if isinstance(rep, (int, float)):
            if rep > 0.8:
                context_info.append("This issuer has a good reputation score.")
            elif rep < 0.3:
                context_info.append("⚠️ This issuer has a low reputation score.")

        if context_info:
            response += "\n\nRegarding your specific certificate: " + " ".join(context_info)

    sources.append(f"knowledge_base:{topic}")

    return {
        "response": response,
        "sources": sources,
        "method": "keyword_matching",
    }


def get_supported_topics() -> List[str]:
    """Return list of supported chatbot topics."""
    return [topic for topic in KNOWLEDGE_BASE.keys() if topic != "general"]
