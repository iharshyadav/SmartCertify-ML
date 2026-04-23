---
title: SmartCertify ML
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# SmartCertify ML Microservice

FastAPI ML service for AI-powered certificate fraud detection.

**Upgraded from Render (lightweight) → Hugging Face Spaces (full models):**
- Image tampering: ResNet-18 CNN (fine-tuned) — NOT ELA stats
- Similarity: BERT sentence-transformers (all-MiniLM-L6-v2) — NOT TF-IDF
- Chat: DistilBERT zero-shot classification — NOT keyword matching

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/ml/verify` | Fraud detection — RF+XGB+LGB ensemble |
| POST | `/api/ml/analyze-image` | Tampering detection — ResNet-18 CNN |
| POST | `/api/ml/similarity` | Duplicate detection — BERT cosine |
| POST | `/api/ml/trust-score` | Issuer trust — Gradient Boosting |
| POST | `/api/ml/anomaly` | Anomaly detection — Isolation Forest |
| POST | `/api/ml/chat` | Q&A chatbot — DistilBERT zero-shot |
| POST | `/api/ml/recommend` | Course recs — BERT similarity |
| GET | `/api/ml/metrics` | Model metrics |

## Authentication

All endpoints require `X-API-Key` header.
Set `ML_API_KEY` as a Space secret in HF settings.

## Models

All trained at Docker build time (baked into image):

| Model | Type | Size |
|-------|------|------|
| Fraud detection | RF + XGBoost + LightGBM | ~15 MB |
| Image tampering | ResNet-18 (fine-tuned, CPU) | ~45 MB |
| Semantic similarity | all-MiniLM-L6-v2 | ~90 MB |
| Chat classification | DistilBERT zero-shot | ~66 MB |
| Trust scoring | Gradient Boosting | ~2 MB |
| Anomaly detection | Isolation Forest | ~1 MB |

## Local Development

```bash
docker build -t smartcertify-ml .
docker run -p 7860:7860 -e ML_API_KEY=dev-key smartcertify-ml
curl http://localhost:7860/health
```

## Deploy to HF Spaces

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SmartCertify-ML
git push hf main
```
