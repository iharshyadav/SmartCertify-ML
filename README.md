# SmartCertify ML Microservice

> Blockchain + AI powered Certificate Verification Platform — ML Service

## Overview

SmartCertify ML is a standalone Python microservice that provides machine learning capabilities for the SmartCertify platform. It integrates with the Express.js backend via REST API.

## Features

- **Certificate Fraud Detection** — Ensemble ML models (RF + XGBoost + LightGBM + Neural Net)
- **Similarity Analysis** — TF-IDF + BERT semantic similarity for duplicate detection
- **Issuer Trust Scoring** — Regression models for institutional trust assessment
- **Image Tampering Detection** — CNN (ResNet-18) for certificate image analysis
- **Anomaly Detection** — Isolation Forest for suspicious pattern identification
- **Recommendation Engine** — Content + collaborative filtering for course suggestions
- **AI Chatbot** — Transformer-based QA for certificate queries
- **Model Monitoring** — Prediction logging, drift detection, performance metrics

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
python -m app.data.generate_synthetic

# 3. Train models
python -m app.models.fraud_detection.train

# 4. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/api/ml/verify` | Certificate fraud detection |
| POST | `/api/ml/similarity` | Certificate similarity check |
| POST | `/api/ml/trust-score` | Issuer trust scoring |
| POST | `/api/ml/analyze-image` | Image tampering detection |
| POST | `/api/ml/recommend` | Certificate recommendations |
| POST | `/api/ml/anomaly` | Anomaly detection |
| POST | `/api/ml/chat` | AI chatbot |
| GET | `/api/ml/metrics` | Model performance metrics |

## Authentication

All API endpoints require an `X-API-Key` header:

```bash
curl -X POST http://localhost:8000/api/ml/verify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_key_here" \
  -d '{"issuer_reputation_score": 0.9, ...}'
```

## Docker

```bash
docker-compose up --build
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ML_API_KEY` | API key for authentication | `smartcertify-dev-key` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MODEL_DIR` | Model storage directory | `saved_models` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |

## Project Structure

```
smartcertify-ml/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── api/routes/             # API route handlers
│   ├── models/                 # ML model modules
│   ├── data/                   # Data generation & preprocessing
│   ├── utils/                  # Utilities (math, viz, IO, monitoring)
│   └── config/                 # Settings & model registry
├── saved_models/               # Serialized model files
├── tests/                      # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Integration with Express Backend

```typescript
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";
const ML_API_KEY = process.env.ML_API_KEY;

async function verifyCertificate(certFeatures: object) {
  const response = await fetch(`${ML_SERVICE_URL}/api/ml/verify`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": ML_API_KEY
    },
    body: JSON.stringify(certFeatures)
  });
  return response.json();
}
```

## License

MIT
