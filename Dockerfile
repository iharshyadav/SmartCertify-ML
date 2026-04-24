FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first (separate index), then everything else
RUN pip install --no-cache-dir --timeout 300 \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

COPY . .

# Create dirs and __init__ files
RUN mkdir -p saved_models plots logs data && \
    touch app/__init__.py \
          app/api/__init__.py \
          app/api/routes/__init__.py \
          app/models/__init__.py \
          app/data/__init__.py \
          app/utils/__init__.py \
          app/config/__init__.py

# ─────────────────────────────────────────────────────────────────────────────
# BUILD STEP 1: Generate 25,000 synthetic CSV rows (no network needed)
# ─────────────────────────────────────────────────────────────────────────────
RUN python -m app.data.generate_synthetic

# ─────────────────────────────────────────────────────────────────────────────
# BUILD STEP 2: Pre-download NLP models from HuggingFace
# NOTE: NO offline env vars yet — we need network access here
# ─────────────────────────────────────────────────────────────────────────────
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
from transformers import pipeline; \
print('Downloading all-MiniLM-L6-v2...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Downloading DistilBERT zero-shot...'); \
pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli'); \
print('NLP models downloaded.') \
"

# ─────────────────────────────────────────────────────────────────────────────
# BUILD STEP 3: Pre-download real certificate image datasets from HuggingFace
# These will be cached in HF_HOME for use during training
# ─────────────────────────────────────────────────────────────────────────────
RUN python -c "\
print('Pre-caching HF image datasets...'); \
from app.data.load_hf_images import load_authentic_images, load_tampered_images; \
auth = load_authentic_images(n_max=300); \
tamp = load_tampered_images(n_max=150); \
print(f'Cached {len(auth)} authentic + {len(tamp)} tampered images'); \
"

# ─────────────────────────────────────────────────────────────────────────────
# BUILD STEP 3.5: Pre-download ResNet18 weights
# ─────────────────────────────────────────────────────────────────────────────
RUN python -c "\
import torchvision.models as tv_models; \
print('Downloading ResNet18 weights...'); \
tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT); \
print('ResNet18 weights downloaded.') \
"

# ─────────────────────────────────────────────────────────────────────────────
# BUILD STEP 4: Train all models (uses cached data — no network calls)
# ─────────────────────────────────────────────────────────────────────────────
RUN python -m app.models.train_all


# ─────────────────────────────────────────────────────────────────────────────
# BUILD STEP 5: Verify all required model files exist — fail build if missing
# ─────────────────────────────────────────────────────────────────────────────
RUN python -c "\
import os; \
from pathlib import Path; \
required = [ \
    'saved_models/fraud_rf.pkl', \
    'saved_models/fraud_xgb.pkl', \
    'saved_models/fraud_lgb.pkl', \
    'saved_models/fraud_features.pkl', \
    'saved_models/image_model.pt', \
    'saved_models/image_classifier_head.pkl', \
    'saved_models/trust_model.pkl', \
    'saved_models/trust_features.pkl', \
    'saved_models/anomaly_model.pkl', \
    'saved_models/anomaly_scaler.pkl', \
    'saved_models/anomaly_features.pkl', \
    'saved_models/similarity_model_name.txt', \
]; \
missing = [f for f in required if not os.path.exists(f)]; \
assert not missing, f'Build failed - missing: {missing}'; \
files = list(Path('saved_models').iterdir()); \
print(f'Build OK - {len(files)} model files saved'); \
[print(f'  {f.name}: {f.stat().st_size/1024:.1f} KB') for f in sorted(files)] \
"

# ─────────────────────────────────────────────────────────────────────────────
# NOW set offline mode — only takes effect at RUNTIME, not during build steps
# This prevents any network calls per inference request
# ─────────────────────────────────────────────────────────────────────────────
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
