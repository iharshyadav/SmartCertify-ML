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
# BUILD STEP 3: Train tabular models ONLY (fraud + trust + anomaly + similarity)
# CNN skipped at build time — image analysis uses ELA heuristics at runtime.
# This avoids OOM from holding thousands of PIL images in memory.
# ─────────────────────────────────────────────────────────────────────────────
RUN python -c "\
import os; \
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2'); \
import joblib, pandas as pd; \
from pathlib import Path; \
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest; \
from sklearn.model_selection import train_test_split; \
from sklearn.preprocessing import LabelEncoder, StandardScaler; \
import xgboost as xgb; \
import lightgbm as lgb; \
\
SAVE_DIR = Path('saved_models'); \
df = pd.read_csv('data/synthetic_certificates.csv'); \
print(f'Training on {len(df)} rows...'); \
\
FRAUD_FEATS = ['issuer_reputation_score','template_match_score','metadata_completeness_score', \
    'domain_verification_status','previous_verification_count','cert_age_days', \
    'issuer_cert_count','has_expiry','name_length','course_name_length', \
    'total_certificates_issued','fraud_rate_historical','avg_metadata_completeness', \
    'domain_age_days','verification_success_rate']; \
TRUST_FEATS = ['total_certificates_issued','fraud_rate_historical','avg_metadata_completeness', \
    'domain_age_days','verification_success_rate']; \
\
le = LabelEncoder(); \
y = le.fit_transform(df['label']); \
label_map = {l:i for i,l in enumerate(le.classes_)}; \
X = df[FRAUD_FEATS].fillna(0); \
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y); \
\
print('  Training RandomForest...'); \
rf = RandomForestClassifier(n_estimators=200,max_depth=12,n_jobs=-1,random_state=42); \
rf.fit(Xtr,ytr); \
print('  Training XGBoost...'); \
xm = xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1, \
    eval_metric='mlogloss',random_state=42,verbosity=0); \
xm.fit(Xtr,ytr); \
print('  Training LightGBM...'); \
lm = lgb.LGBMClassifier(n_estimators=200,max_depth=8,learning_rate=0.1, \
    random_state=42,verbose=-1); \
lm.fit(Xtr,ytr); \
joblib.dump(rf, SAVE_DIR/'fraud_rf.pkl'); \
joblib.dump(xm, SAVE_DIR/'fraud_xgb.pkl'); \
joblib.dump(lm, SAVE_DIR/'fraud_lgb.pkl'); \
joblib.dump(FRAUD_FEATS, SAVE_DIR/'fraud_features.pkl'); \
joblib.dump(label_map, SAVE_DIR/'fraud_label_map.pkl'); \
print('  Fraud models saved.'); \
\
Xt = df[TRUST_FEATS].fillna(0); yt = df['trust_score'].fillna(0.5); \
Xtr2,Xte2,ytr2,yte2 = train_test_split(Xt,yt,test_size=0.2,random_state=42); \
print('  Training trust model...'); \
tm = GradientBoostingRegressor(n_estimators=200,max_depth=5,learning_rate=0.05,random_state=42); \
tm.fit(Xtr2,ytr2); \
joblib.dump(tm, SAVE_DIR/'trust_model.pkl'); \
joblib.dump(TRUST_FEATS, SAVE_DIR/'trust_features.pkl'); \
print('  Trust model saved.'); \
\
sc = StandardScaler(); Xs = sc.fit_transform(X); \
print('  Training anomaly model...'); \
am = IsolationForest(contamination=0.1,n_estimators=200,random_state=42,n_jobs=-1); \
am.fit(Xs); \
joblib.dump(am, SAVE_DIR/'anomaly_model.pkl'); \
joblib.dump(sc, SAVE_DIR/'anomaly_scaler.pkl'); \
joblib.dump(FRAUD_FEATS, SAVE_DIR/'anomaly_features.pkl'); \
print('  Anomaly model saved.'); \
\
from sentence_transformers import SentenceTransformer; \
print('  Setting up similarity model...'); \
sim = SentenceTransformer('all-MiniLM-L6-v2'); \
(SAVE_DIR/'similarity_model_name.txt').write_text('all-MiniLM-L6-v2'); \
joblib.dump({'model_name':'all-MiniLM-L6-v2','embedding_dim':384}, SAVE_DIR/'similarity_meta.pkl'); \
print('  Similarity model saved.'); \
\
from transformers import pipeline as hf_pipeline; \
print('  Setting up chat model...'); \
clf = hf_pipeline('zero-shot-classification',model='typeform/distilbert-base-uncased-mnli',device=-1); \
(SAVE_DIR/'chat_model_name.txt').write_text('typeform/distilbert-base-uncased-mnli'); \
print('All models trained and saved!') \
"

# ─────────────────────────────────────────────────────────────────────────────
# BUILD STEP 4: Verify core model files exist — image model is optional
# ─────────────────────────────────────────────────────────────────────────────
RUN python -c "\
import os; from pathlib import Path; \
required = ['saved_models/fraud_rf.pkl','saved_models/fraud_xgb.pkl', \
    'saved_models/fraud_lgb.pkl','saved_models/fraud_features.pkl', \
    'saved_models/trust_model.pkl','saved_models/trust_features.pkl', \
    'saved_models/anomaly_model.pkl','saved_models/anomaly_scaler.pkl', \
    'saved_models/anomaly_features.pkl']; \
missing = [f for f in required if not os.path.exists(f)]; \
assert not missing, f'Build failed - missing: {missing}'; \
files = list(Path('saved_models').iterdir()); \
print(f'Build OK — {len(files)} model files:'); \
[print(f'  {f.name}: {f.stat().st_size/1024:.1f} KB') for f in sorted(files)] \
"

# Set offline mode for runtime — models are already cached
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
