"""
train_all.py — Build-time training script for SmartCertify-ML.
Called during Docker build: python -m app.models.train_all
Trains every model and saves to saved_models/
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import xgboost as xgb
import lightgbm as lgb

SAVE_DIR = Path("saved_models")
SAVE_DIR.mkdir(exist_ok=True)

DATA_PATH = Path("data/synthetic_certificates.csv")

FRAUD_FEATURE_COLS = [
    "issuer_reputation_score",
    "template_match_score",
    "metadata_completeness_score",
    "domain_verification_status",
    "previous_verification_count",
    "cert_age_days",
    "issuer_cert_count",
    "has_expiry",
    "name_length",
    "course_name_length",
    "total_certificates_issued",
    "fraud_rate_historical",
    "avg_metadata_completeness",
    "domain_age_days",
    "verification_success_rate",
]

TRUST_FEATURE_COLS = [
    "total_certificates_issued",
    "fraud_rate_historical",
    "avg_metadata_completeness",
    "domain_age_days",
    "verification_success_rate",
]

ANOMALY_FEATURE_COLS = FRAUD_FEATURE_COLS  # same feature set


# ─────────────────────────────────────────────────────────────────────────────
# 6A — Fraud Detection (RF + XGB + LGB ensemble)
# ─────────────────────────────────────────────────────────────────────────────

def train_fraud_model(df: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("6A — Training Fraud Detection Models")
    print("=" * 50)

    le = LabelEncoder()
    y = le.fit_transform(df["label"])  # authentic=0, fake=1, tampered=2
    # Remap to: authentic=0, tampered=1, fake=2
    label_map = {l: i for i, l in enumerate(le.classes_)}
    print(f"  Label encoding: {label_map}")

    X = df[FRAUD_FEATURE_COLS].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
    print("  Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    print(classification_report(y_test, rf.predict(X_test),
                                target_names=le.classes_))

    # XGBoost
    print("  Training XGBClassifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    print(classification_report(y_test, xgb_model.predict(X_test),
                                target_names=le.classes_))

    # LightGBM
    print("  Training LGBMClassifier...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        random_state=42, verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    print(classification_report(y_test, lgb_model.predict(X_test),
                                target_names=le.classes_))

    joblib.dump(rf, SAVE_DIR / "fraud_rf.pkl")
    joblib.dump(xgb_model, SAVE_DIR / "fraud_xgb.pkl")
    joblib.dump(lgb_model, SAVE_DIR / "fraud_lgb.pkl")
    joblib.dump(FRAUD_FEATURE_COLS, SAVE_DIR / "fraud_features.pkl")
    joblib.dump(label_map, SAVE_DIR / "fraud_label_map.pkl")
    print("  Fraud models saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 6A(trust) — Trust Score Regression
# ─────────────────────────────────────────────────────────────────────────────

def train_trust_model(df: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("6A — Training Trust Score Model")
    print("=" * 50)

    X = df[TRUST_FEATURE_COLS].fillna(0)
    y = df["trust_score"].fillna(0.5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"  MSE:  {mean_squared_error(y_test, preds):.6f}")
    print(f"  R²:   {r2_score(y_test, preds):.4f}")

    joblib.dump(model, SAVE_DIR / "trust_model.pkl")
    joblib.dump(TRUST_FEATURE_COLS, SAVE_DIR / "trust_features.pkl")
    print("  Trust model saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 6A(anomaly) — Anomaly Detection (Isolation Forest)
# ─────────────────────────────────────────────────────────────────────────────

def train_anomaly_model(df: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("6A — Training Anomaly Detection Model")
    print("=" * 50)

    X = df[ANOMALY_FEATURE_COLS].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.1, n_estimators=200,
                            random_state=42, n_jobs=-1)
    model.fit(X_scaled)
    anomaly_count = (model.predict(X_scaled) == -1).sum()
    print(f"  Anomaly model trained on {len(X)} samples.")
    print(f"  Anomalies detected in training data: {anomaly_count}")

    joblib.dump(model, SAVE_DIR / "anomaly_model.pkl")
    joblib.dump(scaler, SAVE_DIR / "anomaly_scaler.pkl")
    joblib.dump(ANOMALY_FEATURE_COLS, SAVE_DIR / "anomaly_features.pkl")
    print("  Anomaly model saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 6B — Image Tampering (ResNet-18 CNN)
# ─────────────────────────────────────────────────────────────────────────────

def train_image_model() -> None:
    print("\n" + "=" * 50)
    print("6B — Training ResNet-18 CNN for Image Tampering")
    print("=" * 50)

    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader
    from app.utils.cert_image_gen import make_authentic_cert, apply_tampering

    N_PER_CLASS = 400  # keep build time reasonable on HF free tier

    class CertDataset(Dataset):
        def __init__(self, n_per_class: int, transform=None):
            self.data = []
            self.labels = []
            self.transform = transform
            print(f"  Generating {n_per_class * 2} synthetic certificate images...")
            for i in range(n_per_class):
                self.data.append(make_authentic_cert())
                self.labels.append(0)
                self.data.append(apply_tampering(make_authentic_cert()))
                self.labels.append(1)
                if (i + 1) % 100 == 0:
                    print(f"  {(i+1)*2}/{n_per_class*2} images generated")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img = self.data[idx]
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = CertDataset(n_per_class=N_PER_CLASS, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # ResNet-18 — freeze early layers, train layer4 + new FC head
    device = torch.device("cpu")
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)

    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0
    N_EPOCHS = 8

    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        scheduler.step()
        print(f"  Epoch {epoch+1}/{N_EPOCHS} | "
              f"loss: {train_loss/len(train_loader):.4f} | "
              f"val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_DIR / "image_model.pt")
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

    print(f"  ResNet-18 training complete. Best val_acc: {best_val_acc:.4f}")
    joblib.dump(
        {"best_val_acc": best_val_acc, "n_classes": 2, "input_size": (224, 224)},
        SAVE_DIR / "image_classifier_head.pkl",
    )
    print("  Image model saved → saved_models/image_model.pt")


# ─────────────────────────────────────────────────────────────────────────────
# 6C — Similarity: sentence-transformers
# ─────────────────────────────────────────────────────────────────────────────

def train_similarity_model(df: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("6C — Setting up Sentence-Transformers Similarity Model")
    print("=" * 50)

    from sentence_transformers import SentenceTransformer

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    texts = df["course_name"].tolist()
    print(f"  Encoding {len(texts)} course names (warm-up cache)...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)

    (SAVE_DIR / "similarity_model_name.txt").write_text(model_name)
    joblib.dump(
        {
            "model_name": model_name,
            "embedding_dim": int(embeddings[0].shape[0]),
            "n_encoded": len(texts),
        },
        SAVE_DIR / "similarity_meta.pkl",
    )
    print(f"  Similarity model ready. Embedding dim: {embeddings[0].shape[0]}")


# ─────────────────────────────────────────────────────────────────────────────
# 6D — Chat: DistilBERT zero-shot
# ─────────────────────────────────────────────────────────────────────────────

def setup_chat_model() -> None:
    print("\n" + "=" * 50)
    print("6D — Setting up DistilBERT Zero-Shot Chat Model")
    print("=" * 50)

    from transformers import pipeline

    model_name = "typeform/distilbert-base-uncased-mnli"
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1,  # CPU
    )
    test = classifier(
        "How do I verify a certificate?",
        candidate_labels=[
            "verify certificate", "report fraud",
            "check trust score", "general help",
        ],
    )
    print(f"  Chat model test: '{test['labels'][0]}' "
          f"(score: {test['scores'][0]:.3f})")

    (SAVE_DIR / "chat_model_name.txt").write_text(model_name)
    print("  Chat model ready → saved_models/chat_model_name.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("SmartCertify-ML — Build-Time Training (HF Spaces)")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run generate_synthetic first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records from {DATA_PATH}")
    print(df["label"].value_counts().to_string())

    train_fraud_model(df)
    train_trust_model(df)
    train_anomaly_model(df)
    train_image_model()
    train_similarity_model(df)
    setup_chat_model()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"All models trained in {elapsed:.1f}s. Saved files:")
    for f in sorted(SAVE_DIR.iterdir()):
        print(f"  {f.name:<45s} {f.stat().st_size / 1024:8.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
