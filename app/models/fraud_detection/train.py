"""
SmartCertify ML — Fraud Detection Training
Train all classifiers, tune hyperparameters, and create voting ensemble.
"""

import numpy as np
import pandas as pd
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import (
    RANDOM_SEED, CV_FOLDS, MODEL_DIR, PLOTS_DIR,
    NN_EPOCHS, NN_BATCH_SIZE, NN_LEARNING_RATE, NN_WEIGHT_DECAY, NN_PATIENCE,
)
from app.utils.model_io import save_sklearn_model, save_pytorch_model
from app.utils.visualization import plot_learning_curves
from app.config.model_registry import register_model

logger = logging.getLogger(__name__)


# ─── PyTorch Neural Network ──────────────────────────────────

class CertificateFraudNet(nn.Module):
    """Deep neural network for certificate fraud detection."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[CertificateFraudNet, Dict[str, Any]]:
    """Train the PyTorch neural network with early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Neural Network on {device}")

    input_dim = X_train.shape[1]
    model = CertificateFraudNet(input_dim).to(device)

    # Handle class imbalance with pos_weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    criterion = nn.BCELoss(weight=None)  # Using balanced data from SMOTE
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_LEARNING_RATE, weight_decay=NN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train.astype(np.float32)),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val.astype(np.float32)),
    )
    train_loader = DataLoader(train_dataset, batch_size=NN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=NN_BATCH_SIZE, shuffle=False)

    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(NN_EPOCHS):
        # Train
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(X_batch)
            predicted = (output > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += len(y_batch)

        train_loss = epoch_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)
                val_loss += loss.item() * len(X_batch)
                predicted = (output > 0.5).float()
                val_correct += (predicted == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{NN_EPOCHS} — "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= NN_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Save learning curves
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs)

    # Save model
    save_pytorch_model(model, "fraud_nn.pt", optimizer=optimizer, epoch=epoch + 1,
                       metadata={"input_dim": input_dim, "best_val_loss": best_val_loss, "val_acc": val_acc})

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_epoch": epoch + 1 - patience_counter,
        "final_val_acc": val_acc,
    }

    return model, history


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Train all fraud detection models, evaluate with cross-validation, and save."""

    # ── Define Models ───────────────────────────────────────
    sklearn_models = {}

    logger.info("Training Logistic Regression...")
    sklearn_models["logistic_regression"] = LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
    )

    logger.info("Training k-NN...")
    sklearn_models["knn"] = KNeighborsClassifier(n_neighbors=5)

    logger.info("Training SVM...")
    sklearn_models["svm"] = SVC(
        kernel="rbf", probability=True, random_state=RANDOM_SEED, class_weight="balanced"
    )

    logger.info("Training Random Forest...")
    sklearn_models["random_forest"] = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
    )

    try:
        from xgboost import XGBClassifier
        logger.info("Training XGBoost...")
        sklearn_models["xgboost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=RANDOM_SEED, eval_metric="logloss",
            use_label_encoder=False, n_jobs=-1,
        )
    except ImportError:
        logger.warning("XGBoost not installed, skipping")

    try:
        from lightgbm import LGBMClassifier
        logger.info("Training LightGBM...")
        sklearn_models["lightgbm"] = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, random_state=RANDOM_SEED,
            class_weight="balanced", verbose=-1, n_jobs=-1,
        )
    except ImportError:
        logger.warning("LightGBM not installed, skipping")

    # ── Cross-Validation + Training ──────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    benchmark = []

    for name, model in sklearn_models.items():
        start_time = time.time()
        logger.info(f"Cross-validating {name}...")

        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

        # Fit on full training set
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate on test set
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
            "train_time_s": round(train_time, 2),
        }

        benchmark.append({"model": name, **metrics})
        logger.info(f"  {name}: F1={metrics['f1']}, ROC-AUC={metrics['roc_auc']}, CV-F1={metrics['cv_f1_mean']}±{metrics['cv_f1_std']}")

        # Save model
        filename = f"fraud_{name.replace(' ', '_')}.joblib"
        if name == "random_forest":
            filename = "fraud_rf.joblib"
        elif name == "xgboost":
            filename = "fraud_xgb.joblib"
        elif name == "lightgbm":
            filename = "fraud_lgbm.joblib"
        elif name == "logistic_regression":
            filename = "fraud_lr.joblib"

        save_sklearn_model(model, filename, metadata=metrics)
        register_model(name, "1.0", str(MODEL_DIR / filename), metrics=metrics)

    # ── Voting Ensemble ──────────────────────────────────────
    ensemble_estimators = []
    for name in ["random_forest", "xgboost", "lightgbm"]:
        if name in sklearn_models:
            ensemble_estimators.append((name, sklearn_models[name]))

    if len(ensemble_estimators) >= 2:
        logger.info("Building Voting Ensemble (RF + XGB + LGBM)...")
        ensemble = VotingClassifier(estimators=ensemble_estimators, voting="soft", n_jobs=-1)
        ensemble.fit(X_train, y_train)

        y_pred_ens = ensemble.predict(X_test)
        y_proba_ens = ensemble.predict_proba(X_test)[:, 1]

        ens_metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred_ens), 4),
            "precision": round(precision_score(y_test, y_pred_ens, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_ens, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred_ens, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba_ens), 4),
        }

        benchmark.append({"model": "voting_ensemble", **ens_metrics})
        logger.info(f"  Ensemble: F1={ens_metrics['f1']}, ROC-AUC={ens_metrics['roc_auc']}")

        save_sklearn_model(ensemble, "fraud_ensemble.joblib", metadata=ens_metrics)
        register_model("voting_ensemble", "1.0", str(MODEL_DIR / "fraud_ensemble.joblib"), metrics=ens_metrics)

    # ── Train Neural Network ─────────────────────────────────
    logger.info("Training Neural Network...")
    nn_model, nn_history = train_neural_network(X_train, y_train, X_test, y_test)

    # NN evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        nn_predictions = nn_model(X_test_tensor).cpu().numpy().squeeze()

    nn_pred_labels = (nn_predictions > 0.5).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    nn_metrics = {
        "accuracy": round(accuracy_score(y_test, nn_pred_labels), 4),
        "precision": round(precision_score(y_test, nn_pred_labels, zero_division=0), 4),
        "recall": round(recall_score(y_test, nn_pred_labels, zero_division=0), 4),
        "f1": round(f1_score(y_test, nn_pred_labels, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, nn_predictions), 4),
        "val_accuracy": nn_history["final_val_acc"],
    }
    benchmark.append({"model": "neural_network", **nn_metrics})
    register_model("neural_network", "1.0", str(MODEL_DIR / "fraud_nn.pt"), metrics=nn_metrics)

    # ── Save Benchmark ───────────────────────────────────────
    benchmark_df = pd.DataFrame(benchmark)
    benchmark_path = PLOTS_DIR / "model_benchmark.csv"
    benchmark_df.to_csv(benchmark_path, index=False)
    logger.info(f"\n{benchmark_df.to_string(index=False)}")
    logger.info(f"Benchmark saved to {benchmark_path}")

    return {
        "models": sklearn_models,
        "ensemble": ensemble if len(ensemble_estimators) >= 2 else None,
        "nn_model": nn_model,
        "benchmark": benchmark_df,
    }


def main():
    """Run the full training pipeline."""
    from app.data.preprocess import prepare_data

    print("=" * 60)
    print("  SmartCertify ML — Fraud Detection Training Pipeline")
    print("=" * 60)

    # Generate data if needed
    from app.config.settings import DATASET_PATH
    if not Path(DATASET_PATH).exists():
        print("\nGenerating synthetic dataset first...")
        from app.data.generate_synthetic import main as gen_main
        gen_main()

    print("\n📊 Preparing data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    print("\n🤖 Training models...")
    results = train_all_models(X_train, y_train, X_test, y_test)

    print("\n✅ Training complete! Benchmark:")
    print(results["benchmark"].to_string(index=False))
    print(f"\nModels saved to: {MODEL_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    main()
