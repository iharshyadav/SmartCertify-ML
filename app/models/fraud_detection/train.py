"""
SmartCertify ML — Fraud Detection Training (Lightweight)
Train sklearn classifiers and create voting ensemble.
Optimized for 512MB memory environments (Render free tier).
"""

import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import RANDOM_SEED, MODEL_DIR, PLOTS_DIR
from app.utils.model_io import save_sklearn_model
from app.config.model_registry import register_model

logger = logging.getLogger(__name__)

CV_FOLDS = 3
N_ESTIMATORS = 100


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Train fraud detection models (lightweight — fits in 512MB)."""

    sklearn_models = {}

    logger.info("Training Logistic Regression...")
    sklearn_models["logistic_regression"] = LogisticRegression(
        C=1.0, max_iter=500, random_state=RANDOM_SEED, class_weight="balanced"
    )

    logger.info("Training Random Forest...")
    sklearn_models["random_forest"] = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=12,
        random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
    )

    try:
        from xgboost import XGBClassifier
        logger.info("Training XGBoost...")
        sklearn_models["xgboost"] = XGBClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=0.1, max_depth=6,
            random_state=RANDOM_SEED, eval_metric="logloss",
            use_label_encoder=False, n_jobs=-1,
        )
    except ImportError:
        logger.warning("XGBoost not installed, skipping")

    try:
        from lightgbm import LGBMClassifier
        logger.info("Training LightGBM...")
        sklearn_models["lightgbm"] = LGBMClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=0.1,
            random_state=RANDOM_SEED, class_weight="balanced", verbose=-1, n_jobs=-1,
        )
    except ImportError:
        logger.warning("LightGBM not installed, skipping")

    # ── Cross-Validation + Training ──────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    benchmark = []

    for name, model in sklearn_models.items():
        start_time = time.time()
        logger.info(f"Cross-validating {name}...")

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

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
        logger.info(f"  {name}: F1={metrics['f1']}, ROC-AUC={metrics['roc_auc']}")

        filename = {"random_forest": "fraud_rf.joblib", "xgboost": "fraud_xgb.joblib",
                     "lightgbm": "fraud_lgbm.joblib", "logistic_regression": "fraud_lr.joblib"
                    }.get(name, f"fraud_{name}.joblib")

        save_sklearn_model(model, filename, metadata=metrics)
        register_model(name, "1.0", str(MODEL_DIR / filename), metrics=metrics)

    # ── Voting Ensemble ──────────────────────────────────────
    ensemble_estimators = [(n, sklearn_models[n]) for n in ["random_forest", "xgboost", "lightgbm"] if n in sklearn_models]

    ensemble = None
    if len(ensemble_estimators) >= 2:
        logger.info("Building Voting Ensemble...")
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

    # ── Save Benchmark ───────────────────────────────────────
    benchmark_df = pd.DataFrame(benchmark)
    benchmark_path = PLOTS_DIR / "model_benchmark.csv"
    benchmark_df.to_csv(benchmark_path, index=False)
    logger.info(f"\n{benchmark_df.to_string(index=False)}")

    return {"models": sklearn_models, "ensemble": ensemble, "benchmark": benchmark_df}


def main():
    """Run the full training pipeline."""
    from app.data.preprocess import prepare_data

    print("=" * 60)
    print("  SmartCertify ML — Fraud Detection Training")
    print("=" * 60)

    from app.config.settings import DATASET_PATH
    if not Path(DATASET_PATH).exists():
        print("\nGenerating synthetic dataset...")
        from app.data.generate_synthetic import main as gen_main
        gen_main()

    print("\n📊 Preparing data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    print("\n🤖 Training models...")
    results = train_all_models(X_train, y_train, X_test, y_test)

    print("\n✅ Training complete!")
    print(results["benchmark"].to_string(index=False))
    print(f"\nModels saved to: {MODEL_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    main()
