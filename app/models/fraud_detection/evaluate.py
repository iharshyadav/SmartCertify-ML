"""
SmartCertify ML — Fraud Detection Evaluation
Generate comprehensive metrics, reports, confusion matrices, and ROC curves.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import MODEL_DIR, PLOTS_DIR
from app.utils.model_io import load_sklearn_model
from app.utils.visualization import (
    plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_multi_roc,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Evaluate a single model and generate full report."""
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    # Classification report
    report = classification_report(y_test, y_pred, target_names=["Authentic", "Fraudulent"], output_dict=True)

    # Save plots
    cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
    roc_path = plot_roc_curve(y_test, y_proba, model_name)
    pr_path = plot_precision_recall_curve(y_test, y_proba, model_name)

    result = {
        "model_name": model_name,
        "metrics": metrics,
        "classification_report": report,
        "plots": {
            "confusion_matrix": cm_path,
            "roc_curve": roc_path,
            "pr_curve": pr_path,
        },
    }

    return result


def evaluate_all_models(
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Evaluate all saved fraud detection models and create comparison."""
    model_files = {
        "Logistic Regression": "fraud_lr.joblib",
        "k-NN": "fraud_knn.joblib",
        "SVM": "fraud_svm.joblib",
        "Random Forest": "fraud_rf.joblib",
        "XGBoost": "fraud_xgb.joblib",
        "LightGBM": "fraud_lgbm.joblib",
        "Voting Ensemble": "fraud_ensemble.joblib",
    }

    all_results = []
    roc_data = {}

    for name, filename in model_files.items():
        model = load_sklearn_model(filename)
        if model is None:
            logger.warning(f"Model {filename} not found, skipping")
            continue

        logger.info(f"Evaluating {name}...")
        result = evaluate_model(model, X_test, y_test, name)
        all_results.append(result)

        # Collect ROC data for multi-model comparison
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(X_test).astype(float)
        )
        roc_data[name] = {"y_true": y_test, "y_proba": y_proba}

    # Multi-model ROC comparison plot
    if len(roc_data) > 1:
        plot_multi_roc(roc_data)

    # Create benchmark DataFrame
    benchmark_data = []
    for result in all_results:
        benchmark_data.append({
            "Model": result["model_name"],
            **result["metrics"],
        })

    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df = benchmark_df.sort_values("f1", ascending=False)

    # Save benchmark
    benchmark_path = PLOTS_DIR / "evaluation_benchmark.csv"
    benchmark_df.to_csv(benchmark_path, index=False)

    # Save detailed reports as JSON
    reports_path = PLOTS_DIR / "evaluation_reports.json"
    serializable_results = []
    for r in all_results:
        s = {
            "model_name": r["model_name"],
            "metrics": r["metrics"],
            "classification_report": r["classification_report"],
        }
        serializable_results.append(s)

    with open(reports_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\nBenchmark:\n{benchmark_df.to_string(index=False)}")
    return benchmark_df


def main():
    """Run evaluation on all models."""
    from app.data.preprocess import prepare_data

    print("=" * 60)
    print("  SmartCertify ML — Model Evaluation")
    print("=" * 60)

    print("\n📊 Loading data...")
    X_train, X_test, y_train, y_test, _ = prepare_data(apply_smote=False)

    print("\n📈 Evaluating all models...")
    benchmark = evaluate_all_models(X_test, y_test)

    print(f"\n✅ Evaluation complete!")
    print(benchmark.to_string(index=False))
    print(f"\nPlots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
