"""
SmartCertify ML — Trust Score Regression Model
Issuer trust score prediction using multiple regression approaches.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import RANDOM_SEED, TRUST_DATASET_PATH, MODEL_DIR
from app.utils.model_io import save_sklearn_model, load_sklearn_model
from app.config.model_registry import register_model

logger = logging.getLogger(__name__)

TRUST_FEATURES = [
    "total_certificates_issued",
    "fraud_rate_historical",
    "avg_metadata_completeness",
    "domain_age_days",
    "verification_success_rate",
    "response_time_avg",
]

_model = None
_scaler = None


def train_trust_models(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Train all trust score regression models and save the best."""
    if df is None:
        df = pd.read_csv(TRUST_DATASET_PATH)

    X = df[TRUST_FEATURES].values
    y = df["trust_score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    save_sklearn_model(scaler, "trust_scaler.joblib")

    # Train models
    models = {
        "linear_regression": LinearRegression(),
        "polynomial_regression": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lr", LinearRegression()),
        ]),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }

    results = []
    best_model = None
    best_r2 = -float("inf")

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
        results.append({"model": name, **metrics})
        logger.info(f"  {name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = (name, model, metrics)

    # Save best model
    if best_model:
        name, model, metrics = best_model
        save_sklearn_model(model, "trust_regression.joblib", metadata=metrics)
        register_model("trust_regression", "1.0",
                       str(MODEL_DIR / "trust_regression.joblib"), metrics=metrics)
        logger.info(f"Best trust model: {name} (R²={best_r2:.4f})")

    return {
        "results": results,
        "best_model": best_model[0] if best_model else None,
        "best_r2": best_r2,
    }


def predict_trust_score(issuer_features: Dict[str, Any]) -> Dict[str, Any]:
    """Predict trust score for an issuer."""
    global _model, _scaler

    if _model is None:
        _model = load_sklearn_model("trust_regression.joblib")
    if _scaler is None:
        _scaler = load_sklearn_model("trust_scaler.joblib")

    if _model is None:
        return {"error": "Trust score model not trained"}

    # Extract features
    X = np.array([[
        issuer_features.get("total_certificates_issued", 100),
        issuer_features.get("fraud_rate_historical", 0.01),
        issuer_features.get("avg_metadata_completeness", 0.8),
        issuer_features.get("domain_age_days", 365),
        issuer_features.get("verification_success_rate", 0.9),
        issuer_features.get("response_time_avg", 50),
    ]])

    if _scaler:
        X = _scaler.transform(X)

    trust_score = float(_model.predict(X)[0])
    trust_score = max(0, min(100, trust_score))

    # Determine grade
    if trust_score >= 90:
        grade = "A"
    elif trust_score >= 75:
        grade = "B"
    elif trust_score >= 60:
        grade = "C"
    elif trust_score >= 40:
        grade = "D"
    else:
        grade = "F"

    return {
        "trust_score": round(trust_score, 2),
        "trust_grade": grade,
        "confidence": round(min(trust_score / 100, 1.0), 2),
        "factors": {
            "fraud_rate_impact": "low" if issuer_features.get("fraud_rate_historical", 0) < 0.05 else "high",
            "domain_maturity": "established" if issuer_features.get("domain_age_days", 0) > 730 else "new",
            "verification_rate": "good" if issuer_features.get("verification_success_rate", 0) > 0.85 else "poor",
        },
    }


def main():
    """Train trust score models."""
    from app.config.settings import TRUST_DATASET_PATH

    if not Path(TRUST_DATASET_PATH).exists():
        from app.data.generate_synthetic import generate_trust_score_dataset
        df = generate_trust_score_dataset()
        df.to_csv(TRUST_DATASET_PATH, index=False)

    print("Training trust score models...")
    results = train_trust_models()
    print(f"\n✅ Trust score training complete!")
    for r in results["results"]:
        print(f"   {r['model']}: RMSE={r['rmse']}, MAE={r['mae']}, R²={r['r2']}")
    print(f"   Best model: {results['best_model']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
