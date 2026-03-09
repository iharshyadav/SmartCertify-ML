"""
SmartCertify ML — Preprocessing Pipeline
Full sklearn pipeline for data cleaning, transformation, and balancing.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import RANDOM_SEED, TEST_SIZE, DATASET_PATH, MODEL_DIR
from app.utils.model_io import save_sklearn_model, load_sklearn_model

logger = logging.getLogger(__name__)

# ─── Feature Columns ─────────────────────────────────────────

NUMERIC_FEATURES = [
    "issuer_reputation_score",
    "certificate_age_days",
    "metadata_completeness_score",
    "ocr_confidence_score",
    "template_match_score",
    "domain_verification_status",
    "previous_verification_count",
    "time_since_last_verification_days",
]

TEXT_FEATURES = ["issuer_name", "course_name"]
DATE_FEATURES = ["issue_date", "expiry_date"]
TARGET = "label"


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric features from date columns."""
    df = df.copy()

    if "issue_date" in df.columns:
        issue_date = pd.to_datetime(df["issue_date"], errors="coerce")
        df["issue_month"] = issue_date.dt.month.fillna(0).astype(int)
        df["issue_year"] = issue_date.dt.year.fillna(0).astype(int)
        df["issue_dayofweek"] = issue_date.dt.dayofweek.fillna(0).astype(int)
        df["weekend_issued"] = (df["issue_dayofweek"] >= 5).astype(int)

    if "expiry_date" in df.columns and "issue_date" in df.columns:
        expiry_date = pd.to_datetime(df["expiry_date"], errors="coerce")
        days_to_expiry = (expiry_date - issue_date).dt.days
        df["days_to_expiry"] = days_to_expiry.fillna(0).astype(int)
        df["is_expired"] = (days_to_expiry < 0).astype(int)

    # Check for future issue dates (fraud signal)
    if "issue_date" in df.columns:
        now = pd.Timestamp.now()
        df["is_future_issue"] = (issue_date > now).astype(int)

    return df


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build and fit the sklearn ColumnTransformer preprocessor."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # All numeric columns including engineered ones
    numeric_cols = [c for c in X_train.columns if c not in TEXT_FEATURES + DATE_FEATURES + [TARGET, "credential_hash", "recipient_name", "combined_text"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, [c for c in numeric_cols if c in X_train.columns]),
            ("text", TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english"), "combined_text"),
        ],
        remainder="drop",
    )

    return preprocessor


def _combine_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Combine text feature columns into a single column for TF-IDF."""
    df = df.copy()
    text_parts = []
    for col in TEXT_FEATURES:
        if col in df.columns:
            text_parts.append(df[col].fillna("unknown").astype(str))
    if text_parts:
        df["combined_text"] = text_parts[0]
        for part in text_parts[1:]:
            df["combined_text"] = df["combined_text"] + " " + part
    else:
        df["combined_text"] = "unknown"
    return df


def prepare_data(
    df: Optional[pd.DataFrame] = None,
    apply_smote: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Full data preparation pipeline:
    1. Load data
    2. Extract date features
    3. Build and fit preprocessor
    4. Apply SMOTE for class balancing
    5. Return X_train, X_test, y_train, y_test, preprocessor
    """
    if df is None:
        df = pd.read_csv(DATASET_PATH)
        logger.info(f"Loaded dataset: {df.shape}")

    # Extract date features
    df = extract_date_features(df)

    # Combine text columns into a single column
    df = _combine_text_columns(df)

    # Split features and target
    y = df[TARGET].values
    X = df.drop(columns=[TARGET, "credential_hash", "recipient_name"], errors="ignore")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Train class distribution: {np.bincount(y_train)}")

    # Build and fit preprocessor
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse matrices to dense
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # Apply SMOTE on training data only
    if apply_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=RANDOM_SEED)
            X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
            logger.info(f"After SMOTE — Train: {X_train_processed.shape}, Class dist: {np.bincount(y_train)}")
        except ImportError:
            logger.warning("imblearn not installed, skipping SMOTE")

    # Save the preprocessor
    save_sklearn_model(preprocessor, "preprocessor.joblib", metadata={"n_features": X_train_processed.shape[1]})
    logger.info("Saved preprocessor to saved_models/preprocessor.joblib")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def preprocess_single(data: dict, preprocessor=None) -> np.ndarray:
    """Preprocess a single certificate record for inference."""
    if preprocessor is None:
        preprocessor = load_sklearn_model("preprocessor.joblib")

    df = pd.DataFrame([data])
    df = extract_date_features(df)
    df = _combine_text_columns(df)
    df = df.drop(columns=["credential_hash", "recipient_name", TARGET], errors="ignore")

    processed = preprocessor.transform(df)
    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    return processed


def main():
    """Run the preprocessing pipeline."""
    print("Running preprocessing pipeline...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    print(f"\n✅ Preprocessing complete:")
    print(f"   • X_train shape: {X_train.shape}")
    print(f"   • X_test shape:  {X_test.shape}")
    print(f"   • y_train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"   • y_test distribution:  {dict(zip(*np.unique(y_test, return_counts=True)))}")
    print(f"   • Preprocessor saved to: saved_models/preprocessor.joblib")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
