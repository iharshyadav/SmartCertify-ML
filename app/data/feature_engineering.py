"""
SmartCertify ML — Feature Engineering
All feature transformations: date features, text features, risk scores, hash validation, PCA.
"""

import numpy as np
import pandas as pd
import hashlib
import re
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import PLOTS_DIR
from app.utils.visualization import plot_pca_variance

logger = logging.getLogger(__name__)


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract date-based features from issue_date and expiry_date.
    Returns: issue_month, issue_year, days_to_expiry, is_expired, weekend_issued
    """
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

    # Future issue date flag
    if "issue_date" in df.columns:
        now = pd.Timestamp.now()
        df["is_future_issue"] = (issue_date > now).astype(int)

    return df


def compute_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract text-based features from name and course fields.
    Returns: name_length, course_word_count, special_char_ratio
    """
    df = df.copy()

    if "recipient_name" in df.columns:
        df["name_length"] = df["recipient_name"].fillna("").str.len()
        df["name_word_count"] = df["recipient_name"].fillna("").str.split().str.len()

    if "course_name" in df.columns:
        df["course_word_count"] = df["course_name"].fillna("").str.split().str.len()
        df["course_name_length"] = df["course_name"].fillna("").str.len()

    if "issuer_name" in df.columns:
        df["issuer_name_length"] = df["issuer_name"].fillna("").str.len()
        # Special character ratio — unusual characters can indicate fraud
        df["special_char_ratio"] = df["issuer_name"].fillna("").apply(
            lambda x: len(re.findall(r"[^a-zA-Z0-9\s]", x)) / max(len(x), 1)
        )

    return df


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite risk score from reputation, completeness, and template match.
    Higher score = higher risk of fraud.
    """
    df = df.copy()

    rep = 1 - df.get("issuer_reputation_score", pd.Series(0.5, index=df.index)).fillna(0.5)
    comp = 1 - df.get("metadata_completeness_score", pd.Series(0.5, index=df.index)).fillna(0.5)
    tmpl = 1 - df.get("template_match_score", pd.Series(0.5, index=df.index)).fillna(0.5)
    domain = 1 - df.get("domain_verification_status", pd.Series(1, index=df.index)).fillna(1)

    # Weighted composite risk score
    df["risk_score"] = (rep * 0.3 + comp * 0.2 + tmpl * 0.3 + domain * 0.2)

    # Risk category
    df["risk_category"] = pd.cut(
        df["risk_score"],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        include_lowest=True,
    )

    return df


def hash_integrity_check(credential_hash: str) -> Dict[str, Any]:
    """
    Validate hash format, length, and entropy.
    Returns dict with validation results.
    """
    result = {
        "is_valid_format": False,
        "is_valid_length": False,
        "entropy": 0.0,
        "hash_quality": "INVALID",
    }

    if not credential_hash or not isinstance(credential_hash, str):
        return result

    # Check hex format
    result["is_valid_format"] = bool(re.match(r"^[0-9a-f]+$", credential_hash.lower()))

    # Check length (SHA-256 = 64 hex chars)
    result["is_valid_length"] = len(credential_hash) == 64

    # Compute Shannon entropy
    if len(credential_hash) > 0:
        probs = np.array([credential_hash.count(c) / len(credential_hash) for c in set(credential_hash)])
        result["entropy"] = float(-np.sum(probs * np.log2(probs + 1e-10)))

    # Quality assessment
    if result["is_valid_format"] and result["is_valid_length"] and result["entropy"] > 3.5:
        result["hash_quality"] = "VALID"
    elif result["is_valid_format"] and result["entropy"] > 2.5:
        result["hash_quality"] = "SUSPICIOUS"
    else:
        result["hash_quality"] = "INVALID"

    return result


def add_hash_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hash-derived features to the dataframe."""
    df = df.copy()

    if "credential_hash" in df.columns:
        hash_results = df["credential_hash"].fillna("").apply(hash_integrity_check)
        df["hash_valid_format"] = hash_results.apply(lambda x: int(x["is_valid_format"]))
        df["hash_valid_length"] = hash_results.apply(lambda x: int(x["is_valid_length"]))
        df["hash_entropy"] = hash_results.apply(lambda x: x["entropy"])

    return df


def apply_pca(X: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA dimensionality reduction and save variance plot.
    Returns transformed data and fitted PCA object.
    """
    n_components = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    logger.info(f"PCA: {n_components} components explain {pca.explained_variance_ratio_.sum():.2%} variance")

    # Save explained variance plot
    try:
        plot_pca_variance(pca.explained_variance_ratio_)
    except Exception as e:
        logger.warning(f"Could not save PCA plot: {e}")

    return X_pca, pca


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    df = extract_date_features(df)
    df = compute_text_features(df)
    df = compute_risk_score(df)
    df = add_hash_features(df)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def main():
    """Run feature engineering on the dataset."""
    from app.config.settings import DATASET_PATH

    print("Running feature engineering pipeline...")
    df = pd.read_csv(DATASET_PATH)
    df = engineer_all_features(df)

    # Show new features
    new_cols = [c for c in df.columns if c not in pd.read_csv(DATASET_PATH).columns]
    print(f"\n✅ Feature engineering complete:")
    print(f"   • Original columns: {len(pd.read_csv(DATASET_PATH).columns)}")
    print(f"   • New columns: {len(new_cols)}")
    print(f"   • Total columns: {len(df.columns)}")
    print(f"   • New features: {new_cols}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
