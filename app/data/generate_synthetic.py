"""
SmartCertify ML — Synthetic Data Generator
Generates 15,000 synthetic certificate records with realistic fraud patterns.
"""

import numpy as np
import pandas as pd
import hashlib
import string
import logging
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config.settings import DATASET_SIZE, FRAUD_RATIO, RANDOM_SEED, PLOTS_DIR, DATASET_PATH
from app.config.settings import TRUST_DATASET_PATH, TIMESERIES_DATASET_PATH, RECOMMENDATION_DATASET_PATH

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────

ISSUERS = [
    "MIT", "Stanford University", "Harvard University", "Google",
    "Microsoft", "Coursera", "edX", "Udemy", "AWS", "IBM",
    "Oracle", "Cisco", "CompTIA", "PMI", "Salesforce",
    "Adobe", "University of Oxford", "Cambridge University",
    "IIT Delhi", "NUS Singapore", "ETH Zurich", "TU Munich",
    "University of Toronto", "Johns Hopkins University",
    "Columbia University", "Yale University",
    "FakeUniversity123", "DiplomaMill Online", "QuickCert Ltd",
    "InstaDegree Corp", "CertForge Academy",
]

COURSES = [
    "Machine Learning Fundamentals", "Data Science Professional",
    "Cloud Architecture", "Cybersecurity Analyst",
    "Project Management Professional", "Full Stack Development",
    "AI & Deep Learning", "Blockchain Development",
    "DevOps Engineering", "Database Administration",
    "Business Analytics", "Digital Marketing",
    "Network Engineering", "Mobile App Development",
    "UI/UX Design", "Data Engineering",
    "Software Testing", "Agile Methodology",
    "Big Data Analytics", "Computer Vision",
    "Natural Language Processing", "Robotics Engineering",
    "IoT Development", "Quantum Computing Basics",
]

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer",
    "Michael", "Linda", "David", "Elizabeth", "William", "Barbara",
    "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
    "Christopher", "Karen", "Arun", "Priya", "Wei", "Yuki",
    "Mohammed", "Fatima", "Carlos", "Ana", "Pierre", "Sophie",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor",
    "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson",
    "Kumar", "Patel", "Chen", "Wang", "Tanaka", "Kim",
    "Ali", "Hassan", "Santos", "Mueller", "Dubois", "Rossi",
]


def _generate_hash(seed_str: str, corrupt: bool = False) -> str:
    """Generate a SHA-256 hash from a seed string, optionally corrupted."""
    h = hashlib.sha256(seed_str.encode()).hexdigest()
    if corrupt:
        # Truncate or add random chars to simulate hash mismatch
        corruption_type = np.random.choice(["truncate", "alter", "short"])
        if corruption_type == "truncate":
            h = h[:np.random.randint(10, 30)]
        elif corruption_type == "alter":
            pos = np.random.randint(0, len(h))
            replacement = np.random.choice(list(string.ascii_lowercase + string.digits))
            h = h[:pos] + replacement + h[pos + 1:]
        else:
            h = h[:8]
    return h


def generate_certificates_dataset(n_samples: int = DATASET_SIZE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic certificate dataset with realistic fraud patterns."""
    np.random.seed(seed)

    n_fraud = int(n_samples * FRAUD_RATIO)
    n_authentic = n_samples - n_fraud

    records = []
    base_date = datetime(2020, 1, 1)
    now = datetime(2025, 12, 31)

    # ── Generate authentic certificates ──────────────────────
    for i in range(n_authentic):
        issuer = np.random.choice(ISSUERS[:26])  # Legitimate issuers
        recipient = f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}"
        course = np.random.choice(COURSES)

        issue_date = base_date + timedelta(days=np.random.randint(0, (now - base_date).days))
        expiry_date = issue_date + timedelta(days=np.random.randint(365, 1825))

        seed_str = f"{issuer}-{recipient}-{course}-{issue_date.isoformat()}-{i}"
        credential_hash = _generate_hash(seed_str, corrupt=False)

        records.append({
            "issuer_name": issuer,
            "recipient_name": recipient,
            "course_name": course,
            "issue_date": issue_date.strftime("%Y-%m-%d"),
            "expiry_date": expiry_date.strftime("%Y-%m-%d"),
            "credential_hash": credential_hash,
            "issuer_reputation_score": np.clip(np.random.beta(8, 2), 0, 1),
            "certificate_age_days": (now - issue_date).days,
            "metadata_completeness_score": np.clip(np.random.beta(7, 1.5), 0, 1),
            "ocr_confidence_score": np.clip(np.random.normal(0.92, 0.05), 0, 1),
            "template_match_score": np.clip(np.random.beta(9, 1.5), 0, 1),
            "domain_verification_status": int(np.random.random() < 0.95),
            "previous_verification_count": np.random.poisson(5) + 1,
            "time_since_last_verification_days": np.random.exponential(30),
            "label": 0,
        })

    # ── Generate fraudulent certificates ─────────────────────
    for i in range(n_fraud):
        fraud_type = np.random.choice(["low_reputation", "future_date", "zero_verification",
                                        "hash_mismatch", "template_fail", "multi_flag"],
                                       p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2])

        issuer = np.random.choice(ISSUERS[26:]) if np.random.random() < 0.6 else np.random.choice(ISSUERS[:26])
        recipient = f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}"
        course = np.random.choice(COURSES)

        if fraud_type == "future_date":
            issue_date = now + timedelta(days=np.random.randint(1, 365))
        else:
            issue_date = base_date + timedelta(days=np.random.randint(0, (now - base_date).days))

        expiry_date = issue_date + timedelta(days=np.random.randint(30, 1825))

        seed_str = f"{issuer}-{recipient}-{course}-{issue_date.isoformat()}-fraud-{i}"
        corrupt_hash = fraud_type in ("hash_mismatch", "multi_flag") and np.random.random() < 0.7
        credential_hash = _generate_hash(seed_str, corrupt=corrupt_hash)

        # Base scores for fraud — generally worse
        rep_score = np.clip(np.random.beta(2, 8), 0, 1)
        meta_score = np.clip(np.random.beta(2, 5), 0, 1)
        ocr_score = np.clip(np.random.normal(0.65, 0.15), 0, 1)
        template_score = np.clip(np.random.beta(2, 7), 0, 1)
        domain_status = int(np.random.random() < 0.3)
        prev_verifications = 0 if fraud_type in ("zero_verification", "multi_flag") else np.random.poisson(1)
        time_since = np.random.exponential(180) if fraud_type == "zero_verification" else np.random.exponential(60)

        # Override based on fraud type
        if fraud_type == "low_reputation":
            rep_score = np.clip(np.random.beta(1, 12), 0, 0.3)
            template_score = np.clip(np.random.beta(2, 5), 0, 0.5)
        elif fraud_type == "template_fail":
            template_score = np.clip(np.random.beta(1, 10), 0, 0.25)
            meta_score = np.clip(np.random.beta(2, 6), 0, 0.4)
        elif fraud_type == "multi_flag":
            rep_score = np.clip(np.random.beta(1.5, 10), 0, 0.35)
            template_score = np.clip(np.random.beta(1.5, 8), 0, 0.35)
            meta_score = np.clip(np.random.beta(1.5, 6), 0, 0.4)
            domain_status = 0

        records.append({
            "issuer_name": issuer,
            "recipient_name": recipient,
            "course_name": course,
            "issue_date": issue_date.strftime("%Y-%m-%d"),
            "expiry_date": expiry_date.strftime("%Y-%m-%d"),
            "credential_hash": credential_hash,
            "issuer_reputation_score": round(rep_score, 4),
            "certificate_age_days": (now - issue_date).days,
            "metadata_completeness_score": round(meta_score, 4),
            "ocr_confidence_score": round(ocr_score, 4),
            "template_match_score": round(template_score, 4),
            "domain_verification_status": domain_status,
            "previous_verification_count": prev_verifications,
            "time_since_last_verification_days": round(time_since, 2),
            "label": 1,
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Introduce ~2% random missing values in some columns
    for col in ["ocr_confidence_score", "template_match_score", "metadata_completeness_score"]:
        mask = np.random.random(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    return df


def generate_trust_score_dataset(n_issuers: int = 500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic issuer trust score dataset for regression."""
    np.random.seed(seed)

    records = []
    for i in range(n_issuers):
        total_certs = np.random.randint(10, 5000)
        fraud_rate = np.clip(np.random.beta(1, 15), 0, 0.5)
        completeness = np.clip(np.random.beta(6, 2), 0, 1)
        domain_age = np.random.randint(30, 7300)  # 1 month to 20 years
        success_rate = np.clip(np.random.beta(8, 2), 0, 1)
        response_time = np.random.exponential(50)  # avg ms

        # Compute trust score as a function of features + noise
        trust_score = (
            (1 - fraud_rate) * 25
            + completeness * 20
            + min(domain_age / 3650, 1) * 20
            + success_rate * 25
            + max(0, 10 - response_time / 50) * 1
            + np.random.normal(0, 3)
        )
        trust_score = np.clip(trust_score, 0, 100)

        records.append({
            "issuer_id": f"ISS-{i:04d}",
            "total_certificates_issued": total_certs,
            "fraud_rate_historical": round(fraud_rate, 4),
            "avg_metadata_completeness": round(completeness, 4),
            "domain_age_days": domain_age,
            "verification_success_rate": round(success_rate, 4),
            "response_time_avg": round(response_time, 2),
            "trust_score": round(trust_score, 2),
        })

    return pd.DataFrame(records)


def generate_timeseries_dataset(n_days: int = 730, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic daily verification count time series."""
    np.random.seed(seed)

    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    base = 100 + np.arange(n_days) * 0.1  # Slight upward trend
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # Yearly seasonality
    weekly = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # Weekly pattern
    noise = np.random.normal(0, 8, n_days)

    counts = np.maximum(base + seasonal + weekly + noise, 1).astype(int)

    return pd.DataFrame({"date": dates, "verification_count": counts})


def generate_recommendation_dataset(n_students: int = 1000, n_courses: int = 50, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic student-course interaction data for recommendations."""
    np.random.seed(seed)

    records = []
    course_names = COURSES + [f"Advanced {c}" for c in COURSES]
    # Ensure we don't exceed available courses
    n_courses = min(n_courses, len(course_names))
    course_names = course_names[:n_courses]

    for student_id in range(n_students):
        n_completed = np.random.randint(1, 8)
        completed_indices = np.random.choice(n_courses, n_completed, replace=False)

        for idx in completed_indices:
            records.append({
                "student_id": f"STU-{student_id:04d}",
                "course_name": course_names[idx],
                "course_id": idx,
                "rating": np.clip(np.random.normal(3.8, 0.8), 1, 5),
                "completion_pct": np.clip(np.random.normal(85, 15), 10, 100),
                "skills_gained": ", ".join(np.random.choice(
                    ["python", "ml", "data", "cloud", "security", "web",
                     "analytics", "devops", "ai", "database", "networking"],
                    size=np.random.randint(2, 5), replace=False
                )),
            })

    return pd.DataFrame(records)


def main():
    """Generate all synthetic datasets and save to disk."""
    logger.info("Generating synthetic certificate dataset...")
    df = generate_certificates_dataset()
    Path(DATASET_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    logger.info(f"Saved {len(df)} records to {DATASET_PATH}")
    logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")

    # Skip plots in production to save memory
    try:
        from app.utils.visualization import plot_class_distribution, plot_correlation_heatmap
        plot_class_distribution(df["label"].values)
        plot_correlation_heatmap(df)
        logger.info("Saved distribution and correlation plots")
    except Exception:
        logger.info("Skipping plots (memory optimization)")

    # Trust score dataset
    trust_df = generate_trust_score_dataset()
    trust_df.to_csv(TRUST_DATASET_PATH, index=False)
    logger.info(f"Saved {len(trust_df)} issuer records to {TRUST_DATASET_PATH}")

    # Time series dataset
    ts_df = generate_timeseries_dataset()
    ts_df.to_csv(TIMESERIES_DATASET_PATH, index=False)
    logger.info(f"Saved {len(ts_df)} time series records to {TIMESERIES_DATASET_PATH}")

    # Recommendation dataset
    rec_df = generate_recommendation_dataset()
    rec_df.to_csv(RECOMMENDATION_DATASET_PATH, index=False)
    logger.info(f"Saved {len(rec_df)} interaction records to {RECOMMENDATION_DATASET_PATH}")

    print(f"\n✅ Generated all datasets:")
    print(f"   • Certificates:   {len(df):,} records → {DATASET_PATH}")
    print(f"   • Trust Scores:   {len(trust_df):,} records → {TRUST_DATASET_PATH}")
    print(f"   • Time Series:    {len(ts_df):,} records → {TIMESERIES_DATASET_PATH}")
    print(f"   • Interactions:   {len(rec_df):,} records → {RECOMMENDATION_DATASET_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
