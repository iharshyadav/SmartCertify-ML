"""
generate_synthetic.py — Synthetic certificate data generator.
Produces data/synthetic_certificates.csv with exactly 4000 rows.
Run as: python -m app.data.generate_synthetic
"""
from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd

try:
    from faker import Faker
    fake = Faker("en_IN")
    Faker.seed(42)
except ImportError:
    fake = None

random.seed(42)
np.random.seed(42)

INSTITUTIONS = [
    "IIT Delhi", "IIT Bombay", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
    "NIT Trichy", "NIT Surathkal", "NIT Warangal", "BITS Pilani", "VIT Vellore",
    "Manipal University", "Amity University", "SRM University",
    "Lovely Professional University", "Delhi University", "Mumbai University",
    "Anna University", "Pune University", "Osmania University",
    "Jadavpur University", "Coursera Inc", "Udemy Inc",
    "NPTEL", "SWAYAM", "edX Foundation",
]

COURSES = [
    "B.Tech Computer Science", "B.Tech Information Technology",
    "B.Tech Electronics", "M.Tech Software Engineering",
    "M.Tech Data Science", "MBA Finance", "MBA Marketing",
    "BCA", "MCA", "B.Sc Mathematics", "B.Sc Physics",
    "Machine Learning Fundamentals", "Deep Learning Specialization",
    "Full Stack Web Development", "Python Programming",
    "Data Science with Python", "Cloud Computing AWS",
    "DevOps Engineering", "Cybersecurity Essentials",
    "Digital Marketing", "Project Management Professional",
]

FAKE_ISSUERS = [
    "Global Cert Authority", "QuickDegree.com", "EasyCert Solutions",
    "FastTrack University", "Online Degrees Ltd", "CertHub International",
    "InstaCertify.net", "DegreeFactory LLC",
]


def _random_name() -> str:
    if fake:
        return fake.name()
    first = random.choice(["Amit", "Priya", "Rahul", "Sneha", "Vikram", "Kavya",
                           "Arjun", "Meera", "Rohan", "Ananya"])
    last = random.choice(["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel",
                          "Yadav", "Joshi", "Mehta", "Nair"])
    return f"{first} {last}"


def _random_date(start_year: int = 2015, end_year: int = 2023) -> datetime:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def _make_authentic() -> Dict[str, Any]:
    issuer = random.choice(INSTITUTIONS)
    course = random.choice(COURSES)
    recipient = _random_name()
    issue_date = _random_date(2015, 2023)
    has_expiry = random.random() < 0.4
    expiry_date = (issue_date + timedelta(days=random.randint(365, 1825))
                   if has_expiry else None)
    cert_age = (datetime(2024, 1, 1) - issue_date).days

    issuer_rep = round(float(np.clip(np.random.beta(8, 2), 0.6, 1.0)), 4)
    template = round(float(np.clip(np.random.beta(8, 2), 0.7, 1.0)), 4)
    metadata = round(float(np.clip(np.random.beta(9, 2), 0.75, 1.0)), 4)
    domain_ok = 1
    prev_verif = random.randint(0, 15)
    issuer_count = random.randint(500, 50000)
    total_issued = random.randint(1000, 100000)
    fraud_rate = round(float(np.clip(np.random.beta(1, 20), 0.0, 0.08)), 4)
    avg_meta = round(float(np.clip(np.random.beta(9, 2), 0.75, 1.0)), 4)
    domain_age = random.randint(365, 7300)
    verif_success = round(float(np.clip(np.random.beta(9, 2), 0.85, 1.0)), 4)
    trust = round(float(np.clip(
        0.4 * issuer_rep + 0.2 * template + 0.15 * metadata +
        0.1 * domain_ok + 0.15 * verif_success,
        0.6, 1.0
    )), 4)

    return {
        "cert_id": f"CERT-{random.randint(100000, 999999)}",
        "issuer_name": issuer,
        "recipient_name": recipient,
        "course_name": course,
        "issue_date": issue_date.strftime("%Y-%m-%d"),
        "expiry_date": expiry_date.strftime("%Y-%m-%d") if expiry_date else "",
        "issuer_reputation_score": issuer_rep,
        "template_match_score": template,
        "metadata_completeness_score": metadata,
        "domain_verification_status": domain_ok,
        "previous_verification_count": prev_verif,
        "cert_age_days": cert_age,
        "issuer_cert_count": issuer_count,
        "has_expiry": int(has_expiry),
        "name_length": len(recipient),
        "course_name_length": len(course),
        "total_certificates_issued": total_issued,
        "fraud_rate_historical": fraud_rate,
        "avg_metadata_completeness": avg_meta,
        "domain_age_days": domain_age,
        "verification_success_rate": verif_success,
        "trust_score": trust,
        "label": "authentic",
    }


def _make_tampered() -> Dict[str, Any]:
    row = _make_authentic()
    row["label"] = "tampered"
    # Corrupt 1-2 fields
    corruptions = random.sample(
        ["rep", "expiry", "meta", "template", "domain"], k=random.randint(1, 2)
    )
    if "rep" in corruptions:
        # Unrealistically high reputation
        row["issuer_reputation_score"] = round(random.uniform(0.93, 1.0), 4)
    if "expiry" in corruptions and row["has_expiry"]:
        # Expiry before issue date
        issue = datetime.strptime(row["issue_date"], "%Y-%m-%d")
        row["expiry_date"] = (
            issue - timedelta(days=random.randint(1, 365))
        ).strftime("%Y-%m-%d")
    if "meta" in corruptions:
        row["metadata_completeness_score"] = round(random.uniform(0.0, 0.15), 4)
    if "template" in corruptions:
        row["template_match_score"] = round(random.uniform(0.0, 0.25), 4)
    if "domain" in corruptions:
        row["domain_verification_status"] = 0
    # Pull trust score down
    row["trust_score"] = round(float(np.clip(
        row["trust_score"] - random.uniform(0.15, 0.40), 0.05, 0.65
    )), 4)
    return row


def _make_fake() -> Dict[str, Any]:
    recipient = _random_name()
    course = random.choice(COURSES)
    issue_date = _random_date(2010, 2024)
    has_expiry = random.random() < 0.6
    expiry_date = (issue_date + timedelta(days=random.randint(30, 365))
                   if has_expiry else None)
    cert_age = (datetime(2024, 1, 1) - issue_date).days

    issuer_rep = round(random.uniform(0.0, 0.35), 4)
    template = round(random.uniform(0.0, 0.4), 4)
    metadata = round(random.uniform(0.0, 0.3), 4)
    domain_ok = 0
    prev_verif = random.randint(0, 3)
    issuer_count = random.randint(1, 100)
    total_issued = random.randint(1, 200)
    fraud_rate = round(random.uniform(0.15, 0.50), 4)
    avg_meta = round(random.uniform(0.0, 0.3), 4)
    domain_age = random.randint(1, 180)
    verif_success = round(random.uniform(0.0, 0.4), 4)
    trust = round(float(np.clip(
        0.4 * issuer_rep + 0.2 * template + 0.15 * metadata +
        0.1 * domain_ok + 0.15 * verif_success,
        0.0, 0.35
    )), 4)

    return {
        "cert_id": f"CERT-{random.randint(100000, 999999)}",
        "issuer_name": random.choice(FAKE_ISSUERS),
        "recipient_name": recipient,
        "course_name": course,
        "issue_date": issue_date.strftime("%Y-%m-%d"),
        "expiry_date": expiry_date.strftime("%Y-%m-%d") if expiry_date else "",
        "issuer_reputation_score": issuer_rep,
        "template_match_score": template,
        "metadata_completeness_score": metadata,
        "domain_verification_status": domain_ok,
        "previous_verification_count": prev_verif,
        "cert_age_days": cert_age,
        "issuer_cert_count": issuer_count,
        "has_expiry": int(has_expiry),
        "name_length": len(recipient),
        "course_name_length": len(course),
        "total_certificates_issued": total_issued,
        "fraud_rate_historical": fraud_rate,
        "avg_metadata_completeness": avg_meta,
        "domain_age_days": domain_age,
        "verification_success_rate": verif_success,
        "trust_score": trust,
        "label": "fake",
    }


def generate_all(
    n_authentic: int = 12_500,
    n_tampered: int = 7_500,
    n_fake: int = 5_000,
) -> pd.DataFrame:
    """
    Generate synthetic certificate data.
    Default: 25,000 rows (50% authentic, 30% tampered, 20% fake).
    Increase numbers for more data — runs in seconds.
    """
    print(f"  Generating {n_authentic} authentic...")
    records: List[Dict] = [_make_authentic() for _ in range(n_authentic)]
    print(f"  Generating {n_tampered} tampered...")
    records += [_make_tampered() for _ in range(n_tampered)]
    print(f"  Generating {n_fake} fake...")
    records += [_make_fake() for _ in range(n_fake)]
    random.shuffle(records)
    return pd.DataFrame(records)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--authentic", type=int, default=12_500)
    parser.add_argument("--tampered",  type=int, default=7_500)
    parser.add_argument("--fake",      type=int, default=5_000)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    df = generate_all(
        n_authentic=args.authentic,
        n_tampered=args.tampered,
        n_fake=args.fake,
    )
    df.to_csv("data/synthetic_certificates.csv", index=False)
    total = len(df)
    print(f"Generated {total} records -> data/synthetic_certificates.csv")
    print(df["label"].value_counts().to_string())
    print(f"\nColumns: {list(df.columns)}")
