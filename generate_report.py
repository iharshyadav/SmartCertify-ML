"""
SmartCertify ML — Comprehensive Model Evaluation Report Generator
Generates both Markdown (.md) and PDF (.pdf) reports with:
  • Full model accuracy benchmarks
  • Confusion matrices, ROC curves, PR curves
  • Dataset statistics & class distributions
  • Image analysis & similarity methodology
  • Trust score regression results
"""

import sys
import os
import json
import logging
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# ── Setup paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.config.settings import (
    MODEL_DIR, PLOTS_DIR, DATASET_PATH,
    TRUST_DATASET_PATH, RANDOM_SEED, TEST_SIZE,
    FRAUD_THRESHOLD, HIGH_RISK_THRESHOLD,
)
from app.utils.model_io import load_sklearn_model
from app.utils.visualization import (
    plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_multi_roc,
    plot_class_distribution, plot_feature_importance,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. DATA LOADING & EVALUATION
# ═══════════════════════════════════════════════════════════════

def load_and_evaluate():
    """Load data, evaluate all models, and return results dict."""
    from app.data.preprocess import prepare_data

    logger.info("Loading and preparing dataset...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(apply_smote=False)

    # ── Dataset stats ────────────────────────────────────────
    raw_df = pd.read_csv(DATASET_PATH)
    dataset_stats = {
        "total_samples": len(raw_df),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "n_features": X_train.shape[1],
        "train_authentic": int(np.sum(y_train == 0)),
        "train_fraudulent": int(np.sum(y_train == 1)),
        "test_authentic": int(np.sum(y_test == 0)),
        "test_fraudulent": int(np.sum(y_test == 1)),
        "fraud_ratio": round(float(np.mean(raw_df["label"] == 1)) * 100, 2),
    }

    # ── Class distribution plot ──────────────────────────────
    plot_class_distribution(y_test, "report_class_distribution.png")

    # ── Evaluate each model ──────────────────────────────────
    model_files = {
        "Logistic Regression": "fraud_lr.joblib",
        "Random Forest":       "fraud_rf.joblib",
        "XGBoost":             "fraud_xgb.joblib",
        "LightGBM":            "fraud_lgbm.joblib",
        "SVM":                 "fraud_svm.joblib",
        "k-NN":                "fraud_knn.joblib",
        "Voting Ensemble":     "fraud_ensemble.joblib",
    }

    all_results = []
    roc_data = {}

    for name, filename in model_files.items():
        model = load_sklearn_model(filename)
        if model is None:
            logger.warning(f"Model {filename} not found, skipping")
            continue

        logger.info(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred.astype(float)
        )

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        }

        cls_report = classification_report(
            y_test, y_pred,
            target_names=["Authentic", "Fraudulent"],
            output_dict=True,
        )

        # Generate plots
        safe_name = name.lower().replace(" ", "_")
        cm_path  = plot_confusion_matrix(y_test, y_pred, name, f"report_cm_{safe_name}.png")
        roc_path = plot_roc_curve(y_test, y_proba, name, f"report_roc_{safe_name}.png")
        pr_path  = plot_precision_recall_curve(y_test, y_proba, name, f"report_pr_{safe_name}.png")

        all_results.append({
            "name": name,
            "metrics": metrics,
            "classification_report": cls_report,
            "plots": {"cm": cm_path, "roc": roc_path, "pr": pr_path},
        })

        roc_data[name] = {"y_true": y_test, "y_proba": y_proba}

        # Feature importance for tree-based
        if hasattr(model, "feature_importances_"):
            n_feat = len(model.feature_importances_)
            feat_names = [f"feature_{i}" for i in range(n_feat)]
            plot_feature_importance(
                model.feature_importances_, feat_names, name,
                filename=f"report_fi_{safe_name}.png",
            )

    # Multi-model ROC comparison
    if len(roc_data) > 1:
        plot_multi_roc(roc_data, "report_multi_roc.png")

    # ── Trust Score evaluation ───────────────────────────────
    trust_results = None
    if Path(TRUST_DATASET_PATH).exists():
        try:
            trust_df = pd.read_csv(TRUST_DATASET_PATH)
            trust_model = load_sklearn_model("trust_regression.joblib")
            trust_scaler = load_sklearn_model("trust_scaler.joblib")

            if trust_model is not None and trust_scaler is not None:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                TRUST_FEATURES = [
                    "total_certificates_issued", "fraud_rate_historical",
                    "avg_metadata_completeness", "domain_age_days",
                    "verification_success_rate", "response_time_avg",
                ]
                X_trust = trust_df[TRUST_FEATURES].values
                y_trust = trust_df["trust_score"].values

                from sklearn.model_selection import train_test_split
                _, X_t_test, _, y_t_test = train_test_split(
                    X_trust, y_trust, test_size=0.2, random_state=RANDOM_SEED
                )
                X_t_test_scaled = trust_scaler.transform(X_t_test)
                y_t_pred = trust_model.predict(X_t_test_scaled)

                trust_results = {
                    "rmse": round(float(np.sqrt(mean_squared_error(y_t_test, y_t_pred))), 4),
                    "mae":  round(float(mean_absolute_error(y_t_test, y_t_pred)), 4),
                    "r2":   round(float(r2_score(y_t_test, y_t_pred)), 4),
                    "n_samples": len(trust_df),
                }
        except Exception as e:
            logger.warning(f"Trust score evaluation failed: {e}")

    return {
        "dataset_stats": dataset_stats,
        "model_results": all_results,
        "trust_results": trust_results,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ═══════════════════════════════════════════════════════════════
# 2. MARKDOWN REPORT GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_markdown_report(results: dict) -> str:
    """Generate a comprehensive Markdown report string."""

    ds = results["dataset_stats"]
    ts = results["timestamp"]

    md = []
    md.append("# SmartCertify ML — Model Evaluation Report")
    md.append(f"\n**Generated:** {ts}  ")
    md.append(f"**Project:** SmartCertify Certificate Fraud Detection System  ")
    md.append(f"**Author:** SmartCertify ML Team\n")

    md.append("---\n")

    # ── Table of Contents ────────────────────────────────────
    md.append("## Table of Contents\n")
    md.append("1. [Executive Summary](#1-executive-summary)")
    md.append("2. [Dataset Overview](#2-dataset-overview)")
    md.append("3. [Methodology](#3-methodology)")
    md.append("4. [Fraud Detection Models](#4-fraud-detection-models)")
    md.append("5. [Model Comparison Benchmark](#5-model-comparison-benchmark)")
    md.append("6. [Confusion Matrices](#6-confusion-matrices)")
    md.append("7. [ROC & Precision-Recall Curves](#7-roc--precision-recall-curves)")
    md.append("8. [Image Analysis Module](#8-image-analysis-module)")
    md.append("9. [Similarity Detection Module](#9-similarity-detection-module)")
    md.append("10. [Trust Score Module](#10-trust-score-module)")
    md.append("11. [Conclusions & Recommendations](#11-conclusions--recommendations)")
    md.append("")

    # ── 1. Executive Summary ─────────────────────────────────
    md.append("---\n")
    md.append("## 1. Executive Summary\n")
    md.append("SmartCertify ML is an AI-powered certificate verification platform that uses machine learning")
    md.append("to detect fraudulent certificates, analyze certificate images for tampering, compute similarity")
    md.append("between certificates, and predict issuer trust scores.\n")

    best = max(results["model_results"], key=lambda r: r["metrics"]["f1"])
    md.append(f"**Key Findings:**")
    md.append(f"- **Best Performing Model:** {best['name']} (F1 = {best['metrics']['f1']}, ROC-AUC = {best['metrics']['roc_auc']})")
    md.append(f"- **Total Models Evaluated:** {len(results['model_results'])}")
    md.append(f"- **Dataset Size:** {ds['total_samples']} samples ({ds['fraud_ratio']}% fraudulent)")
    md.append(f"- **Test Set Size:** {ds['test_samples']} samples")
    md.append("")

    # ── 2. Dataset Overview ──────────────────────────────────
    md.append("---\n")
    md.append("## 2. Dataset Overview\n")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| Total Samples | {ds['total_samples']} |")
    md.append(f"| Training Samples | {ds['train_samples']} |")
    md.append(f"| Test Samples | {ds['test_samples']} |")
    md.append(f"| Number of Features | {ds['n_features']} |")
    md.append(f"| Train — Authentic | {ds['train_authentic']} |")
    md.append(f"| Train — Fraudulent | {ds['train_fraudulent']} |")
    md.append(f"| Test — Authentic | {ds['test_authentic']} |")
    md.append(f"| Test — Fraudulent | {ds['test_fraudulent']} |")
    md.append(f"| Fraud Ratio | {ds['fraud_ratio']}% |")
    md.append(f"| Train/Test Split | {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} |")
    md.append(f"| Random Seed | {RANDOM_SEED} |")
    md.append("")
    md.append(f"![Class Distribution]({PLOTS_DIR / 'report_class_distribution.png'})\n")

    # ── 3. Methodology ───────────────────────────────────────
    md.append("---\n")
    md.append("## 3. Methodology\n")
    md.append("### 3.1 Data Preprocessing\n")
    md.append("- **Text Features:** Issuer name and course name are combined and vectorized using TF-IDF (max 500 features, unigrams + bigrams)")
    md.append("- **Numeric Features:** Standardized using `StandardScaler` with median imputation for missing values")
    md.append("- **Date Features:** Extracted month, year, day-of-week, weekend flag, days-to-expiry, expiration status, and future-issue flag")
    md.append("- **Class Balancing:** SMOTE (Synthetic Minority Over-sampling Technique) available but disabled by default to preserve memory\n")

    md.append("### 3.2 Models Trained\n")
    md.append("| # | Model | Type | Key Hyperparameters |")
    md.append("|---|---|---|---|")
    md.append("| 1 | Logistic Regression | Linear | C=1.0, max_iter=500, balanced weights |")
    md.append("| 2 | k-Nearest Neighbors | Instance-based | Default sklearn parameters |")
    md.append("| 3 | Support Vector Machine | Kernel-based | Default sklearn parameters |")
    md.append("| 4 | Random Forest | Ensemble (Bagging) | n_estimators=100, max_depth=12 |")
    md.append("| 5 | XGBoost | Ensemble (Boosting) | n_estimators=100, lr=0.1, max_depth=6 |")
    md.append("| 6 | LightGBM | Ensemble (Boosting) | n_estimators=100, lr=0.1, balanced weights |")
    md.append("| 7 | Voting Ensemble | Meta-Ensemble | Soft voting over RF + XGBoost + LightGBM |")
    md.append("")

    md.append("### 3.3 Evaluation Metrics\n")
    md.append("- **Accuracy:** Overall correctness of predictions")
    md.append("- **Precision:** Of predicted fraudulent, how many are truly fraudulent")
    md.append("- **Recall (Sensitivity):** Of truly fraudulent, how many are detected")
    md.append("- **F1-Score:** Harmonic mean of precision and recall")
    md.append("- **ROC-AUC:** Area under the Receiver Operating Characteristic curve")
    md.append(f"- **Fraud Threshold:** {FRAUD_THRESHOLD} | **High Risk Threshold:** {HIGH_RISK_THRESHOLD}\n")

    # ── 4. Fraud Detection Models ────────────────────────────
    md.append("---\n")
    md.append("## 4. Fraud Detection Models\n")

    for i, result in enumerate(results["model_results"], 1):
        name = result["name"]
        m = result["metrics"]
        cr = result["classification_report"]

        md.append(f"### 4.{i} {name}\n")

        md.append("| Metric | Score |")
        md.append("|---|---|")
        md.append(f"| Accuracy | {m['accuracy']} |")
        md.append(f"| Precision | {m['precision']} |")
        md.append(f"| Recall | {m['recall']} |")
        md.append(f"| F1-Score | {m['f1']} |")
        md.append(f"| ROC-AUC | {m['roc_auc']} |")
        md.append("")

        # Per-class report
        md.append("**Per-Class Classification Report:**\n")
        md.append("| Class | Precision | Recall | F1-Score | Support |")
        md.append("|---|---|---|---|---|")
        for cls_name in ["Authentic", "Fraudulent"]:
            if cls_name in cr:
                c = cr[cls_name]
                md.append(f"| {cls_name} | {c.get('precision', 'N/A'):.4f} | {c.get('recall', 'N/A'):.4f} | {c.get('f1-score', 'N/A'):.4f} | {int(c.get('support', 0))} |")
        md.append("")

    # ── 5. Model Comparison Benchmark ────────────────────────
    md.append("---\n")
    md.append("## 5. Model Comparison Benchmark\n")
    md.append("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
    md.append("|---|---|---|---|---|---|")
    for r in sorted(results["model_results"], key=lambda x: x["metrics"]["f1"], reverse=True):
        m = r["metrics"]
        md.append(f"| **{r['name']}** | {m['accuracy']} | {m['precision']} | {m['recall']} | {m['f1']} | {m['roc_auc']} |")
    md.append("")

    md.append(f"![Multi-Model ROC Comparison]({PLOTS_DIR / 'report_multi_roc.png'})\n")

    # ── 6. Confusion Matrices ────────────────────────────────
    md.append("---\n")
    md.append("## 6. Confusion Matrices\n")
    for r in results["model_results"]:
        if "cm" in r["plots"]:
            md.append(f"### {r['name']}\n")
            md.append(f"![Confusion Matrix — {r['name']}]({r['plots']['cm']})\n")

    # ── 7. ROC & PR Curves ───────────────────────────────────
    md.append("---\n")
    md.append("## 7. ROC & Precision-Recall Curves\n")
    for r in results["model_results"]:
        md.append(f"### {r['name']}\n")
        if "roc" in r["plots"]:
            md.append(f"![ROC Curve — {r['name']}]({r['plots']['roc']})\n")
        if "pr" in r["plots"]:
            md.append(f"![PR Curve — {r['name']}]({r['plots']['pr']})\n")

    # ── 8. Image Analysis Module ─────────────────────────────
    md.append("---\n")
    md.append("## 8. Image Analysis Module\n")
    md.append("The image analysis module performs **pixel-level tampering detection** on certificate images.\n")
    md.append("### Methodology\n")
    md.append("- Decodes base64-encoded certificate images")
    md.append("- Computes statistical features: mean brightness, standard deviation, channel means")
    md.append("- Uses heuristic scoring rules:")
    md.append("  - Low/high contrast detection (std < 0.05 or > 0.35) → +0.3 tampering score")
    md.append("  - Uneven channel distribution (range > 80) → +0.2 tampering score")
    md.append("  - Excessive uniform regions indicating compression artifacts (> 70%) → +0.2 tampering score")
    md.append("- **Tampering Threshold:** Score > 0.4 → flagged as tampered")
    md.append("- **Method:** Pixel statistics (lightweight, no deep learning required)\n")

    md.append("### Output Fields\n")
    md.append("| Field | Description |")
    md.append("|---|---|")
    md.append("| `is_tampered` | Boolean flag for detected tampering |")
    md.append("| `tamper_probability` | Score from 0.0 to 1.0 |")
    md.append("| `confidence` | Confidence in the prediction |")
    md.append("| `analysis.mean_brightness` | Average pixel brightness |")
    md.append("| `analysis.std_brightness` | Standard deviation of brightness |")
    md.append("| `analysis.channel_means` | Per-channel (RGB) mean values |")
    md.append("")

    # ── 9. Similarity Detection ──────────────────────────────
    md.append("---\n")
    md.append("## 9. Similarity Detection Module\n")
    md.append("The similarity module detects **duplicate or near-duplicate certificates** using text similarity.\n")
    md.append("### Methodology\n")
    md.append("- Builds text representations from certificate fields (issuer, course, recipient, credential hash)")
    md.append("- Vectorizes using **TF-IDF** (max 1000 features, unigrams + bigrams)")
    md.append("- Computes **cosine similarity** between the query certificate and corpus")
    md.append("- Certificates with similarity > 0.9 are flagged as **duplicates**")
    md.append("- Returns top-N most similar certificates ranked by score\n")

    # ── 10. Trust Score Module ───────────────────────────────
    md.append("---\n")
    md.append("## 10. Trust Score Module\n")
    md.append("The trust score module predicts **issuer reliability** using regression models.\n")

    md.append("### Input Features\n")
    md.append("| Feature | Description |")
    md.append("|---|---|")
    md.append("| `total_certificates_issued` | Total certs issued by the institution |")
    md.append("| `fraud_rate_historical` | Historical fraud rate |")
    md.append("| `avg_metadata_completeness` | Average metadata quality |")
    md.append("| `domain_age_days` | Age of the issuer's domain |")
    md.append("| `verification_success_rate` | Rate of successful verifications |")
    md.append("| `response_time_avg` | Average response time (ms) |")
    md.append("")

    if results["trust_results"]:
        tr = results["trust_results"]
        md.append("### Regression Results\n")
        md.append("| Metric | Value |")
        md.append("|---|---|")
        md.append(f"| RMSE | {tr['rmse']} |")
        md.append(f"| MAE | {tr['mae']} |")
        md.append(f"| R² Score | {tr['r2']} |")
        md.append(f"| Dataset Size | {tr['n_samples']} |")
        md.append("")

    md.append("### Trust Grade Scale\n")
    md.append("| Grade | Score Range | Interpretation |")
    md.append("|---|---|---|")
    md.append("| A | 90-100 | Highly trusted issuer |")
    md.append("| B | 75-89 | Trusted issuer |")
    md.append("| C | 60-74 | Moderately trusted |")
    md.append("| D | 40-59 | Low trust |")
    md.append("| F | 0-39 | Untrusted / suspicious |")
    md.append("")

    # ── 11. Conclusions ──────────────────────────────────────
    md.append("---\n")
    md.append("## 11. Conclusions & Recommendations\n")

    md.append("### Key Takeaways\n")
    all_f1 = [r["metrics"]["f1"] for r in results["model_results"]]
    avg_f1 = round(np.mean(all_f1), 4)
    md.append(f"1. **Average F1-Score across all models:** {avg_f1}")
    md.append(f"2. **Best model:** {best['name']} with F1 = {best['metrics']['f1']} and ROC-AUC = {best['metrics']['roc_auc']}")
    md.append(f"3. **All {len(results['model_results'])} models** achieve high accuracy on the test set")
    md.append(f"4. The **Voting Ensemble** combines the strengths of RF, XGBoost, and LightGBM for robust predictions")
    md.append(f"5. The image analysis module provides a lightweight, heuristic-based tampering detection pipeline")
    md.append("")

    md.append("### Recommendations\n")
    md.append("- Deploy the **Voting Ensemble** model for production fraud detection")
    md.append("- Continuously retrain models as new certificate data becomes available")
    md.append("- Consider upgrading image analysis to a CNN-based approach for higher accuracy on complex tampering")
    md.append("- Monitor model drift using periodic evaluation against fresh data")
    md.append("- Expand the trust score dataset for improved issuer reliability predictions\n")

    md.append("---\n")
    md.append(f"*Report generated automatically by SmartCertify ML Evaluation Pipeline — {ts}*")

    return "\n".join(md)


# ═══════════════════════════════════════════════════════════════
# 3. PDF GENERATION
# ═══════════════════════════════════════════════════════════════

def _sanitize(text: str) -> str:
    """Replace Unicode characters with ASCII equivalents for PDF rendering."""
    replacements = {
        '\u2014': '--',   # em dash
        '\u2013': '-',    # en dash
        '\u2019': "'",    # right single quote
        '\u2018': "'",    # left single quote
        '\u201c': '"',    # left double quote
        '\u201d': '"',    # right double quote
        '\u2022': '-',    # bullet
        '\u00b7': '-',    # middle dot
        '\u2192': '->',   # right arrow
        '\u2190': '<-',   # left arrow
        '\u2265': '>=',   # greater than or equal
        '\u2264': '<=',   # less than or equal
        '\u00b2': '2',    # superscript 2
        '\u00b3': '3',    # superscript 3
        '\u2026': '...',  # ellipsis
        '\u00a0': ' ',    # non-breaking space
        '\u2248': '~=',   # approximately equal
        '\u00d7': 'x',    # multiplication sign
    }
    for uni, ascii_char in replacements.items():
        text = text.replace(uni, ascii_char)
    # Remove any remaining non-latin1 characters
    return text.encode('latin-1', errors='replace').decode('latin-1')


def markdown_to_pdf(md_path: Path, pdf_path: Path):
    """Convert Markdown to PDF. Tries multiple methods."""

    # Method 1: Use markdown + fpdf2 (pure Python)
    try:
        from fpdf import FPDF

        class ReportPDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(100, 100, 100)
                self.cell(0, 8, _sanitize("SmartCertify ML -- Model Evaluation Report"), align="C")
                self.ln(10)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(150, 150, 150)
                self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

        pdf = ReportPDF()
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        with open(md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.rstrip("\n")

            try:
                # Skip image lines
                if line.startswith("!["):
                    import re
                    match = re.search(r'\!\[.*?\]\((.*?)\)', line)
                    if match:
                        img_path = match.group(1)
                        if os.path.exists(img_path):
                            try:
                                pdf.ln(4)
                                pdf.image(img_path, w=170)
                                pdf.ln(4)
                            except Exception:
                                pdf.set_font("Helvetica", "I", 9)
                                pdf.set_text_color(150, 0, 0)
                                pdf.cell(0, 6, f"[Image: {os.path.basename(img_path)}]", new_x="LMARGIN", new_y="NEXT")
                    continue

                # Headings
                if line.startswith("# ") and not line.startswith("## "):
                    pdf.ln(6)
                    pdf.set_font("Helvetica", "B", 20)
                    pdf.set_text_color(30, 60, 120)
                    pdf.cell(0, 12, _sanitize(line[2:].strip()), new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(2)
                    continue

                if line.startswith("## "):
                    pdf.ln(6)
                    pdf.set_font("Helvetica", "B", 16)
                    pdf.set_text_color(40, 80, 150)
                    pdf.cell(0, 10, _sanitize(line[3:].strip()), new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(2)
                    continue

                if line.startswith("### "):
                    pdf.ln(4)
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.set_text_color(60, 60, 60)
                    pdf.cell(0, 8, _sanitize(line[4:].strip()), new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(1)
                    continue

                # Horizontal rules
                if line.startswith("---"):
                    pdf.ln(3)
                    pdf.set_draw_color(200, 200, 200)
                    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                    pdf.ln(3)
                    continue

                # Table rows
                if line.startswith("|"):
                    cells = [c.strip() for c in line.split("|")[1:-1]]
                    if all(set(c) <= set("-| ") for c in cells):
                        continue

                    is_header = any(c.startswith("**") for c in cells)
                    pdf.set_font("Courier", "B" if is_header else "", 7)
                    pdf.set_text_color(0, 0, 0)

                    rendered_cells = []
                    for cell in cells:
                        cell_text = cell.replace("**", "").replace("`", "")
                        cell_text = _sanitize(cell_text).strip()
                        if len(cell_text) > 22:
                            cell_text = cell_text[:20] + ".."
                        rendered_cells.append(cell_text.center(22))
                    row_text = " | ".join(rendered_cells)
                    pdf.cell(0, 4, row_text, new_x="LMARGIN", new_y="NEXT")
                    continue

                # Bold text
                if line.startswith("**") and line.endswith("**"):
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(0, 6, _sanitize(line.replace("**", ""))[:80], new_x="LMARGIN", new_y="NEXT")
                    continue

                # Bullet points
                if line.startswith("- ") or line.startswith("  - "):
                    pdf.set_font("Helvetica", "", 9)
                    pdf.set_text_color(30, 30, 30)
                    text = line.lstrip(" -").strip()
                    pdf.cell(0, 5, _sanitize(f"  - {text}")[:120], new_x="LMARGIN", new_y="NEXT")
                    continue

                # Numbered lists
                if len(line) > 2 and line[0].isdigit() and line[1] == ".":
                    pdf.set_font("Helvetica", "", 9)
                    pdf.set_text_color(30, 30, 30)
                    pdf.cell(0, 5, _sanitize(line)[:120], new_x="LMARGIN", new_y="NEXT")
                    continue

                # TOC links / skip markdown links
                if line.strip().startswith("[") or line.strip().startswith("1. ["):
                    continue

                # Normal text
                if line.strip():
                    pdf.set_font("Helvetica", "", 9)
                    pdf.set_text_color(30, 30, 30)
                    clean = line.replace("**", "").replace("*", "").replace("`", "")
                    pdf.cell(0, 5, _sanitize(clean)[:120], new_x="LMARGIN", new_y="NEXT")
                else:
                    pdf.ln(3)

            except Exception as e:
                # Skip any line that causes rendering issues
                logger.debug(f"PDF render skipped line: {e}")

        pdf.output(str(pdf_path))
        logger.info(f"PDF report saved: {pdf_path}")
        return True

    except ImportError:
        logger.warning("fpdf2 not installed. Trying alternative method...")

    # Method 2: If no PDF library available, inform user
    logger.error("No PDF generation library available. Install fpdf2: pip install fpdf2")
    return False


# ═══════════════════════════════════════════════════════════════
# 4. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  SmartCertify ML -- Comprehensive Model Evaluation Report")
    print("=" * 64)

    # Step 1: Evaluate all models
    print("\n[*] Evaluating all models...")
    results = load_and_evaluate()

    # Step 2: Generate Markdown
    print("\n[*] Generating Markdown report...")
    md_content = generate_markdown_report(results)
    md_path = REPORT_DIR / "SmartCertify_ML_Report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"   [OK] Markdown report saved: {md_path}")

    # Step 3: Generate PDF
    print("\n[*] Generating PDF report...")
    pdf_path = REPORT_DIR / "SmartCertify_ML_Report.pdf"
    success = markdown_to_pdf(md_path, pdf_path)
    if success:
        print(f"   [OK] PDF report saved: {pdf_path}")
    else:
        print("   [!!] PDF generation failed. Install fpdf2: pip install fpdf2")
        print("        Then re-run: python generate_report.py")

    # Step 4: Save evaluation data as JSON
    json_path = REPORT_DIR / "evaluation_data.json"
    json_data = {
        "timestamp": results["timestamp"],
        "dataset_stats": results["dataset_stats"],
        "models": [
            {"name": r["name"], "metrics": r["metrics"]}
            for r in results["model_results"]
        ],
        "trust_results": results["trust_results"],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"   [OK] Raw evaluation data: {json_path}")

    # Summary
    print("\n" + "=" * 64)
    print("  EVALUATION SUMMARY")
    print("=" * 64)
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>8}")
    print("  " + "-" * 70)
    for r in sorted(results["model_results"], key=lambda x: x["metrics"]["f1"], reverse=True):
        m = r["metrics"]
        print(f"  {r['name']:<25} {m['accuracy']:>10} {m['precision']:>10} {m['recall']:>8} {m['f1']:>8} {m['roc_auc']:>8}")

    print(f"\n  Reports saved to: {REPORT_DIR}")
    print(f"  Plots saved to:   {PLOTS_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    main()

