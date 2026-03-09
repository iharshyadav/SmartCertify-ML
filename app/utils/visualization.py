"""
SmartCertify ML — Visualization Utilities
Matplotlib/Seaborn plotting functions for model evaluation and data analysis.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from app.config.settings import PLOTS_DIR


def _save_plot(fig: plt.Figure, filename: str) -> str:
    """Save a figure to the plots directory and return the path."""
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def plot_class_distribution(labels: np.ndarray, filename: str = "class_distribution.png") -> str:
    """Plot and save class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    unique, counts = np.unique(labels, return_counts=True)
    colors = ["#2ecc71", "#e74c3c"]
    label_names = ["Authentic", "Fraudulent"]

    bars = ax.bar(label_names[:len(unique)], counts, color=colors[:len(unique)], edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{count}\n({count/sum(counts)*100:.1f}%)",
                ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_title("Certificate Class Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Count", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(counts) * 1.2)
    return _save_plot(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame, filename: str = "correlation_heatmap.png") -> str:
    """Plot and save feature correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
    return _save_plot(fig, filename)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    filename: Optional[str] = None,
) -> str:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Authentic", "Fraud"],
                yticklabels=["Authentic", "Fraud"],
                linewidths=1, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=15, fontweight="bold", pad=12)

    fname = filename or f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    return _save_plot(fig, fname)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    filename: Optional[str] = None,
) -> str:
    """Plot and save ROC-AUC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#3498db", lw=2.5, label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#bdc3c7", lw=1.5, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#3498db")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fname = filename or f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    return _save_plot(fig, fname)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    filename: Optional[str] = None,
) -> str:
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#e74c3c", lw=2.5, label=model_name)
    ax.fill_between(recall, precision, alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fname = filename or f"pr_curve_{model_name.lower().replace(' ', '_')}.png"
    return _save_plot(fig, fname)


def plot_multi_roc(
    results: Dict[str, Dict],
    filename: str = "multi_roc_comparison.png",
) -> str:
    """Plot ROC curves for multiple models on the same figure."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("husl", len(results))

    for (name, data), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve Comparison — All Models", fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _save_plot(fig, filename)


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    filename: str = "nn_learning_curve.png",
) -> str:
    """Plot neural network learning curves (loss and optional accuracy)."""
    n_plots = 2 if train_accs is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, "b-", lw=2, label="Train Loss")
    axes[0].plot(epochs, val_losses, "r-", lw=2, label="Val Loss")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].spines[["top", "right"]].set_visible(False)

    if train_accs is not None:
        axes[1].plot(epochs, train_accs, "b-", lw=2, label="Train Acc")
        axes[1].plot(epochs, val_accs, "r-", lw=2, label="Val Acc")
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=11)
        axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save_plot(fig, filename)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    model_name: str,
    top_n: int = 15,
    filename: Optional[str] = None,
) -> str:
    """Plot and save feature importance bar chart."""
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(indices)), importances[indices], color="#3498db", edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel("Importance", fontsize=13)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=15, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)

    fname = filename or f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    return _save_plot(fig, fname)


def plot_anomaly_distribution(
    scores: np.ndarray,
    threshold: float,
    filename: str = "anomaly_distribution.png",
) -> str:
    """Plot anomaly score distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores, bins=60, color="#3498db", alpha=0.7, edgecolor="white", label="Anomaly Scores")
    ax.axvline(threshold, color="#e74c3c", lw=2, linestyle="--", label=f"Threshold ({threshold:.3f})")
    ax.set_xlabel("Anomaly Score", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title("Anomaly Score Distribution", fontsize=15, fontweight="bold", pad=12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    return _save_plot(fig, filename)


def plot_pca_variance(
    explained_variance: np.ndarray,
    filename: str = "pca_explained_variance.png",
) -> str:
    """Plot PCA explained variance ratio."""
    cumulative = np.cumsum(explained_variance)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, len(explained_variance) + 1), explained_variance,
           alpha=0.7, color="#3498db", label="Individual", edgecolor="white")
    ax.step(range(1, len(explained_variance) + 1), cumulative,
            where="mid", color="#e74c3c", lw=2, label="Cumulative")
    ax.set_xlabel("Principal Component", fontsize=13)
    ax.set_ylabel("Explained Variance Ratio", fontsize=13)
    ax.set_title("PCA Explained Variance", fontsize=15, fontweight="bold", pad=12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    return _save_plot(fig, filename)
