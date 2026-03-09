"""
SmartCertify ML — Math Utilities
Linear algebra, statistics, and probability utilities.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Optional


# ─── Linear Algebra Utilities ─────────────────────────────────

def cosine_similarity_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def matrix_rank(matrix: np.ndarray) -> int:
    """Compute rank of a matrix."""
    return int(np.linalg.matrix_rank(matrix))


def compute_svd(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Singular Value Decomposition."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U, S, Vt


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# ─── Statistics Utilities ─────────────────────────────────────

def compute_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute confidence interval for the mean of data."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (float(mean - h), float(mean + h))


def compute_z_score(value: float, mean: float, std: float) -> float:
    """Compute z-score for a value given mean and standard deviation."""
    if std == 0:
        return 0.0
    return (value - mean) / std


def compute_p_value(z_score: float, two_tailed: bool = True) -> float:
    """Compute p-value from z-score."""
    p = 2 * (1 - stats.norm.cdf(abs(z_score))) if two_tailed else (1 - stats.norm.cdf(z_score))
    return float(p)


def ks_test(data: np.ndarray, distribution: str = "norm") -> Tuple[float, float]:
    """Kolmogorov-Smirnov test for distribution fit."""
    statistic, p_value = stats.kstest(data, distribution)
    return float(statistic), float(p_value)


def compute_entropy(probabilities: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence D(P || Q)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Avoid division by zero
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


# ─── Probability Utilities ────────────────────────────────────

def gaussian_probability(x: float, mean: float, std: float) -> float:
    """Compute probability density of x under Gaussian distribution."""
    return float(stats.norm.pdf(x, loc=mean, scale=std))


def bayesian_update(
    prior: float, likelihood: float, evidence: float
) -> float:
    """Apply Bayes' theorem: P(H|E) = P(E|H) * P(H) / P(E)."""
    if evidence == 0:
        return 0.0
    return (likelihood * prior) / evidence


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ─── Feature Analysis ─────────────────────────────────────────

def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix."""
    return np.corrcoef(data, rowvar=False)


def compute_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Compute mutual information between two variables."""
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
    return mi
