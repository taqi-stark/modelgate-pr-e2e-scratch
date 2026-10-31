"""ModelGate statistics engine.

Reusable statistical utilities: paired bootstrap, confidence intervals,
Holm-Bonferroni correction, and Population Stability Index (PSI).
All functions are deterministic given a numpy RandomState / seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a bootstrap confidence interval estimation."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    confidence: float
    p_value: float = 1.0


def paired_bootstrap_ci(
    y_true: NDArray[np.int_],
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute a paired bootstrap CI for the difference metric_fn(A) - metric_fn(B).

    Both models are evaluated on the same resampled rows to reduce variance.

    Args:
        y_true: True labels.
        scores_a: Predictions from model A (champion).
        scores_b: Predictions from model B (candidate).
        metric_fn: Metric function(y_true, y_pred) -> float.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level for the CI.
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with the point estimate and confidence interval.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        metric_a = metric_fn(y_true[idx], scores_a[idx])
        metric_b = metric_fn(y_true[idx], scores_b[idx])
        diffs[i] = metric_a - metric_b

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    point_estimate = float(metric_fn(y_true, scores_a) - metric_fn(y_true, scores_b))
    p_value = float(np.mean(diffs <= 0))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        p_value=p_value,
    )


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Holm-Bonferroni correction to a list of p-values.

    Args:
        p_values: Raw p-values from multiple hypothesis tests.
        alpha: Family-wise error rate.

    Returns:
        List of booleans: True if the null hypothesis is rejected (significant).
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values while tracking original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * n

    for rank, (original_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            rejected[original_idx] = True
        else:
            # Once we fail to reject, stop (Holm step-down)
            break

    return rejected


def compute_psi(
    expected: NDArray[np.float64],
    actual: NDArray[np.float64],
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    Args:
        expected: Reference distribution values.
        actual: Current distribution values.
        n_bins: Number of bins for histogram.
        eps: Small value to avoid log(0).

    Returns:
        PSI value. Values > 0.25 indicate significant shift.
    """
    # Create bins from the expected distribution
    breakpoints = np.linspace(
        min(np.min(expected), np.min(actual)),
        max(np.max(expected), np.max(actual)),
        n_bins + 1,
    )

    expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
    actual_counts = np.histogram(actual, bins=breakpoints)[0].astype(float)

    # Normalize to proportions
    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi


def compute_psi_categorical(
    expected: NDArray[np.object_],
    actual: NDArray[np.object_],
    eps: float = 1e-4,
) -> float:
    """Compute PSI for categorical features.

    Args:
        expected: Reference categorical values.
        actual: Current categorical values.
        eps: Small value to avoid log(0).

    Returns:
        PSI value.
    """
    # Get all unique categories from both
    all_categories = np.union1d(expected, actual)

    expected_counts = np.array([np.sum(expected == cat) for cat in all_categories], dtype=float)
    actual_counts = np.array([np.sum(actual == cat) for cat in all_categories], dtype=float)

    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi


def ks_test(
    expected: NDArray[np.float64],
    actual: NDArray[np.float64],
) -> tuple[float, float]:
    """Perform a two-sample Kolmogorov-Smirnov test.

    Args:
        expected: Reference distribution.
        actual: Current distribution.

    Returns:
        Tuple of (KS statistic, p-value).
    """
    result = sp_stats.ks_2samp(expected, actual)
    return float(result.statistic), float(result.pvalue)


def demographic_parity_diff(
    y_pred: NDArray[np.int_],
    protected: NDArray[np.object_],
) -> float:
    """Compute the demographic parity difference.

    The maximum difference in positive prediction rate across groups.

    Args:
        y_pred: Binary predictions.
        protected: Protected group labels.

    Returns:
        Demographic parity difference value.
    """
    groups = np.unique(protected)
    if len(groups) < 2:
        return 0.0

    rates = []
    for g in groups:
        mask = protected == g
        if np.sum(mask) == 0:
            continue
        rates.append(float(np.mean(y_pred[mask])))

    return max(rates) - min(rates) if rates else 0.0


def equal_opportunity_diff(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    protected: NDArray[np.object_],
) -> float:
    """Compute equal opportunity difference.

    The maximum difference in true positive rates across groups.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        protected: Protected group labels.

    Returns:
        Equal opportunity difference value.
    """
    groups = np.unique(protected)
    if len(groups) < 2:
        return 0.0

    tpr_rates = []
    for g in groups:
        mask = (protected == g) & (y_true == 1)
        if np.sum(mask) == 0:
            continue
        tpr_rates.append(float(np.mean(y_pred[mask])))

    return max(tpr_rates) - min(tpr_rates) if tpr_rates else 0.0
