"""ModelGate statistics engine.

Reusable statistical utilities: paired bootstrap, confidence intervals,
Holm-Bonferroni correction, and Population Stability Index (PSI).
All functions are deterministic given a numpy RandomState / seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def _vectorized_paired_bootstrap(
    y_true: NDArray[Any],
    scores_a: NDArray[Any],
    scores_b: NDArray[Any],
    metric_name: str,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> BootstrapResult | None:
    """Vectorized paired bootstrap using multinomial weight matrix and confusion counts."""
    n = len(y_true)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, n_bootstrap, confidence, 1.0)

    rng = np.random.RandomState(seed)
    try:
        weights = rng.multinomial(n, [1.0 / n] * n, size=n_bootstrap).astype(np.float64)
    except Exception:
        return None

    if metric_name in ("f1", "weighted_f1"):
        classes = np.unique(y_true)

        def _calc_f1(scores: NDArray[Any]) -> NDArray[np.float64]:
            weighted_f1_sum = np.zeros(n_bootstrap, dtype=np.float64)
            total_support = np.zeros(n_bootstrap, dtype=np.float64)
            for c in classes:
                tp_mask = ((y_true == c) & (scores == c)).astype(np.float64)
                fp_mask = ((y_true != c) & (scores == c)).astype(np.float64)
                fn_mask = ((y_true == c) & (scores != c)).astype(np.float64)
                supp_mask = (y_true == c).astype(np.float64)

                tp = weights @ tp_mask
                fp = weights @ fp_mask
                fn = weights @ fn_mask
                supp = weights @ supp_mask

                denom = 2.0 * tp + fp + fn
                f1_c = np.where(denom > 0, (2.0 * tp) / np.maximum(denom, 1e-12), 0.0)
                weighted_f1_sum += f1_c * supp
                total_support += supp
            return np.where(
                total_support > 0,
                weighted_f1_sum / np.maximum(total_support, 1e-12),
                0.0,
            )

        f1_a = _calc_f1(scores_a)
        f1_b = _calc_f1(scores_b)
        diffs = f1_a - f1_b

        def _full_f1(scores: NDArray[Any]) -> float:
            tot_sum = 0.0
            tot_supp = 0.0
            for c in classes:
                tp = float(np.sum((y_true == c) & (scores == c)))
                fp = float(np.sum((y_true != c) & (scores == c)))
                fn = float(np.sum((y_true == c) & (scores != c)))
                supp = float(np.sum(y_true == c))
                denom = 2.0 * tp + fp + fn
                f1_c = (2.0 * tp / denom) if denom > 0 else 0.0
                tot_sum += f1_c * supp
                tot_supp += supp
            return tot_sum / tot_supp if tot_supp > 0 else 0.0

        point_estimate = float(_full_f1(scores_a) - _full_f1(scores_b))

    elif metric_name in ("accuracy", "acc"):
        correct_a = (y_true == scores_a).astype(np.float64)
        correct_b = (y_true == scores_b).astype(np.float64)
        acc_a = (weights @ correct_a) / float(n)
        acc_b = (weights @ correct_b) / float(n)
        diffs = acc_a - acc_b
        point_estimate = float(np.mean(correct_a) - np.mean(correct_b))
    else:
        return None

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    p_value = float(np.mean(diffs <= 0))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        p_value=p_value,
    )


def paired_bootstrap_ci(
    y_true: NDArray[np.int_],
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    metric_fn: Any,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    metric_name: str | None = None,
) -> BootstrapResult:
    """Compute a paired bootstrap CI for the difference metric_fn(A) - metric_fn(B).

    Both models are evaluated on the same resampled rows to reduce variance.
    Uses vectorized multinomial sampling for F1 and accuracy metrics.

    Args:
        y_true: True labels.
        scores_a: Predictions from model A (champion).
        scores_b: Predictions from model B (candidate).
        metric_fn: Metric function(y_true, y_pred) -> float.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level for the CI.
        seed: Random seed for reproducibility.
        metric_name: Optional metric name ('f1', 'accuracy') to enable vectorization.

    Returns:
        BootstrapResult with the point estimate and confidence interval.
    """
    if metric_name is None:
        fn_name = getattr(metric_fn, "__name__", "")
        if "accuracy" in fn_name:
            metric_name = "accuracy"
        elif "f1" in fn_name:
            metric_name = "f1"

    if metric_name in ("f1", "weighted_f1", "accuracy", "acc"):
        vec_res = _vectorized_paired_bootstrap(
            y_true, scores_a, scores_b, metric_name, n_bootstrap, confidence, seed
        )
        if vec_res is not None:
            return vec_res

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

    return max(tpr_rates) - min(tpr_rates) if len(tpr_rates) >= 2 else 0.0


def equalized_odds_diff(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    protected: NDArray[np.object_],
) -> float:
    """Compute equalized odds difference across protected groups.

    Defined as max(max_g TPR_g - min_g TPR_g, max_g FPR_g - min_g FPR_g).

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        protected: Protected group labels.

    Returns:
        Equalized odds difference in [0, 1].
    """
    groups = np.unique(protected)
    if len(groups) < 2:
        return 0.0

    tpr_rates: list[float] = []
    fpr_rates: list[float] = []
    for g in groups:
        pos_mask = (protected == g) & (y_true == 1)
        neg_mask = (protected == g) & (y_true == 0)
        if np.sum(pos_mask) > 0:
            tpr_rates.append(float(np.mean(y_pred[pos_mask])))
        if np.sum(neg_mask) > 0:
            fpr_rates.append(float(np.mean(y_pred[neg_mask])))

    tpr_gap = (max(tpr_rates) - min(tpr_rates)) if len(tpr_rates) >= 2 else 0.0
    fpr_gap = (max(fpr_rates) - min(fpr_rates)) if len(fpr_rates) >= 2 else 0.0
    return float(max(tpr_gap, fpr_gap))


def per_group_calibration(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    protected: NDArray[np.object_],
) -> list[dict[str, Any]]:
    """Compute per-group calibration metrics (mean predicted probability, base rate, calibration error, Brier score)."""
    groups = np.unique(protected)
    y_true_f = np.asarray(y_true, dtype=np.float64)
    y_prob_f = np.asarray(y_prob, dtype=np.float64)
    results: list[dict[str, Any]] = []
    for g in groups:
        mask = protected == g
        n_g = int(np.sum(mask))
        if n_g == 0:
            continue
        mean_prob = float(np.mean(y_prob_f[mask]))
        base_rate = float(np.mean(y_true_f[mask]))
        cal_err = abs(mean_prob - base_rate)
        brier = float(np.mean((y_prob_f[mask] - y_true_f[mask]) ** 2))
        results.append(
            {
                "group": str(g),
                "n": n_g,
                "mean_pred_prob": round(mean_prob, 4),
                "base_rate": round(base_rate, 4),
                "calibration_error": round(cal_err, 4),
                "brier_score": round(brier, 4),
            }
        )
    return results


def calibration_gap_diff(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    protected: NDArray[np.object_],
) -> float:
    """Compute the maximum difference in group calibration error |E[p_hat|G=g] - E[Y|G=g]| across protected groups."""
    groups = np.unique(protected)
    if len(groups) < 2:
        return 0.0

    y_true_f = np.asarray(y_true, dtype=np.float64)
    y_prob_f = np.asarray(y_prob, dtype=np.float64)
    errors: list[float] = []
    for g in groups:
        mask = protected == g
        if np.sum(mask) == 0:
            continue
        errors.append(abs(float(np.mean(y_prob_f[mask]) - np.mean(y_true_f[mask]))))

    return float(max(errors) - min(errors)) if len(errors) >= 2 else 0.0


def paired_bootstrap_gap_ci(
    y_true: NDArray[np.int_],
    champ_scores: NDArray[Any],
    cand_scores: NDArray[Any],
    protected: NDArray[np.object_],
    metric: str = "demographic_parity_diff",
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute a paired bootstrap CI on the change in group fairness gap: gap(candidate) - gap(champion).

    A positive point_estimate / ci_lower > 0 means the candidate model has a larger (worse)
    between-group disparity than the champion model on the same resampled test rows.
    """
    y_arr = np.asarray(y_true, dtype=np.float64)
    a_arr = np.asarray(champ_scores, dtype=np.float64)
    b_arr = np.asarray(cand_scores, dtype=np.float64)
    prot_arr = np.asarray(protected)
    n = len(y_arr)
    groups = np.unique(prot_arr)
    if n == 0 or len(groups) < 2:
        return BootstrapResult(0.0, 0.0, 0.0, n_bootstrap, confidence, 1.0)

    def _point_gap(scores: NDArray[np.float64]) -> float:
        if metric == "demographic_parity_diff":
            return demographic_parity_diff(scores.astype(int), prot_arr)
        if metric == "equal_opportunity_diff":
            return equal_opportunity_diff(y_arr.astype(int), scores.astype(int), prot_arr)
        if metric == "equalized_odds_diff":
            return equalized_odds_diff(y_arr.astype(int), scores.astype(int), prot_arr)
        if metric == "calibration_gap_diff":
            return calibration_gap_diff(y_arr.astype(int), scores, prot_arr)
        return demographic_parity_diff(scores.astype(int), prot_arr)

    point_estimate = float(_point_gap(b_arr) - _point_gap(a_arr))

    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_bootstrap, n))
    y_boot = y_arr[idx]
    a_boot = a_arr[idx]
    b_boot = b_arr[idx]
    g_Masks = [(prot_arr == g)[idx].astype(np.float64) for g in groups]

    def _boot_gaps(scores_boot: NDArray[np.float64]) -> NDArray[np.float64]:
        if metric in ("demographic_parity_diff", "calibration_gap_diff"):
            vals = scores_boot - y_boot if metric == "calibration_gap_diff" else scores_boot
            rates = []
            for m in g_Masks:
                denom = np.maximum(np.sum(m, axis=1), 1e-12)
                r = np.sum(m * vals, axis=1) / denom
                if metric == "calibration_gap_diff":
                    r = np.abs(r)
                rates.append(r)
            stacked = np.vstack(rates)
            return np.max(stacked, axis=0) - np.min(stacked, axis=0)

        # equal_opportunity_diff or equalized_odds_diff
        tprs = []
        fprs = []
        for m in g_Masks:
            pos_m = m * (y_boot == 1.0)
            pos_denom = np.maximum(np.sum(pos_m, axis=1), 1e-12)
            tprs.append(np.sum(pos_m * scores_boot, axis=1) / pos_denom)
            if metric == "equalized_odds_diff":
                neg_m = m * (y_boot == 0.0)
                neg_denom = np.maximum(np.sum(neg_m, axis=1), 1e-12)
                fprs.append(np.sum(neg_m * scores_boot, axis=1) / neg_denom)
        tpr_stack = np.vstack(tprs)
        tpr_gap = np.max(tpr_stack, axis=0) - np.min(tpr_stack, axis=0)
        if metric == "equal_opportunity_diff":
            return tpr_gap
        fpr_stack = np.vstack(fprs)
        fpr_gap = np.max(fpr_stack, axis=0) - np.min(fpr_stack, axis=0)
        return np.maximum(tpr_gap, fpr_gap)

    gap_a = _boot_gaps(a_boot)
    gap_b = _boot_gaps(b_boot)
    diffs = gap_b - gap_a

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    p_value = float(np.mean(diffs <= 0))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        p_value=p_value,
    )
