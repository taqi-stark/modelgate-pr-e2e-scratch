"""Fairness check.

Computes demographic parity difference, equal opportunity difference,
equalized odds difference, and per-group calibration disparity across
protected groups, and compares champion vs candidate group gaps using
a paired bootstrap confidence interval on the change in gap.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd

from modelgate.config import FairnessConfig
from modelgate.stats import (
    calibration_gap_diff,
    demographic_parity_diff,
    equal_opportunity_diff,
    equalized_odds_diff,
    paired_bootstrap_gap_ci,
    per_group_calibration,
)
from modelgate.verdict import CheckResult


def _get_positive_probs(model: Any, X: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Return predicted probability of positive class if available, else binary predictions as float."""
    if hasattr(model, "predict_proba"):
        try:
            probs = np.asarray(model.predict_proba(X), dtype=np.float64)
            if probs.ndim == 2 and probs.shape[1] > 1:
                return probs[:, 1]
            return probs.ravel()
        except Exception:
            pass
    return np.asarray(preds, dtype=np.float64)


def _compute_gap_suite(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    protected: np.ndarray,
) -> dict[str, float]:
    """Compute all four fairness gap metrics."""
    return {
        "demographic_parity_diff": demographic_parity_diff(y_pred, protected),
        "equal_opportunity_diff": equal_opportunity_diff(y_true, y_pred, protected),
        "equalized_odds_diff": equalized_odds_diff(y_true, y_pred, protected),
        "calibration_gap_diff": calibration_gap_diff(y_true, y_prob, protected),
    }


def run_fairness_check(
    test_df: pd.DataFrame,
    target_column: str,
    candidate_model: Any,
    champion_model: Any | None,
    config: FairnessConfig,
    seed: int = 42,
) -> CheckResult:
    """Run fairness check on the candidate model and compare against champion if available."""
    start = time.time()

    if not config.enabled:
        return CheckResult(
            name="fairness",
            status="skip",
            summary="Fairness check disabled",
            duration_s=time.time() - start,
        )

    if not config.protected:
        return CheckResult(
            name="fairness",
            status="skip",
            summary="No protected attributes configured",
            duration_s=time.time() - start,
        )

    X_test = test_df.drop(columns=[target_column])
    y_test = np.asarray(test_df[target_column].values, dtype=int)

    candidate_preds = np.asarray(candidate_model.predict(X_test), dtype=int)
    candidate_probs = _get_positive_probs(candidate_model, X_test, candidate_preds)

    champion_preds: np.ndarray | None = None
    champion_probs: np.ndarray | None = None
    if champion_model is not None:
        champion_preds = np.asarray(champion_model.predict(X_test), dtype=int)
        champion_probs = _get_positive_probs(champion_model, X_test, champion_preds)

    group_results: list[dict[str, Any]] = []
    max_metric_value = 0.0
    max_eqodds_value = 0.0
    max_cal_gap_value = 0.0
    max_gap_delta = 0.0
    bootstrap_fail_reasons: list[str] = []
    bootstrap_warn_reasons: list[str] = []

    for idx_col, protected_col in enumerate(config.protected):
        if protected_col not in test_df.columns:
            group_results.append(
                {
                    "protected_attribute": protected_col,
                    "status": "skip",
                    "message": f"Column '{protected_col}' not found",
                }
            )
            continue

        protected_values = test_df[protected_col].values

        cand_suite = _compute_gap_suite(y_test, candidate_preds, candidate_probs, protected_values)
        candidate_value = cand_suite.get(config.metric, cand_suite["demographic_parity_diff"])

        max_metric_value = max(max_metric_value, candidate_value)
        max_eqodds_value = max(max_eqodds_value, cand_suite["equalized_odds_diff"])
        max_cal_gap_value = max(max_cal_gap_value, cand_suite["calibration_gap_diff"])

        cand_cal_by_group = {
            row["group"]: row
            for row in per_group_calibration(y_test, candidate_probs, protected_values)
        }
        champ_cal_by_group = (
            {
                row["group"]: row
                for row in per_group_calibration(y_test, champion_probs, protected_values)
            }
            if champion_probs is not None
            else {}
        )

        # Per-group rates, TPR/FPR, and calibration
        groups = np.unique(protected_values)
        per_group: list[dict[str, Any]] = []
        for g in groups:
            mask = protected_values == g
            n_group = int(np.sum(mask))
            pos_rate = float(np.mean(candidate_preds[mask])) if n_group > 0 else 0.0
            pos_mask = mask & (y_test == 1)
            neg_mask = mask & (y_test == 0)
            tpr_g = float(np.mean(candidate_preds[pos_mask])) if np.sum(pos_mask) > 0 else 0.0
            fpr_g = float(np.mean(candidate_preds[neg_mask])) if np.sum(neg_mask) > 0 else 0.0
            cal_info = cand_cal_by_group.get(str(g), {})

            group_info: dict[str, Any] = {
                "group": str(g),
                "n": n_group,
                "positive_rate": round(pos_rate, 4),
                "tpr": round(tpr_g, 4),
                "fpr": round(fpr_g, 4),
                "base_rate": cal_info.get("base_rate", 0.0),
                "mean_pred_prob": cal_info.get("mean_pred_prob", round(pos_rate, 4)),
                "calibration_error": cal_info.get("calibration_error", 0.0),
                "brier_score": cal_info.get("brier_score", 0.0),
            }
            if champion_preds is not None:
                champ_rate = float(np.mean(champion_preds[mask])) if n_group > 0 else 0.0
                group_info["champion_positive_rate"] = round(champ_rate, 4)
                champ_cal = champ_cal_by_group.get(str(g), {})
                if champ_cal:
                    group_info["champion_mean_pred_prob"] = champ_cal["mean_pred_prob"]
                    group_info["champion_calibration_error"] = champ_cal["calibration_error"]
            per_group.append(group_info)

        result_entry: dict[str, Any] = {
            "protected_attribute": protected_col,
            "metric": config.metric,
            "candidate_value": round(candidate_value, 4),
            "equalized_odds_diff": round(cand_suite["equalized_odds_diff"], 4),
            "calibration_gap_diff": round(cand_suite["calibration_gap_diff"], 4),
            "candidate_gaps": {k: round(v, 4) for k, v in cand_suite.items()},
            "per_group": per_group,
        }

        if champion_preds is not None and champion_probs is not None:
            champ_suite = _compute_gap_suite(
                y_test, champion_preds, champion_probs, protected_values
            )
            champion_value = champ_suite.get(config.metric, champ_suite["demographic_parity_diff"])
            result_entry["champion_value"] = round(champion_value, 4)
            result_entry["delta"] = round(candidate_value - champion_value, 4)
            result_entry["champion_gaps"] = {k: round(v, 4) for k, v in champ_suite.items()}

            monitored_metrics = list(
                dict.fromkeys(
                    [
                        config.metric,
                        "demographic_parity_diff",
                        "equalized_odds_diff",
                        "calibration_gap_diff",
                    ]
                )
            )
            boot_details: dict[str, dict[str, float]] = {}
            for m_idx, m_name in enumerate(monitored_metrics):
                a_scores = champion_probs if m_name == "calibration_gap_diff" else champion_preds
                b_scores = candidate_probs if m_name == "calibration_gap_diff" else candidate_preds
                boot = paired_bootstrap_gap_ci(
                    y_test,
                    a_scores,
                    b_scores,
                    protected_values,
                    metric=m_name,
                    n_bootstrap=config.n_bootstrap,
                    confidence=config.confidence,
                    seed=seed + idx_col * 17 + m_idx,
                )
                boot_details[m_name] = {
                    "point_estimate": round(boot.point_estimate, 4),
                    "ci_lower": round(boot.ci_lower, 4),
                    "ci_upper": round(boot.ci_upper, 4),
                    "p_value": round(boot.p_value, 4),
                }
                max_gap_delta = max(max_gap_delta, boot.point_estimate)

                if boot.ci_lower > 0.0 and (
                    boot.point_estimate >= config.fail_gap_increase
                    or boot.ci_lower >= config.max_gap_increase
                ):
                    bootstrap_fail_reasons.append(
                        f"{protected_col} {m_name} gap +{boot.point_estimate:.3f} "
                        f"(95% CI [+{boot.ci_lower:.3f}, +{boot.ci_upper:.3f}])"
                    )
                elif boot.ci_lower > 0.0 and boot.point_estimate >= config.max_gap_increase:
                    bootstrap_warn_reasons.append(
                        f"{protected_col} {m_name} gap +{boot.point_estimate:.3f} "
                        f"(95% CI [+{boot.ci_lower:.3f}, +{boot.ci_upper:.3f}])"
                    )

            result_entry["gap_bootstrap"] = boot_details[config.metric]
            result_entry["all_gap_bootstraps"] = boot_details

        group_results.append(result_entry)

    duration = time.time() - start

    status: Literal["pass", "warn", "fail", "skip"]
    if max_metric_value >= config.fail_at:
        status = "fail"
        summary = f"{config.metric} {max_metric_value:.2f} >= {config.fail_at}"
    elif bootstrap_fail_reasons:
        status = "fail"
        summary = f"Fairness gap regression: {'; '.join(bootstrap_fail_reasons[:2])}"
    elif max_metric_value >= config.warn_at:
        status = "warn"
        summary = f"{config.metric} {max_metric_value:.2f} >= {config.warn_at}"
    elif bootstrap_warn_reasons:
        status = "warn"
        summary = f"Fairness gap increase: {'; '.join(bootstrap_warn_reasons[:2])}"
    else:
        status = "pass"
        summary = (
            f"{config.metric} {max_metric_value:.2f}, "
            f"eq_odds {max_eqodds_value:.2f}, cal_gap {max_cal_gap_value:.2f}"
        )

    return CheckResult(
        name="fairness",
        status=status,
        summary=summary,
        metrics={
            config.metric: max_metric_value,
            "equalized_odds_diff": max_eqodds_value,
            "calibration_gap_diff": max_cal_gap_value,
            "max_gap_delta": max_gap_delta,
        },
        details={"groups": group_results},
        duration_s=duration,
    )
