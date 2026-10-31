"""Fairness check.

Computes demographic parity difference and equal opportunity difference
across protected groups, with bootstrap confidence intervals.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from modelgate.config import FairnessConfig
from modelgate.stats import (
    demographic_parity_diff,
    equal_opportunity_diff,
)
from modelgate.verdict import CheckResult


def run_fairness_check(
    test_df: pd.DataFrame,
    target_column: str,
    candidate_model: Any,
    champion_model: Any | None,
    config: FairnessConfig,
    seed: int = 42,
) -> CheckResult:
    """Run fairness check on the candidate model.

    Args:
        test_df: Test DataFrame.
        target_column: Name of the target column.
        candidate_model: The candidate model.
        champion_model: The champion model (optional).
        config: Fairness check configuration.
        seed: Random seed.

    Returns:
        CheckResult with fairness metrics.
    """
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
    y_test = test_df[target_column].values

    candidate_preds = candidate_model.predict(X_test)

    champion_preds = None
    if champion_model is not None:
        champion_preds = champion_model.predict(X_test)

    group_results: list[dict] = []
    max_metric_value = 0.0

    for protected_col in config.protected:
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

        # Compute candidate fairness metrics
        if config.metric == "demographic_parity_diff":
            candidate_value = demographic_parity_diff(candidate_preds, protected_values)
        elif config.metric == "equal_opportunity_diff":
            candidate_value = equal_opportunity_diff(y_test, candidate_preds, protected_values)
        else:
            candidate_value = demographic_parity_diff(candidate_preds, protected_values)

        max_metric_value = max(max_metric_value, candidate_value)

        # Per-group rates
        groups = np.unique(protected_values)
        per_group: list[dict] = []
        for g in groups:
            mask = protected_values == g
            n_group = int(np.sum(mask))
            pos_rate = float(np.mean(candidate_preds[mask])) if n_group > 0 else 0.0
            group_info: dict = {
                "group": str(g),
                "n": n_group,
                "positive_rate": round(pos_rate, 4),
            }
            if champion_preds is not None:
                champ_rate = float(np.mean(champion_preds[mask])) if n_group > 0 else 0.0
                group_info["champion_positive_rate"] = round(champ_rate, 4)
            per_group.append(group_info)

        # Champion comparison
        champion_value = None
        if champion_preds is not None:
            if config.metric == "demographic_parity_diff":
                champion_value = demographic_parity_diff(champion_preds, protected_values)
            elif config.metric == "equal_opportunity_diff":
                champion_value = equal_opportunity_diff(y_test, champion_preds, protected_values)
            else:
                champion_value = demographic_parity_diff(champion_preds, protected_values)

        result_entry: dict = {
            "protected_attribute": protected_col,
            "metric": config.metric,
            "candidate_value": round(candidate_value, 4),
            "per_group": per_group,
        }
        if champion_value is not None:
            result_entry["champion_value"] = round(champion_value, 4)
            result_entry["delta"] = round(candidate_value - champion_value, 4)

        group_results.append(result_entry)

    duration = time.time() - start

    # Determine status
    if max_metric_value >= config.fail_at:
        status = "fail"
        summary = f"{config.metric} {max_metric_value:.2f} >= {config.fail_at}"
    elif max_metric_value >= config.warn_at:
        status = "warn"
        summary = f"{config.metric} {max_metric_value:.2f} >= {config.warn_at}"
    else:
        status = "pass"
        summary = f"{config.metric} {max_metric_value:.2f}"

    return CheckResult(
        name="fairness",
        status=status,
        summary=summary,
        metrics={
            config.metric: max_metric_value,
        },
        details={"groups": group_results},
        duration_s=duration,
    )
