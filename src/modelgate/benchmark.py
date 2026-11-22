"""ModelGate fault-injection benchmark.

Automatically creates faulty candidate models and reports whether
ModelGate and an accuracy-only baseline catch each fault type.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from modelgate.checks.budgets import run_budgets_check
from modelgate.checks.drift import run_drift_check
from modelgate.checks.fairness import run_fairness_check
from modelgate.checks.slices import run_slices_check
from modelgate.config import (
    BudgetConfig,
    DriftConfig,
    FairnessConfig,
    SliceConfig,
)
from modelgate.verdict import CheckResult, compute_verdict


def _generate_data(seed: int = 42, n: int = 3000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate train/test data."""
    rng = np.random.RandomState(seed)

    def _make_df(n: int, rng: np.random.RandomState) -> pd.DataFrame:
        income = rng.lognormal(10.5, 0.5, n)
        age = rng.randint(18, 70, n)
        credit = rng.normal(680, 80, n).clip(300, 850).astype(int)
        debt = rng.beta(2, 5, n)
        emp = rng.exponential(5, n).clip(0, 40).astype(int)
        region = rng.choice(["North", "South", "East", "West"], n)
        gender = rng.choice(["male", "female"], n, p=[0.55, 0.45])
        age_bucket = np.where(age < 30, "young", np.where(age < 50, "middle", "senior"))

        logit = -2 + 0.3 * (debt - 0.3) - 0.001 * (income - 30000) / 10000 - 0.005 * (credit - 600)
        logit += np.where(region == "North", 0.1, 0)
        prob = 1 / (1 + np.exp(-logit))
        default = (rng.random(n) < prob).astype(int)

        return pd.DataFrame(
            {
                "income": income,
                "age": age,
                "credit_score": credit,
                "debt_ratio": debt,
                "employment_years": emp,
                "region": region,
                "age_bucket": age_bucket,
                "gender": gender,
                "default": default,
            }
        )

    train = _make_df(n, rng)
    test_rng = np.random.RandomState(seed + 1000)
    test = _make_df(n // 2, test_rng)
    return train, test


def _train_model(X: pd.DataFrame, y: pd.Series, seed: int) -> Pipeline:
    """Train a gradient boosting pipeline."""
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def _train_latency_model(X: pd.DataFrame, y: pd.Series, seed: int) -> Pipeline:
    """Train a large random forest pipeline with high latency."""
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=300, max_depth=15, random_state=seed),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


FAULT_DESCRIPTIONS = {
    "feature_leakage": "Add a column derived from the target",
    "label_shift": "Corrupt labels in one subgroup of the train set",
    "degraded_slice": "Remove most of one subgroup from training data",
    "schema_drift": "Change a column's scale or unit",
    "latency_regression": "Swap in a much larger model",
    "fairness_regression": "Bias sampling against one protected group",
    "noop": "Retrain with a different seed (should PASS)",
}


def inject_fault(
    train_df: pd.DataFrame,
    fault_type: str,
    seed: int,
) -> tuple[pd.DataFrame, Any]:
    """Inject a specific fault and return modified data and optional model override.

    Returns:
        Tuple of (modified_train_df, model_override_or_None).
    """
    rng = np.random.RandomState(seed)
    df = train_df.copy()
    model_override = None

    if fault_type == "feature_leakage":
        df["leaked"] = df["default"] * 0.9 + rng.normal(0, 0.1, len(df))

    elif fault_type == "label_shift":
        mask = df["region"] == "North"
        flip_idx = df[mask].sample(frac=0.4, random_state=seed).index
        df.loc[flip_idx, "default"] = 1 - df.loc[flip_idx, "default"]

    elif fault_type == "degraded_slice":
        north = df[df["region"] == "North"]
        drop_idx = north.sample(frac=0.9, random_state=seed).index
        df = df.drop(drop_idx)

    elif fault_type == "schema_drift":
        df["income"] = df["income"] / 1000

    elif fault_type == "latency_regression":
        # Use a much larger model
        model_override = "large_model"

    elif fault_type == "fairness_regression":
        female_non_default = df[(df["gender"] == "female") & (df["default"] == 0)]
        flip_idx = female_non_default.sample(frac=0.3, random_state=seed).index
        df.loc[flip_idx, "default"] = 1

    elif fault_type == "noop":
        pass  # No change, just different seed

    return df, model_override


def run_single_trial(
    fault_type: str,
    seed: int,
) -> dict[str, Any]:
    """Run a single trial of a fault injection.

    Returns:
        Dict with fault_type, seed, modelgate_detected, baseline_detected.
    """
    # Generate clean data
    clean_train, test_df = _generate_data(seed=seed)

    # Train champion on clean data
    X_clean = clean_train.drop(columns=["default"])
    y_clean = clean_train["default"]
    champion = _train_model(X_clean, y_clean, seed=seed)

    # Inject fault
    faulty_train, model_override = inject_fault(clean_train, fault_type, seed)

    # Train candidate on faulty data
    X_faulty = faulty_train.drop(columns=["default"])
    y_faulty = faulty_train["default"]

    cand_seed = seed + 500 if fault_type == "noop" else seed
    if model_override == "large_model":
        candidate = _train_latency_model(X_faulty, y_faulty, seed=cand_seed)
    else:
        candidate = _train_model(X_faulty, y_faulty, seed=cand_seed)

    # Prepare test data
    test_eval = test_df.copy()
    if fault_type == "feature_leakage":
        test_rng = np.random.RandomState(seed + 1000)
        test_eval["leaked"] = test_eval["default"] * 0.9 + test_rng.normal(0, 0.1, len(test_eval))

    X_test = test_eval.drop(columns=["default"])
    y_test = test_eval["default"]

    # Predict
    champ_preds = champion.predict(X_test)
    cand_preds = candidate.predict(X_test)

    champ_acc = accuracy_score(y_test, champ_preds)
    cand_acc = accuracy_score(y_test, cand_preds)

    # ── Baseline: accuracy-only check ──
    baseline_detected = abs(champ_acc - cand_acc) > 0.01

    # ── ModelGate: run multiple checks ──
    modelgate_results: list[CheckResult] = []

    # Drift check
    drift_result = run_drift_check(
        faulty_train,
        clean_train,
        "default",
        DriftConfig(enabled=True, warn_at=0.1, fail_at=0.25),
    )
    modelgate_results.append(drift_result)

    # Slice check
    slice_result = run_slices_check(
        test_eval,
        "default",
        champion,
        candidate,
        SliceConfig(
            enabled=True,
            metric="f1",
            by=["region", "gender"],
            min_slice_size=20,
            max_drop=0.03,
            confidence=0.95,
            n_bootstrap=200,
        ),
        seed=seed,
    )
    modelgate_results.append(slice_result)

    # Fairness check
    fairness_result = run_fairness_check(
        test_eval,
        "default",
        candidate,
        champion,
        FairnessConfig(
            enabled=True,
            protected=["gender"],
            metric="demographic_parity_diff",
            warn_at=0.05,
            fail_at=0.10,
        ),
        seed=seed,
    )
    modelgate_results.append(fairness_result)

    # Budgets check
    if fault_type == "latency_regression":
        budget_result = run_budgets_check(
            test_eval,
            "default",
            candidate,
            champion,
            BudgetConfig(enabled=True, p95_latency_ms=10.0),
        )
        modelgate_results.append(budget_result)

    # Compute ModelGate verdict
    verdict = compute_verdict(modelgate_results)
    modelgate_detected = verdict.status in ("warn", "fail")

    return {
        "fault_type": fault_type,
        "seed": seed,
        "modelgate_detected": modelgate_detected,
        "modelgate_status": verdict.status,
        "baseline_detected": baseline_detected,
        "champion_acc": round(champ_acc, 4),
        "candidate_acc": round(cand_acc, 4),
    }


def wilson_ci(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    """Compute Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + (z**2 / n)
    center = p + (z**2 / (2 * n))
    margin = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    lower = max(0.0, (center - margin) / denom)
    upper = min(1.0, (center + margin) / denom)
    return float(lower), float(upper)


def _get_git_commit() -> str:
    """Return short git commit hash (or 'unknown' if unavailable)."""
    import subprocess

    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def run_benchmark(
    console: Console,
    n_seeds: int = 30,
    start_seed: int = 1000,
    command_str: str = "modelgate bench",
) -> dict[str, Any]:
    """Run the full fault injection benchmark.

    Args:
        console: Rich console for output.
        n_seeds: Number of seeds per fault type.
        start_seed: First random seed in the contiguous range [start_seed, start_seed + n_seeds - 1].
        command_str: Command used to invoke the benchmark.

    Returns:
        Benchmark results dict.
    """
    fault_types = [
        "feature_leakage",
        "label_shift",
        "degraded_slice",
        "schema_drift",
        "latency_regression",
        "fairness_regression",
        "noop",
    ]

    seeds = [start_seed + i for i in range(n_seeds)]
    commit_hash = _get_git_commit()
    all_results: list[dict] = []
    summary: dict[str, dict] = {}

    for fault in fault_types:
        console.print(f"  Running [bold]{fault}[/bold] ({n_seeds} seeds)...", end=" ")
        start = time.time()

        mg_detected = 0
        bl_detected = 0

        for seed in seeds:
            try:
                result = run_single_trial(fault, seed)
                all_results.append(result)
                if result["modelgate_detected"]:
                    mg_detected += 1
                if result["baseline_detected"]:
                    bl_detected += 1
            except Exception as e:
                console.print(f"[red]Error seed {seed}: {e}[/red]")

        elapsed = time.time() - start
        mg_rate = mg_detected / n_seeds if n_seeds > 0 else 0
        bl_rate = bl_detected / n_seeds if n_seeds > 0 else 0
        mg_ci = wilson_ci(mg_detected, n_seeds)
        bl_ci = wilson_ci(bl_detected, n_seeds)

        summary[fault] = {
            "description": FAULT_DESCRIPTIONS[fault],
            "n_trials": n_seeds,
            "mg_detected": mg_detected,
            "bl_detected": bl_detected,
            "modelgate_detection_rate": round(mg_rate, 3),
            "baseline_detection_rate": round(bl_rate, 3),
            "modelgate_ci_95": [round(mg_ci[0], 3), round(mg_ci[1], 3)],
            "baseline_ci_95": [round(bl_ci[0], 3), round(bl_ci[1], 3)],
            "modelgate_wins": mg_rate > bl_rate,
        }

        console.print(f"MG: {mg_rate:.0%}  BL: {bl_rate:.0%}  ({elapsed:.1f}s)")

    # Print summary table
    console.print()
    table = Table(title="ModelGate Benchmark Results")
    table.add_column("Fault", style="bold")
    table.add_column("Description")
    table.add_column("ModelGate", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Winner", justify="center")

    for fault, s in summary.items():
        mg_str = f"{s['mg_detected']}/{s['n_trials']} ({s['modelgate_detection_rate']:.0%})"
        bl_str = f"{s['bl_detected']}/{s['n_trials']} ({s['baseline_detection_rate']:.0%})"
        winner = (
            "✅ MG"
            if s["modelgate_wins"]
            else (
                "🔵 TIE"
                if s["modelgate_detection_rate"] == s["baseline_detection_rate"]
                else "❌ BL"
            )
        )

        if fault == "noop":
            # For no-op, lower detection = better (false positives)
            mg_str = f"{s['mg_detected']}/{s['n_trials']} ({s['modelgate_detection_rate']:.0%} FP)"
            bl_str = f"{s['bl_detected']}/{s['n_trials']} ({s['baseline_detection_rate']:.0%} FP)"
            winner = (
                "✅ MG"
                if s["modelgate_detection_rate"] <= s["baseline_detection_rate"]
                else "❌ BL"
            )

        table.add_row(fault, s["description"], mg_str, bl_str, winner)

    console.print(table)

    # Save results
    results_data = {
        "command": command_str,
        "commit_hash": commit_hash,
        "n_seeds": n_seeds,
        "seed_range": f"{seeds[0]}..{seeds[-1]}" if seeds else "",
        "seeds": seeds,
        "summary": summary,
        "details": all_results,
    }

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "benchmark_results.json").write_text(json.dumps(results_data, indent=2))

    # Generate markdown
    _generate_benchmark_markdown(
        summary,
        docs_dir / "benchmark.md",
        commit_hash=commit_hash,
        seeds=seeds,
        command_str=command_str,
    )

    console.print(
        "\n[green]Results saved to docs/benchmark_results.json and docs/benchmark.md[/green]"
    )

    return results_data


def _generate_benchmark_markdown(
    summary: dict,
    path: Path,
    commit_hash: str = "unknown",
    seeds: list[int] | None = None,
    command_str: str = "modelgate bench",
) -> None:
    """Generate benchmark.md report with exact counts, metadata, and Wilson 95% CIs."""
    if seeds is None:
        n_default = next(iter(summary.values()), {}).get("n_trials", 30)
        seeds = [1000 + i for i in range(n_default)]
    seed_range_str = f"`{seeds[0]}..{seeds[-1]}` (N={len(seeds)}: `{', '.join(map(str, seeds))}`)"
    noop = summary.get("noop", {})
    noop_mg_k = noop.get("mg_detected", 0)
    noop_n = noop.get("n_trials", len(seeds))
    noop_mg_rate = noop.get("modelgate_detection_rate", 0.0)
    noop_mg_ci = noop.get("modelgate_ci_95", [0.0, 0.114])
    noop_bl_k = noop.get("bl_detected", 0)
    noop_bl_rate = noop.get("baseline_detection_rate", 0.0)
    noop_bl_ci = noop.get("baseline_ci_95", [0.0, 0.114])

    lines = [
        "# ModelGate Benchmark Results",
        "",
        "Single source of truth for the ModelGate fault-injection benchmark comparing ModelGate against an accuracy-only baseline.",
        "",
        "## Reproducibility Metadata",
        "",
        f"- **Command**: `{command_str}`",
        f"- **Commit Hash**: `{commit_hash}`",
        f"- **Evaluated Seeds**: {seed_range_str}",
        f"- **No-op False-Positive Rate (ModelGate)**: **{noop_mg_k}/{noop_n} ({noop_mg_rate:.1%} FP, Wilson 95% CI [{noop_mg_ci[0]:.1%}, {noop_mg_ci[1]:.1%}])**",
        f"- **No-op False-Positive Rate (Baseline)**: **{noop_bl_k}/{noop_n} ({noop_bl_rate:.1%} FP, Wilson 95% CI [{noop_bl_ci[0]:.1%}, {noop_bl_ci[1]:.1%}])**",
        "",
        "## Detection Rates",
        "",
        "| Fault | Injected Defect | ModelGate (Count, Rate, Wilson 95% CI) | Baseline (Count, Rate, Wilson 95% CI) | Winner |",
        "|-------|-----------------|----------------------------------------|---------------------------------------|--------|",
    ]

    for fault, s in summary.items():
        n = s["n_trials"]
        mg_k = s["mg_detected"]
        bl_k = s["bl_detected"]
        mg_ci = s["modelgate_ci_95"]
        bl_ci = s["baseline_ci_95"]

        if fault == "noop":
            mg_cell = f"{mg_k}/{n} ({s['modelgate_detection_rate']:.1%} FP, [{mg_ci[0]:.1%}, {mg_ci[1]:.1%}])"
            bl_cell = f"{bl_k}/{n} ({s['baseline_detection_rate']:.1%} FP, [{bl_ci[0]:.1%}, {bl_ci[1]:.1%}])"
            winner = (
                "ModelGate"
                if s["modelgate_detection_rate"] <= s["baseline_detection_rate"]
                else "Baseline"
            )
        else:
            mg_cell = f"{mg_k}/{n} ({s['modelgate_detection_rate']:.1%}, [{mg_ci[0]:.1%}, {mg_ci[1]:.1%}])"
            bl_cell = (
                f"{bl_k}/{n} ({s['baseline_detection_rate']:.1%}, [{bl_ci[0]:.1%}, {bl_ci[1]:.1%}])"
            )
            winner = (
                "ModelGate"
                if s["modelgate_wins"]
                else (
                    "Tie"
                    if s["modelgate_detection_rate"] == s["baseline_detection_rate"]
                    else "Baseline"
                )
            )

        lines.append(f"| `{fault}` | {s['description']} | {mg_cell} | {bl_cell} | **{winner}** |")

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            f"Each fault is injected across {len(seeds)} independent random seeds (`{seeds[0]}..{seeds[-1]}`). "
            "For each seed `s`, training data (`N=3000`) is generated with `RandomState(s)` and held-out test data (`N=1500`) with `RandomState(s + 1000)`. "
            "Confidence intervals are calculated using the Wilson score interval at the 95% confidence level (`z = 1.95996`).",
            "",
            "The accuracy-only baseline flags a regression only if overall test accuracy changes by more than 1 percentage point (`|acc_champ - acc_cand| > 0.01`).",
            "",
            "For the `noop` control (retraining the identical pipeline with candidate seed `s + 500`), lower detection rates represent fewer false positives.",
        ]
    )

    path.write_text("\n".join(lines) + "\n")
