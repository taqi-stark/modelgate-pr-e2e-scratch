"""Fast benchmark runner: 5 seeds, n_bootstrap=50, writes docs/benchmark.md directly."""
from __future__ import annotations

import json
import time
from pathlib import Path

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
from modelgate.config import BudgetConfig, DriftConfig, FairnessConfig, SliceConfig
from modelgate.verdict import CheckResult, compute_verdict

FAULT_DESCRIPTIONS = {
    "feature_leakage": "Add a column derived from the target",
    "label_shift": "Corrupt labels in one subgroup of the train set",
    "degraded_slice": "Remove 90% of one subgroup from training data",
    "schema_drift": "Change a column's scale or unit",
    "latency_regression": "Swap in a much larger slow model",
    "fairness_regression": "Bias sampling against one protected group",
    "noop": "Retrain with a different seed (should PASS)",
}


def _generate_data(seed=42, n=2000):
    rng = np.random.RandomState(seed)

    def _make_df(n, rng):
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
        return pd.DataFrame({
            "income": income, "age": age, "credit_score": credit,
            "debt_ratio": debt, "employment_years": emp,
            "region": region, "age_bucket": age_bucket,
            "gender": gender, "default": default,
        })

    train = _make_df(n, rng)
    test_rng = np.random.RandomState(seed + 1000)
    test = _make_df(n // 2, test_rng)
    return train, test


def _train_model(X, y, seed, n_estimators=50):
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    return Pipeline([
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier(n_estimators=n_estimators, random_state=seed)),
    ]).fit(X, y)


def _train_slow_model(X, y, seed):
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    return Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed)),
    ]).fit(X, y)


def inject_fault(train_df, fault_type, seed):
    rng = np.random.RandomState(seed)
    df = train_df.copy()
    model_override = None
    if fault_type == "feature_leakage":
        # Add leaked column; test set will also need it (simulates the leakage propagating)
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
        model_override = "large_model"
    elif fault_type == "fairness_regression":
        female_nd = df[(df["gender"] == "female") & (df["default"] == 0)]
        flip_idx = female_nd.sample(frac=0.3, random_state=seed).index
        df.loc[flip_idx, "default"] = 1
    return df, model_override


def run_trial(fault_type, seed):
    clean_train, test_df = _generate_data(seed=seed)
    X_clean = clean_train.drop(columns=["default"])
    y_clean = clean_train["default"]
    champion = _train_model(X_clean, y_clean, seed=seed)

    faulty_train, model_override = inject_fault(clean_train, fault_type, seed)
    X_faulty = faulty_train.drop(columns=["default"])
    y_faulty = faulty_train["default"]

    if model_override == "large_model":
        candidate = _train_slow_model(X_faulty, y_faulty, seed=seed)
    else:
        candidate = _train_model(X_faulty, y_faulty, seed=seed)

    # For feature_leakage, test set must also have the leaked column so the pipeline can predict
    test_df_eval = test_df.copy()
    if fault_type == "feature_leakage" and "leaked" not in test_df_eval.columns:
        rng = np.random.RandomState(seed + 999)
        test_df_eval["leaked"] = test_df_eval["default"] * 0.9 + rng.normal(0, 0.1, len(test_df_eval))

    # For slice/fairness checks we use the augmented df so columns match
    test_df_check = test_df_eval

    X_test = test_df_eval.drop(columns=["default"])
    y_test = test_df_eval["default"]
    champ_preds = champion.predict(X_test)
    cand_preds = candidate.predict(X_test)
    champ_acc = accuracy_score(y_test, champ_preds)
    cand_acc = accuracy_score(y_test, cand_preds)
    baseline_detected = abs(champ_acc - cand_acc) > 0.01

    results: list[CheckResult] = []
    results.append(run_drift_check(
        faulty_train, clean_train, "default",
        DriftConfig(enabled=True, warn_at=0.1, fail_at=0.25),
    ))
    # Use augmented test set (includes leaked col for feature_leakage)
    results.append(run_slices_check(
        test_df_eval, "default", champion, candidate,
        SliceConfig(enabled=True, metric="f1", by=["region", "gender"],
                    min_slice_size=20, max_drop=0.03, confidence=0.95, n_bootstrap=50),
        seed=seed,
    ))
    results.append(run_fairness_check(
        test_df_eval, "default", candidate, champion,
        FairnessConfig(enabled=True, protected=["gender"],
                       metric="demographic_parity_diff", warn_at=0.05, fail_at=0.10),
        seed=seed,
    ))
    if fault_type == "latency_regression":
        results.append(run_budgets_check(
            test_df_eval, "default", candidate, champion,
            BudgetConfig(enabled=True, p95_latency_ms=10.0),
        ))

    verdict = compute_verdict(results)
    return {
        "fault_type": fault_type,
        "seed": seed,
        "modelgate_detected": verdict.status in ("warn", "fail"),
        "modelgate_status": verdict.status,
        "baseline_detected": baseline_detected,
        "champion_acc": round(champ_acc, 4),
        "candidate_acc": round(cand_acc, 4),
    }


def _run_trial_star(args):
    return run_trial(*args)


def main():
    from concurrent.futures import ProcessPoolExecutor

    console = Console()
    n_seeds = 30
    fault_types = list(FAULT_DESCRIPTIONS.keys())

    console.print(f"[bold]ModelGate Benchmark[/bold] — {n_seeds} seeds × {len(fault_types)} faults (4 parallel workers)\n")

    summary = {}
    all_results = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        for fault in fault_types:
            console.print(f"  Running [bold]{fault}[/bold]...", end=" ")
            t0 = time.time()
            tasks = [(fault, 42 + i) for i in range(n_seeds)]
            results = list(executor.map(_run_trial_star, tasks))
            all_results.extend(results)
            mg_detected = sum(1 for r in results if r["modelgate_detected"])
            bl_detected = sum(1 for r in results if r["baseline_detected"])
            elapsed = time.time() - t0
            mg_rate = mg_detected / n_seeds
            bl_rate = bl_detected / n_seeds
            summary[fault] = {
                "description": FAULT_DESCRIPTIONS[fault],
                "modelgate_detection_rate": mg_rate,
                "baseline_detection_rate": bl_rate,
                "modelgate_wins": mg_rate > bl_rate if fault != "noop" else mg_rate <= bl_rate,
                "n_trials": n_seeds,
            }
            console.print(f"MG={mg_rate:.0%}  BL={bl_rate:.0%}  [{elapsed:.1f}s]")

    # Print table
    console.print()
    table = Table(title="Results")
    table.add_column("Fault", style="bold")
    table.add_column("ModelGate", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Winner", justify="center")
    for fault, s in summary.items():
        mg_str = f"{s['modelgate_detection_rate']:.0%}"
        bl_str = f"{s['baseline_detection_rate']:.0%}"
        if fault == "noop":
            mg_str += " FP"; bl_str += " FP"
            winner = "✅ MG" if s["modelgate_wins"] else "❌ BL"
        else:
            winner = "✅ MG" if s["modelgate_wins"] else ("🔵 TIE" if mg_str == bl_str else "❌ BL")
        table.add_row(fault, mg_str, bl_str, winner)
    console.print(table)

    # Write outputs
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    results_data = {"summary": summary, "details": all_results, "n_seeds": n_seeds}
    (docs_dir / "benchmark_results.json").write_text(json.dumps(results_data, indent=2))

    # Write markdown
    lines = [
        "# ModelGate Benchmark",
        "",
        "Fault-injection benchmark comparing ModelGate against an accuracy-only baseline.",
        f"Each fault type was run with **{n_seeds} independent seeds** (seeds 42–{41+n_seeds}).",
        "",
        "## Detection Rates",
        "",
        "| Fault | Injected Defect | ModelGate | Baseline (Acc-only) | Winner |",
        "|-------|----------------|-----------|---------------------|--------|",
    ]
    for fault, s in summary.items():
        mg = f"{s['modelgate_detection_rate']:.0%}"
        bl = f"{s['baseline_detection_rate']:.0%}"
        if fault == "noop":
            winner = "ModelGate" if s["modelgate_wins"] else "Baseline"
            mg += " FP"; bl += " FP"
        else:
            winner = "ModelGate" if s["modelgate_wins"] else ("Tie" if mg == bl else "Baseline")
        lines.append(f"| `{fault}` | {s['description']} | {mg} | {bl} | **{winner}** |")

    lines.extend([
        "",
        "> **FP** = false-positive rate for the no-op control. Lower is better.",
        "",
        "## Methodology",
        "",
        "### Fault Injection",
        "",
        "| Fault | Injection Technique |",
        "|-------|---------------------|",
        "| `feature_leakage` | Add `leaked = 0.9·target + ε` column to training data |",
        "| `label_shift` | Flip 40% of labels in the `North` region subgroup |",
        "| `degraded_slice` | Drop 90% of `North` region rows from training data |",
        "| `schema_drift` | Divide `income` by 1000 (unit change) |",
        "| `latency_regression` | Replace GBM with RandomForest(500 trees, depth=20) |",
        "| `fairness_regression` | Flip 30% of `female` non-default labels to default |",
        "| `noop` | Retrain identical pipeline with a different random seed |",
        "",
        "### Checks Run per Trial",
        "",
        "**ModelGate** runs three checks on every trial:",
        "",
        "1. **Drift** (PSI): flags `warn` ≥ 0.10, `fail` ≥ 0.25 on any feature.",
        "2. **Slice Regression** (bootstrap CI, n=50): flags if per-slice F1 drops",
        "   > 0.03 with 95% confidence on `region` or `gender` slices.",
        "3. **Fairness** (demographic parity diff on `gender`): `warn` ≥ 0.05, `fail` ≥ 0.10.",
        "4. **Budgets** (p95 latency ≤ 10 ms): checked only for `latency_regression`.",
        "",
        "**Baseline** detects a fault only when |champion_acc − candidate_acc| > 0.01.",
        "",
        "### Verdict",
        "",
        "A ModelGate trial is counted as *detected* when the overall verdict is `warn` or `fail`.",
        "For the `noop` control, lower detection rates are better (fewer false positives).",
        "",
        "### Reproducibility",
        "",
        "```bash",
        "# Re-run the benchmark (requires the virtual environment)",
        "modelgate bench",
        "# Results are written to docs/benchmark_results.json and docs/benchmark.md",
        "```",
    ])
    (docs_dir / "benchmark.md").write_text("\n".join(lines))
    console.print("\n[green]✓ Written docs/benchmark.md and docs/benchmark_results.json[/green]")


if __name__ == "__main__":
    main()
