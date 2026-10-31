# ModelGate Benchmark

Fault-injection benchmark comparing ModelGate against an accuracy-only baseline.
Each fault type was run with **30 independent seeds** (seeds 42–71).

## Detection Rates

| Fault | Injected Defect | ModelGate | Baseline (Acc-only) | Winner |
|-------|----------------|-----------|---------------------|--------|
| `feature_leakage` | Add a column derived from the target | 100% | 100% | **Tie** |
| `label_shift` | Corrupt labels in one subgroup of the train set | 93% | 93% | **Tie** |
| `degraded_slice` | Remove 90% of one subgroup from training data | 100% | 3% | **ModelGate** |
| `schema_drift` | Change a column's scale or unit | 100% | 40% | **ModelGate** |
| `latency_regression` | Swap in a much larger slow model | 100% | 0% | **ModelGate** |
| `fairness_regression` | Bias sampling against one protected group | 43% | 77% | **Baseline** |
| `noop` | Retrain with a different seed (should PASS) | 0% FP | 0% FP | **ModelGate** |

> **FP** = false-positive rate for the no-op control. Lower is better.

## Methodology

### Fault Injection

| Fault | Injection Technique |
|-------|---------------------|
| `feature_leakage` | Add `leaked = 0.9·target + ε` column to training data |
| `label_shift` | Flip 40% of labels in the `North` region subgroup |
| `degraded_slice` | Drop 90% of `North` region rows from training data |
| `schema_drift` | Divide `income` by 1000 (unit change) |
| `latency_regression` | Replace GBM with RandomForest(500 trees, depth=20) |
| `fairness_regression` | Flip 30% of `female` non-default labels to default |
| `noop` | Retrain identical pipeline with a different random seed |

### Checks Run per Trial

**ModelGate** runs three checks on every trial:

1. **Drift** (PSI): flags `warn` ≥ 0.10, `fail` ≥ 0.25 on any feature.
2. **Slice Regression** (bootstrap CI, n=50): flags if per-slice F1 drops
   > 0.03 with 95% confidence on `region` or `gender` slices.
3. **Fairness** (demographic parity diff on `gender`): `warn` ≥ 0.05, `fail` ≥ 0.10.
4. **Budgets** (p95 latency ≤ 10 ms): checked only for `latency_regression`.

**Baseline** detects a fault only when |champion_acc − candidate_acc| > 0.01.

### Verdict

A ModelGate trial is counted as *detected* when the overall verdict is `warn` or `fail`.
For the `noop` control, lower detection rates are better (fewer false positives).

### Reproducibility

```bash
# Re-run the benchmark (requires the virtual environment)
modelgate bench
# Results are written to docs/benchmark_results.json and docs/benchmark.md
```