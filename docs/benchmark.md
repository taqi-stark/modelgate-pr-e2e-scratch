# ModelGate Benchmark Results

Single source of truth for the ModelGate fault-injection benchmark comparing ModelGate against an accuracy-only baseline.

## Reproducibility Metadata

- **Command**: `modelgate bench`
- **Commit Hash**: `3ba8ba9`
- **Evaluated Seeds**: `1000..1029` (N=30: `1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029`)
- **No-op False-Positive Rate (ModelGate)**: **0/30 (0.0% FP, Wilson 95% CI [0.0%, 11.4%])**
- **No-op False-Positive Rate (Baseline)**: **0/30 (0.0% FP, Wilson 95% CI [0.0%, 11.4%])**

## Detection Rates

| Fault | Injected Defect | ModelGate (Count, Rate, Wilson 95% CI) | Baseline (Count, Rate, Wilson 95% CI) | Winner |
|-------|-----------------|----------------------------------------|---------------------------------------|--------|
| `feature_leakage` | Add a column derived from the target | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | **Tie** |
| `label_shift` | Corrupt labels in one subgroup of the train set | 28/30 (93.3%, [78.7%, 98.2%]) | 28/30 (93.3%, [78.7%, 98.2%]) | **Tie** |
| `degraded_slice` | Remove most of one subgroup from training data | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | **ModelGate** |
| `schema_drift` | Change a column's scale or unit | 30/30 (100.0%, [88.6%, 100.0%]) | 17/30 (56.7%, [39.2%, 72.6%]) | **ModelGate** |
| `latency_regression` | Swap in a much larger model | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | **ModelGate** |
| `fairness_regression` | Bias sampling against one protected group | 2/30 (6.7%, [1.8%, 21.3%]) | 9/30 (30.0%, [16.7%, 47.9%]) | **Baseline** |
| `noop` | Retrain with a different seed (should PASS) | 0/30 (0.0% FP, [0.0%, 11.4%]) | 0/30 (0.0% FP, [0.0%, 11.4%]) | **ModelGate** |

## Methodology

Each fault is injected across 30 independent random seeds (`1000..1029`). For each seed `s`, training data (`N=3000`) is generated with `RandomState(s)` and held-out test data (`N=1500`) with `RandomState(s + 1000)`. Confidence intervals are calculated using the Wilson score interval at the 95% confidence level (`z = 1.95996`).

The accuracy-only baseline flags a regression only if overall test accuracy changes by more than 1 percentage point (`|acc_champ - acc_cand| > 0.01`).

For the `noop` control (retraining the identical pipeline with candidate seed `s + 500`), lower detection rates represent fewer false positives.
