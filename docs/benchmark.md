# ModelGate Benchmark Results

Fault injection benchmark comparing ModelGate against an accuracy-only baseline.
Each fault type was evaluated across 30 independent seeds.

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

Each fault is injected 30 times with different random seeds. ModelGate executes data contract, drift, slice regression, behavioral, performance budgets, and fairness checks. Confidence intervals are calculated using the Wilson score interval at 95% confidence level.

The accuracy-only baseline fails only if overall accuracy drops more than 1 percentage point (|acc_champ - acc_cand| > 0.01).

For the `noop` control (retraining with a different seed), lower detection rates represent fewer false positives.