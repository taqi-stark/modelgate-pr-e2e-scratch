# ModelGate Benchmark Results

Single source of truth for the ModelGate fault-injection benchmark comparing ModelGate against an accuracy-only baseline.

## Reproducibility Metadata

- **Command**: `modelgate bench`
- **Commit Hash**: `62e622e`
- **Evaluated Seeds**: `1000..1029` (N=30: `1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029`)
- **No-op False-Positive Rate (ModelGate)**: **0/30 (0.0% observed FP; Wilson 95% CI [0.0%, 11.4%], i.e. an 11.4% 95% upper bound at N=30)**
- **No-op False-Positive Rate (Baseline)**: **0/30 (0.0% observed FP; Wilson 95% CI [0.0%, 11.4%], i.e. an 11.4% 95% upper bound at N=30)**

## Detection Rates (Synthetic Credit Benchmark, Seeds `1000..1029`)

A winner is declared only when the 95% Wilson confidence intervals do not overlap; overlapping 95% CIs are reported as a statistical **Tie (overlapping 95% CIs)**.

| Fault | Injected Defect | ModelGate (Count, Rate, Wilson 95% CI) | Baseline (Count, Rate, Wilson 95% CI) | Winner |
|-------|-----------------|----------------------------------------|---------------------------------------|--------|
| `feature_leakage` | Add a column derived from the target | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | **Tie (overlapping 95% CIs)** |
| `label_shift` | Corrupt labels in one subgroup of the train set | 30/30 (100.0%, [88.6%, 100.0%]) | 28/30 (93.3%, [78.7%, 98.2%]) | **Tie (overlapping 95% CIs)** |
| `degraded_slice` | Remove most of one subgroup from training data | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | **ModelGate** |
| `schema_drift` | Change a column's scale or unit | 30/30 (100.0%, [88.6%, 100.0%]) | 17/30 (56.7%, [39.2%, 72.6%]) | **ModelGate** |
| `latency_regression` | Swap in a much larger model | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | **ModelGate** |
| `fairness_regression` | Bias sampling against one protected group | 30/30 (100.0%, [88.6%, 100.0%]) | 9/30 (30.0%, [16.7%, 47.9%]) | **ModelGate** |
| `noop` | Retrain with a different seed (should PASS) | 0/30 (0.0% FP, [0.0%, 11.4%]) | 0/30 (0.0% FP, [0.0%, 11.4%]) | **Tie (11.4% 95% CI upper bound)** |

## Baseline Definitions

1. **Two-Sided Accuracy-Change Baseline (`|acc_champ - acc_cand| > 0.01`, shown in the table above)**: Flags any trial where overall test accuracy changes by more than 1 percentage point in either direction (drop or increase).
   - Under this definition, `feature_leakage` (`leaked = 0.9 * default + N(0, 0.1)`) is caught `30/30` by the baseline because the leaked feature artificially boosts test accuracy by ~8.5 percentage points (`~91.5% -> ~100.0%`), resulting in a **Tie (`30/30` vs `30/30`, overlapping 95% CIs `[88.6%, 100.0%]`)**.
   - On `label_shift` (corrupting 40% of training labels in `region=North`), ModelGate detects `30/30 (100.0%, [88.6%, 100.0%])` and the baseline detects `28/30 (93.3%, [78.7%, 98.2%])`. Because their 95% Wilson confidence intervals overlap (`[88.6%, 100.0%]` vs `[78.7%, 98.2%]`), this is also a statistical **Tie (overlapping 95% CIs)** at `N=30`.
2. **One-Sided Accuracy-Drop Baseline (`acc_champ - acc_cand > 0.01`)**: Flags only when overall test accuracy *drops* by more than 1 percentage point.
   - Under a one-sided accuracy-drop baseline, `feature_leakage` is detected in **0/30 (0.0%, Wilson 95% CI [0.0%, 11.4%])** trials (because leakage increases accuracy), whereas ModelGate's dedicated `run_leakage_check` detects **30/30 (100.0%, Wilson 95% CI [88.6%, 100.0%])**.

## No-op False-Positive Rate & 11.4% Upper Bound

For the `noop` control (retraining the identical pipeline on clean data with candidate seed `s + 500`), both ModelGate and the baseline produced **0/30 (0.0%)** observed false positives across seeds `1000..1029`. At sample size `N=30`, `0/30` corresponds to a **Wilson 95% confidence interval of `[0.0%, 11.4%]`**, meaning the data is consistent with a true false-positive rate of up to **11.4%** at the 95% confidence level.

## Independent Fairness Validation on Public Datasets (Frozen Thresholds)

Evaluated across **30 independent seeds (`3000..3029`)** on **UCI Adult (Census Income)** (`sex`, target `>50K`) and **UCI German Credit (`credit-g`)** (`sex`, target `bad`), plus **Synthetic Credit (`1000..1029`)**, with all `FairnessConfig` thresholds frozen at their defaults (`warn_at=0.05`, `fail_at=0.10`, `max_gap_increase=0.02`, `fail_gap_increase=0.05`, `confidence=0.95`, `n_bootstrap=200`). Two new uncalibrated fairness faults were tested:
- `fairness_threshold_shift` (`τ_group = 0.25` vs `0.50`, probabilities unchanged) and `fairness_threshold_shift_hi` (`τ_group = 0.75` vs `0.50`, probabilities unchanged).
- `fairness_proxy_feature` (explicit protected attribute redacted; correlated `proxy_score` added with 35% label corruption on high-proxy negative rows).

| Dataset (Seeds) | Fault | Paired Bootstrap Gap CI Only | Full Fairness Check (Static + Relative) | Accuracy Baseline (`\|Δacc\| > 1%`) | Mean DP (`champ -> cand`) |
|-----------------|-------|------------------------------|-----------------------------------------|--------------------------------------|---------------------------|
| **UCI Adult** (`3000..3029`) | `fairness_threshold_shift` (`τ=0.25`) | 4/30 (13.3%, [5.3%, 29.7%]) | 29/30 (96.7%, [83.3%, 99.4%]) | 5/30 (16.7%, [7.3%, 33.6%]) | `0.162 -> 0.093` |
| **UCI Adult** (`3000..3029`) | `fairness_threshold_shift_hi` (`τ=0.75`) | 29/30 (96.7%, [83.3%, 99.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | `0.162 -> 0.194` |
| **UCI Adult** (`3000..3029`) | `fairness_proxy_feature` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 22/30 (73.3%, [55.6%, 85.8%]) | `0.162 -> 0.078` |
| **UCI Adult** (`3000..3029`) | `fairness_regression` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 14/30 (46.7%, [30.2%, 63.9%]) | `0.162 -> 0.086` |
| **UCI Adult** (`3000..3029`) | `noop` (FP rate) | **0/30 (0.0% FP, [0.0%, 11.4%])** | **30/30 (100.0% FP, [88.6%, 100.0%])** | **0/30 (0.0% FP, [0.0%, 11.4%])** | `0.162 -> 0.162` |
| **German Credit** (`3000..3029`) | `fairness_threshold_shift` (`τ=0.25`) | 29/30 (96.7%, [83.3%, 99.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 20/30 (66.7%, [48.8%, 80.8%]) | `0.047 -> 0.351` |
| **German Credit** (`3000..3029`) | `fairness_threshold_shift_hi` (`τ=0.75`) | 19/30 (63.3%, [45.5%, 78.1%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 23/30 (76.7%, [59.1%, 88.2%]) | `0.047 -> 0.155` |
| **German Credit** (`3000..3029`) | `fairness_proxy_feature` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 25/30 (83.3%, [66.4%, 92.7%]) | `0.047 -> 0.376` |
| **German Credit** (`3000..3029`) | `fairness_regression` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 26/30 (86.7%, [70.3%, 94.7%]) | `0.047 -> 0.347` |
| **German Credit** (`3000..3029`) | `noop` (FP rate) | **0/30 (0.0% FP, [0.0%, 11.4%])** | **10/30 (33.3% FP, [19.2%, 51.2%])** | **0/30 (0.0% FP, [0.0%, 11.4%])** | `0.047 -> 0.046` |
| **Synthetic Credit** (`1000..1029`) | `fairness_threshold_shift` (`τ=0.25`) | 8/30 (26.7%, [14.2%, 44.4%]) | 8/30 (26.7%, [14.2%, 44.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | `0.002 -> 0.015` |
| **Synthetic Credit** (`1000..1029`) | `fairness_threshold_shift_hi` (`τ=0.75`) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | `0.002 -> 0.002` |
| **Synthetic Credit** (`1000..1029`) | `fairness_proxy_feature` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 29/30 (96.7%, [83.3%, 99.4%]) | `0.002 -> 0.064` |
| **Synthetic Credit** (`1000..1029`) | `fairness_regression` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 9/30 (30.0%, [16.7%, 47.9%]) | `0.002 -> 0.024` |
| **Synthetic Credit** (`1000..1029`) | `noop` (FP rate) | **0/30 (0.0% FP, [0.0%, 11.4%])** | **0/30 (0.0% FP, [0.0%, 11.4%])** | **0/30 (0.0% FP, [0.0%, 11.4%])** | `0.002 -> 0.002` |

## Methodology

Each fault is injected across 30 independent random seeds (`1000..1029`). For each seed `s`, training data (`N=3000`) is generated with `RandomState(s)` and held-out test data (`N=1500`) with `RandomState(s + 1000)`. Confidence intervals are calculated using the Wilson score interval at the 95% confidence level (`z = 1.95996`).
