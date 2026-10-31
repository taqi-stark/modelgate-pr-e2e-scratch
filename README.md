# ModelGate

[![CI](https://github.com/your-org/ModelGate/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/ModelGate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Coverage: 96% Core](https://img.shields.io/badge/coverage-96%25%20core-brightgreen.svg)](https://github.com/your-org/ModelGate)

**ModelGate** is an open-source CI/CD quality gate for machine learning models that intercepts pull requests changing model code, hyperparameters, or training data. Acting as automated code review and unit testing for ML, it runs a battery of six statistical and operational checks—data contracts, feature drift, slice regressions with paired bootstrap confidence intervals, behavioral invariance, performance budgets, and algorithmic fairness—blocking regressions before they reach production.

---

## Visual Previews

### 1. GitHub PR Comment Bot
When a PR is opened, ModelGate executes the quality gate and posts a comprehensive status report directly onto the pull request:

```markdown
### 🛡️ ModelGate Quality Gate Verdict: ❌ FAIL

> **Summary:** Candidate model failed 2 of 6 checks (slice regression detected in 'North' region, p95 latency budget exceeded).

| Check | Status | Value / Score | Threshold | Details |
| :--- | :---: | :---: | :---: | :--- |
| **Data Contract** | ✅ PASS | Valid | Strict | Pandera schema validation succeeded |
| **Feature Drift** | ✅ PASS | Max PSI: 0.042 | Warn ≥ 0.10, Fail ≥ 0.25 | No significant distribution shift |
| **Slice Regression** | ❌ FAIL | ΔF1: -0.068 (p=0.002) | Max drop: 0.030 (95% CI) | Severe degradation in slice `region=North` |
| **Behavioral Tests** | ✅ PASS | 0.0% flip rate | Max flip: 0.0% | Invariance tests passed |
| **Performance Budgets**| ❌ FAIL | p95: 58.9 ms | Max p95: 10.0 ms | 48.9 ms above latency limit |
| **Fairness** | ⚠️ WARN | DP Diff: 0.062 | Warn ≥ 0.05, Fail ≥ 0.10 | Disparity observed for `gender` |

<details>
<summary><b>🔍 How to Reproduce Locally</b></summary>

```bash
git checkout feature/update-credit-risk-model
dvc checkout data.dvc
modelgate reproduce --run-id 09b01094598a448781657d622a4e64a0
```
</details>
```

---

### 2. Interactive React Dashboard (`http://localhost:3000`)
The web dashboard provides interactive exploration of evaluation runs, champion-challenger comparisons, and drift distributions:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MODELGATE DASHBOARD                     [ Runs ]   [ Models ]   [ Compare ]   [ Benchmark ]   [ Settings ]│
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  RUN #09b01094   •   credit-risk-gate   •   Commit: 8946af2   •   Status: ❌ REJECTED                      │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  OVERVIEW CARDS                                                                                        │
│  ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐  ┌──────────────────┐ │
│  │ Champion Accuracy     │  │ Candidate Accuracy    │  │ Slices Analyzed       │  │ Gate Verdict     │ │
│  │ 90.9% (GBM baseline)  │  │ 91.5% (+0.6% overall) │  │ 6 Subgroups (F1)      │  │ ❌ FAIL (Blocked)│ │
│  └───────────────────────┘  └───────────────────────┘  └───────────────────────┘  └──────────────────┘ │
│                                                                                                        │
│  PER-SLICE REGRESSION BREAKDOWN (Paired Bootstrap, B=500, α=0.05)                                      │
│  Slice                Champion F1   Candidate F1   Difference (Δ)   p-value   Holm-Bonf.   Status      │
│  ──────────────────   ───────────   ────────────   ──────────────   ───────   ──────────   ─────────   │
│  region = North              0.88           0.81           -0.070    0.0020       Yes      ❌ FAIL     │
│  region = South              0.91           0.92           +0.010    0.4120       No       ✅ PASS     │
│  gender = female             0.89           0.89            0.000    0.8500       No       ✅ PASS     │
│  age_bucket = senior         0.85           0.86           +0.010    0.3200       No       ✅ PASS     │
│                                                                                                        │
│  LATENCY & RESOURCE BUDGETS                                                                            │
│  • Single-Row Prediction Latency:  p50 = 30.9 ms  |  p95 = 58.9 ms (Budget: 10.0 ms)  ❌ EXCEEDED     │
│  • Serialized Artifact Size:       2.4 MB (Budget: 50.0 MB)                           ✅ OK           │
│  • Peak Inference Memory:          14.2 MB (Budget: 500.0 MB)                         ✅ OK           │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart (3 Commands)

Get ModelGate running locally with the credit-risk demo in 3 commands:

```bash
# 1. Install ModelGate with development dependencies
pip install -e ".[dev]"

# 2. Train baseline model and generate demo datasets
python demo/train.py

# 3. Execute the full ModelGate quality check suite
modelgate run --config modelgate.yaml
```

*To launch the full API, MLflow, and Dashboard stack via Docker:*
```bash
docker compose up -d
```

---

## Architecture Diagram

```mermaid
flowchart TD
    subgraph Developer Workflow
        A[Git Push / Pull Request] --> B[GitHub Actions CI / action.yml]
    end

    subgraph ModelGate Execution
        B --> C[modelgate run --config modelgate.yaml]
        C --> D[Lineage Capture: Git SHA + DVC Hash]
        C --> E[MLflow Registry: Load Champion vs Candidate]
        C --> F{Quality Gate Checks}
    end

    subgraph Quality Gate Suite
        F --> G1[1. Data Contract: Pandera Schema]
        F --> G2[2. Feature Drift: PSI & KS-Test]
        F --> G3[3. Slice Regression: Paired Bootstrap CI]
        F --> G4[4. Behavioral Tests: Invariance / Directional]
        F --> G5[5. Performance Budgets: Latency & Memory]
        F --> G6[6. Algorithmic Fairness: Demographic Parity]
    end

    subgraph Verdict & Reporting
        G1 & G2 & G3 & G4 & G5 & G6 --> H[Verdict Aggregator]
        H --> I[PASS / WARN / FAIL Verdict]
        I --> J[Artifacts: verdict.json & report.html]
        I --> K[GitHub PR Comment Bot]
        I --> L[MLflow Experiment Tracking & Tags]
        I --> M[React Dashboard via FastAPI]
    end
```

---

## Benchmark Headline Result

A comprehensive fault-injection benchmark comparing **ModelGate** against a standard **Accuracy-Only Baseline** across **30 independent seeds** per fault mode:

| Fault Injected | Defect Description | ModelGate Detection Rate | Accuracy-Only Baseline | Winner |
| :--- | :--- | :---: | :---: | :---: |
| **`feature_leakage`** | Target derived column added to training data | **100%** | 100% | 🔵 Tie |
| **`label_shift`** | 40% label flips in `North` region subgroup | **93%** | 93% | 🔵 Tie |
| **`degraded_slice`** | 90% data drop on `North` region subgroup | **100%** | 3% | **✅ ModelGate (+97%)** |
| **`schema_drift`** | Unit scale change on `income` feature | **100%** | 40% | **✅ ModelGate (+60%)** |
| **`latency_regression`**| Substituted high-latency ensemble model | **100%** | 0% | **✅ ModelGate (+100%)** |
| **`fairness_regression`**| Disproportionate label flip on protected group | **43%** | 77% | ❌ Baseline |
| **`noop` (Control)** | Re-trained identical pipeline with new seed | **0% FP** | **0% FP** | **✅ ModelGate (No FPs)** |

> **Key Takeaway:** An accuracy-only metric catches only **3%** of silent subgroup degradations and **0%** of latency blowups. ModelGate detects **100%** of these defects while maintaining a **0% false-positive rate** on clean models. Full methodology available in [`docs/benchmark.md`](docs/benchmark.md).

---

## Project Structure

```
ModelGate/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Lint (ruff), type check (mypy), tests (pytest), dashboard build
│       └── modelgate.yml          # Reusable quality gate action invocation on PRs
├── action.yml                     # Reusable GitHub Composite Action
├── docker-compose.yml             # Local multi-service stack (API, MLflow, Dashboard, Seed)
├── Dockerfile                     # Multi-stage container for ModelGate API & CLI
├── pyproject.toml                 # Package configuration, Hatchling build, pytest & ruff settings
├── modelgate.yaml                 # Quality gate configuration file
├── data.dvc                       # DVC lockfile tracking SHA-256 data hash
├── data/                          # Reference, train, and test datasets
├── demo/
│   ├── train.py                   # Credit-risk demo dataset generation & training
│   └── contracts/
│       └── schema.py              # Pandera data contract schema
├── dashboard/                     # React + Vite TypeScript Dashboard
│   ├── src/
│   │   ├── pages/                 # Runs, Models, Compare, Benchmark, Settings
│   │   └── App.tsx                # Main router & layout
│   ├── nginx.conf                 # Production reverse-proxy config
│   └── Dockerfile                 # Multi-stage Node/Nginx container
├── docs/
│   ├── architecture.md            # System design, data flow, and check mechanics
│   ├── usage.md                   # CLI commands, config schema, and CI/CD guide
│   ├── decisions.md               # Architecture Decision Records (ADRs)
│   ├── benchmark.md               # Fault-injection benchmark methodology & findings
│   └── benchmark_results.json     # Machine-readable 30-seed benchmark dataset
├── scripts/
│   ├── seed.py                    # Docker Compose seed script for initial demo data
│   └── run_bench_fast.py          # Parallel 30-seed benchmark runner
├── src/modelgate/                 # Core Python package
│   ├── cli.py                     # Typer CLI (modelgate run / bench / reproduce / init)
│   ├── config.py                  # Pydantic models for modelgate.yaml
│   ├── verdict.py                 # Verdict aggregator (PASS / WARN / FAIL)
│   ├── stats.py                   # Paired bootstrap CI, PSI, KS-test, Holm-Bonferroni
│   ├── report.py                  # HTML report & GitHub Markdown PR comment generation
│   ├── registry.py                # MLflow model registry champion/challenger interface
│   ├── lineage.py                 # Git SHA & DVC data hash extraction
│   ├── loader.py                  # Dynamic module & object loader
│   ├── benchmark.py               # Benchmark fault generators & runner
│   ├── api/                       # FastAPI backend
│   │   ├── main.py                # REST endpoints (/runs, /models, /settings, /health)
│   │   └── models.py              # SQLModel & Pydantic database models
│   ├── checks/                    # Quality gate check modules
│   │   ├── data_contract.py       # Pandera schema contract check
│   │   ├── drift.py               # Feature PSI & KS-test drift check
│   │   ├── slices.py              # Subgroup slice regression check with bootstrap CI
│   │   ├── behavioral.py          # Invariance and directional metamorphic check
│   │   ├── budgets.py             # Latency, model size, and peak memory check
│   │   └── fairness.py            # Demographic parity and equal opportunity check
│   └── integrations/
│       └── github/
│           └── pr_bot.py          # GitHub PR comment poster
└── tests/
    └── unit/                      # 63 unit tests across API, CLI, checks, stats, and reports
```

---

## License

ModelGate is open-source software licensed under the [MIT License](LICENSE).
