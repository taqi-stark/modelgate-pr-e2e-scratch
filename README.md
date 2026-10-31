# ModelGate

CI/CD quality gate for machine learning models. Think of it as pytest plus code review for ML models.

[![CI](https://github.com/your-org/ModelGate/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/ModelGate/actions/workflows/ci.yml)

## Overview

ModelGate is a Python tool that acts as a CI/CD quality gate for machine learning models. On every pull request that changes a model, data, or training code, ModelGate trains or loads the candidate model, compares it to the current production (champion) model, runs a battery of automated checks, and returns a PASS, WARN or FAIL verdict.

---

## Quick Start (local)

```bash
pip install -e ".[dev]"
modelgate run --config modelgate.yaml
```

## Quick Start (Docker Compose)

```bash
docker compose up          # starts API :8000, MLflow :5000, dashboard :3000
# The seed service runs once and populates demo data automatically
```

---

## GitHub Actions — Reusable Composite Action

Use ModelGate as a drop-in GitHub Actions step in any ML repository:

```yaml
# .github/workflows/modelgate.yml
name: ModelGate Quality Gate
on:
  pull_request:
    paths: ['src/**', 'data/**', 'modelgate.yaml']

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run ModelGate
        uses: your-org/ModelGate@main
        with:
          config: modelgate.yaml        # path to your modelgate.yaml
          output-dir: modelgate_output  # where to write reports
          fail-on-verdict: 'true'       # exit 1 if verdict is FAIL
```

### Action inputs

| Input | Default | Description |
|-------|---------|-------------|
| `config` | `modelgate.yaml` | Path to your ModelGate config |
| `python-version` | `3.11` | Python version to set up |
| `output-dir` | `modelgate_output` | Directory for `report.html` and `pr_comment.md` |
| `fail-on-verdict` | `true` | Exit non-zero when verdict is `fail` |

### Action outputs

| Output | Description |
|--------|-------------|
| `verdict` | `pass`, `warn`, or `fail` |
| `summary` | Human-readable reason string |
| `report-html` | Path to the generated HTML report |
| `pr-comment` | Path to the generated Markdown PR comment |

---

## CI Workflow (`ci.yml`)

The built-in CI workflow runs on every push/PR to `main`:

| Job | What it does |
|-----|-------------|
| **lint** | `ruff check` + `ruff format --check` + `mypy` |
| **test** | `pytest tests/unit/` with coverage gate |
| **dashboard-build** | `npm run build` in `dashboard/` |

---

## DVC — Reproducible Data

Dataset hashes are tracked in `data.dvc`. To reproduce a run against the exact data used in CI:

```bash
dvc pull               # downloads data/ matching the locked SHA
modelgate reproduce    # calls dvc repro then runs the gate
```

---

## Features

- **Data Contract Validation** – Schema enforcement with Pandera
- **Feature Drift Detection** – PSI and KS test monitoring
- **Slice Regression Testing** – Per-subgroup metric analysis with bootstrap CIs
- **Behavioral Testing** – Invariance and directional tests
- **Performance Budgets** – Latency, model size, memory limits
- **Fairness Checks** – Demographic parity and equal opportunity
- **Full Lineage** – Data hashes, git commits, seeds, environment tracking
- **Model Registry** – MLflow-backed with champion/challenger workflow
- **Dashboard** – React-based UI for run inspection and approval
- **GitHub Integration** – PR comments and status checks

---

## Documentation

- [Architecture](docs/architecture.md) — System design, data flow, and extending ModelGate
- [Benchmark](docs/benchmark.md) — Fault-injection detection rates vs accuracy-only baseline
