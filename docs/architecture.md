# ModelGate — Architecture

## Overview

ModelGate is a **CI/CD quality gate for ML models**. It intercepts every PR that changes model code, data, or configuration, executes a configurable set of checks, and blocks the merge if the model doesn't meet production standards.

```
┌──────────────────────────────────────────────────────────────────┐
│                       Developer Workflow                         │
│                                                                  │
│  git push  →  GitHub Actions  →  modelgate run  →  PR comment   │
│                                        │                         │
│                              verdict.json / report.html         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
ModelGate/
├── src/modelgate/          # Core Python package
│   ├── cli.py              # Typer CLI (modelgate run / bench / reproduce …)
│   ├── config.py           # Pydantic models for modelgate.yaml
│   ├── loader.py           # Dynamic import of model entrypoints
│   ├── lineage.py          # Git SHA + DVC hash capture
│   ├── registry.py         # MLflow champion/challenger registry
│   ├── verdict.py          # Aggregates CheckResults → PASS/WARN/FAIL
│   ├── report.py           # HTML + Markdown PR comment generation
│   ├── stats.py            # Bootstrap CI, PSI, paired-bootstrap helpers
│   ├── benchmark.py        # Fault-injection benchmark suite
│   ├── api/                # FastAPI REST layer
│   │   ├── main.py         # Endpoints: runs, models, settings, health
│   │   └── models.py       # Pydantic request/response schemas
│   ├── checks/             # Individual quality-gate checks
│   │   ├── data_contract.py   # Pandera schema validation
│   │   ├── drift.py           # PSI / KS / Wasserstein drift detection
│   │   ├── slices.py          # Per-slice metric regression (bootstrap CI)
│   │   ├── behavioral.py      # Invariance & directional tests
│   │   ├── budgets.py         # Latency, model size, peak memory budgets
│   │   └── fairness.py        # Demographic parity / equal opportunity
│   └── integrations/
│       └── github/
│           └── pr_bot.py      # GitHub PR comment posting
├── dashboard/              # Vite + React TypeScript dashboard
│   ├── src/pages/          # Runs, Models, Compare, Benchmark, Settings
│   └── Dockerfile          # Multi-stage nginx build
├── demo/                   # End-to-end credit-risk demo
│   ├── train.py            # Generates synthetic data + trains model
│   └── contracts/schema.py # Pandera schema for demo dataset
├── data/                   # CSV datasets (tracked by DVC)
├── data.dvc                # DVC lock file (SHA-256 hash of data/)
├── tests/
│   ├── unit/               # Pure-Python, no I/O
│   └── integration/        # Smoke tests that call the API
├── scripts/seed.py         # Docker Compose seed script
├── .github/workflows/
│   ├── ci.yml              # Ruff, mypy, pytest, dashboard build
│   └── modelgate.yml       # ModelGate quality-gate on PRs
├── action.yml              # Reusable GitHub composite action
├── docker-compose.yml      # api + mlflow + dashboard + seed
├── Dockerfile              # API container
└── modelgate.yaml          # Project-level configuration
```

---

## Core Data Flow

```
modelgate run
     │
     ▼
config.py ──► loader.py
                  │  load model entrypoint + datasets
                  ▼
            ┌─────────────────────────────────────┐
            │            Check pipeline           │
            │  ┌──────────────┐                  │
            │  │ data_contract│  Pandera schema   │
            │  ├──────────────┤                  │
            │  │ drift        │  PSI / KS / Wass  │
            │  ├──────────────┤                  │
            │  │ slices       │  Bootstrap CI     │
            │  ├──────────────┤                  │
            │  │ behavioral   │  Invariance/Dir.  │
            │  ├──────────────┤                  │
            │  │ budgets      │  Lat/Mem/Size     │
            │  ├──────────────┤                  │
            │  │ fairness     │  Dem. parity      │
            │  └──────────────┘                  │
            └──────────┬──────────────────────────┘
                       │  list[CheckResult]
                       ▼
                  verdict.py  ──► PASS / WARN / FAIL
                       │
                       ├──► report.py ──► report.html
                       │               └► pr_comment.md
                       ├──► registry.py ──► MLflow alias update
                       └──► api/main.py ──► POST /runs
```

---

## Checks

| Check | Trigger | Default Action |
|-------|---------|----------------|
| **Data Contract** | Schema column types, nullable constraints (Pandera) | `fail` on violation |
| **Drift** | PSI > threshold vs reference dataset | `warn` at 0.10, `fail` at 0.25 |
| **Slice Regression** | Per-group F1 drop with 95% bootstrap CI | `fail` if drop > `max_drop` |
| **Behavioral** | Invariance (protected attr flip) & directional (income ↑ ⇒ risk ↓) | `fail` if flip > `max_flip_rate` |
| **Budgets** | p95 inference latency, pickle size, peak RSS | `fail` if any budget exceeded |
| **Fairness** | Demographic parity difference on protected groups | `warn` at 0.05, `fail` at 0.10 |

---

## Verdict Aggregation

`verdict.py` collects all `CheckResult(status, check, details)` objects and applies:

1. Any `FAIL` → overall verdict is **FAIL**  
2. Any `WARN` (no `FAIL`) → overall verdict is **WARN**  
3. All `PASS` / `SKIP` → overall verdict is **PASS**

The `fail_if_any_fail` flag in `modelgate.yaml` controls whether the CI step exits non-zero.

---

## REST API

The FastAPI layer (`api/main.py`) exposes:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness check |
| `POST` | `/runs` | Create a gate run record |
| `GET` | `/runs` | List all runs (with optional filters) |
| `GET` | `/runs/{id}` | Fetch a single run |
| `GET` | `/models` | List registered models |
| `GET` | `/settings` | Read project settings |
| `PUT` | `/settings` | Update project settings |
| `GET` | `/stats/summary` | Aggregate pass/warn/fail counts |

Data is persisted in a SQLite file (`modelgate.db`). In Docker Compose this file is stored in a named volume.

---

## MLflow Integration

ModelGate uses a **single tracking URI** configured in `modelgate.yaml`:

```yaml
mlflow:
  tracking_uri: sqlite:///mlflow.db   # local dev
  artifact_location: ./mlruns
```

When running via Docker Compose the API passes `MLFLOW_TRACKING_URI=http://mlflow:5000` so all services use the same MLflow server. The `registry.py` module promotes a run to the `champion` alias after a PASS verdict.

---

## DVC Data Tracking

`data/` is tracked via DVC (`data.dvc`). To reproduce a run against the exact data used in CI:

```bash
dvc pull          # downloads data/ matching data.dvc SHA
modelgate reproduce
```

`modelgate reproduce` calls `dvc repro` before running the gate, ensuring data integrity.

---

## Docker Compose Services

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   dashboard  │────▶│     api      │────▶│    mlflow    │
│  :3000 nginx │     │  :8000 uvicorn│     │ :5000 server │
└──────────────┘     └──────────────┘     └──────────────┘
                              ▲
                      ┌───────┴───────┐
                      │     seed      │
                      │ (one-shot run)│
                      └───────────────┘
```

- **api** — FastAPI service; reads/writes `modelgate.db` via a named volume.  
- **mlflow** — MLflow tracking server; persists runs in `mlflow-data` volume.  
- **dashboard** — Vite/React app served via nginx; proxies API calls.  
- **seed** — Runs once on startup; generates demo data and posts results to the API.

Start everything:

```bash
docker compose up
```

---

## GitHub Actions

### CI (`ci.yml`)

Runs on every push and PR to `main`:

1. **Ruff** — lint and format check  
2. **Mypy** — static type checking  
3. **Pytest** — unit tests with coverage gate (≥ 55% total)  
4. **Dashboard build** — `npm run build` in `dashboard/`

### Quality Gate (`modelgate.yml`)

Triggered on PRs that touch `src/`, `demo/`, `data/`, or `modelgate.yaml`:

1. Runs `modelgate run`  
2. Uploads `report.html` as an artifact  
3. Posts/updates a PR comment with the verdict table  
4. Sets a commit status (`success` / `failure`)

### Reusable Composite Action (`action.yml`)

Any downstream repository can use ModelGate as a reusable action:

```yaml
- uses: your-org/ModelGate@main
  with:
    config: modelgate.yaml
    output-dir: modelgate_output
    fail-on-verdict: 'true'
```

---

## Extending ModelGate

To add a new check:

1. Create `src/modelgate/checks/mycheck.py` with a `run_mycheck(…) -> CheckResult` function.  
2. Add a `MyCheckConfig` Pydantic model in `config.py`.  
3. Wire it into `cli.py` (inside `run_gate`).  
4. Add a unit test under `tests/unit/test_checks.py`.

The `CheckResult` dataclass:

```python
@dataclass
class CheckResult:
    check: str          # e.g. "drift"
    status: str         # "pass" | "warn" | "fail" | "skip"
    details: dict       # arbitrary payload for the report
    reason: str = ""    # human-readable summary line
```
