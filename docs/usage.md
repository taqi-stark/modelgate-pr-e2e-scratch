# ModelGate — User Guide & Usage

This guide covers getting started, CLI commands, configuration options, GitHub Actions integration, and local Docker deployment.

---

## 1. Quickstart

Install ModelGate in a virtual environment:

```bash
git clone https://github.com/your-org/ModelGate.git
cd ModelGate
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the quality gate on the included credit-risk demo:

```bash
# 1. Generate synthetic dataset and train demo model
python demo/train.py

# 2. Run the full ModelGate quality check suite
modelgate run --config modelgate.yaml

# 3. View the generated report
open modelgate_output/report.html
```

---

## 2. CLI Reference

### `modelgate run`
Executes all configured quality checks against candidate and champion models.

```bash
modelgate run [OPTIONS]

Options:
  --config PATH      Path to modelgate.yaml (default: modelgate.yaml)
  --candidate PATH   Path or reference to candidate model artifact
  --champion PATH    Path or reference to champion model artifact
  --output-dir PATH  Directory to write verdict.json and report.html (default: modelgate_output)
  --post-pr-comment  Post verdict table to active GitHub PR (requires GITHUB_TOKEN)
```

Outputs:
- `modelgate_output/verdict.json`: Machine-readable check verdicts and scores.
- `modelgate_output/report.html`: Self-contained HTML report with interactive cards.
- Terminal output: Rich-formatted summary table and exit code (0 for PASS/WARN, 1 for FAIL).

### `modelgate bench`
Executes the fault-injection benchmark suite across multiple seeds and fault modes.

```bash
modelgate bench [OPTIONS]

Options:
  --seeds INT        Number of random seeds per fault (default: 30)
  --output-dir PATH  Directory to store benchmark results (default: docs/)
```

### `modelgate reproduce`
Reproduces a previous run's evaluation using recorded Git commit SHA, DVC data hash, and model parameters.

```bash
modelgate reproduce --run-id <RUN_ID>
# or
modelgate reproduce --data-hash <DVC_DATA_HASH>
```

---

## 3. Configuration (`modelgate.yaml`)

ModelGate is configured via a single declarative `modelgate.yaml` file:

```yaml
version: "1"

# MLflow Tracking Configuration
mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "credit-risk-gate"

# Data contract validation (Pandera)
data_contract:
  enabled: true
  schema_path: "demo/contracts/schema.py"

# Drift detection on features & target
drift:
  enabled: true
  method: "psi"               # psi, ks, or wasserstein
  warn_at: 0.10
  fail_at: 0.25

# Performance slices with bootstrap confidence intervals
slices:
  enabled: true
  metric: "f1"
  by: ["region", "gender"]
  min_slice_size: 20
  max_drop: 0.03
  confidence: 0.95
  n_bootstrap: 50

# Behavioral & metamorphic tests
behavioral:
  enabled: true
  invariance_features: ["region"]
  directional_features:
    - feature: "income"
      direction: "increasing"

# Resource and latency budgets
budgets:
  enabled: true
  p95_latency_ms: 15.0
  max_memory_mb: 512.0
  max_model_size_mb: 100.0

# Target leakage detection
leakage:
  enabled: true
  warn_corr: 0.50
  fail_corr: 0.75
  warn_auc: 0.85
  fail_auc: 0.92

# Overall & subgroup label-shift detection
label_shift:
  enabled: true
  by: ["region", "gender", "age_bucket"]
  min_slice_size: 30
  warn_delta: 0.05
  fail_delta: 0.10
  alpha: 0.01

# Algorithmic fairness (demographic parity, equalized odds, per-group calibration & paired bootstrap gap CI)
fairness:
  enabled: true
  protected: ["gender"]
  metric: "demographic_parity_diff"
  warn_at: 0.05
  fail_at: 0.10
  max_gap_increase: 0.02
  fail_gap_increase: 0.05
  confidence: 0.95
  n_bootstrap: 200
```

---

## 4. GitHub Actions Integration

Use ModelGate's composite action in `.github/workflows/modelgate.yml`:

```yaml
name: ModelGate Evaluation

on:
  pull_request:
    paths:
      - 'src/**'
      - 'demo/**'
      - 'modelgate.yaml'
      - 'data/**'

jobs:
  gate:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@v4

      - name: Run ModelGate Quality Gate
        uses: ./action.yml
        with:
          config-path: 'modelgate.yaml'
          post-pr-comment: 'true'
          github-token: ${{ secrets.GITHUB_TOKEN }}
```

---

## 5. Docker Compose Services

ModelGate includes a production-ready Docker Compose environment:

```bash
docker compose up -d
```

Services:
- **API**: http://localhost:8000 (FastAPI Swagger docs at `/docs`)
- **Dashboard**: http://localhost:3000 (React + Vite web interface)
- **MLflow Tracking Server**: http://localhost:5000
- **Seed Script**: Automatically seeds demo runs on startup.

---

## 6. Authentication, RBAC, Audit Log & HTTPS Deployment

ModelGate enforces bearer-token authentication and role-based access control (RBAC) across mutating API endpoints:

- **No Default Admin Token**: ModelGate never ships with a default or hardcoded admin token. Provision tokens explicitly via the CLI:
  ```bash
  modelgate users add alice --role engineer
  modelgate users add bob --role reviewer
  modelgate users add carol --role admin
  modelgate users list
  ```
  Alternatively, set `MODELGATE_ADMIN_TOKEN` in the server environment at startup to bootstrap the initial `admin` user. Plaintext tokens are displayed once at creation; only their SHA-256 hex digests are stored in the `users` table.
- **Role Permissions**:
  - `POST /runs`: `engineer`, `reviewer`, or `admin`
  - `POST /runs/{id}/approve` (or `/approval`): `reviewer` or `admin` (reviewer identity is bound from the authenticated token, not the request body). Overriding a `FAIL` run requires `admin` role and a justification comment of at least 20 characters.
  - `POST /runs/{id}/promote`: `admin`
  - `PUT /settings`: `admin`
  - `GET /audit`, `POST /users`, `GET /users`: `admin`
  - Read endpoints (`GET /runs`, `GET /models`, `GET /compare`, `GET /settings`, `GET /bench`, `GET /health`) remain unauthenticated for read-only dashboard access.
- **Append-Only Audit Log (`GET /audit`)**: Every allowed and denied (`401`/`403`/`400`) action on protected endpoints is recorded in the immutable `audit_log` table with `timestamp`, `actor`, `role`, `action`, `target`, `outcome`, and `detail`.
- **Restricted CORS & HTTPS Requirement**:
  - CORS origins are restricted to `http://localhost:3000,http://localhost:5173` by default (override via `MODELGATE_CORS_ORIGINS`), with methods restricted to `GET, POST, PUT, OPTIONS` and headers restricted to `Authorization, Content-Type, Accept`.
  - **HTTPS in Production**: Because bearer tokens are sent via `Authorization: Bearer <token>`, any production or non-localhost deployment **must** run behind TLS/HTTPS (e.g. Nginx/Caddy/ALB terminating TLS and setting `MODELGATE_CORS_ORIGINS=https://modelgate.example.com`) so tokens are encrypted in transit.
