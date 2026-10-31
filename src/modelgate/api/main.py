"""ModelGate FastAPI application.

Provides REST API for run ingestion, querying, comparison, approval,
and model promotion.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, SQLModel, col, create_engine, select

from modelgate.api.models import Approval, CheckResultRecord, Run, Settings

# Database setup
DATABASE_URL = "sqlite:///modelgate.db"
engine = create_engine(DATABASE_URL, echo=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup."""
    SQLModel.metadata.create_all(engine)
    yield


app = FastAPI(
    title="ModelGate API",
    description="CI/CD quality gate for ML models",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──── Request/Response models ────


class RunCreateRequest(BaseModel):
    project: str = ""
    branch: str = ""
    pr_number: int | None = None
    timestamp: str = ""
    git_sha: str = ""
    seed: int = 42
    verdict: dict = {}
    lineage: dict = {}
    metrics: dict = {}
    run_id: int | str | None = None


class ApprovalRequest(BaseModel):
    reviewer: str
    decision: str  # "approve" or "reject"
    comment: str


class SettingsUpdateRequest(BaseModel):
    config: dict


# ──── Routes ────


@app.post("/runs")
def create_run(req: RunCreateRequest) -> dict:
    """Ingest a run result."""
    verdict = req.verdict or {}
    metrics: dict[str, float] = {}

    # Extract metrics from check results
    for cr in verdict.get("check_results", []):
        for k, v in cr.get("metrics", {}).items():
            metrics[f"{cr['name']}_{k}"] = v

    run = Run(
        project=req.project,
        branch=req.branch,
        pr_number=req.pr_number,
        git_sha=req.lineage.get("git_sha", req.git_sha) if req.lineage else req.git_sha,
        seed=req.lineage.get("seed", req.seed) if req.lineage else req.seed,
        status=verdict.get("status", ""),
        summary=verdict.get("summary", ""),
        verdict_json=json.dumps(verdict),
        lineage_json=json.dumps(req.lineage or {}),
        metrics_json=json.dumps(metrics),
    )

    with Session(engine) as session:
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = int(run.id or 0)

        # Save individual check results
        for cr in verdict.get("check_results", []):
            record = CheckResultRecord(
                run_id=run_id,
                name=cr.get("name", ""),
                status=cr.get("status", ""),
                summary=cr.get("summary", ""),
                metrics_json=json.dumps(cr.get("metrics", {})),
                details_json=json.dumps(cr.get("details", {})),
                duration_s=cr.get("duration_s", 0.0),
            )
            session.add(record)
        session.commit()

    return {"id": run_id, "status": verdict.get("status", "")}


@app.get("/runs")
def list_runs(
    status: str | None = Query(None),
    branch: str | None = Query(None),
    project: str | None = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
) -> list[dict]:
    """List runs with optional filters."""
    with Session(engine) as session:
        query = select(Run).order_by(col(Run.id).desc())

        if status:
            query = query.where(Run.status == status)
        if branch:
            query = query.where(Run.branch == branch)
        if project:
            query = query.where(Run.project == project)

        query = query.offset(offset).limit(limit)
        runs = session.exec(query).all()

        return [
            {
                "id": r.id,
                "project": r.project,
                "branch": r.branch,
                "pr_number": r.pr_number,
                "git_sha": r.git_sha,
                "status": r.status,
                "summary": r.summary,
                "created_at": r.created_at,
                "metrics": r.metrics,
            }
            for r in runs
        ]


@app.get("/runs/{run_id}")
def get_run(run_id: int) -> dict:
    """Get full run detail with all check results."""
    with Session(engine) as session:
        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Get check results
        checks = session.exec(
            select(CheckResultRecord).where(CheckResultRecord.run_id == run_id)
        ).all()

        # Get approvals
        approvals = session.exec(select(Approval).where(Approval.run_id == run_id)).all()

        return {
            "id": run.id,
            "project": run.project,
            "branch": run.branch,
            "pr_number": run.pr_number,
            "git_sha": run.git_sha,
            "seed": run.seed,
            "status": run.status,
            "summary": run.summary,
            "created_at": run.created_at,
            "verdict": run.verdict,
            "lineage": run.lineage,
            "metrics": run.metrics,
            "check_results": [
                {
                    "name": cr.name,
                    "status": cr.status,
                    "summary": cr.summary,
                    "metrics": json.loads(cr.metrics_json),
                    "details": json.loads(cr.details_json),
                    "duration_s": cr.duration_s,
                }
                for cr in checks
            ],
            "approvals": [
                {
                    "reviewer": a.reviewer,
                    "decision": a.decision,
                    "comment": a.comment,
                    "is_override": a.is_override,
                    "created_at": a.created_at,
                }
                for a in approvals
            ],
        }


@app.get("/runs/{run_id}/slices")
def get_run_slices(run_id: int) -> dict:
    """Get slice data for the explorer."""
    with Session(engine) as session:
        checks = session.exec(
            select(CheckResultRecord)
            .where(CheckResultRecord.run_id == run_id)
            .where(CheckResultRecord.name == "slices")
        ).all()

        if not checks:
            raise HTTPException(status_code=404, detail="No slice data found")

        details = json.loads(checks[0].details_json)
        return {
            "run_id": run_id,
            "slices": details.get("slices", []),
            "worst_5": details.get("worst_5", []),
        }


@app.get("/compare")
def compare_runs(
    a: int = Query(..., description="First run ID"),
    b: int = Query(..., description="Second run ID"),
) -> dict:
    """Side-by-side comparison of two runs."""
    with Session(engine) as session:
        run_a = session.get(Run, a)
        run_b = session.get(Run, b)

        if not run_a or not run_b:
            raise HTTPException(status_code=404, detail="Run not found")

        metrics_a = run_a.metrics
        metrics_b = run_b.metrics

        all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
        comparison = []
        for key in all_keys:
            val_a = metrics_a.get(key)
            val_b = metrics_b.get(key)
            delta = None
            if val_a is not None and val_b is not None:
                try:
                    delta = float(val_b) - float(val_a)
                except (ValueError, TypeError):
                    pass
            comparison.append(
                {
                    "name": key,
                    "value_a": val_a,
                    "value_b": val_b,
                    "delta": delta,
                }
            )

        return {
            "run_a": {"id": a, "status": run_a.status, "branch": run_a.branch},
            "run_b": {"id": b, "status": run_b.status, "branch": run_b.branch},
            "metrics": comparison,
        }


@app.post("/runs/{run_id}/approval")
@app.post("/runs/{run_id}/approve")
def approve_run(run_id: int, req: ApprovalRequest) -> dict:
    """Approve or reject a run with a comment."""
    with Session(engine) as session:
        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        is_override = run.status == "fail" and req.decision == "approve"

        if is_override and len(req.comment) < 20:
            raise HTTPException(
                status_code=400,
                detail="Override approval requires a justification of at least 20 characters",
            )

        approval = Approval(
            run_id=run_id,
            reviewer=req.reviewer,
            decision=req.decision,
            comment=req.comment,
            is_override=is_override,
        )

        session.add(approval)
        session.commit()
        session.refresh(approval)

        return {
            "id": approval.id,
            "run_id": run_id,
            "decision": req.decision,
            "is_override": is_override,
        }


@app.post("/runs/{run_id}/promote")
def promote_run(run_id: int) -> dict:
    """Promote a run's model to champion."""
    with Session(engine) as session:
        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Check if run can be promoted
        if run.status == "fail":
            # Check for override approval
            approvals = session.exec(
                select(Approval)
                .where(Approval.run_id == run_id)
                .where(Approval.decision == "approve")
            ).all()
            if not approvals:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot promote a FAIL run without override approval",
                )

        return {
            "run_id": run_id,
            "promoted": True,
            "message": f"Model from run {run_id} promoted to champion",
        }


@app.get("/models")
def list_models(
    project: str | None = Query(None),
) -> list[dict]:
    """Registry view with champion marked."""
    with Session(engine) as session:
        query = select(Run).where(col(Run.status).in_(["pass", "warn"])).order_by(col(Run.id).desc())
        if project:
            query = query.where(Run.project == project)
        runs = session.exec(query.limit(50)).all()

        return [
            {
                "version": f"v{r.id}",
                "run_id": r.id,
                "status": r.status,
                "created_at": r.created_at,
                "branch": r.branch,
                "git_sha": r.git_sha,
                "is_champion": i == 0,  # Latest passing run is champion
            }
            for i, r in enumerate(runs)
        ]


@app.get("/settings")
def get_settings(project: str = Query("default")) -> dict:
    """Get current settings."""
    with Session(engine) as session:
        settings = session.exec(select(Settings).where(Settings.project == project)).first()

        if not settings:
            return {"project": project, "config": {}}

        return {
            "project": settings.project,
            "config": json.loads(settings.config_json),
            "updated_at": settings.updated_at,
        }


@app.put("/settings")
def update_settings(project: str = Query("default"), req: SettingsUpdateRequest = Body(...)) -> dict:
    """Update settings."""
    with Session(engine) as session:
        settings = session.exec(select(Settings).where(Settings.project == project)).first()

        if settings:
            settings.config_json = json.dumps(req.config)
            settings.updated_at = datetime.now(UTC).isoformat()
        else:
            settings = Settings(
                project=project,
                config_json=json.dumps(req.config),
            )
            session.add(settings)

        session.commit()

        return {"project": project, "status": "saved"}


@app.get("/bench")
def get_benchmark() -> dict:
    """Get latest benchmark results."""
    results_path = Path("docs/benchmark_results.json")
    if results_path.exists():
        return json.loads(results_path.read_text())
    return {"message": "No benchmark results found. Run 'modelgate bench' first."}


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "modelgate-api"}
