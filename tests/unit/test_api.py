"""Unit tests for ModelGate FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from modelgate.api.main import app, engine


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    with TestClient(app) as client:
        yield client


def test_health_check(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_create_and_get_run(client: TestClient) -> None:
    payload = {
        "project": "test-project",
        "branch": "feat/test",
        "git_sha": "abc1234",
        "seed": 42,
        "verdict": {
            "status": "pass",
            "summary": "All checks passed",
            "reasons": [],
            "check_results": [
                {
                    "name": "data_contract",
                    "status": "pass",
                    "summary": "0 violations",
                    "metrics": {"violations": 0},
                    "details": {},
                    "duration_s": 0.5,
                }
            ],
        },
        "metrics": {"accuracy": 0.95},
        "lineage": {"git_sha": "abc1234", "seed": 42},
    }

    create_resp = client.post("/runs", json=payload)
    assert create_resp.status_code == 200
    created = create_resp.json()
    assert created["status"] == "pass"
    assert "id" in created

    run_id = created["id"]
    get_resp = client.get(f"/runs/{run_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == run_id
    assert get_resp.json()["project"] == "test-project"

    # List runs
    list_resp = client.get("/runs?project=test-project")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1


def test_run_approval_flow(client: TestClient) -> None:
    # Create a WARN run
    payload = {
        "project": "test-project",
        "branch": "feat/warn",
        "git_sha": "warn123",
        "seed": 42,
        "verdict": {
            "status": "warn",
            "summary": "1 warning",
            "reasons": [{"check_name": "drift", "severity": "warn", "message": "minor drift"}],
            "check_results": [],
        },
    }
    run = client.post("/runs", json=payload).json()
    run_id = run["id"]

    # Approve with comment
    approve_resp = client.post(
        f"/runs/{run_id}/approve",
        json={"reviewer": "reviewer1", "decision": "approve", "comment": "Approved after reviewing slice metrics"},
    )
    assert approve_resp.status_code == 200
    assert approve_resp.json()["decision"] == "approve"


def test_settings_endpoints(client: TestClient) -> None:
    # Get settings
    get_resp = client.get("/settings")
    assert get_resp.status_code == 200

    # Update settings
    update_resp = client.put("/settings", json={"config": {"checks": {"budgets": {"p95_latency_ms": 75.0}}}})
    assert update_resp.status_code == 200
    assert update_resp.json()["status"] == "saved"
