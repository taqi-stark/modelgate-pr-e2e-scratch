"""Unit tests for ModelGate CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from modelgate.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "CI/CD quality gate for machine learning models" in result.stdout


def test_cli_init() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(app, ["init", tmpdir])
        assert result.exit_code == 0
        assert (Path(tmpdir) / "modelgate.yaml").exists()
        assert (Path(tmpdir) / "demo" / "train.py").exists()

        # Running again reports already exists
        result2 = runner.invoke(app, ["init", tmpdir])
        assert result2.exit_code == 0
        assert "already exists" in result2.stdout


def test_cli_run_missing_config() -> None:
    result = runner.invoke(app, ["run", "--config", "nonexistent_config.yaml"])
    assert result.exit_code == 1
    assert "Config file not found" in result.stdout


def test_cli_run_success(monkeypatch) -> None:
    # Test modelgate run with a passing mock
    from modelgate.loader import LineageInfo, LoadResult
    from modelgate.verdict import CheckResult, Verdict

    dummy_lineage = LineageInfo(
        git_sha="abcdef123456",
        data_hashes={"train": "hash123"},
        dvc_hash="dvc999",
        python_version="3.11.0",
        package_versions={},
        seed=42,
    )
    dummy_verdict = Verdict(
        status="pass",
        reasons=[],
        check_results=[
            CheckResult(name="data_contract", status="pass", summary="Passed"),
            CheckResult(name="drift", status="pass", summary="Passed"),
            CheckResult(name="slices", status="pass", summary="Passed"),
            CheckResult(name="behavioral", status="pass", summary="Passed"),
            CheckResult(name="budgets", status="pass", summary="Passed"),
            CheckResult(name="fairness", status="pass", summary="Passed"),
        ],
        summary="All checks passed",
    )

    dummy_load_result = LoadResult(
        candidate_model=MagicMock(),
        champion_model=MagicMock(),
        train_df=MagicMock(),
        test_df=MagicMock(),
        reference_df=MagicMock(),
        target_column="default",
        train_time_s=0.5,
        lineage=dummy_lineage,
        is_first_run=False,
    )

    monkeypatch.setattr(
        "modelgate.loader.load_candidate_and_champion", lambda cfg: dummy_load_result
    )
    monkeypatch.setattr(
        "modelgate.checks.data_contract.run_data_contract_check",
        lambda *args, **kwargs: dummy_verdict.check_results[0],
    )
    monkeypatch.setattr(
        "modelgate.checks.drift.run_drift_check",
        lambda *args, **kwargs: dummy_verdict.check_results[1],
    )
    monkeypatch.setattr(
        "modelgate.checks.leakage.run_leakage_check",
        lambda *args, **kwargs: CheckResult(name="leakage", status="pass", summary="Passed"),
    )
    monkeypatch.setattr(
        "modelgate.checks.label_shift.run_label_shift_check",
        lambda *args, **kwargs: CheckResult(name="label_shift", status="pass", summary="Passed"),
    )
    monkeypatch.setattr(
        "modelgate.checks.slices.run_slices_check",
        lambda *args, **kwargs: dummy_verdict.check_results[2],
    )
    monkeypatch.setattr(
        "modelgate.checks.behavioral.run_behavioral_check",
        lambda *args, **kwargs: dummy_verdict.check_results[3],
    )
    monkeypatch.setattr(
        "modelgate.checks.budgets.run_budgets_check",
        lambda *args, **kwargs: dummy_verdict.check_results[4],
    )
    monkeypatch.setattr(
        "modelgate.checks.fairness.run_fairness_check",
        lambda *args, **kwargs: dummy_verdict.check_results[5],
    )
    monkeypatch.setattr(
        "modelgate.verdict.compute_verdict", lambda results, cfg=None: dummy_verdict
    )
    monkeypatch.setattr(
        "modelgate.registry.log_run_to_mlflow", lambda *args, **kwargs: "run-test-123"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "output"
        result = runner.invoke(app, ["run", "--config", "modelgate.yaml", "--output", str(out_dir)])
        assert result.exit_code == 0
        assert (out_dir / "verdict.json").exists()
        assert (out_dir / "pr_comment.md").exists()
        assert (out_dir / "report.html").exists()
        assert "Verdict:" in result.stdout


def test_cli_compare() -> None:
    # When API is not running, falls back cleanly
    result = runner.invoke(app, ["compare", "run_1", "run_2"])
    assert result.exit_code == 0
    assert "Comparing runs" in result.stdout


def test_cli_promote_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "modelgate.registry.promote_to_champion",
        lambda project, run_id, alias="champion", tracking_uri=None: True,
    )
    result = runner.invoke(app, ["promote", "run-123"])
    assert result.exit_code == 0
    assert "promoted to champion" in result.stdout


def test_cli_promote_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "modelgate.registry.promote_to_champion",
        lambda project, run_id, alias="champion", tracking_uri=None: False,
    )
    result = runner.invoke(app, ["promote", "run-failed"])
    assert result.exit_code == 1
    assert "Failed to promote" in result.stdout


def test_cli_reproduce_success(monkeypatch) -> None:
    mock_run = MagicMock()
    mock_run.data.params = {"seed": "42", "git_sha": "abc1234", "dvc_hash": "dvc98765"}
    mock_run.data.metrics = {"accuracy": 0.88}

    mock_client = MagicMock()
    mock_client.get_run.return_value = mock_run

    mock_mlflow = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = mock_client
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda *args, **kwargs: mock_client)
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda *args, **kwargs: None)

    result = runner.invoke(app, ["reproduce", "run-123"])
    assert result.exit_code == 0
    assert "Reproducing run" in result.stdout
    assert "Seed: 42" in result.stdout
