"""Unit tests for ModelGate CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path

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
