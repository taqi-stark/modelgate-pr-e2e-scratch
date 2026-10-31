"""Unit tests for ModelGate fault injection benchmark."""

from __future__ import annotations

import pandas as pd

from modelgate.benchmark import FAULT_DESCRIPTIONS, _generate_data, inject_fault, run_single_trial


def test_generate_data() -> None:
    train_df, test_df = _generate_data(seed=42, n=100)
    assert len(train_df) == 100
    assert len(test_df) == 50
    assert "income" in train_df.columns
    assert "default" in train_df.columns


def test_inject_faults() -> None:
    train_df, _ = _generate_data(seed=42, n=100)

    for fault in FAULT_DESCRIPTIONS:
        faulty, override = inject_fault(train_df, fault, seed=42)
        assert isinstance(faulty, pd.DataFrame)
        if fault == "feature_leakage":
            assert "leaked" in faulty.columns
        elif fault == "latency_regression":
            assert override == "large_model"


def test_run_single_trial_noop() -> None:
    res = run_single_trial("noop", seed=42)
    assert res["fault_type"] == "noop"
    assert "modelgate_detected" in res
    assert "baseline_detected" in res
