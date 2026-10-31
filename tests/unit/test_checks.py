"""Unit tests for all 6 ModelGate checks."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from modelgate.checks.data_contract import run_data_contract_check
from modelgate.checks.drift import run_drift_check
from modelgate.checks.slices import run_slices_check
from modelgate.checks.behavioral import run_behavioral_check
from modelgate.checks.budgets import run_budgets_check
from modelgate.checks.fairness import run_fairness_check
from modelgate.config import (
    DataContractConfig,
    DriftConfig,
    SliceConfig,
    BehavioralConfig,
    InvarianceTest,
    DirectionalTest,
    BudgetConfig,
    FairnessConfig,
)


@pytest.fixture
def sample_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create sample train, test, and reference datasets."""
    rng = np.random.RandomState(42)
    n = 200

    def make_df(seed_offset: int = 0) -> pd.DataFrame:
        r = np.random.RandomState(42 + seed_offset)
        return pd.DataFrame({
            "applicant_id": np.arange(1, n + 1),
            "income": r.uniform(20000, 100000, n),
            "credit_score": r.randint(500, 850, n),
            "region": r.choice(["North", "South", "East", "West"], n),
            "gender": r.choice(["male", "female"], n),
            "default": r.choice([0, 1], n, p=[0.8, 0.2]),
        })

    return make_df(0), make_df(1), make_df(2)


def test_data_contract_check_passes(sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df, _ = sample_dataset
    schema = DataFrameSchema({
        "applicant_id": Column(int, Check.greater_than(0)),
        "income": Column(float, Check.greater_than(0)),
        "credit_score": Column(int, Check.in_range(300, 850)),
        "default": Column(int, Check.isin([0, 1])),
    }, strict=False)

    cfg = DataContractConfig(enabled=True, on_violation="fail")
    res = run_data_contract_check(train_df, test_df, schema, cfg)
    assert res.status == "pass"
    assert res.metrics["violations"] == 0


def test_data_contract_check_violation(sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df, _ = sample_dataset
    bad_train = train_df.copy()
    bad_train.loc[0, "income"] = -500  # violates > 0 check

    schema = DataFrameSchema({
        "income": Column(float, Check.greater_than(0)),
    }, strict=False)

    cfg = DataContractConfig(enabled=True, on_violation="fail")
    res = run_data_contract_check(bad_train, test_df, schema, cfg)
    assert res.status == "fail"
    assert res.metrics["violations"] > 0


def test_drift_check_no_drift(sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    train_df, _, _ = sample_dataset
    ref_df = train_df.copy()
    cfg = DriftConfig(enabled=True, warn_at=0.1, fail_at=0.25)
    res = run_drift_check(train_df, ref_df, "default", cfg)
    assert res.status == "pass"
    assert res.metrics["psi_max"] < 0.1


def test_drift_check_detects_drift(sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    train_df, _, ref_df = sample_dataset
    drifted_train = train_df.copy()
    drifted_train["income"] = drifted_train["income"] * 10  # huge scale shift

    cfg = DriftConfig(enabled=True, warn_at=0.1, fail_at=0.25)
    res = run_drift_check(drifted_train, ref_df, "default", cfg)
    assert res.status in ("warn", "fail")
    assert res.metrics["psi_max"] > 0.1


def test_slices_check(sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df, _ = sample_dataset
    X_train = train_df[["income", "credit_score"]]
    y_train = train_df["default"]
    X_test = test_df[["income", "credit_score", "region"]]
    y_test = test_df["default"]

    # Champion and candidate identical
    champ = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    cand = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    cfg = SliceConfig(
        enabled=True,
        metric="accuracy",
        by=["region"],
        min_slice_size=10,
        max_drop=0.05,
        confidence=0.95,
        n_bootstrap=100,
    )

    res = run_slices_check(test_df, "default", champ, cand, cfg, seed=42)
    assert res.status == "pass"


def test_behavioral_check_invariance() -> None:
    df = pd.DataFrame({
        "applicant_id": [1, 2, 3, 4, 5],
        "income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "default": [0, 0, 1, 0, 1],
    })

    # Model that ignores applicant_id
    model = LogisticRegression().fit(df[["income"]], df["default"])

    class ModelWrapper:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return model.predict(X[["income"]])
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return model.predict_proba(X[["income"]])

    cfg = BehavioralConfig(
        enabled=True,
        invariance=[InvarianceTest(feature="applicant_id", perturb="random_id", max_flip_rate=0.0)],
        directional=[],
    )

    res = run_behavioral_check(df, "default", ModelWrapper(), cfg)
    assert res.status == "pass"


def test_budgets_check() -> None:
    df = pd.DataFrame({
        "income": [50000.0, 60000.0],
        "credit_score": [650, 700],
        "default": [0, 1],
    })
    model = LogisticRegression().fit(df[["income", "credit_score"]], df["default"])

    class BudgetModelWrapper:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return model.predict(X[["income", "credit_score"]])

    cfg = BudgetConfig(
        enabled=True,
        p95_latency_ms=500.0,
        max_model_mb=100.0,
        max_peak_mem_mb=1000.0,
    )

    res = run_budgets_check(df, "default", BudgetModelWrapper(), None, cfg)
    assert res.status == "pass"
    assert "p95_ms" in res.metrics


def test_fairness_check(sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df, _ = sample_dataset
    model = DummyClassifier(strategy="most_frequent").fit(train_df[["income"]], train_df["default"])

    class DummyWrapper:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return model.predict(X[["income"]])

    cfg = FairnessConfig(
        enabled=True,
        protected=["gender"],
        metric="demographic_parity_diff",
        warn_at=0.1,
        fail_at=0.2,
    )

    res = run_fairness_check(test_df, "default", DummyWrapper(), None, cfg)
    assert res.status == "pass"
