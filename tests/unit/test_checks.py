"""Unit tests for all 6 ModelGate checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandera.pandas import Check, Column, DataFrameSchema
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from modelgate.checks.behavioral import run_behavioral_check
from modelgate.checks.budgets import run_budgets_check
from modelgate.checks.data_contract import run_data_contract_check
from modelgate.checks.drift import run_drift_check
from modelgate.checks.fairness import run_fairness_check
from modelgate.checks.slices import run_slices_check
from modelgate.config import (
    BehavioralConfig,
    BudgetConfig,
    DataContractConfig,
    DirectionalTest,
    DriftConfig,
    FairnessConfig,
    InvarianceTest,
    SliceConfig,
)


@pytest.fixture
def sample_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create sample train, test, and reference datasets."""
    np.random.RandomState(42)
    n = 200

    def make_df(seed_offset: int = 0) -> pd.DataFrame:
        r = np.random.RandomState(42 + seed_offset)
        return pd.DataFrame(
            {
                "applicant_id": np.arange(1, n + 1),
                "income": r.uniform(20000, 100000, n),
                "credit_score": r.randint(500, 850, n),
                "region": r.choice(["North", "South", "East", "West"], n),
                "gender": r.choice(["male", "female"], n),
                "default": r.choice([0, 1], n, p=[0.8, 0.2]),
            }
        )

    return make_df(0), make_df(1), make_df(2)


def test_data_contract_check_passes(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df, _ = sample_dataset
    schema = DataFrameSchema(
        {
            "applicant_id": Column(int, Check.greater_than(0)),
            "income": Column(float, Check.greater_than(0)),
            "credit_score": Column(int, Check.in_range(300, 850)),
            "default": Column(int, Check.isin([0, 1])),
        },
        strict=False,
    )

    cfg = DataContractConfig(enabled=True, on_violation="fail")
    res = run_data_contract_check(train_df, test_df, schema, cfg)
    assert res.status == "pass"
    assert res.metrics["violations"] == 0


def test_data_contract_check_violation(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df, _ = sample_dataset
    bad_train = train_df.copy()
    bad_train.loc[0, "income"] = -500  # violates > 0 check

    schema = DataFrameSchema(
        {
            "income": Column(float, Check.greater_than(0)),
        },
        strict=False,
    )

    cfg = DataContractConfig(enabled=True, on_violation="fail")
    res = run_data_contract_check(bad_train, test_df, schema, cfg)
    assert res.status == "fail"
    assert res.metrics["violations"] > 0


def test_drift_check_no_drift(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, _, _ = sample_dataset
    ref_df = train_df.copy()
    cfg = DriftConfig(enabled=True, warn_at=0.1, fail_at=0.25)
    res = run_drift_check(train_df, ref_df, "default", cfg)
    assert res.status == "pass"
    assert res.metrics["psi_max"] < 0.1


def test_drift_check_detects_drift(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
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
    test_df[["income", "credit_score", "region"]]
    test_df["default"]

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
    df = pd.DataFrame(
        {
            "applicant_id": [1, 2, 3, 4, 5],
            "income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            "default": [0, 0, 1, 0, 1],
        }
    )

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
    df = pd.DataFrame(
        {
            "income": [50000.0, 60000.0],
            "credit_score": [650, 700],
            "default": [0, 1],
        }
    )
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


def test_slices_check_fail(sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    _, test_df, _ = sample_dataset

    # Champion predicts ground truth perfectly
    class PerfectChampion:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return test_df.loc[X.index, "default"].values

    # Candidate always predicts 0
    class DegradedCandidate:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

    cfg = SliceConfig(
        enabled=True,
        metric="accuracy",
        by=["region"],
        min_slice_size=5,
        max_drop=0.01,
        confidence=0.95,
        n_bootstrap=100,
    )

    res = run_slices_check(test_df, "default", PerfectChampion(), DegradedCandidate(), cfg, seed=42)
    assert res.status == "fail"
    assert res.metrics["n_failing_slices"] > 0


def test_slices_check_no_champion_skip(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, test_df, _ = sample_dataset
    candidate = DummyClassifier().fit(test_df[["income"]], test_df["default"])
    cfg = SliceConfig(enabled=True, by=["region"])
    res = run_slices_check(test_df, "default", None, candidate, cfg)
    assert res.status == "skip"
    assert "No champion" in res.summary


def test_budgets_check_fail_latency() -> None:
    df = pd.DataFrame({"income": [50000.0], "default": [0]})

    class SlowModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            import time

            time.sleep(0.005)
            return np.array([0])

    # Unrealistic 0.0001 ms budget to force fail
    cfg = BudgetConfig(enabled=True, p95_latency_ms=0.0001)
    res = run_budgets_check(df, "default", SlowModel(), None, cfg)
    assert res.status == "fail"
    assert "p95" in res.summary


class BigPicklableModel:
    def __init__(self) -> None:
        self.weights = np.zeros(200_000)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


def test_budgets_check_fail_size() -> None:
    df = pd.DataFrame({"income": [50000.0], "default": [0]})
    cfg = BudgetConfig(enabled=True, max_model_mb=0.01)
    res = run_budgets_check(df, "default", BigPicklableModel(), None, cfg)
    assert res.status == "fail"
    assert "model size" in res.summary


def test_fairness_check_fail() -> None:
    # Dataset with protected attribute
    df = pd.DataFrame(
        {
            "gender": ["male"] * 50 + ["female"] * 50,
            "income": [50000.0] * 100,
            "default": [0] * 100,
        }
    )

    # Disparate model: predicts 1 for males and 0 for females
    class BiasedModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return (X["gender"] == "male").astype(int).values

    cfg = FairnessConfig(
        enabled=True,
        protected=["gender"],
        metric="demographic_parity_diff",
        warn_at=0.1,
        fail_at=0.2,
    )
    res = run_fairness_check(df, "default", BiasedModel(), None, cfg)
    assert res.status == "fail"
    assert "demographic_parity_diff" in res.summary


def test_behavioral_check_fail_invariance() -> None:
    df = pd.DataFrame(
        {
            "applicant_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "income": [50000.0] * 8,
            "default": [0] * 8,
        }
    )

    # Model sensitive to applicant_id (should be invariant)
    class IDSensitiveModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return (X["applicant_id"] % 2 == 0).astype(int).values

    cfg = BehavioralConfig(
        enabled=True,
        invariance=[InvarianceTest(feature="applicant_id", perturb="random_id", max_flip_rate=0.0)],
        directional=[],
    )
    res = run_behavioral_check(df, "default", IDSensitiveModel(), cfg)
    assert res.status == "fail"
    assert res.metrics["n_fail"] > 0


def test_behavioral_check_directional() -> None:
    df = pd.DataFrame(
        {
            "debt": [0.1, 0.2, 0.5, 0.8],
            "default": [0, 0, 1, 1],
        }
    )

    class MonotonicModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return (X["debt"] > 0.4).astype(int).values

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            p = X["debt"].values
            return np.column_stack([1 - p, p])

    cfg_pass = BehavioralConfig(
        enabled=True,
        directional=[DirectionalTest(feature="debt", direction="increase", expect="prob_increase")],
    )
    res_pass = run_behavioral_check(df, "default", MonotonicModel(), cfg_pass)
    assert res_pass.status == "pass"

    # Expect probability to decrease when debt increases -> fail
    cfg_fail = BehavioralConfig(
        enabled=True,
        directional=[DirectionalTest(feature="debt", direction="increase", expect="prob_decrease")],
    )
    res_fail = run_behavioral_check(df, "default", MonotonicModel(), cfg_fail)
    assert res_fail.status == "fail"


def test_data_contract_edge_cases(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df, _ = sample_dataset

    # Disabled
    res_disabled = run_data_contract_check(
        train_df, test_df, "dummy:schema", DataContractConfig(enabled=False)
    )
    assert res_disabled.status == "skip"

    # No schema
    res_no_schema = run_data_contract_check(train_df, test_df, "", DataContractConfig(enabled=True))
    assert res_no_schema.status == "skip"

    # Schema import error
    res_bad_import = run_data_contract_check(
        train_df, test_df, "nonexistent.mod:schema", DataContractConfig(enabled=True)
    )
    assert res_bad_import.status == "fail"


def test_fairness_equal_opportunity(
    sample_dataset: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df, _ = sample_dataset
    model = DummyClassifier(strategy="most_frequent").fit(train_df[["income"]], train_df["default"])

    class ModelWrapper:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return model.predict(X[["income"]])

    cfg = FairnessConfig(
        enabled=True,
        protected=["gender"],
        metric="equal_opportunity_diff",
        warn_at=0.1,
        fail_at=0.2,
    )
    res = run_fairness_check(test_df, "default", ModelWrapper(), None, cfg)
    assert res.status == "pass"

    # Missing protected column -> skipped
    cfg_missing = FairnessConfig(enabled=True, protected=["nonexistent_col"])
    res_missing = run_fairness_check(test_df, "default", ModelWrapper(), None, cfg_missing)
    assert res_missing.status == "pass"
    assert res_missing.details["groups"][0]["status"] == "skip"

    # Disabled
    res_disabled = run_fairness_check(
        test_df, "default", ModelWrapper(), None, FairnessConfig(enabled=False)
    )
    assert res_disabled.status == "skip"


def test_behavioral_min_functionality(tmp_path: Path) -> None:
    csv_file = tmp_path / "mf.csv"
    csv_file.write_text("income,default,expected\n50000.0,0,0\n60000.0,1,1\n")

    class SimpleModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([0, 0])  # second case fails

    cfg = BehavioralConfig(enabled=True, min_functionality_csv=str(csv_file))
    df = pd.DataFrame({"income": [50000.0], "default": [0]})
    res = run_behavioral_check(df, "default", SimpleModel(), cfg)
    assert res.status == "fail"
    assert res.metrics["n_fail"] > 0
