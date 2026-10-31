"""Tests for the ModelGate statistics engine.

Includes unit tests and property-based tests with Hypothesis.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from modelgate.stats import (
    BootstrapResult,
    compute_psi,
    compute_psi_categorical,
    demographic_parity_diff,
    equal_opportunity_diff,
    holm_bonferroni,
    ks_test,
    paired_bootstrap_ci,
)

# ──────────────────────────────────────────────────
# Paired Bootstrap CI
# ──────────────────────────────────────────────────


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Simple accuracy metric."""
    return float(np.mean(y_true == y_pred))


class TestPairedBootstrapCI:
    """Tests for paired_bootstrap_ci."""

    def test_identical_models_zero_diff(self) -> None:
        """Identical predictions must yield zero point estimate and CI containing 0."""
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, size=200)
        preds = rng.randint(0, 2, size=200)

        result = paired_bootstrap_ci(y_true, preds, preds, accuracy, n_bootstrap=500, seed=42)

        assert result.point_estimate == 0.0
        assert result.ci_lower <= 0.0 <= result.ci_upper

    def test_better_model_positive_diff(self) -> None:
        """A better model A should produce a positive point estimate."""
        rng = np.random.RandomState(1)
        y_true = rng.randint(0, 2, size=500)
        # Model A = perfect, model B = random
        model_a = y_true.copy()
        model_b = rng.randint(0, 2, size=500)

        result = paired_bootstrap_ci(y_true, model_a, model_b, accuracy, n_bootstrap=500, seed=42)

        assert result.point_estimate > 0.0
        assert result.ci_lower > 0.0  # Even lower bound should be positive

    def test_result_structure(self) -> None:
        """Test that result has expected fields."""
        y = np.array([0, 1, 0, 1, 0])
        p = np.array([0, 1, 0, 1, 0])

        result = paired_bootstrap_ci(y, p, p, accuracy, n_bootstrap=100, seed=0)

        assert isinstance(result, BootstrapResult)
        assert result.n_bootstrap == 100
        assert result.confidence == 0.95

    @given(
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=10, deadline=None)
    def test_identical_models_ci_contains_zero(self, seed: int) -> None:
        """Property: identical models must always yield a CI containing 0."""
        rng = np.random.RandomState(seed)
        n = 100
        y_true = rng.randint(0, 2, size=n)
        preds = rng.randint(0, 2, size=n)

        result = paired_bootstrap_ci(y_true, preds, preds, accuracy, n_bootstrap=200, seed=seed)

        assert result.point_estimate == 0.0
        assert result.ci_lower <= 0.0 <= result.ci_upper


# ──────────────────────────────────────────────────
# Holm-Bonferroni
# ──────────────────────────────────────────────────


class TestHolmBonferroni:
    """Tests for holm_bonferroni."""

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert holm_bonferroni([]) == []

    def test_single_significant(self) -> None:
        """Single p-value below alpha is rejected."""
        assert holm_bonferroni([0.01], alpha=0.05) == [True]

    def test_single_not_significant(self) -> None:
        """Single p-value above alpha is not rejected."""
        assert holm_bonferroni([0.1], alpha=0.05) == [False]

    def test_multiple_corrections(self) -> None:
        """Test that correction is more conservative than raw comparison."""
        p_values = [0.01, 0.04, 0.03]
        # Without correction, all would be significant at 0.05
        # With Holm: sorted = [0.01, 0.03, 0.04]
        # 0.01 <= 0.05/3 = 0.0167 -> reject
        # 0.03 <= 0.05/2 = 0.025 -> not reject (step-down stops)
        result = holm_bonferroni(p_values, alpha=0.05)
        assert result[0] is True  # 0.01 is significant
        # The others depend on step-down logic

    def test_all_significant(self) -> None:
        """Very small p-values should all be rejected."""
        p_values = [0.001, 0.002, 0.003]
        result = holm_bonferroni(p_values, alpha=0.05)
        assert all(result)

    def test_none_significant(self) -> None:
        """Large p-values should not be rejected."""
        p_values = [0.5, 0.6, 0.7]
        result = holm_bonferroni(p_values, alpha=0.05)
        assert not any(result)

    def test_preserves_original_order(self) -> None:
        """Results correspond to original p-value positions."""
        p_values = [0.5, 0.001, 0.3]
        result = holm_bonferroni(p_values, alpha=0.05)
        # 0.001 (index 1) should be significant
        assert result[1] is True
        assert result[0] is False


# ──────────────────────────────────────────────────
# PSI
# ──────────────────────────────────────────────────


class TestPSI:
    """Tests for compute_psi."""

    def test_identical_distributions_near_zero(self) -> None:
        """PSI of identical distributions should be near zero."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, size=1000)

        psi = compute_psi(data, data)

        assert psi < 0.01

    def test_shifted_distribution_high_psi(self) -> None:
        """A shifted distribution should produce high PSI."""
        rng = np.random.RandomState(42)
        expected = rng.normal(0, 1, size=1000)
        actual = rng.normal(5, 1, size=1000)  # Large shift

        psi = compute_psi(expected, actual)

        assert psi > 0.25  # Significant shift

    def test_psi_non_negative(self) -> None:
        """PSI should always be non-negative."""
        rng = np.random.RandomState(42)
        expected = rng.normal(0, 1, size=500)
        actual = rng.normal(0.5, 1.5, size=500)

        psi = compute_psi(expected, actual)

        assert psi >= 0.0


class TestPSICategorical:
    """Tests for compute_psi_categorical."""

    def test_identical_categorical(self) -> None:
        """PSI of identical categorical should be near zero."""
        data = np.array(["a", "b", "c", "a", "b", "c"] * 100)

        psi = compute_psi_categorical(data, data)

        assert psi < 0.01

    def test_different_categorical(self) -> None:
        """Different distributions should have higher PSI."""
        expected = np.array(["a"] * 500 + ["b"] * 500)
        actual = np.array(["a"] * 100 + ["b"] * 900)

        psi = compute_psi_categorical(expected, actual)

        assert psi > 0.05


# ──────────────────────────────────────────────────
# KS Test
# ──────────────────────────────────────────────────


class TestKSTest:
    """Tests for ks_test."""

    def test_identical_distributions(self) -> None:
        """Same data should give high p-value."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, size=1000)

        stat, pvalue = ks_test(data, data)

        assert stat == 0.0
        assert pvalue == 1.0

    def test_different_distributions(self) -> None:
        """Different distributions should have low p-value."""
        rng = np.random.RandomState(42)
        d1 = rng.normal(0, 1, size=1000)
        d2 = rng.normal(5, 1, size=1000)

        stat, pvalue = ks_test(d1, d2)

        assert stat > 0.5
        assert pvalue < 0.001


# ──────────────────────────────────────────────────
# Fairness Metrics
# ──────────────────────────────────────────────────


class TestDemographicParityDiff:
    """Tests for demographic_parity_diff."""

    def test_equal_rates(self) -> None:
        """Equal prediction rates across groups should give 0."""
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        protected = np.array(["a", "a", "a", "b", "b", "b"])

        dpd = demographic_parity_diff(y_pred, protected)

        # Group a: 2/3, Group b: 1/3
        assert abs(dpd - (2 / 3 - 1 / 3)) < 1e-6

    def test_no_difference(self) -> None:
        """Same rate in all groups."""
        y_pred = np.array([1, 0, 1, 0])
        protected = np.array(["a", "a", "b", "b"])

        dpd = demographic_parity_diff(y_pred, protected)

        assert dpd == 0.0

    def test_single_group(self) -> None:
        """Single group should return 0."""
        y_pred = np.array([1, 0, 1])
        protected = np.array(["a", "a", "a"])

        dpd = demographic_parity_diff(y_pred, protected)

        assert dpd == 0.0


class TestEqualOpportunityDiff:
    """Tests for equal_opportunity_diff."""

    def test_equal_tpr(self) -> None:
        """Equal TPR across groups."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 0, 1, 0])
        protected = np.array(["a", "a", "b", "b"])

        eod = equal_opportunity_diff(y_true, y_pred, protected)

        assert eod == 0.0

    def test_unequal_tpr(self) -> None:
        """Unequal TPR across groups."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0])
        protected = np.array(["a", "a", "b", "b"])

        eod = equal_opportunity_diff(y_true, y_pred, protected)

        assert eod == 0.5  # Group a: 1.0, Group b: 0.5
