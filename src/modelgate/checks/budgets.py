"""Performance budget check.

Measures p50/p95 single-row latency, serialized model size, and peak memory.
Compares against configured budgets.
"""

from __future__ import annotations

import gc
import io
import pickle
import time
import tracemalloc
from typing import Any, Literal

import numpy as np
import pandas as pd

from modelgate.config import BudgetConfig
from modelgate.verdict import CheckResult


def _measure_latency(
    model: Any,
    X_sample: pd.DataFrame,
    n_calls: int = 100,
    n_warmup: int = 10,
) -> tuple[float, float]:
    """Measure single-row prediction latency.

    Args:
        model: The model to benchmark.
        X_sample: Sample data (uses first row).
        n_calls: Number of timed prediction calls.
        n_warmup: Number of warmup calls before timing.

    Returns:
        Tuple of (p50_ms, p95_ms) latency in milliseconds.
    """
    single_row = X_sample.iloc[[0]]

    # Warmup
    for _ in range(n_warmup):
        model.predict(single_row)

    gc.collect()

    # Timed calls
    latencies = []
    for _ in range(n_calls):
        start = time.perf_counter()
        model.predict(single_row)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    arr = np.array(latencies)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    return p50, p95


def _measure_paired_latency(
    candidate_model: Any,
    champion_model: Any,
    X_sample: pd.DataFrame,
    n_calls: int = 100,
    n_warmup: int = 10,
) -> tuple[float, float, float, float]:
    """Measure candidate and champion single-row latency interleaved in the same job.

    Alternates call order (candidate-first vs champion-first) on every iteration so both
    models experience identical CPU scheduling and cache conditions.

    Returns:
        Tuple of (cand_p50_ms, cand_p95_ms, champ_p50_ms, champ_p95_ms).
    """
    single_row = X_sample.iloc[[0]]

    for _ in range(n_warmup):
        champion_model.predict(single_row)
        candidate_model.predict(single_row)

    gc.collect()

    cand_latencies: list[float] = []
    champ_latencies: list[float] = []
    for i in range(n_calls):
        if i % 2 == 0:
            t0 = time.perf_counter()
            candidate_model.predict(single_row)
            cand_latencies.append((time.perf_counter() - t0) * 1000)

            t1 = time.perf_counter()
            champion_model.predict(single_row)
            champ_latencies.append((time.perf_counter() - t1) * 1000)
        else:
            t1 = time.perf_counter()
            champion_model.predict(single_row)
            champ_latencies.append((time.perf_counter() - t1) * 1000)

            t0 = time.perf_counter()
            candidate_model.predict(single_row)
            cand_latencies.append((time.perf_counter() - t0) * 1000)

    cand_arr = np.array(cand_latencies)
    champ_arr = np.array(champ_latencies)
    return (
        float(np.percentile(cand_arr, 50)),
        float(np.percentile(cand_arr, 95)),
        float(np.percentile(champ_arr, 50)),
        float(np.percentile(champ_arr, 95)),
    )


def _measure_model_size(model: Any) -> float:
    """Measure the serialized model size in MB.

    Args:
        model: The model to measure.

    Returns:
        Size in megabytes.
    """
    try:
        buf = io.BytesIO()
        pickle.dump(model, buf)
        size_bytes = buf.tell()
        return size_bytes / (1024 * 1024)
    except (pickle.PicklingError, TypeError, AttributeError):
        return 0.0


def _measure_peak_memory(model: Any, X_sample: pd.DataFrame) -> float:
    """Measure peak memory usage during prediction in MB.

    Args:
        model: The model to benchmark.
        X_sample: Sample data.

    Returns:
        Peak memory in megabytes.
    """
    tracemalloc.start()

    # Run some predictions
    for _ in range(10):
        model.predict(X_sample.iloc[[0]])

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak / (1024 * 1024)


def run_budgets_check(
    test_df: pd.DataFrame,
    target_column: str,
    candidate_model: Any,
    champion_model: Any | None,
    config: BudgetConfig,
) -> CheckResult:
    """Run performance budget checks.

    Args:
        test_df: Test DataFrame.
        target_column: Name of the target column.
        candidate_model: The candidate model.
        champion_model: The champion model (optional, for delta reporting).
        config: Budget check configuration.

    Returns:
        CheckResult with latency, size, and memory metrics.
    """
    start = time.time()

    if not config.enabled:
        return CheckResult(
            name="budgets",
            status="skip",
            summary="Budget check disabled",
            duration_s=time.time() - start,
        )

    X_test = test_df.drop(columns=[target_column])

    # Measure latency: interleaved champion + candidate in the same job when champion exists
    champion_metrics: dict[str, float] = {}
    latency_ratio: float | None = None
    champ_p95: float | None = None
    if champion_model is not None:
        p50, p95, champ_p50, champ_p95 = _measure_paired_latency(
            candidate_model, champion_model, X_test
        )
        latency_ratio = round(p95 / max(champ_p95, 1e-6), 3)
        p50_ratio = round(p50 / max(champ_p50, 1e-6), 3)
        champ_size = _measure_model_size(champion_model)
        champion_metrics = {
            "champion_p50_ms": round(champ_p50, 2),
            "champion_p95_ms": round(champ_p95, 2),
            "latency_ratio": latency_ratio,
            "p50_latency_ratio": p50_ratio,
            "champion_model_mb": round(champ_size, 2),
        }
    else:
        p50, p95 = _measure_latency(candidate_model, X_test)

    model_size_mb = _measure_model_size(candidate_model)
    peak_mem_mb = _measure_peak_memory(candidate_model, X_test)

    # Check against budgets
    violations: list[str] = []

    if (
        latency_ratio is not None
        and champ_p95 is not None
        and config.max_latency_ratio is not None
        and latency_ratio > config.max_latency_ratio
    ):
        violations.append(
            f"p95 latency ratio {latency_ratio:.2f}x ({p95:.2f}ms vs champion {champ_p95:.2f}ms) "
            f"> budget {config.max_latency_ratio:.2f}x"
        )

    if config.p95_latency_ms is not None and p95 > config.p95_latency_ms:
        violations.append(f"p95 {p95:.1f}ms > budget {config.p95_latency_ms}ms")

    if model_size_mb > config.max_model_mb:
        violations.append(f"model size {model_size_mb:.1f}MB > budget {config.max_model_mb}MB")

    if peak_mem_mb > config.max_peak_mem_mb:
        violations.append(f"peak memory {peak_mem_mb:.1f}MB > budget {config.max_peak_mem_mb}MB")

    duration = time.time() - start

    metrics = {
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "model_size_mb": round(model_size_mb, 2),
        "peak_mem_mb": round(peak_mem_mb, 2),
        **champion_metrics,
    }

    status: Literal["pass", "warn", "fail", "skip"]
    if violations:
        status = "fail"
        summary = "; ".join(violations)
    else:
        status = "pass"
        if latency_ratio is not None and champ_p95 is not None:
            summary = (
                f"p95 {p95:.2f}ms ({latency_ratio:.2f}x vs champion {champ_p95:.2f}ms), "
                f"size {model_size_mb:.1f}MB, mem {peak_mem_mb:.1f}MB"
            )
        else:
            summary = f"p95 {p95:.1f}ms, size {model_size_mb:.1f}MB, mem {peak_mem_mb:.1f}MB"

    return CheckResult(
        name="budgets",
        status=status,
        summary=summary,
        metrics=metrics,
        details={"violations": violations},
        duration_s=duration,
    )
