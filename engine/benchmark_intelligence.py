"""Reproducible single-process latency and memory microbenchmark.

This script measures synthetic demo workloads; it is not a production load test.
Run the same command on the same machine before comparing code revisions.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tempfile
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from engine.assessment import assess_transaction
from engine.behaviour_intelligence import analyze_behaviour
from engine.fraud_twin import run_fraud_twin
from engine.graph_intelligence import analyze_graph
from engine.ledger import AuditLedger
from engine.simulator_and_model import (
    BASELINE, _torch_bundle, anomaly_score, explain, generate_normal_fraud_txn, predict_score,
)
from engine.verdict import evaluate


def _memory_mb() -> float | None:
    """Return process RSS on Windows/Linux without adding a monitoring dependency."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            ok = psapi.GetProcessMemoryInfo(
                kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb,
            )
            return round(counters.WorkingSetSize / (1024 * 1024), 2) if ok else None
        if platform.system() == "Linux":
            with open("/proc/self/statm", encoding="ascii") as stream:
                resident_pages = int(stream.read().split()[1])
            return round(resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 2)
    except (OSError, AttributeError, ValueError):
        return None
    return None


def _measure(function, iterations: int) -> dict:
    function()  # one warm-up, excluded from timings
    elapsed_ms = []
    started_all = time.perf_counter()
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        elapsed_ms.append((time.perf_counter() - started) * 1000)
    total_seconds = time.perf_counter() - started_all
    ordered = sorted(elapsed_ms)
    p95_index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return {
        "iterations": iterations,
        "median_ms": round(statistics.median(elapsed_ms), 3),
        "mean_ms": round(statistics.mean(elapsed_ms), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "sequential_throughput_per_second": round(iterations / total_seconds, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--graph-rows", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1 or not 1 <= args.graph_rows <= 5000:
        parser.error("iterations must be positive and graph-rows must be 1..5000")

    transaction = generate_normal_fraud_txn()
    now = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
    history = []
    for index in range(60):
        history.append({
            "txn_id": f"bench-{index}", "account_id": "bench-account",
            "beneficiary_account_id": f"bench-recipient-{index % 8}",
            "amount": 80.0 + (index % 30),
            "occurred_at": now - timedelta(hours=(60 - index) * 8),
        })
    current = {
        "txn_id": "bench-current", "account_id": "bench-account",
        "beneficiary_account_id": "bench-recipient-1", "amount": 100.0,
        "occurred_at": now,
    }
    transaction["timestamp_hour"] = now.hour
    transaction["txn_id"] = "bench-current"
    current["amount"] = float(transaction["amount"])
    context = {**current}
    explanation_score = predict_score(transaction)

    graph_rows = []
    for index in range(args.graph_rows):
        graph_rows.append({
            "txn_id": f"graph-{index}",
            "account_id": f"node-{index % 40}",
            "beneficiary_account_id": f"node-{(index * 7 + 3) % 40}",
            "amount": float(25 + index % 700),
            "occurred_at": now - timedelta(minutes=args.graph_rows - index),
        })

    memory_before = _memory_mb()
    iterations = args.iterations
    expensive_iterations = max(1, min(iterations, 3))
    with tempfile.TemporaryDirectory(prefix="finguard-benchmark-") as folder:
        ledger = AuditLedger(os.path.join(folder, "benchmark.db"))

        def end_to_end_with_ledger():
            result, payload = assess_transaction(transaction.to_dict(), context, history)
            ledger.append(payload)
            return result

        original_explain = explain

        def legacy_explain(*args, **kwargs):
            # Recreate the prior audit path, which recalculated AE scores for
            # explanation-only perturbations instead of reusing safe outputs.
            kwargs["include_anomaly"] = True
            return original_explain(*args, **kwargs)

        def legacy_ablation(row, features, baseline=None):
            old_baseline = predict_score(row)
            shifts = {}
            for feature in features:
                masked = row.copy(deep=True)
                masked[feature] = BASELINE[feature]
                shifts[feature] = abs(old_baseline - predict_score(masked))
            return shifts

        def legacy_trust_audit():
            with patch("engine.verdict.explain", side_effect=legacy_explain), \
                    patch("engine.verdict._ablation", side_effect=legacy_ablation):
                return evaluate(transaction)

        benchmarks = {
            "prediction": _measure(lambda: predict_score(transaction), iterations),
            "anomaly_detection": _measure(lambda: anomaly_score(transaction), iterations),
            "explanation": _measure(
                lambda: explain(
                    transaction, score_override=explanation_score, include_anomaly=False,
                ),
                iterations,
            ),
            "trust_audit_before_safe_reuse_optimisation": _measure(
                legacy_trust_audit, expensive_iterations,
            ),
            "perturbation_and_trust_audit": _measure(lambda: evaluate(transaction), expensive_iterations),
            "fraud_twin_four_scenarios": _measure(lambda: run_fraud_twin(transaction, seed=42), expensive_iterations),
            "graph_analysis": _measure(lambda: analyze_graph(graph_rows), iterations),
            "behaviour_analysis": _measure(lambda: analyze_behaviour(current, history), iterations),
            "full_assessment_without_http_or_ledger": _measure(
                lambda: assess_transaction(transaction.to_dict(), context, history), expensive_iterations,
            ),
            "end_to_end_assessment_with_ledger": _measure(end_to_end_with_ledger, expensive_iterations),
        }
    memory_after = _memory_mb()
    print(json.dumps({
        "benchmark": "FinGuard synthetic single-process microbenchmark",
        "warning": "Sequential microbenchmark, not concurrent capacity or production evidence.",
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
            "model_backend": "pytorch" if _torch_bundle is not None else "xgboost + isolation_forest fallback",
            "torch_bundle_data_classification": (
                _torch_bundle.metadata.get("training", {}).get("data_classification")
                if _torch_bundle is not None else "synthetic in-memory demo"
            ),
            "fixed_seed": 42, "graph_rows": args.graph_rows,
        },
        "measurements": benchmarks,
        "process_rss_mb": {"before": memory_before, "after": memory_after},
    }, indent=2))


if __name__ == "__main__":
    main()
