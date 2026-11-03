"""One request-level orchestrator for model, graph, behaviour, and trust results."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from engine.behaviour_intelligence import analyze_behaviour
from engine.graph_intelligence import analyze_graph
from engine.simulator_and_model import _torch_bundle, anomaly_score, explain, predict_score
from engine.verdict import evaluate


def _milliseconds(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def assess_transaction(
    transaction: dict[str, Any],
    current_context: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    enable_counterfactual: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all currently available FinGuard signals without collapsing them.

    Returns a frontend-safe response and a richer, pseudonymous payload for the
    SQLite hash-chain ledger. Fraud probability, anomaly, graph evidence,
    behaviour and explanation trust remain separate fields.
    """
    started = time.perf_counter()
    txn_id = str(transaction.get("txn_id") or "")
    model_row = {**transaction, "txn_id": txn_id}
    current = {**current_context, "txn_id": txn_id}
    occurred_at = current["occurred_at"]
    if isinstance(occurred_at, str):
        occurred_at = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
    if occurred_at.tzinfo is None:
        raise ValueError("occurred_at must include a timezone offset")
    occurred_at = occurred_at.astimezone(timezone.utc)
    current["occurred_at"] = occurred_at

    earlier_history = []
    for row in history:
        row_time = row["occurred_at"]
        if isinstance(row_time, str):
            row_time = datetime.fromisoformat(row_time.replace("Z", "+00:00"))
        if row_time.tzinfo is not None and row_time.astimezone(timezone.utc) < occurred_at:
            earlier_history.append({**row, "occurred_at": row_time})

    timings: dict[str, float] = {}
    mark = time.perf_counter()
    probability = float(predict_score(model_row))
    timings["prediction"] = _milliseconds(mark)
    mark = time.perf_counter()
    anomaly = float(anomaly_score(model_row))
    timings["anomaly"] = _milliseconds(mark)
    mark = time.perf_counter()
    explanation = explain(
        model_row, score_override=probability, anomaly_override=anomaly,
    )
    timings["explanation"] = _milliseconds(mark)
    mark = time.perf_counter()
    trust = evaluate(
        pd.Series(model_row), enable_cf=enable_counterfactual,
        original_explanation=explanation,
    )
    timings["perturbation_and_trust"] = _milliseconds(mark)

    graph_context = [*earlier_history, current]
    mark = time.perf_counter()
    graph = analyze_graph(graph_context, target_transaction_id=txn_id)
    timings["graph"] = _milliseconds(mark)
    mark = time.perf_counter()
    behaviour = analyze_behaviour(current, earlier_history)
    timings["behaviour"] = _milliseconds(mark)

    model_provenance = (
        _torch_bundle.metadata.get("training", {}).get("data_classification", "unknown")
        if _torch_bundle is not None else "synthetic"
    )
    if model_provenance == "synthetic":
        disclaimer = "Synthetic demonstration model; scores are not validated fraud probabilities and do not block payments."
    elif model_provenance == "approved":
        disclaimer = "Approved model artifact; scores support review and do not by themselves prove fraud or block payments."
    else:
        disclaimer = "Training data provenance is unverified; scores are not validated fraud probabilities."

    trust_safe = {
        "verdict": trust["verdict"],
        "trust_score": trust["trust_score"],
        "trust_classification": trust["trust_classification"],
        "manipulation_risk": trust["manipulation_risk"],
        "vetto_metrics": trust["vetto_metrics"],
        "flip_count": trust["flip_count"],
        "perturbed_runs": trust["perturbed_runs"],
    }
    explanation_safe = {
        "top_features": explanation["top3_features"],
        "top_feature_values": [round(float(value), 6) for value in explanation["top3_values"]],
        "narrative": trust["narrative"],
        "withheld": trust["narrative"] is None,
    }
    graph_evidence = [item["explanation"] for item in graph["target_indicators"]]
    behaviour_evidence = [item["explanation"] for item in behaviour["indicators"]]
    forensic_sections = [trust["forensic_report"]]
    forensic_sections.append(
        "Graph review indicators: " + ("; ".join(graph_evidence) if graph_evidence else "none in supplied prior graph data.")
    )
    forensic_sections.append(
        "Behaviour review indicators: " + ("; ".join(behaviour_evidence) if behaviour_evidence else "none in sufficient prior history.")
    )
    forensic_sections.append(
        "These separate signals support review; none is a confirmed fraud determination."
    )
    timings["total"] = _milliseconds(started)
    risk_level = "HIGH" if probability >= 0.70 else "MEDIUM" if probability >= 0.40 else "LOW"
    result = {
        "txn_id": txn_id,
        "model_name": (
            "PyTorch DNN + legitimate-only autoencoder"
            if _torch_bundle is not None else "XGBoost + Isolation Forest development fallback"
        ),
        "model_provenance": (
            model_provenance
        ),
        "disclaimer": disclaimer,
        "fraud_probability": round(probability, 6),
        "fraud_risk_level": risk_level,
        "anomaly_score": round(anomaly, 6),
        "anomaly_detected": anomaly < 0,
        "graph_intelligence": graph,
        "behavioural_intelligence": behaviour,
        "explanation": explanation_safe,
        "trust_verification": trust_safe,
        "forensic_report": "\n".join(forensic_sections),
        "audit_hash": "",
        "component_latency_ms": timings,
        "latency_ms": timings["total"],
    }
    audit_payload = {
        **trust,
        "fraud_probability": round(probability, 6),
        "model_name": result["model_name"],
        "model_provenance": result["model_provenance"],
        "anomaly_score": round(anomaly, 6),
        "graph_intelligence": graph,
        "behavioural_intelligence": behaviour,
        "forensic_report": result["forensic_report"],
        "component_latency_ms": timings,
    }
    return result, audit_payload
