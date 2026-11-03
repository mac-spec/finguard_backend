"""Explanation robustness, ablation faithfulness, consistency, and verdict scoring."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from engine.simulator_and_model import (
    explain, predict_score, BASELINE, FEATURES, HIGH_RISK, MANIPULABLE,
    REFERENCE_SHAP_CENTROID,
)
from engine.perturb import generate_perturbations
from engine.counterfactual import counterfactual

ENABLE_COUNTERFACTUAL = False
N_PERTURBATIONS = 5

def _ablation(txn_row, features, baseline=None):
    baseline = predict_score(txn_row) if baseline is None else float(baseline)
    shifts = {}
    for feature in features:
        masked = txn_row.copy(deep=True)
        masked[feature] = BASELINE[feature]
        shifts[feature] = abs(baseline - predict_score(masked))
    return shifts

def evaluate(
    txn_row: pd.Series,
    enable_cf: bool = ENABLE_COUNTERFACTUAL,
    simulate_proxy_masking: bool = False,
    *,
    original_explanation: dict | None = None,
) -> dict:
    started = time.perf_counter()
    original = original_explanation or explain(
        txn_row, simulate_proxy_masking=simulate_proxy_masking,
    )
    original_top3 = original["top3_features"]
    base_set = set(original_top3)
    perturbations = generate_perturbations(txn_row, n=N_PERTURBATIONS, seed=42)
    perturbed_runs = []
    flip_count = 0
    for i, row in enumerate(perturbations, 1):
        top3 = explain(
            row, simulate_proxy_masking=simulate_proxy_masking, include_anomaly=False,
        )["top3_features"]
        perturbed_runs.append({"run": i, "top3": top3})
        if set(top3) != base_set:
            flip_count += 1
    verdict = "STABLE" if flip_count <= 1 else "UNTRUSTWORTHY"
    stability_score = max(0.0, 100.0 * (1.0 - flip_count / N_PERTURBATIONS))

    baseline_score = float(original["fraud_score"])
    top_shifts = _ablation(txn_row, original_top3[:2], baseline=baseline_score)
    hidden_candidates = [f for f in HIGH_RISK if f not in original_top3]
    hidden_shifts = _ablation(txn_row, hidden_candidates, baseline=baseline_score)
    top_drop = max(top_shifts.values(), default=0.0)
    hidden_max = max(hidden_shifts.values(), default=0.0)
    total_sensitivity = top_drop + hidden_max + 1e-9
    faithfulness_score = float(np.clip(100.0 * top_drop / total_sensitivity, 0, 100))
    # Penalize reported features that barely affect the model while hidden risk features do.
    if top_drop < .025 and hidden_max > .08:
        faithfulness_score *= .35
    hidden_sorted = sorted(hidden_shifts, key=hidden_shifts.get, reverse=True)
    # A hidden driver need not exceed the strongest reported feature's ablation:
    # report the most sensitive unreported risk features when their score impact
    # is material in absolute terms.
    true_hidden_drivers = [f for f in hidden_sorted if hidden_shifts[f] >= .05][:2]

    # Compare attribution vectors to fraud-reference attributions; raw feature
    # values and SHAP values use incompatible scales and should not be compared.
    shap_map = original["shap_values_full"]
    profile = np.array([shap_map[f] for f in FEATURES], dtype=float)
    reference = np.asarray(REFERENCE_SHAP_CENTROID, dtype=float)
    norm = float(np.linalg.norm(profile))
    reference_norm = float(np.linalg.norm(reference))
    if norm == 0 or reference_norm == 0:
        consistency_score = 0.0
    else:
        cosine_similarity = float(np.dot(profile, reference) / (norm * reference_norm))
        consistency_score = float(np.clip(50.0 * (1.0 + cosine_similarity), 0, 100))

    trust_score = float(np.clip(
        .40 * faithfulness_score + .30 * stability_score + .30 * consistency_score, 0, 100
    ))
    discrepancy = bool(true_hidden_drivers and hidden_max > max(top_drop, .02) * 1.5)
    if trust_score < 45 or discrepancy:
        trust_classification = "Manipulated"
    elif trust_score >= 75 and verdict == "STABLE":
        trust_classification = "Trusted"
    else:
        trust_classification = "Suspicious"
    if verdict == "UNTRUSTWORTHY" or trust_score < 45:
        manipulation_risk = "HIGH"
    elif trust_score > 75:
        manipulation_risk = "LOW"
    else:
        manipulation_risk = "MEDIUM"

    top_names = original_top3[:2]
    top_values = [shap_map[f] for f in top_names]
    if verdict == "STABLE" and trust_classification != "Manipulated":
        narrative = (
            f"Flagged because {top_names[0]} (SHAP {top_values[0]:+.2f}) and "
            f"{top_names[1]} (SHAP {top_values[1]:+.2f}) dominated the decision. "
            f"Explanation was stable across 5 bounded perturbations of "
            f"{sum(f in txn_row.index for f in MANIPULABLE)} manipulable features."
        )
    else:
        narrative = None

    if verdict == "UNTRUSTWORTHY" or trust_classification == "Manipulated":
        forensic_report = (
            "ALERT: Explanation Poisoning / Instability Detected. "
            f"Standard narrative withheld (flipped in {flip_count}/5 bounded perturbations). "
            f"Reported explanation emphasized [{top_names[0]}], which contributed only "
            f"{100.0 * top_drop / (top_drop + hidden_max + 1e-9):.1f}% to actual model sensitivity. "
            f"Counterfactual ablation reveals true hidden drivers: {true_hidden_drivers}. "
            f"Explanation Trust Score: {trust_score:.1f}% ({manipulation_risk} MANIPULATION RISK)."
        )
    else:
        forensic_report = (
            f"Audit completed. No material explanation instability detected. "
            f"Trust score: {trust_score:.1f}%; set flips: {flip_count}/5."
        )

    cf_text = counterfactual(txn_row) if verdict == "UNTRUSTWORTHY" and enable_cf else None
    return {
        "txn_id": original["txn_id"],
        "fraud_score": round(original["fraud_score"], 6),
        "risk_level": original["fraud_risk_level"],
        "original_top3": original_top3,
        "perturbed_runs": perturbed_runs,
        "flip_count": flip_count,
        "verdict": verdict,
        "trust_score": round(trust_score, 2),
        "trust_classification": trust_classification,
        "manipulation_risk": manipulation_risk,
        "vetto_metrics": {
            "faithfulness_score": round(faithfulness_score, 2),
            "stability_score": round(stability_score, 2),
            "consistency_score": round(consistency_score, 2),
        },
        "true_hidden_drivers": true_hidden_drivers,
        "narrative": narrative,
        "counterfactual_explanation": cf_text,
        "forensic_report": forensic_report,
        "audit_hash": "",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
