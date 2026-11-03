"""Schema-driven trust audit for numeric, non-v1 model bundles.

The Kaggle credit-card fields V1..V28 are anonymized PCA components. This module
reports those names as statistical drivers and never assigns them business meaning.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from engine.torch_model import TorchModelBundle

RUNS = 5


def _top3(values: dict[str, float]) -> list[str]:
    return sorted(values, key=lambda name: abs(values[name]), reverse=True)[:3]


def evaluate_generic(
    bundle: TorchModelBundle,
    transaction: dict[str, Any],
    *,
    enable_counterfactual: bool = False,
) -> dict:
    started = time.perf_counter()
    features = bundle.metadata["features"]
    row = {name: float(transaction[name]) for name in features}
    txn_id = str(transaction.get("txn_id") or "")
    score = bundle.predict_score(row)
    shap_values = bundle.shap_values(row)
    top = _top3(shap_values)

    # The PDF schema has no MCC or merchant categories. Only perturb the two
    # observed transaction-time variables; keep PCA-derived history fixed.
    mutable = [name for name in ("Amount", "Time") if name in features]
    perturbations: list[dict] = []
    flip_count = 0
    seed = int.from_bytes(txn_id.encode("utf-8")[:8].ljust(8, b"0"), "big")
    rng = np.random.default_rng(seed)
    for index in range(RUNS):
        candidate = dict(row)
        if mutable:
            feature = mutable[index % len(mutable)]
            original = row[feature]
            if feature == "Amount":
                factor = float(rng.choice([.95, 1.05]))
                candidate[feature] = max(0.0, original * factor)
            else:  # Time is seconds elapsed in this dataset.
                shift = int(rng.choice([-300, 300]))
                candidate[feature] = max(0.0, original + shift)
        perturbed_top = _top3(bundle.shap_values(candidate))
        perturbations.append({"run": index + 1, "top3": perturbed_top})
        if set(perturbed_top) != set(top):
            flip_count += 1

    verdict = "STABLE" if flip_count <= 1 else "UNTRUSTWORTHY"
    stability = max(0.0, 100.0 * (1.0 - flip_count / RUNS))
    medians = bundle.metadata["medians"]
    shifts: dict[str, float] = {}
    for feature in features:
        ablated = dict(row)
        ablated[feature] = float(medians[feature])
        shifts[feature] = abs(score - bundle.predict_score(ablated))

    reported = top[:2]
    top_drop = max((shifts[name] for name in reported), default=0.0)
    hidden_names = [name for name in features if name not in top]
    hidden_max = max((shifts[name] for name in hidden_names), default=0.0)
    faithfulness = 100.0 * top_drop / (top_drop + hidden_max + 1e-9)
    if top_drop < .025 and hidden_max > .08:
        faithfulness *= .35

    profile = np.asarray([shap_values[name] for name in features], dtype=float)
    reference = np.asarray(bundle.metadata["reference_shap_centroid"], dtype=float)
    denominator = float(np.linalg.norm(profile) * np.linalg.norm(reference))
    cosine = float(np.dot(profile, reference) / denominator) if denominator > 1e-12 else 0.0
    consistency = float(np.clip(50.0 * (cosine + 1.0), 0.0, 100.0))
    trust = float(np.clip(.4 * faithfulness + .3 * stability + .3 * consistency, 0, 100))
    hidden = [name for name in hidden_names if shifts[name] > max(top_drop * .5, .02)]
    hidden.sort(key=shifts.get, reverse=True)
    hidden = hidden[:5]
    suspicious_discrepancy = bool(hidden and hidden_max > top_drop * 1.5)
    if verdict == "UNTRUSTWORTHY" or trust < 45 or suspicious_discrepancy:
        classification, manipulation = "Manipulated", "HIGH"
    elif trust >= 75 and verdict == "STABLE":
        classification, manipulation = "Trusted", "LOW"
    elif trust >= 45:
        classification, manipulation = "Suspicious", "MEDIUM"
    else:
        classification, manipulation = "Manipulated", "HIGH"

    narrative = None
    if verdict == "STABLE" and classification != "Manipulated":
        narrative = (
            f"Flagged because {top[0]} and {top[1]} dominated the model explanation. "
            f"The explanation remained stable across {RUNS} bounded amount/time perturbations. "
            "V1-V28 are anonymized PCA components and have no direct business interpretation."
        )
    forensic = (
        f"Explanation audit: {verdict}; top-3 set changed in {flip_count}/{RUNS} bounded "
        f"perturbations. Top reported ablation shift={top_drop:.4f}; strongest unreported "
        f"shift={hidden_max:.4f}. Anonymized PCA features are statistical indicators only."
    )

    cf_text = None
    if enable_counterfactual:
        cf_text = _generic_counterfactual(bundle, row, score)

    return {
        "txn_id": txn_id,
        "fraud_score": round(score, 6),
        "risk_level": "HIGH" if score >= .70 else "MEDIUM" if score >= .40 else "LOW",
        "original_top3": top,
        "perturbed_runs": perturbations,
        "flip_count": flip_count,
        "verdict": verdict,
        "trust_score": round(trust, 2),
        "trust_classification": classification,
        "manipulation_risk": manipulation,
        "vetto_metrics": {
            "faithfulness_score": round(float(np.clip(faithfulness, 0, 100)), 2),
            "stability_score": round(stability, 2),
            "consistency_score": round(consistency, 2),
        },
        "true_hidden_drivers": hidden,
        "narrative": narrative,
        "counterfactual_explanation": cf_text,
        "forensic_report": forensic,
        "audit_hash": "",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _generic_counterfactual(bundle: TorchModelBundle, row: dict, score: float) -> str:
    if score < .5:
        return "This transaction is already below the 0.5 fraud threshold."
    candidates: list[tuple[str, float]] = []
    if "Amount" in row:
        amount = row["Amount"]
        candidates.extend(("Amount", float(value)) for value in np.linspace(
            max(0.0, amount * .95), amount * 1.05, 21
        ))
    if "Time" in row:
        candidates.extend(("Time", max(0.0, row["Time"] + shift)) for shift in (-300, -150, 150, 300))
    for feature, value in candidates[:200]:
        candidate = dict(row)
        candidate[feature] = value
        if bundle.predict_score(candidate) < .5:
            return (
                f"A bounded what-if change to {feature}={value:.2f} moves the score below "
                "0.5; this statistical counterfactual is not a recommended payment action."
            )
    return f"No single-variable counterfactual found within {min(200, len(candidates))} evaluations."
