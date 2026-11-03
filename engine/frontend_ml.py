"""Map browser transaction records to a demo ML model and audit its explanation."""
from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np

from engine.frontend_schema import (
    CATEGORY_MCC_CODES,
    METHOD_CODES,
    SIMULATED_CONTEXT_FEATURES,
)
from engine.torch_model import TorchModelBundle

PERTURBATION_RUNS = 5


def model_row(transaction: dict) -> dict:
    date_text = transaction["date"]
    hour = 12
    timestamp_known = 0
    if "T" in date_text or " " in date_text:
        try:
            hour = datetime.fromisoformat(date_text.replace("Z", "+00:00")).hour
            timestamp_known = 1
        except ValueError:
            pass
    category = transaction["category"].strip().lower()
    method = transaction["method"].strip().lower()
    identity = "|".join((transaction["id"], transaction["receiver"], transaction["description"]))
    seed = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    account_age = int(rng.integers(1, 3001))
    new_device = int(rng.random() < .12)
    card_age = int(rng.integers(1, min(account_age, 2500) + 1))
    max_device_age = max(1, min(account_age, 1500))
    device_age = int(rng.integers(0, min(2, max_device_age) + 1)) if new_device else int(
        rng.integers(0, max_device_age + 1)
    )
    row = {
        "amount": float(transaction["amount"]),
        "timestamp_hour": hour,
        "mcc_code": CATEGORY_MCC_CODES.get(category, 5999),
        "merchant_id": 100 + (seed % 40),
        "account_age_days": account_age,
        "device_fingerprint_age": device_age,
        "historical_txn_count_30d": int(rng.poisson(12)),
        "card_age_days": card_age,
        "is_new_device": new_device,
        "foreign_location_mismatch": int(rng.random() < .05),
        "velocity_1h": float(rng.poisson(1.2)),
        "risky_beneficiary_score": float(rng.beta(2, 5)),
        "is_recurring_subscription_window": int(rng.random() < .25),
        "merchant_category_loyalty": float(rng.beta(4, 2)),
        "timestamp_known": timestamp_known,
        "direction_code": 1 if transaction["direction"] == "credit" else 0,
        "method_code": METHOD_CODES.get(method, 7),
    }
    return row


def _top3(shap_values: dict[str, float]) -> list[str]:
    return sorted(shap_values, key=lambda key: abs(shap_values[key]), reverse=True)[:3]


def audit_frontend_row(bundle: TorchModelBundle, row: dict, score: float, anomaly: float) -> dict:
    features = bundle.metadata["features"]
    shap_values = bundle.shap_values(row)
    original_top3 = _top3(shap_values)
    perturbations = []
    flips = 0
    for run in range(PERTURBATION_RUNS):
        perturbed = dict(row)
        if row["timestamp_known"] and run % 2:
            perturbed["timestamp_hour"] = min(
                23, max(0, row["timestamp_hour"] + (1 if run % 4 == 1 else -1))
            )
        else:
            multiplier = 1.05 if run % 2 == 0 else .95
            perturbed["amount"] = max(.01, row["amount"] * multiplier)
        top = _top3(bundle.shap_values(perturbed))
        perturbations.append({"run": run + 1, "top3": top})
        flips += set(top) != set(original_top3)

    shifts = {}
    for feature in features:
        ablated = dict(row)
        ablated[feature] = float(bundle.metadata["medians"][feature])
        shifts[feature] = abs(score - bundle.predict_score(ablated))
    top_drop = max((shifts[key] for key in original_top3[:2]), default=0.0)
    hidden_names = [key for key in features if key not in original_top3]
    hidden_max = max((shifts[key] for key in hidden_names), default=0.0)
    faithfulness = 100 * top_drop / (top_drop + hidden_max + 1e-9)
    if top_drop < .025 and hidden_max > .08:
        faithfulness *= .35

    reference = np.asarray(bundle.metadata["reference_shap_centroid"], dtype=float)
    profile = np.asarray([shap_values[name] for name in features], dtype=float)
    denominator = float(np.linalg.norm(profile) * np.linalg.norm(reference))
    cosine = float(np.dot(profile, reference) / denominator) if denominator > 1e-12 else 0.0
    consistency = float(np.clip(50 * (cosine + 1), 0, 100))
    stability = 100 * (1 - flips / PERTURBATION_RUNS)
    trust = float(np.clip(.4 * faithfulness + .3 * stability + .3 * consistency, 0, 100))
    verdict = "STABLE" if flips <= 1 else "UNTRUSTWORTHY"
    hidden = sorted(
        (name for name in hidden_names if shifts[name] > max(top_drop * .5, .02)),
        key=shifts.get, reverse=True,
    )[:3]
    trust_class = "Trusted" if trust >= 75 and verdict == "STABLE" else (
        "Manipulated" if verdict == "UNTRUSTWORTHY" or trust < 45 else "Suspicious"
    )
    narrative = None if verdict == "UNTRUSTWORTHY" or trust_class == "Manipulated" else (
        f"Model signals were led by {original_top3[0]} and {original_top3[1]}; "
        "the explanation remained stable under bounded amount/time changes."
    )
    return {
        "fraud_score": round(float(score), 6),
        "fraud_risk_level": "HIGH" if score >= .70 else "MEDIUM" if score >= .40 else "LOW",
        "anomaly_score": round(float(anomaly), 6),
        "top3_features": original_top3,
        "top3_values": [round(shap_values[key], 6) for key in original_top3],
        "explanation_verdict": verdict,
        "trust_score": round(trust, 2),
        "trust_classification": trust_class,
        "stability_score": round(float(stability), 2),
        "faithfulness_score": round(float(np.clip(faithfulness, 0, 100)), 2),
        "consistency_score": round(consistency, 2),
        "flip_count": int(flips),
        "perturbed_runs": perturbations,
        "true_hidden_drivers": hidden,
        "narrative": narrative,
        "model_provenance": bundle.metadata["training"].get("data_classification", "unknown"),
        "synthetic_context_features": SIMULATED_CONTEXT_FEATURES,
    }
