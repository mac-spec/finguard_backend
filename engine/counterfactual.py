"""Bounded single-feature counterfactual search (maximum 200 model evaluations)."""
from __future__ import annotations
import numpy as np
from engine.simulator_and_model import predict_score
from engine.perturb import MCC_NEIGHBORS, MERCHANT_CATEGORY, MERCHANT_GROUPS

MAX_EVALUATIONS = 200

def counterfactual(txn_row) -> str:
    baseline_score = predict_score(txn_row)
    if baseline_score < .5:
        return "This payment is already below the 0.5 fraud threshold."
    amount = float(txn_row.get("amount", 1.0))
    hour = int(txn_row.get("timestamp_hour", 0))
    merchant = int(txn_row.get("merchant_id", -1))
    merchant_group = MERCHANT_CATEGORY.get(merchant)
    candidates = {
        "amount": np.linspace(max(.01, amount * .95), amount * 1.05, 21),
        "timestamp_hour": sorted({(hour + shift) % 24 for shift in (-2, -1, 1, 2)}),
        "mcc_code": MCC_NEIGHBORS.get(int(txn_row.get("mcc_code", -1)), []),
        "merchant_id": [
            value for value in MERCHANT_GROUPS.get(merchant_group, []) if value != merchant
        ],
    }
    evaluated = 0
    best = None
    best_cost = float("inf")
    for feature, values in candidates.items():
        if feature not in txn_row.index:
            continue
        original = float(txn_row[feature])
        for value in values:
            if evaluated >= MAX_EVALUATIONS:
                break
            value = float(value)
            if value == original:
                continue
            candidate = txn_row.copy(deep=True)
            candidate[feature] = value
            score = predict_score(candidate)
            evaluated += 1
            if score < .5:
                if feature == "amount":
                    cost = abs(value - original) / max(original * .05, .01)
                elif feature == "timestamp_hour":
                    hours = min((int(value) - int(original)) % 24,
                                (int(original) - int(value)) % 24)
                    cost = hours / 2.0
                else:
                    cost = 1.0
                if cost < best_cost:
                    best, best_cost = (feature, value), cost
        if evaluated >= MAX_EVALUATIONS:
            break
    if best is None:
        return f"No single-feature counterfactual found within {evaluated} evaluations."
    feature, value = best
    shown = int(value) if feature in ("timestamp_hour", "mcc_code", "merchant_id") else round(value, 2)
    return f"This payment would likely be legitimate if {feature} were {shown}."
