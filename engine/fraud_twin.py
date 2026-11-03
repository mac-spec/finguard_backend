"""Controlled, local-only adversarial scenarios for FinGuard's synthetic demo.

These functions create candidate transactions. They never connect to a bank or
change historical records. Model detections are measured after each scenario;
an attack is called successful only when an originally detected transaction
becomes completely undetected under the configured demo thresholds.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from datetime import datetime, timedelta, timezone

from engine.graph_intelligence import analyze_graph
from engine.schema import FEATURES, IMMUTABLE, MANIPULABLE
from engine.simulator_and_model import anomaly_score, predict_score

FRAUD_THRESHOLD = 0.50
MAX_SPLIT_PARTS = 20
MAX_SCENARIO_TRANSACTIONS = 24


def _copy_record(transaction: dict[str, Any]) -> dict[str, Any]:
    """Copy input before editing so callers' historical/source records stay intact."""
    return deepcopy(transaction)


def _feature_changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Return model feature names whose values actually changed."""
    return [name for name in FEATURES if before.get(name) != after.get(name)]


def transaction_splitting_attack(
    transaction: dict[str, Any], *, parts: int = 4,
) -> list[dict[str, Any]]:
    """Split one payment into smaller valid payments while preserving its sum."""
    if not 2 <= parts <= MAX_SPLIT_PARTS:
        raise ValueError(f"parts must be between 2 and {MAX_SPLIT_PARTS}.")
    amount = float(transaction["amount"])
    if amount < parts * 0.01:
        raise ValueError("amount is too small to split into positive cent-valued payments.")
    each = round(amount / parts, 2)
    amounts = [each] * (parts - 1)
    amounts.append(round(amount - sum(amounts), 2))
    output = []
    for index, split_amount in enumerate(amounts, start=1):
        candidate = _copy_record(transaction)
        candidate["amount"] = split_amount
        candidate["txn_id"] = f"{transaction.get('txn_id', 'twin')}-split-{index}"
        output.append(candidate)
    return output


def behaviour_mimicry_attack(
    transaction: dict[str, Any], normal_profile: dict[str, Any],
) -> dict[str, Any]:
    """Copy selected mutable values from a supplied synthetic normal profile.

    Historical fields such as account age and prior transaction counts are
    deliberately retained from the source transaction.
    """
    candidate = _copy_record(transaction)
    for feature in MANIPULABLE:
        if feature in normal_profile:
            candidate[feature] = deepcopy(normal_profile[feature])
    return candidate


def velocity_manipulation_attack(
    transactions: list[dict[str, Any]], *, spacing_hours: int = 2,
) -> list[dict[str, Any]]:
    """Spread an existing burst across distinct hours without changing history."""
    if not transactions:
        raise ValueError("at least one transaction is required.")
    if len(transactions) > MAX_SCENARIO_TRANSACTIONS:
        raise ValueError(f"at most {MAX_SCENARIO_TRANSACTIONS} transactions are supported.")
    if not 1 <= spacing_hours <= 23:
        raise ValueError("spacing_hours must be between 1 and 23.")
    if len(transactions) > 24 // spacing_hours:
        raise ValueError("spacing_hours would cause repeated hours in this scenario.")
    start_hour = int(transactions[0]["timestamp_hour"])
    output = []
    for index, transaction in enumerate(transactions):
        candidate = _copy_record(transaction)
        candidate["timestamp_hour"] = (start_hour + index * spacing_hours) % 24
        candidate["txn_id"] = f"{transaction.get('txn_id', 'twin')}-velocity-{index + 1}"
        output.append(candidate)
    return output


def coordinated_account_attack(
    transaction: dict[str, Any], *, account_count: int = 4,
) -> list[dict[str, Any]]:
    """Create synthetic transfers along a connected account ring.

    Account identifiers are scenario metadata, not model inputs in the current
    v1 schema. This function tests row-level model visibility only; graph
    detection is a later module and is not claimed here.
    """
    if not 3 <= account_count <= MAX_SCENARIO_TRANSACTIONS:
        raise ValueError(f"account_count must be between 3 and {MAX_SCENARIO_TRANSACTIONS}.")
    accounts = [f"synthetic-account-{index + 1}" for index in range(account_count)]
    output = []
    for index in range(account_count):
        candidate = _copy_record(transaction)
        candidate["txn_id"] = f"{transaction.get('txn_id', 'twin')}-coordinated-{index + 1}"
        candidate["account_id"] = accounts[index]
        candidate["beneficiary_account_id"] = accounts[(index + 1) % account_count]
        output.append(candidate)
    return output


def _assess(transaction: dict[str, Any]) -> dict[str, Any]:
    """Run both existing detectors; keep probability and anomaly separate."""
    probability = float(predict_score(transaction))
    anomaly = float(anomaly_score(transaction))
    fraud_model_detected = probability >= FRAUD_THRESHOLD
    anomaly_detected = anomaly < 0.0
    return {
        "fraud_probability": round(probability, 6),
        "fraud_threshold": FRAUD_THRESHOLD,
        "fraud_model_detected": fraud_model_detected,
        "anomaly_score": round(anomaly, 6),
        "anomaly_detected": anomaly_detected,
        "pipeline_detected": fraud_model_detected or anomaly_detected,
    }


def _run_attack(
    name: str,
    originals: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    graph_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_results = [_assess(row) for row in originals]
    candidate_results = [_assess(row) for row in candidates]
    original_graph_detected = bool(
        graph_comparison and graph_comparison["original"]["target_flagged"]
    )
    modified_graph_detected = bool(
        graph_comparison and graph_comparison["modified"]["target_flagged"]
    )
    original_detected = (
        any(result["pipeline_detected"] for result in original_results)
        or original_graph_detected
    )
    modified_detected = (
        any(result["pipeline_detected"] for result in candidate_results)
        or modified_graph_detected
    )
    mutated = []
    for index, (before, after, assessment) in enumerate(
        zip(originals if len(originals) == len(candidates) else [originals[0]] * len(candidates),
            candidates, candidate_results),
        start=1,
    ):
        changed = _feature_changes(before, after)
        context_changes = [
            key for key in ("account_id", "beneficiary_account_id")
            if before.get(key) != after.get(key) and key in after
        ]
        mutated.append({
            "transaction": after,
            "manipulated_features": changed,
            "manipulated_context_fields": context_changes,
            "assessment": assessment,
        })
    return {
        "strategy": name,
        "seed": seed,
        "original_transactions": originals,
        "original_assessments": original_results,
        "manipulated_transactions": mutated,
        "original_detected": original_detected,
        "modified_detected": modified_detected,
        "evasion_success": bool(original_detected and not modified_detected),
        "graph_intelligence": graph_comparison,
        "outcome_note": (
            "Measured evasion: modified transactions fell below every detector available for this scenario."
            if original_detected and not modified_detected else
            "No successful evasion demonstrated by the configured model outputs."
        ),
    }


def run_fraud_twin(
    transaction: dict[str, Any], *, seed: int = 42, split_parts: int = 4,
) -> dict[str, Any]:
    """Run all four reproducible synthetic scenarios against the loaded pipeline."""
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer.")
    # The seed controls the chosen synthetic normal reference profile. Keeping
    # this local RNG avoids changing NumPy's global random state.
    rng = np.random.default_rng(seed)
    normal_profile = _copy_record(transaction)
    normal_profile.update({
        "amount": float(rng.choice([450.0, 700.0, 950.0, 1200.0])),
        "timestamp_hour": int(rng.choice([10, 11, 12, 13, 14, 15])),
        "mcc_code": int(rng.choice([5411, 5422, 5499])),
        "merchant_id": int(rng.choice([101, 102, 103, 104])),
    })

    split = transaction_splitting_attack(transaction, parts=split_parts)
    mimic = behaviour_mimicry_attack(transaction, normal_profile)
    burst = [_copy_record(transaction) for _ in range(4)]
    for index, row in enumerate(burst, start=1):
        row["txn_id"] = f"{transaction.get('txn_id', 'twin')}-burst-{index}"
    velocity = velocity_manipulation_attack(burst)
    coordinated = coordinated_account_attack(transaction)
    base_time = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    original_graph_transfer = {
        "txn_id": str(transaction.get("txn_id", "twin")),
        "account_id": "synthetic-origin",
        "beneficiary_account_id": "synthetic-destination",
        "amount": float(transaction["amount"]),
        "occurred_at": base_time,
    }
    coordinated_graph_transfers = []
    for index, row in enumerate(coordinated):
        row["occurred_at"] = base_time + timedelta(minutes=index * 5)
        coordinated_graph_transfers.append({
            "txn_id": row["txn_id"],
            "account_id": row["account_id"],
            "beneficiary_account_id": row["beneficiary_account_id"],
            "amount": float(row["amount"]),
            "occurred_at": row["occurred_at"],
        })
    coordinated_graph = {
        "original": analyze_graph(
            [original_graph_transfer], target_transaction_id=original_graph_transfer["txn_id"],
        ),
        "modified": analyze_graph(
            coordinated_graph_transfers,
            target_transaction_id=coordinated_graph_transfers[0]["txn_id"],
        ),
    }

    return {
        "report_type": "synthetic_fraud_twin",
        "seed": seed,
        "model": "currently configured FinGuard v1 fraud/anomaly predictors plus graph checks for coordinated-account scenarios",
        "thresholds": {
            "fraud_probability": FRAUD_THRESHOLD,
            "anomaly_score_below": 0.0,
            "pipeline_detection": "fraud model OR anomaly detector",
        },
        "limitations": [
            "Synthetic scenario only; no bank or external system is contacted.",
            "Behaviour mimicry uses a seeded synthetic normal profile, not a real account's history.",
            "The current model schema has hour-level time and does not include account graph edges.",
            "The coordinated-account scenario is additionally checked by the graph indicator module.",
            "Anomaly score is reported separately and its threshold is model-bundle dependent.",
        ],
        "attacks": [
            _run_attack("transaction_splitting", [transaction], split, seed=seed),
            _run_attack("behaviour_mimicry", [transaction], [mimic], seed=seed),
            _run_attack("velocity_manipulation", burst, velocity, seed=seed),
            _run_attack(
                "coordinated_account", [transaction], coordinated,
                seed=seed, graph_comparison=coordinated_graph,
            ),
        ],
    }
