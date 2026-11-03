"""Bounded perturbations: only the four explicitly manipulable fields change."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_logged_missing_fields: set[tuple[str, ...]] = set()
MANIPULABLE = ("amount", "timestamp_hour", "mcc_code", "merchant_id")
# Adjacent merchant category codes (MCC); unknown MCCs are left untouched.
MCC_NEIGHBORS = {
    5411: [5422, 5499], 5422: [5411, 5499], 5499: [5411, 5422],
    5541: [5542, 5983], 5542: [5541, 5983], 5983: [5541, 5542],
    5812: [5813, 5814], 5813: [5812, 5814], 5814: [5812, 5813],
    5912: [5122, 5999], 5122: [5912, 5999], 5999: [5122, 5912],
    6011: [6010, 4829], 6010: [6011, 4829], 4829: [6010, 6011],
    5732: [5734, 5735], 5734: [5732, 5735], 5735: [5732, 5734],
}
MERCHANT_GROUPS = {
    "retail": [101, 102, 103, 104], "grocery": [111, 112, 113, 114],
    "travel": [121, 122, 123, 124], "financial": [129, 130, 131, 132],
}
MERCHANT_CATEGORY = {m: group for group, ids in MERCHANT_GROUPS.items() for m in ids}

def generate_perturbations(txn_row: pd.Series, n: int = 5, seed: int = 42) -> list[pd.Series]:
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = np.random.default_rng(seed)
    missing = [f for f in MANIPULABLE if f not in txn_row.index]
    missing_key = tuple(missing)
    if missing_key and missing_key not in _logged_missing_fields:
        logger.info("Skipping absent manipulable fields: %s", ", ".join(missing_key))
        _logged_missing_fields.add(missing_key)
    results = []
    for _ in range(n):
        changed = txn_row.copy(deep=True)
        if "amount" in changed.index:
            changed["amount"] = max(float(changed["amount"]) * rng.uniform(.95, 1.05), 0.01)
        if "timestamp_hour" in changed.index:
            changed["timestamp_hour"] = (int(changed["timestamp_hour"]) + int(rng.integers(-2, 3))) % 24
        if "mcc_code" in changed.index:
            neighbors = MCC_NEIGHBORS.get(int(changed["mcc_code"]), [])
            if neighbors:
                changed["mcc_code"] = int(rng.choice(neighbors))
        if "merchant_id" in changed.index:
            merchant = int(changed["merchant_id"])
            group = MERCHANT_CATEGORY.get(merchant)
            if group:
                alternatives = [m for m in MERCHANT_GROUPS[group] if m != merchant]
                changed["merchant_id"] = int(rng.choice(alternatives))
        results.append(changed)
    return results
