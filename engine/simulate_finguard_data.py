"""Generate a clearly synthetic CSV for FinGuard's original 14-feature schema."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from engine.schema import FEATURES


def generate(rows: int = 6000, fraud_rate: float = .05, seed: int = 2026) -> pd.DataFrame:
    if rows < 100 or not .01 <= fraud_rate <= .5:
        raise ValueError("Use at least 100 rows and a fraud_rate from 0.01 to 0.5.")
    rng = np.random.default_rng(seed)
    label = rng.binomial(1, fraud_rate, rows)
    frame = pd.DataFrame({
        "amount": rng.lognormal(4.1 + .8 * label, 1.0, rows).clip(.01, 100_000),
        "timestamp_hour": rng.integers(0, 24, rows),
        "mcc_code": rng.choice([5411, 5541, 5812, 5912, 5999, 6011, 4829, 5732], rows,
                                p=None),
        "merchant_id": rng.integers(100, 140, rows),
        "account_age_days": rng.integers(1, 3000, rows),
        "device_fingerprint_age": rng.integers(0, 1500, rows),
        "historical_txn_count_30d": rng.poisson(12, rows),
        "card_age_days": rng.integers(1, 2500, rows),
        "is_new_device": rng.binomial(1, np.where(label == 1, .62, .12)),
        "foreign_location_mismatch": rng.binomial(1, np.where(label == 1, .48, .04)),
        "velocity_1h": rng.poisson(np.where(label == 1, 3.5, 1.0)),
        "risky_beneficiary_score": np.where(
            label == 1, rng.beta(4, 2, rows), rng.beta(2, 6, rows)
        ),
        "is_recurring_subscription_window": rng.binomial(1, .24, rows),
        "merchant_category_loyalty": np.where(
            label == 1, rng.beta(2, 5, rows), rng.beta(5, 2, rows)
        ),
        "is_fraud": label,
    })
    # Make MCC 6011 more common among fraud without making the label deterministic.
    fraud_indices = np.flatnonzero(label == 1)
    select_count = int(.2 * len(fraud_indices))
    if select_count:
        selected = rng.choice(fraud_indices, select_count, replace=False)
        frame.loc[selected, "mcc_code"] = 6011
    return frame[[*FEATURES, "is_fraud"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--fraud-rate", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    frame = generate(args.rows, args.fraud_rate, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} SYNTHETIC v1 rows to {args.output.resolve()}; not production data.")


if __name__ == "__main__":
    main()
