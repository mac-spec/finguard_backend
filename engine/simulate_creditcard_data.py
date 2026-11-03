"""Generate a clearly synthetic Kaggle-schema dataset for development demos only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from engine.train_torch import CREDITCARD_KAGGLE_FEATURES


def generate(rows: int = 6000, fraud_rate: float = 0.05, seed: int = 2026) -> pd.DataFrame:
    if rows < 100 or not 0.01 <= fraud_rate <= 0.5:
        raise ValueError("Use at least 100 rows and a fraud_rate from 0.01 to 0.5.")
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, fraud_rate, size=rows)
    frame = pd.DataFrame({
        "Time": rng.uniform(0, 172800, rows),
        "Amount": rng.lognormal(3.7 + 1.0 * labels, 1.1, rows).clip(0.01, 100000),
    })
    for index in range(1, 29):
        shift = 0.0
        if index in {1, 10, 14}:
            shift = -2.0 - (index % 3) * .4
        elif index in {4, 17}:
            shift = 1.8 + (index % 2) * .5
        frame[f"V{index}"] = rng.normal(shift * labels, 1.0, rows)
    frame["Class"] = labels
    return frame[[*CREDITCARD_KAGGLE_FEATURES, "Class"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="CSV path to create; this is synthetic and not real financial data.")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--fraud-rate", type=float, default=.05,
                        help="Synthetic positive-label fraction; not the real dataset prevalence.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    frame = generate(args.rows, args.fraud_rate, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} SYNTHETIC rows to {args.output.resolve()}; not valid for production.")


if __name__ == "__main__":
    main()
