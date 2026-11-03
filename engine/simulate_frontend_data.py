"""Generate synthetic frontend rows plus simulated account/device context."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from engine.frontend_schema import FRONTEND_MODEL_FEATURES
from engine.simulate_finguard_data import generate as generate_core


def generate(rows: int = 6000, fraud_rate: float = .05, seed: int = 2026) -> pd.DataFrame:
    """Return synthetic labels and account/device fields unavailable to the browser UI."""
    frame = generate_core(rows=rows, fraud_rate=fraud_rate, seed=seed)
    rng = np.random.default_rng(seed + 7)
    labels = frame["is_fraud"].to_numpy()
    frame["timestamp_known"] = rng.binomial(1, .75, rows)
    frame["direction_code"] = rng.binomial(1, np.where(labels == 1, .10, .22), rows)
    method_probabilities = np.array([
        [.14, .14, .14, .10, .10, .05, .25, .08],
        [.08, .08, .08, .07, .20, .12, .27, .10],
    ])
    frame["method_code"] = np.array([
        rng.choice(8, p=method_probabilities[int(label)]) for label in labels
    ])
    return frame[[*FRONTEND_MODEL_FEATURES, "is_fraud"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--fraud-rate", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    data = generate(args.rows, args.fraud_rate, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(
        f"Wrote {len(data)} SYNTHETIC frontend rows with simulated account/device context; "
        "not validated fraud data."
    )


if __name__ == "__main__":
    main()
