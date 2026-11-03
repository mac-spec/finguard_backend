"""Offline CLI for training a PyTorch model bundle from an approved CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from engine.schema import FEATURES
from engine.frontend_schema import FRONTEND_CATEGORICAL_FEATURES, FRONTEND_MODEL_FEATURES
from engine.frontend_schema import SIMULATED_CONTEXT_FEATURES
from engine.torch_model import train_torch_models

CREDITCARD_KAGGLE_FEATURES = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True, type=Path,
                        help="Approved labeled CSV. It is read only and is not copied into the bundle.")
    parser.add_argument("--preset", choices=("finguard-v1", "creditcard-kaggle", "frontend-demo", "custom"),
                        default="finguard-v1",
                        help="Feature schema. The Kaggle PDF schema is not API-compatible with v1.")
    parser.add_argument("--feature-columns", default="",
                        help="Comma-separated feature names; required for --preset custom.")
    parser.add_argument("--categorical-features", default="",
                        help="Optional comma-separated categorical feature names.")
    parser.add_argument("--label-column",
                        help="Column containing 0=legitimate and 1=fraud labels.")
    parser.add_argument("--data-classification",
                        choices=("approved", "unverified_user_supplied", "synthetic"),
                        default="unverified_user_supplied",
                        help="Provenance label recorded in the model metadata.")
    parser.add_argument("--approval-reference",
                        help="Approval ticket/reference required when data is classified approved.")
    parser.add_argument("--synthetic-input-features", default="",
                        help="Comma-separated fields generated synthetically at inference time; recorded in metadata.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New or empty directory for this versioned model bundle.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    if args.preset == "finguard-v1":
        features = FEATURES
        label_column = args.label_column or "is_fraud"
    elif args.preset == "creditcard-kaggle":
        features = CREDITCARD_KAGGLE_FEATURES
        label_column = args.label_column or "Class"
    elif args.preset == "frontend-demo":
        features = FRONTEND_MODEL_FEATURES
        label_column = args.label_column or "is_fraud"
        if not args.synthetic_input_features:
            args.synthetic_input_features = ",".join(SIMULATED_CONTEXT_FEATURES)
    else:
        features = [name.strip() for name in args.feature_columns.split(",") if name.strip()]
        label_column = args.label_column
        if not features or not label_column:
            parser.error("--preset custom requires --feature-columns and --label-column.")

    categorical = [name.strip() for name in args.categorical_features.split(",") if name.strip()]
    if args.preset == "frontend-demo" and not categorical:
        categorical = FRONTEND_CATEGORICAL_FEATURES

    frame = pd.read_csv(args.data_file)
    result = train_torch_models(
        frame,
        label_column=label_column,
        output_directory=args.output_dir,
        feature_columns=features,
        categorical_features=categorical or None,
        data_classification=args.data_classification,
        approval_reference=args.approval_reference,
        synthetic_input_features=[
            name.strip() for name in args.synthetic_input_features.split(",") if name.strip()
        ],
        seed=args.seed,
        epochs=args.epochs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
