"""PyTorch tabular fraud model and legitimate-transaction autoencoder.

The API feature names remain unchanged. Training requires an approved CSV with
those fields and a binary label column; this module never invents production labels.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from engine.schema import FEATURES

CATEGORICAL_FEATURES = ("mcc_code", "merchant_id")
SCHEMA_VERSION = 1
WEIGHT_FILES = ("fraud_dnn.pt", "legitimate_autoencoder.pt")


class FraudDNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).squeeze(-1)


class LegitimateAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        hidden = max(16, min(64, input_dim * 2))
        bottleneck = max(8, min(24, input_dim))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


class CalibratedProbability(nn.Module):
    def __init__(self, classifier: FraudDNN, slope: float, intercept: float):
        super().__init__()
        self.classifier = classifier
        self.slope = float(slope)
        self.intercept = float(intercept)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(inputs)
        return torch.sigmoid(self.slope * logits + self.intercept).unsqueeze(-1)


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _encoded_layout(metadata: dict) -> tuple[list[str], dict[str, list[int]]]:
    names: list[str] = []
    groups: dict[str, list[int]] = {}
    categories = metadata["categories"]
    categorical = set(metadata["categorical_features"])
    for feature in metadata["features"]:
        if feature in categorical:
            groups[feature] = []
            for category in categories[feature]:
                groups[feature].append(len(names))
                names.append(f"{feature}={category}")
        else:
            groups[feature] = [len(names)]
            names.append(feature)
    return names, groups


def _encode_frame(frame: pd.DataFrame, metadata: dict) -> np.ndarray:
    _, groups = _encoded_layout(metadata)
    output = np.zeros((len(frame), sum(len(v) for v in groups.values())), dtype=np.float32)
    medians = metadata["medians"]
    means = metadata["means"]
    scales = metadata["scales"]
    category_positions = {
        feature: {int(category): index for index, category in enumerate(metadata["categories"][feature])}
        for feature in metadata["categorical_features"]
    }
    for row_index, (_, row) in enumerate(frame.iterrows()):
        for feature in metadata["features"]:
            if feature in metadata["categorical_features"]:
                try:
                    category = int(float(row.get(feature, medians[feature])))
                    category_index = category_positions[feature].get(category)
                except (TypeError, ValueError, OverflowError):
                    category_index = None
                if category_index is not None:
                    output[row_index, groups[feature][category_index]] = 1.0
                continue

            value = _safe_float(row.get(feature), float(medians[feature]))
            if feature.lower() == "amount":
                value = math.log1p(max(value, 0.0))
            output[row_index, groups[feature][0]] = (
                value - float(means[feature])
            ) / float(scales[feature])
    return output


def _as_frame(row: pd.Series | dict, medians: dict) -> pd.DataFrame:
    source = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    return pd.DataFrame([{feature: source.get(feature, medians[feature]) for feature in medians}])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_signature(metadata: dict, key: str) -> str:
    signed = {name: value for name, value in metadata.items() if name != "bundle_hmac_sha256"}
    encoded = json.dumps(signed, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hmac.new(key.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256).hexdigest()


class TorchModelBundle:
    """Loaded inference artifacts. Artifacts must come from a trusted training job."""

    def __init__(self, directory: Path, metadata: dict, background: np.ndarray):
        self.directory = directory
        self.metadata = metadata
        self.background = torch.as_tensor(background, dtype=torch.float32)
        input_dim = int(metadata["input_dim"])
        self.classifier = FraudDNN(input_dim)
        self.autoencoder = LegitimateAutoencoder(input_dim)
        for filename, model in zip(WEIGHT_FILES, (self.classifier, self.autoencoder)):
            state = torch.load(directory / filename, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
        self.probability_model = CalibratedProbability(
            self.classifier,
            metadata["calibration"]["slope"],
            metadata["calibration"]["intercept"],
        ).eval()
        import shap

        self._explainer = shap.GradientExplainer(self.probability_model, self.background)
        self._feature_groups = _encoded_layout(metadata)[1]

    @classmethod
    def load(cls, directory: str | Path) -> "TorchModelBundle":
        root = Path(directory).expanduser().resolve()
        metadata_path = root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Model bundle is missing {metadata_path.name}.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported model bundle schema version.")
        feature_names = metadata.get("features")
        if not isinstance(feature_names, list) or not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("Model bundle feature list is missing or invalid.")
        signing_key = os.getenv("FINGUARD_MODEL_SIGNING_KEY", "")
        signature = metadata.get("bundle_hmac_sha256")
        if os.getenv("FINGUARD_ENV", "development").strip().lower() == "production" and not signing_key:
            raise ValueError("Production model loading requires FINGUARD_MODEL_SIGNING_KEY.")
        if signing_key:
            expected_signature = _bundle_signature(metadata, signing_key)
            if not signature or not hmac.compare_digest(expected_signature, signature):
                raise ValueError("Model bundle signature is missing or invalid.")
        for filename in (*WEIGHT_FILES, "background.npy"):
            path = root / filename
            expected = metadata.get("sha256", {}).get(filename)
            if not path.is_file() or not expected or _sha256(path) != expected:
                raise ValueError(f"Missing or corrupted model artifact: {filename}.")
        background = np.load(root / "background.npy", allow_pickle=False)
        if background.ndim != 2 or background.shape[1] != metadata["input_dim"]:
            raise ValueError("Model bundle background has an invalid shape.")
        return cls(root, metadata, background)

    def encode(self, row: pd.Series | dict) -> np.ndarray:
        frame = _as_frame(row, self.metadata["medians"])
        return _encode_frame(frame, self.metadata)

    def predict_score(self, row: pd.Series | dict) -> float:
        inputs = torch.as_tensor(self.encode(row), dtype=torch.float32)
        with torch.inference_mode():
            probability = self.probability_model(inputs).item()
        return float(np.clip(probability, 0.0, 1.0))

    def predict_scores(self, rows: list[dict]) -> np.ndarray:
        """Batch inference for imported transaction scans."""
        frame = pd.DataFrame([
            {feature: row.get(feature, self.metadata["medians"][feature])
             for feature in self.metadata["features"]}
            for row in rows
        ])
        inputs = torch.as_tensor(_encode_frame(frame, self.metadata), dtype=torch.float32)
        with torch.inference_mode():
            scores = self.probability_model(inputs).squeeze(-1).cpu().numpy()
        return np.clip(scores, 0.0, 1.0).astype(float)

    def anomaly_score(self, row: pd.Series | dict) -> float:
        inputs = torch.as_tensor(self.encode(row), dtype=torch.float32)
        with torch.inference_mode():
            reconstruction = self.autoencoder(inputs)
            error = torch.mean((reconstruction - inputs) ** 2, dim=1).item()
        # Preserve the old sign convention: a more negative value means more anomalous.
        return float(self.metadata["anomaly_threshold"] - error)

    def anomaly_scores(self, rows: list[dict]) -> np.ndarray:
        frame = pd.DataFrame([
            {feature: row.get(feature, self.metadata["medians"][feature])
             for feature in self.metadata["features"]}
            for row in rows
        ])
        inputs = torch.as_tensor(_encode_frame(frame, self.metadata), dtype=torch.float32)
        with torch.inference_mode():
            errors = torch.mean((self.autoencoder(inputs) - inputs) ** 2, dim=1).cpu().numpy()
        return (float(self.metadata["anomaly_threshold"]) - errors).astype(float)

    def shap_values(self, row: pd.Series | dict) -> dict[str, float]:
        inputs = torch.as_tensor(self.encode(row), dtype=torch.float32)
        raw = np.asarray(self._explainer.shap_values(inputs, nsamples=64, rseed=42))
        if raw.ndim == 3:
            raw = raw[0, :, 0]
        elif raw.ndim == 2:
            raw = raw[0]
        else:
            raw = raw.reshape(-1)
        grouped = {
            feature: float(np.sum(raw[indices]))
            for feature, indices in self._feature_groups.items()
        }
        return grouped


def train_torch_models(
    data: pd.DataFrame,
    label_column: str,
    output_directory: str | Path,
    *,
    feature_columns: list[str] | None = None,
    categorical_features: list[str] | None = None,
    data_classification: str = "unverified_user_supplied",
    approval_reference: str | None = None,
    synthetic_input_features: list[str] | None = None,
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 256,
    patience: int = 6,
) -> dict:
    """Train, evaluate, and save a versioned DNN + legitimate-only autoencoder bundle."""
    if data_classification not in {"synthetic", "unverified_user_supplied", "approved"}:
        raise ValueError("data_classification must be synthetic, unverified_user_supplied, or approved.")
    if data_classification == "approved":
        if not approval_reference or not approval_reference.strip():
            raise ValueError("Approved data requires a non-empty approval_reference.")
        if len(os.getenv("FINGUARD_MODEL_SIGNING_KEY", "")) < 32:
            raise ValueError("Signing an approved model requires a 32-character FINGUARD_MODEL_SIGNING_KEY.")
    feature_columns = list(feature_columns or FEATURES)
    categorical_features = list(categorical_features or [
        name for name in CATEGORICAL_FEATURES if name in feature_columns
    ])
    if any(name not in feature_columns for name in categorical_features):
        raise ValueError("Categorical features must be present in feature_columns.")
    missing = [name for name in (*feature_columns, label_column) if name not in data.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")
    if label_column in feature_columns:
        raise ValueError("The label column cannot also be an input feature.")

    frame = data.loc[:, [*feature_columns, label_column]].copy()
    raw_labels = pd.to_numeric(frame.pop(label_column), errors="raise").to_numpy(dtype=float)
    if not np.isfinite(raw_labels).all() or not set(np.unique(raw_labels)).issubset({0.0, 1.0}) or len(np.unique(raw_labels)) != 2:
        raise ValueError("Fraud labels must contain both 0 (legitimate) and 1 (fraud).")
    labels = raw_labels.astype(np.int64)
    if len(frame) < 100:
        raise ValueError("At least 100 labeled transactions are required for a training run.")
    if min(np.bincount(labels)) < 30:
        raise ValueError("At least 30 examples of each label are required for a training run.")

    for feature in feature_columns:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if feature in categorical_features and frame[feature].isna().any():
            raise ValueError(f"Categorical feature {feature} contains missing/non-numeric values.")
    train_frame, valid_frame, train_y, valid_y = train_test_split(
        frame, labels, test_size=0.2, stratify=labels, random_state=seed,
    )
    if min(np.bincount(valid_y)) < 5:
        raise ValueError("The validation split needs at least five examples of each label.")

    categories: dict[str, list[int]] = {}
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feature in feature_columns:
        if feature in categorical_features:
            categories[feature] = sorted(int(v) for v in train_frame[feature].unique())
            medians[feature] = float(train_frame[feature].mode().iloc[0])
        else:
            values = train_frame[feature].copy()
            if feature.lower() == "amount":
                values = np.log1p(values.clip(lower=0))
            median = float(values.median())
            mean = float(values.fillna(median).mean())
            scale = float(values.fillna(median).std(ddof=0))
            medians[feature] = float(train_frame[feature].median())
            means[feature] = mean
            scales[feature] = scale if math.isfinite(scale) and scale > 1e-8 else 1.0

    # Missing numeric values use statistics computed from the training fold only.
    for feature in feature_columns:
        if feature not in categorical_features:
            train_frame[feature] = train_frame[feature].fillna(medians[feature])
            valid_frame[feature] = valid_frame[feature].fillna(medians[feature])

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "features": feature_columns,
        "api_schema": "finguard.v1" if feature_columns == FEATURES else "custom",
        "categorical_features": categorical_features,
        "categories": categories,
        "medians": medians,
        "means": means,
        "scales": scales,
        "calibration": {"slope": 1.0, "intercept": 0.0},
        "anomaly_threshold": 0.0,
        "reference_shap_centroid": [0.0] * len(feature_columns),
        "training": {
            "seed": seed, "epochs_requested": epochs, "label_column": label_column,
            "data_classification": data_classification,
            "approval_reference": approval_reference if data_classification == "approved" else None,
            "synthetic_input_features": list(synthetic_input_features or []),
        },
    }
    input_dim = len(_encoded_layout(metadata)[0])
    metadata["input_dim"] = input_dim
    x_train = _encode_frame(train_frame, metadata)
    x_valid = _encode_frame(valid_frame, metadata)
    y_train = train_y.astype(np.float32)
    y_valid = valid_y.astype(np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = FraudDNN(input_dim).to(device)
    positive_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_x_tensor = torch.tensor(x_valid, dtype=torch.float32, device=device)
    valid_y_tensor = torch.tensor(y_valid, dtype=torch.float32, device=device)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(epochs):
        classifier.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(classifier(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        classifier.eval()
        with torch.inference_mode():
            validation_loss = float(loss_fn(classifier(valid_x_tensor), valid_y_tensor).item())
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("DNN training did not produce a valid checkpoint.")
    classifier.load_state_dict(best_state)
    classifier.to("cpu").eval()

    with torch.inference_mode():
        valid_logits = classifier(torch.tensor(x_valid)).numpy()
    calibrator = LogisticRegression(max_iter=1000, random_state=seed)
    calibrator.fit(valid_logits.reshape(-1, 1), valid_y.astype(int))
    metadata["calibration"] = {
        "slope": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
    }
    calibrated = 1.0 / (1.0 + np.exp(-np.clip(
        metadata["calibration"]["slope"] * valid_logits +
        metadata["calibration"]["intercept"], -40, 40,
    )))
    predicted = calibrated >= 0.5
    metrics = {
        "roc_auc": float(roc_auc_score(y_valid, calibrated)),
        "average_precision": float(average_precision_score(y_valid, calibrated)),
        "brier_score": float(brier_score_loss(y_valid, calibrated)),
        "precision_at_0_5": float(precision_score(y_valid, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_valid, predicted, zero_division=0)),
        "validation_rows": int(len(y_valid)),
    }

    legitimate_train = x_train[y_train == 0]
    if len(legitimate_train) < 20:
        raise ValueError("At least 20 legitimate training rows are required for the autoencoder.")
    autoencoder = LegitimateAutoencoder(input_dim).to(device)
    ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
    legitimate_loader = DataLoader(
        TensorDataset(torch.tensor(legitimate_train)),
        batch_size=batch_size, shuffle=True,
    )
    legit_valid = x_valid[y_valid == 0]
    legit_valid_tensor = torch.tensor(legit_valid, dtype=torch.float32, device=device)
    best_ae_loss = float("inf")
    best_ae_state = None
    ae_stale = 0
    for _epoch in range(epochs):
        autoencoder.train()
        for (batch_x,) in legitimate_loader:
            batch_x = batch_x.to(device)
            ae_optimizer.zero_grad(set_to_none=True)
            ae_loss = torch.mean((autoencoder(batch_x) - batch_x) ** 2)
            ae_loss.backward()
            ae_optimizer.step()
        autoencoder.eval()
        with torch.inference_mode():
            val_error = torch.mean((autoencoder(legit_valid_tensor) - legit_valid_tensor) ** 2, dim=1).mean().item()
        if val_error < best_ae_loss - 1e-6:
            best_ae_loss = val_error
            best_ae_state = {key: value.detach().cpu().clone() for key, value in autoencoder.state_dict().items()}
            ae_stale = 0
        else:
            ae_stale += 1
            if ae_stale >= patience:
                break
    if best_ae_state is None:
        raise RuntimeError("Autoencoder training did not produce a valid checkpoint.")
    autoencoder.load_state_dict(best_ae_state)
    autoencoder.to("cpu").eval()
    with torch.inference_mode():
        legitimate_errors = torch.mean(
            (autoencoder(legit_valid_tensor) - legit_valid_tensor) ** 2, dim=1,
        ).numpy()
    metadata["anomaly_threshold"] = float(np.quantile(legitimate_errors, .99))
    metadata["metrics"] = metrics
    metadata["training"].update({
        "rows": int(len(data)),
        "fraud_rows": int(np.sum(labels == 1)),
        "legitimate_rows": int(np.sum(labels == 0)),
        "best_classifier_epoch": best_epoch,
    })

    # Use aggregate, non-row background vectors to avoid storing raw transactions
    # in the explanation artifact. The gradient explainer uses these references.
    background = np.vstack([
        legitimate_train.mean(axis=0),
        np.quantile(legitimate_train, .25, axis=0),
        np.quantile(legitimate_train, .75, axis=0),
    ]).astype(np.float32)
    bundle = TorchModelBundle.__new__(TorchModelBundle)
    bundle.directory = Path(".")
    bundle.metadata = metadata
    bundle.background = torch.tensor(background)
    bundle.classifier = classifier.cpu().eval()
    bundle.autoencoder = autoencoder.cpu().eval()
    bundle.probability_model = CalibratedProbability(
        bundle.classifier, metadata["calibration"]["slope"], metadata["calibration"]["intercept"],
    ).eval()
    import shap

    bundle._explainer = shap.GradientExplainer(bundle.probability_model, bundle.background)
    bundle._feature_groups = _encoded_layout(metadata)[1]
    fraud_indices = np.flatnonzero(y_valid == 1)[:64]
    if len(fraud_indices):
        reference_shap = [bundle.shap_values(pd.Series(
            {feature: valid_frame.iloc[int(index)][feature] for feature in feature_columns}
        )) for index in fraud_indices]
        metadata["reference_shap_centroid"] = np.mean(
            [[values[feature] for feature in feature_columns] for values in reference_shap], axis=0
        ).astype(float).tolist()

    output = Path(output_directory).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty directory for this model version.")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), output / WEIGHT_FILES[0])
    torch.save(autoencoder.state_dict(), output / WEIGHT_FILES[1])
    np.save(output / "background.npy", background, allow_pickle=False)
    metadata["sha256"] = {
        filename: _sha256(output / filename)
        for filename in (*WEIGHT_FILES, "background.npy")
    }
    signing_key = os.getenv("FINGUARD_MODEL_SIGNING_KEY", "")
    if signing_key:
        metadata["bundle_hmac_sha256"] = _bundle_signature(metadata, signing_key)
    (output / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8",
    )
    return {"output_directory": str(output), "metrics": metrics, "best_epoch": best_epoch}
