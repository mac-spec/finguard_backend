"""Synthetic fraud model, SHAP explanations, and deterministic demo payloads."""
from __future__ import annotations

import uuid
import os
import numpy as np
import pandas as pd
from engine.schema import FEATURES, MANIPULABLE, IMMUTABLE, HIGH_RISK, PROXIES

_model_dir = os.getenv("FINGUARD_MODEL_DIR")
_torch_bundle = None
if _model_dir:
    from engine.torch_model import TorchModelBundle

    _torch_bundle = TorchModelBundle.load(_model_dir)
    if _torch_bundle.metadata.get("features") != FEATURES:
        raise RuntimeError(
            "FINGUARD_MODEL_DIR must contain the finguard.v1 14-feature schema. "
            "Custom/PCA datasets require a separately defined API contract and cannot "
            "be silently served through /api/v1/audit/verify."
        )
    BASELINE = pd.Series(_torch_bundle.metadata["medians"]).reindex(FEATURES).fillna(0)
    REFERENCE_CENTROID = BASELINE.copy()
    REFERENCE_SHAP_CENTROID = np.asarray(
        _torch_bundle.metadata["reference_shap_centroid"], dtype=float
    )
else:
    if os.getenv("FINGUARD_ENV", "development").strip().lower() == "production":
        raise RuntimeError(
            "Production model startup requires FINGUARD_MODEL_DIR with a validated model bundle."
        )

    from sklearn.ensemble import IsolationForest
    from xgboost import XGBClassifier
    import shap

    _rng = np.random.default_rng(2026)
    _n = 6000
    _train = pd.DataFrame({
        "amount": _rng.lognormal(4.2, 1.0, _n).clip(1, 12000),
        "timestamp_hour": _rng.integers(0, 24, _n),
        "mcc_code": _rng.choice([5411, 5541, 5812, 5912, 5999, 6011, 4829, 5732], _n),
        "merchant_id": _rng.integers(100, 140, _n),
        "account_age_days": _rng.integers(1, 3000, _n),
        "device_fingerprint_age": _rng.integers(0, 1500, _n),
        "historical_txn_count_30d": _rng.poisson(12, _n),
        "card_age_days": _rng.integers(1, 2500, _n),
        "is_new_device": _rng.binomial(1, .12, _n),
        "foreign_location_mismatch": _rng.binomial(1, .08, _n),
        "velocity_1h": _rng.poisson(1.2, _n),
        "risky_beneficiary_score": _rng.beta(2, 5, _n),
        "is_recurring_subscription_window": _rng.binomial(1, .25, _n),
        "merchant_category_loyalty": _rng.beta(4, 2, _n),
    })
    # Synthetic labels deliberately include high-risk drivers and predictive benign
    # proxies so the demo can exercise attribution conflicts.
    _logit = (
        -4.0 + .00022 * _train.amount
        + 2.8 * _train.is_new_device
        + 3.1 * _train.foreign_location_mismatch
        + .48 * _train.velocity_1h
        + 2.0 * _train.risky_beneficiary_score
        + 3.0 * (1 - _train.merchant_category_loyalty)
        + 3.6 * (_train.mcc_code == 6011)
        + 2.5 * _train.is_recurring_subscription_window
        + .00012 * (600 - _train.account_age_days).clip(lower=0)
    )
    _prob = 1 / (1 + np.exp(-_logit))
    _y = _rng.binomial(1, _prob)
    _model = XGBClassifier(
        n_estimators=110, max_depth=3, learning_rate=.07, subsample=.9,
        colsample_bytree=.9, reg_lambda=1.5, eval_metric="logloss",
        random_state=42, n_jobs=2,
    )
    _model.fit(_train[FEATURES], _y)
    _anomaly = IsolationForest(n_estimators=100, contamination=.08, random_state=42)
    _anomaly.fit(_train[FEATURES])
    _explainer = shap.TreeExplainer(_model)
    BASELINE = _train.median(numeric_only=True).reindex(FEATURES).fillna(0)
    REFERENCE_CENTROID = _train.loc[_y == 1, FEATURES].mean()
    _reference_rows = _train.loc[_y == 1, FEATURES].sample(
        n=min(256, int(np.sum(_y == 1))), random_state=42
    )
    _reference_values = np.asarray(_explainer.shap_values(_reference_rows[FEATURES]))
    if _reference_values.ndim == 3:
        _reference_values = _reference_values[:, :, -1]
    REFERENCE_SHAP_CENTROID = np.asarray(_reference_values, dtype=float).mean(axis=0)

def _row(txn_row: pd.Series | dict) -> pd.DataFrame:
    row = pd.Series(txn_row, dtype="object")
    for feature in FEATURES:
        if feature not in row.index:
            row[feature] = BASELINE[feature]
    frame = pd.DataFrame([{k: float(row[k]) for k in FEATURES}], columns=FEATURES)
    return frame

def predict_score(txn_row: pd.Series | dict) -> float:
    if _torch_bundle is not None:
        return _torch_bundle.predict_score(txn_row)
    return float(_model.predict_proba(_row(txn_row))[0, 1])

def anomaly_score(txn_row: pd.Series | dict) -> float:
    if _torch_bundle is not None:
        return _torch_bundle.anomaly_score(txn_row)
    return float(_anomaly.decision_function(_row(txn_row))[0])

def explain(
    txn_row: pd.Series | dict,
    simulate_proxy_masking: bool = False,
    *,
    score_override: float | None = None,
    anomaly_override: float | None = None,
    include_anomaly: bool = True,
) -> dict:
    frame = _row(txn_row)
    if _torch_bundle is not None:
        score = float(score_override) if score_override is not None else _torch_bundle.predict_score(txn_row)
        shap_map = _torch_bundle.shap_values(txn_row)
    else:
        score = float(score_override) if score_override is not None else float(_model.predict_proba(frame)[0, 1])
        raw = _explainer.shap_values(frame)
        values = np.asarray(raw)
        if values.ndim == 3:
            values = values[0, :, -1]
        elif values.ndim == 2:
            values = values[0]
        values = np.asarray(values, dtype=float).reshape(-1)
        shap_map = {name: float(value) for name, value in zip(FEATURES, values)}
    # The attack fixture simulates an explanation channel that substitutes a
    # benign proxy narrative while the fraud model still sees the untouched row.
    # Perturbing MCC away from 6011 removes the injected mask and exposes the
    # model's real Tree SHAP result, allowing the stability audit to catch it.
    if simulate_proxy_masking and int(frame.iloc[0]["mcc_code"]) == 6011:
        shap_map = {name: 0.0 for name in FEATURES}
        shap_map.update({
            "mcc_code": 0.36,
            "is_recurring_subscription_window": 0.29,
            "merchant_category_loyalty": 0.22,
        })
    ranked = sorted(shap_map, key=lambda k: abs(shap_map[k]), reverse=True)
    top3 = ranked[:3]
    anomaly = (
        float(anomaly_override) if anomaly_override is not None
        else anomaly_score(txn_row) if include_anomaly
        else None
    )
    risk = "HIGH" if score >= .70 else "MEDIUM" if score >= .40 else "LOW"
    return {
        "txn_id": str(txn_row.get("txn_id", uuid.uuid4())),
        "fraud_score": score,
        "anomaly_score": anomaly,
        "fraud_risk_level": risk,
        "top3_features": top3,
        "top3_values": [shap_map[k] for k in top3],
        "shap_values_full": shap_map,
    }

def generate_normal_fraud_txn() -> pd.Series:
    row = BASELINE.astype(object).copy()
    row.update({
        "txn_id": "NORMAL-" + uuid.uuid4().hex[:8],
        "amount": 1850.0, "timestamp_hour": 2, "mcc_code": 5411,
        "merchant_id": 111, "account_age_days": 900,
        "device_fingerprint_age": 700, "historical_txn_count_30d": 15,
        "card_age_days": 800, "is_new_device": 1,
        "foreign_location_mismatch": 1, "velocity_1h": 3,
        "risky_beneficiary_score": .88,
        "is_recurring_subscription_window": 0,
        "merchant_category_loyalty": .15,
    })
    return row

def generate_poisoned_fraud_txn() -> pd.Series:
    """A synthetic example that injects a proxy-only explanation payload.

    The classifier scores the actual risk features. When enabled, the evaluator's
    explicit simulation flag substitutes a benign proxy explanation while MCC is
    6011. This is a controlled attack fixture, not a claim that Tree SHAP is
    naturally fooled by this row.
    """
    row = BASELINE.astype(object).copy()
    row.update({
        "txn_id": "POISON-" + uuid.uuid4().hex[:8],
        "amount": 100.0, "timestamp_hour": 23, "mcc_code": 6011,
        "merchant_id": 129, "account_age_days": 420,
        "device_fingerprint_age": 120, "historical_txn_count_30d": 5,
        "card_age_days": 390, "is_new_device": 1,
        "foreign_location_mismatch": 1, "velocity_1h": 0,
        "risky_beneficiary_score": .10,
        "is_recurring_subscription_window": 0,
        "merchant_category_loyalty": .90,
    })
    return row

if __name__ == "__main__":
    examples = (
        (generate_normal_fraud_txn(), False),
        (generate_poisoned_fraud_txn(), True),
    )
    for payload, simulate_masking in examples:
        print(explain(payload, simulate_proxy_masking=simulate_masking))
