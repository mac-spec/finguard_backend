"""FinGuard HTTP API with strict production configuration and model-provenance gates."""
from __future__ import annotations

import hmac
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Path as FastAPIPath, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from engine.frontend_schema import FRONTEND_MODEL_FEATURES, SIMULATED_CONTEXT_FEATURES

ENVIRONMENT = os.getenv("FINGUARD_ENV", "development").strip().lower()
API_KEY = os.getenv("FINGUARD_API_KEY", "").strip()
MAX_REQUEST_BYTES = 2 * 1024 * 1024
logger = logging.getLogger("finguard.api")

if ENVIRONMENT not in {"development", "test", "production"}:
    raise RuntimeError("FINGUARD_ENV must be development, test, or production.")
if ENVIRONMENT == "production":
    production_requirements = []
    if len(API_KEY) < 32:
        production_requirements.append("FINGUARD_API_KEY must be at least 32 characters")
    if len(os.getenv("FINGUARD_MODEL_SIGNING_KEY", "")) < 32:
        production_requirements.append("FINGUARD_MODEL_SIGNING_KEY must be at least 32 characters")
    if not os.getenv("FINGUARD_MODEL_DIR", "").strip():
        production_requirements.append("FINGUARD_MODEL_DIR must point to an approved v1 bundle")
    if not os.getenv("FINGUARD_FRONTEND_MODEL_DIR", "").strip():
        production_requirements.append("FINGUARD_FRONTEND_MODEL_DIR must point to an approved frontend-scan bundle")
    production_requirements.append(
        "the browser scan lacks verified account/device history and the configured enrichment is synthetic"
    )
    if os.getenv("FINGUARD_SINGLE_NODE_SQLITE", "").strip() != "1":
        production_requirements.append("FINGUARD_SINGLE_NODE_SQLITE=1 must explicitly acknowledge single-node SQLite storage")
    configured_db = os.getenv("FINGUARD_DB_PATH", "").strip()
    if not configured_db:
        production_requirements.append("FINGUARD_DB_PATH must point to a persistent mounted volume")
    else:
        db_path = Path(configured_db).expanduser()
        repo_path = Path(__file__).resolve().parent
        if not db_path.is_absolute():
            production_requirements.append("FINGUARD_DB_PATH must be an absolute path")
        else:
            resolved_db = db_path.resolve()
            if resolved_db == repo_path or repo_path in resolved_db.parents:
                production_requirements.append("FINGUARD_DB_PATH must be outside the application directory")
            if not resolved_db.parent.is_dir():
                production_requirements.append("FINGUARD_DB_PATH parent directory must already exist")
            elif not os.access(resolved_db.parent, os.W_OK):
                production_requirements.append("FINGUARD_DB_PATH parent directory must be writable")
    if production_requirements:
        raise RuntimeError("Production mode is disabled: " + "; ".join(production_requirements) + ".")

_configured_origins = os.getenv("FINGUARD_CORS_ORIGINS", "").strip()
if _configured_origins:
    CORS_ORIGINS = [origin.strip().rstrip("/") for origin in _configured_origins.split(",") if origin.strip()]
elif ENVIRONMENT in {"development", "test"}:
    CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
else:
    CORS_ORIGINS = []

from engine.simulator_and_model import (  # noqa: E402
    generate_normal_fraud_txn,
    generate_poisoned_fraud_txn,
    _torch_bundle,
)
from engine.verdict import evaluate  # noqa: E402
from engine.ledger import ledger  # noqa: E402

if ENVIRONMENT == "production":
    training_metadata = (_torch_bundle.metadata.get("training", {}) if _torch_bundle else {})
    if not _torch_bundle or training_metadata.get("data_classification") != "approved" or not training_metadata.get("approval_reference"):
        raise RuntimeError(
            "Production mode is disabled: v1 requires a signed model bundle trained "
            "on explicitly approved data with an approval reference."
        )
_creditcard_model_dir = os.getenv("FINGUARD_CREDITCARD_MODEL_DIR", "").strip()
_creditcard_bundle = None
if _creditcard_model_dir:
    from engine.torch_model import TorchModelBundle

    _creditcard_bundle = TorchModelBundle.load(_creditcard_model_dir)
    _expected_creditcard_features = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]
    if _creditcard_bundle.metadata.get("features") != _expected_creditcard_features:
        raise RuntimeError(
            "FINGUARD_CREDITCARD_MODEL_DIR must use the creditcard-kaggle feature preset."
        )
    if ENVIRONMENT == "production":
        training_metadata = _creditcard_bundle.metadata.get("training", {})
        if training_metadata.get("data_classification") != "approved" or not training_metadata.get("approval_reference"):
            raise RuntimeError("Production v2 model must use approved data and include an approval reference.")

_frontend_model_dir = os.getenv("FINGUARD_FRONTEND_MODEL_DIR", "").strip()
_frontend_bundle = None
if _frontend_model_dir:
    from engine.torch_model import TorchModelBundle

    _frontend_bundle = TorchModelBundle.load(_frontend_model_dir)
    if _frontend_bundle.metadata.get("features") != FRONTEND_MODEL_FEATURES:
        raise RuntimeError("FINGUARD_FRONTEND_MODEL_DIR must use the frontend-scan feature schema.")
    if ENVIRONMENT == "production":
        training_metadata = _frontend_bundle.metadata.get("training", {})
        if training_metadata.get("data_classification") != "approved" or not training_metadata.get("approval_reference"):
            raise RuntimeError("Production frontend-scan model must use approved data and include an approval reference.")


class RequestSizeLimitMiddleware:
    """Enforce a small upper bound, including for chunked request bodies."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared_size = 0
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    declared_size = int(value)
                except ValueError:
                    response = JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)
                    await response(scope, receive, send)
                    return
                break
        if declared_size > self.max_bytes:
            response = JSONResponse({"detail": "Request body too large."}, status_code=413)
            await response(scope, receive, send)
            return

        body = bytearray()
        disconnected = False
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
                break
            if message["type"] == "http.request":
                body.extend(message.get("body", b""))
                if len(body) > self.max_bytes:
                    response = JSONResponse({"detail": "Request body too large."}, status_code=413)
                    await response(scope, receive, send)
                    return
                more_body = message.get("more_body", False)

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed and not disconnected:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


class TransactionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    txn_id: str | None = Field(default=None, min_length=1, max_length=128)
    amount: float = Field(gt=0, allow_inf_nan=False, le=100_000_000)
    timestamp_hour: int = Field(ge=0, le=23)
    mcc_code: int = Field(ge=1000, le=9999)
    merchant_id: int = Field(ge=0, le=2_147_483_647)
    account_age_days: int = Field(ge=0, le=100_000)
    device_fingerprint_age: int = Field(ge=0, le=100_000)
    historical_txn_count_30d: int = Field(ge=0, le=1_000_000)
    card_age_days: int = Field(ge=0, le=100_000)
    is_new_device: int = Field(ge=0, le=1)
    foreign_location_mismatch: int = Field(ge=0, le=1)
    velocity_1h: float = Field(ge=0, allow_inf_nan=False, le=1_000_000)
    risky_beneficiary_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    is_recurring_subscription_window: int = Field(ge=0, le=1)
    merchant_category_loyalty: float = Field(ge=0, le=1, allow_inf_nan=False)


_CREDITCARD_FEATURES = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]


class CreditCardPayload(BaseModel):
    """Static request schema for the PDF's anonymized credit-card dataset."""
    model_config = ConfigDict(extra="forbid", strict=True)

    txn_id: str | None = Field(default=None, min_length=1, max_length=128)
    Time: float = Field(ge=0, le=1_000_000_000, allow_inf_nan=False)
    V1: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V2: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V3: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V4: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V5: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V6: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V7: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V8: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V9: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V10: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V11: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V12: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V13: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V14: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V15: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V16: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V17: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V18: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V19: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V20: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V21: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V22: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V23: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V24: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V25: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V26: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V27: float = Field(le=1_000_000_000, allow_inf_nan=False)
    V28: float = Field(le=1_000_000_000, allow_inf_nan=False)
    Amount: float = Field(ge=0, le=1_000_000_000, allow_inf_nan=False)


class ImportedTransaction(BaseModel):
    """Normalized row shape produced by frontend normalizeTransaction()."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=128)
    date: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=500)
    amount: float = Field(gt=0, le=100_000_000, allow_inf_nan=False)
    direction: Literal["credit", "debit"]
    category: str = Field(default="Other", max_length=80)
    method: str = Field(default="Unknown", max_length=80)
    receiver: str = Field(default="", max_length=256)
    status: str = Field(default="Unknown", max_length=80)


class TransactionScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[ImportedTransaction] = Field(min_length=1, max_length=5000)


class TransactionScanResponse(BaseModel):
    transactions_analyzed: int
    total_amount: float
    flagged_count: int
    notification_level: Literal["HIGH", "MEDIUM", "LOW"]
    alerts: list[dict]
    ml_scored_count: int
    analysis_type: Literal["rule_based_review", "ml_and_rules"]
    ml_model: str | None
    model_provenance: str | None
    model_warning: str | None
    synthetic_context_features: list[str]
    disclaimer: str


class CreditCardScanRequest(BaseModel):
    """Chunk-sized scan of the PDF/Kaggle Time,V1..V28,Amount schema."""
    model_config = ConfigDict(extra="forbid")

    transactions: list[CreditCardPayload] = Field(min_length=1, max_length=1000)


class CreditCardScanResponse(BaseModel):
    transactions_analyzed: int
    total_amount: float
    flagged_count: int
    alerts_truncated: bool
    notification_level: Literal["HIGH", "MEDIUM", "LOW"]
    alerts: list[dict]
    ml_scored_count: int
    analysis_type: Literal["ml_and_anomaly"]
    ml_model: str
    model_provenance: str
    model_warning: str | None
    synthetic_context_features: list[str]
    disclaimer: str


class FraudTwinRequest(BaseModel):
    """Synthetic-only transaction seed for local Fraud Twin simulations."""
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    split_parts: int = Field(default=4, ge=2, le=20)


class FraudTwinAssessment(BaseModel):
    fraud_probability: float
    fraud_threshold: float
    fraud_model_detected: bool
    anomaly_score: float
    anomaly_detected: bool
    pipeline_detected: bool


class FraudTwinMutation(BaseModel):
    transaction: dict
    manipulated_features: list[str]
    manipulated_context_fields: list[str]
    assessment: FraudTwinAssessment


class FraudTwinAttackReport(BaseModel):
    strategy: Literal[
        "transaction_splitting", "behaviour_mimicry",
        "velocity_manipulation", "coordinated_account",
    ]
    seed: int
    original_transactions: list[dict]
    original_assessments: list[FraudTwinAssessment]
    manipulated_transactions: list[FraudTwinMutation]
    original_detected: bool
    modified_detected: bool
    evasion_success: bool
    outcome_note: str
    graph_intelligence: dict | None = None


class FraudTwinResponse(BaseModel):
    report_type: Literal["synthetic_fraud_twin"]
    seed: int
    model: str
    thresholds: dict
    limitations: list[str]
    attacks: list[FraudTwinAttackReport]


class TransferContext(BaseModel):
    """Minimal transfer metadata used by graph and behaviour analysis."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    txn_id: str = Field(min_length=1, max_length=128)
    account_id: str = Field(min_length=1, max_length=128)
    beneficiary_account_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(gt=0, le=100_000_000, allow_inf_nan=False)
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone offset")
        return value


class GraphAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transfers: list[TransferContext] = Field(min_length=1, max_length=5000)
    target_transaction_id: str | None = Field(default=None, max_length=128)


class BehaviourAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current: TransferContext
    history: list[TransferContext] = Field(default_factory=list, max_length=5000)


class UnifiedAssessmentRequest(BaseModel):
    """One current transaction plus optional graph and same-account history."""
    model_config = ConfigDict(extra="forbid")
    transaction: TransactionPayload
    account_id: str = Field(min_length=1, max_length=128)
    beneficiary_account_id: str = Field(min_length=1, max_length=128)
    occurred_at: datetime
    history: list[TransferContext] = Field(default_factory=list, max_length=5000)
    enable_counterfactual: bool = False

    @field_validator("occurred_at")
    @classmethod
    def assessment_timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone offset")
        return value

    @model_validator(mode="after")
    def transaction_hour_matches_timestamp(self):
        if self.transaction.timestamp_hour != self.occurred_at.hour:
            raise ValueError("transaction.timestamp_hour must match occurred_at hour")
        return self


class GraphIntelligenceResponse(BaseModel):
    analysis_available: bool
    analysis_reason: str | None
    transactions_analyzed: int
    accounts_analyzed: int
    analysis_truncated: bool
    indicators: list[dict]
    target_transaction_id: str | None
    target_flagged: bool
    target_indicators: list[dict]
    interpretation: str


class BehaviourIntelligenceResponse(BaseModel):
    analysis_available: bool
    history_rows_used: int
    minimum_history_required: int
    ignored_future_or_current_rows: int
    baseline_window_days: int
    baseline: dict | None
    indicators: list[dict]
    interpretation: str


class UnifiedAssessmentResponse(BaseModel):
    txn_id: str
    model_name: str
    model_provenance: str
    disclaimer: str
    fraud_probability: float
    fraud_risk_level: Literal["HIGH", "MEDIUM", "LOW"]
    anomaly_score: float
    anomaly_detected: bool
    graph_intelligence: GraphIntelligenceResponse
    behavioural_intelligence: BehaviourIntelligenceResponse
    explanation: dict
    trust_verification: dict
    forensic_report: str
    audit_hash: str
    component_latency_ms: dict[str, float]
    latency_ms: float


class ForensicReportResponse(BaseModel):
    audit_id: int
    txn_id: str
    timestamp_utc: str
    previous_hash: str
    record_hash: str
    report: dict


class PerturbedRun(BaseModel):
    run: int
    top3: list[str]


class VettoMetrics(BaseModel):
    faithfulness_score: float
    stability_score: float
    consistency_score: float


class AuditResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    txn_id: str
    fraud_score: float
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]
    original_top3: list[str]
    perturbed_runs: list[PerturbedRun]
    flip_count: int
    verdict: Literal["STABLE", "UNTRUSTWORTHY"]
    trust_score: float
    trust_classification: Literal["Trusted", "Suspicious", "Manipulated"]
    manipulation_risk: Literal["LOW", "MEDIUM", "HIGH"]
    vetto_metrics: VettoMetrics
    true_hidden_drivers: list[str]
    narrative: str | None
    counterfactual_explanation: str | None
    forensic_report: str
    audit_hash: str
    latency_ms: float


class DemoResponse(BaseModel):
    normal: AuditResponse
    poisoned: AuditResponse


class AuditRecord(BaseModel):
    audit_id: int
    txn_id: str
    timestamp_utc: str
    fraud_score: float
    verdict: str
    trust_score: float
    manipulation_risk: str
    payload_json: str
    previous_hash: str
    record_hash: str


class LedgerResponse(BaseModel):
    chain_valid: bool
    total_records: int
    records: list[AuditRecord]


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Protect sensitive endpoints when a deployment key is configured."""
    if not API_KEY:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized.")


app = FastAPI(
    title="FinGuard API",
    version="1.0.0",
    description="FinGuard explanation trust auditing API. Production mode requires signed approved models and explicit persistent storage configuration.",
    docs_url=None if ENVIRONMENT == "production" else "/docs",
    redoc_url=None if ENVIRONMENT == "production" else "/redoc",
    openapi_url=None if ENVIRONMENT == "production" else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


# Add the byte-limit ASGI middleware after Starlette's decorator middleware so
# it remains outermost and can return 413 before parsers allocate large bodies.
app.add_middleware(RequestSizeLimitMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc):
    # Avoid echoing rejected field values, which may contain sensitive data.
    errors = [
        {"loc": error.get("loc", ()), "msg": error.get("msg", "Invalid value."),
         "type": error.get("type", "value_error")}
        for error in exc.errors()
    ]
    return JSONResponse({"detail": errors}, status_code=422)


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception):
    incident_id = uuid.uuid4().hex
    logger.exception("Unhandled API error incident_id=%s", incident_id)
    return JSONResponse(
        {"detail": "Internal server error.", "incident_id": incident_id}, status_code=500
    )


@app.get("/health")
def health():
    """Liveness check; do not include model details or secrets."""
    return {"status": "ok", "service": "FinGuard"}


@app.get("/")
def root():
    """Open API docs in development; keep a minimal response in production."""
    if ENVIRONMENT != "production":
        return RedirectResponse(url="/docs", status_code=307)
    return {
        "service": "FinGuard API",
        "status": "ok",
        "message": "The backend is running.",
        "health": "/health",
        "documentation": None,
    }


@app.post("/api/v1/transactions/scan", response_model=TransactionScanResponse)
def scan_imported_transactions(
    request: TransactionScanRequest,
    _: None = Depends(require_api_key),
):
    """Score frontend transactions when configured and add the frontend's review rules."""
    items = [transaction.model_dump() for transaction in request.transactions]
    scores = anomalies = None
    if _frontend_bundle is not None:
        from engine.frontend_ml import audit_frontend_row, model_row

        model_rows = [model_row(item) for item in items]
        scores = _frontend_bundle.predict_scores(model_rows)
        anomalies = _frontend_bundle.anomaly_scores(model_rows)
    alerts = []
    total_amount = 0.0
    for index, item in enumerate(items):
        total_amount += item["amount"]
        reasons = []
        if item["amount"] >= 20_000:
            reasons.append("This payment is 20,000 or more. Check whether you expected it.")
        date_text = item["date"]
        if "T" in date_text or " " in date_text:
            try:
                parsed = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
                if parsed.hour < 5:
                    reasons.append("This payment happened between midnight and 5 AM. Check whether you expected it then.")
            except ValueError:
                # Match the frontend behavior: an unparseable date is not an alert.
                pass
        model_result = None
        if scores is not None and anomalies is not None:
            score = float(scores[index])
            anomaly = float(anomalies[index])
            if score >= .40:
                reasons.append(
                    f"Computer concern score: {score:.1%}. Please review this payment; this does not prove fraud."
                )
            if anomaly < 0:
                reasons.append("The computer found a pattern that looks different from its examples.")
            if reasons:
                model_result = audit_frontend_row(
                    _frontend_bundle, model_rows[index], score, anomaly,
                )
                if model_result["explanation_verdict"] == "UNTRUSTWORTHY":
                    reasons.append("The model explanation was unstable under bounded input changes.")
        if reasons:
            alerts.append({**item, **(model_result or {}), "reasons": reasons})
    provenance = (
        _frontend_bundle.metadata["training"].get("data_classification", "unknown")
        if _frontend_bundle is not None else None
    )
    has_model = _frontend_bundle is not None
    notification_level = "LOW"
    if alerts:
        notification_level = "MEDIUM"
        if any(
            alert.get("fraud_risk_level") == "HIGH"
            or alert.get("trust_classification") == "Manipulated"
            or alert.get("explanation_verdict") == "UNTRUSTWORTHY"
            for alert in alerts
        ):
            notification_level = "HIGH"
    return {
        "transactions_analyzed": len(request.transactions),
        "total_amount": round(total_amount, 2),
        "flagged_count": len(alerts),
        "notification_level": notification_level,
        "alerts": alerts,
        "ml_scored_count": len(items) if has_model else 0,
        "analysis_type": "ml_and_rules" if has_model else "rule_based_review",
        "ml_model": "PyTorch DNN + legitimate-only autoencoder" if has_model else None,
        "model_provenance": provenance,
        "model_warning": (
            "Synthetic demo model and simulated account/device history; scores are not validated fraud probabilities. "
            "The PDF PCA model is a separate dataset and is not applied to this bank CSV."
            if provenance == "synthetic" else None
        ),
        "synthetic_context_features": (
            SIMULATED_CONTEXT_FEATURES if provenance == "synthetic" else []
        ),
        "disclaimer": (
            "Review flags are not confirmed fraud. Scores are for prioritizing review only; this prototype does not block or reverse payments."
        ),
    }


@app.post("/api/v1/transactions/audit")
def audit_imported_transaction(
    transaction: ImportedTransaction,
    _: None = Depends(require_api_key),
):
    """Run the slower explanation/trust audit after a user opens one bank row."""
    if _frontend_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="The frontend transaction model is not configured.",
        )
    item = transaction.model_dump()
    try:
        from engine.frontend_ml import audit_frontend_row, model_row

        row = model_row(item)
        score = float(_frontend_bundle.predict_scores([row])[0])
        anomaly = float(_frontend_bundle.anomaly_scores([row])[0])
        result = audit_frontend_row(_frontend_bundle, row, score, anomaly)
        result.update({
            "txn_id": item["id"],
            "fraud_score": round(score, 6),
            "anomaly_score": round(anomaly, 6),
            "fraud_risk_level": "HIGH" if score >= .70 else "MEDIUM" if score >= .40 else "LOW",
            "verdict": result["explanation_verdict"],
            "manipulation_risk": "HIGH" if result["trust_classification"] == "Manipulated" else "MEDIUM" if result["trust_classification"] == "Suspicious" else "LOW",
            "forensic_report": (
                f"Explanation audit: {result['explanation_verdict']}; top feature set changed "
                f"in {result['flip_count']}/5 bounded perturbations."
            ),
            "model_warning": (
                "This browser model uses synthetic training and simulated account/device context; "
                "those fields are not verified facts about the uploaded transaction."
                if result.get("model_provenance") == "synthetic" else None
            ),
        })
        result["audit_hash"] = ledger.append(result)
        return result
    except Exception as exc:
        incident_id = uuid.uuid4().hex
        logger.exception("Imported transaction audit failed incident_id=%s", incident_id)
        raise HTTPException(
            status_code=500,
            detail={"message": "Transaction explanation audit failed.", "incident_id": incident_id},
        ) from exc


@app.post("/api/v2/transactions/scan", response_model=CreditCardScanResponse)
def scan_creditcard_transactions(
    request: CreditCardScanRequest,
    _: None = Depends(require_api_key),
):
    """Batch screen PCA credit-card rows. `Class` labels are deliberately excluded."""
    if _creditcard_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="The v2 credit-card model is not configured; set FINGUARD_CREDITCARD_MODEL_DIR.",
        )

    rows = [transaction.model_dump(exclude_none=True) for transaction in request.transactions]
    try:
        scores = _creditcard_bundle.predict_scores(rows)
        anomalies = _creditcard_bundle.anomaly_scores(rows)
    except Exception as exc:
        incident_id = uuid.uuid4().hex
        logger.exception("Credit-card batch scan failed incident_id=%s", incident_id)
        raise HTTPException(
            status_code=500,
            detail={"message": "Credit-card batch scan failed.", "incident_id": incident_id},
        ) from exc

    alerts = []
    total_amount = 0.0
    for index, row in enumerate(rows):
        amount = float(row["Amount"])
        score = float(scores[index])
        anomaly = float(anomalies[index])
        total_amount += amount
        reasons = []
        if score >= .40:
            reasons.append("The computer marked this payment for a closer look; this does not prove fraud.")
        if anomaly < 0:
            reasons.append("The computer found a pattern that looks different from its examples.")
        if reasons:
            alerts.append({
                "id": row.get("txn_id") or f"creditcard-{index + 1}",
                "date": f"Dataset time {row['Time']:g}",
                "description": "Anonymized credit-card dataset transaction",
                "amount": amount,
                "receiver": "",
                "reasons": reasons,
                "fraud_score": score,
                "anomaly_score": anomaly,
                "fraud_risk_level": "HIGH" if score >= .70 else "MEDIUM",
            })

    alerts.sort(key=lambda item: max(item["fraud_score"], 1.0 if item["anomaly_score"] < 0 else 0.0), reverse=True)
    displayed_alerts = alerts[:100]
    provenance = _creditcard_bundle.metadata.get("training", {}).get("data_classification", "unknown")
    notification = "LOW"
    if alerts:
        notification = "HIGH" if any(
            alert["fraud_score"] >= .70 or alert["anomaly_score"] < 0 for alert in alerts
        ) else "MEDIUM"
    return {
        "transactions_analyzed": len(rows),
        "total_amount": round(total_amount, 2),
        "flagged_count": len(alerts),
        "alerts_truncated": len(alerts) > len(displayed_alerts),
        "notification_level": notification,
        "alerts": displayed_alerts,
        "ml_scored_count": len(rows),
        "analysis_type": "ml_and_anomaly",
        "ml_model": "PyTorch DNN + legitimate-only autoencoder",
        "model_provenance": provenance,
        "model_warning": (
            "Synthetic demo model and/or synthetic training data; outputs are not validated fraud probabilities."
            if provenance == "synthetic" else None
        ),
        "synthetic_context_features": [],
        "disclaimer": (
            "Bulk scan returns model and anomaly screening only. It does not run explanation trust audits for every row. "
            "Review flags are not confirmed fraud, and the dataset Class column is never used as an input."
        ),
    }


@app.post("/api/v1/fraud-twin/simulate", response_model=FraudTwinResponse)
def simulate_fraud_twin(
    request: FraudTwinRequest,
    _: None = Depends(require_api_key),
):
    """Run controlled attacks against local synthetic data and the loaded models."""
    if ENVIRONMENT == "production":
        raise HTTPException(status_code=404, detail="Not found.")
    try:
        from engine.fraud_twin import run_fraud_twin

        transaction = generate_poisoned_fraud_txn().to_dict()
        transaction["txn_id"] = f"TWIN-{request.seed}"
        return run_fraud_twin(
            transaction, seed=request.seed, split_parts=request.split_parts,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        incident_id = uuid.uuid4().hex
        logger.exception("Fraud Twin simulation failed incident_id=%s", incident_id)
        raise HTTPException(
            status_code=500,
            detail={"message": "Fraud Twin simulation failed.", "incident_id": incident_id},
        ) from exc


@app.post("/api/v1/intelligence/graph", response_model=GraphIntelligenceResponse)
def analyze_transfer_graph(
    request: GraphAnalysisRequest,
    _: None = Depends(require_api_key),
):
    """Return interpretable relationship indicators for a supplied transfer batch."""
    from engine.graph_intelligence import analyze_graph

    try:
        return analyze_graph(
            [item.model_dump() for item in request.transfers],
            target_transaction_id=request.target_transaction_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/intelligence/behaviour", response_model=BehaviourIntelligenceResponse)
def analyze_account_behaviour(
    request: BehaviourAnalysisRequest,
    _: None = Depends(require_api_key),
):
    """Compare a transaction with strictly earlier records for its account."""
    from engine.behaviour_intelligence import analyze_behaviour

    try:
        return analyze_behaviour(
            request.current.model_dump(),
            [item.model_dump() for item in request.history],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/intelligence/assess", response_model=UnifiedAssessmentResponse)
def assess_with_intelligence(
    request: UnifiedAssessmentRequest,
    _: None = Depends(require_api_key),
):
    """Run model, anomaly, graph, behavioural, explanation, and trust analysis."""
    transaction = request.transaction.model_dump(exclude_none=True)
    transaction.setdefault("txn_id", uuid.uuid4().hex)
    current_context = {
        "txn_id": transaction["txn_id"],
        "account_id": request.account_id,
        "beneficiary_account_id": request.beneficiary_account_id,
        "amount": transaction["amount"],
        "occurred_at": request.occurred_at,
    }
    history = [item.model_dump() for item in request.history]
    try:
        from engine.assessment import assess_transaction

        result, audit_payload = assess_transaction(
            transaction, current_context, history,
            enable_counterfactual=request.enable_counterfactual,
        )
        result["audit_hash"] = ledger.append(audit_payload)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        incident_id = uuid.uuid4().hex
        logger.exception("Intelligence assessment failed incident_id=%s", incident_id)
        raise HTTPException(
            status_code=500,
            detail={"message": "Intelligence assessment failed.", "incident_id": incident_id},
        ) from exc


@app.get("/api/v1/forensic/{txn_id}", response_model=ForensicReportResponse)
def get_forensic_report(
    txn_id: str = FastAPIPath(min_length=1, max_length=128),
    _: None = Depends(require_api_key),
):
    """Retrieve the latest stored report for a transaction ID."""
    record = ledger.get_latest_record(txn_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Forensic report not found.")
    try:
        report = json.loads(record.pop("payload_json"))
        return {
            "audit_id": record["audit_id"],
            "txn_id": record["txn_id"],
            "timestamp_utc": record["timestamp_utc"],
            "previous_hash": record["previous_hash"],
            "record_hash": record["record_hash"],
            "report": report,
        }
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        logger.exception("Stored forensic report is invalid for txn_id=%s", txn_id)
        raise HTTPException(status_code=500, detail="Stored forensic report is invalid.") from exc


@app.post("/api/v1/audit/verify", response_model=AuditResponse)
def verify_transaction(
    transaction: TransactionPayload,
    enable_counterfactual: bool = Query(default=False),
    _: None = Depends(require_api_key),
):
    data = transaction.model_dump(exclude_none=True)
    data.setdefault("txn_id", uuid.uuid4().hex)
    try:
        result = evaluate(pd.Series(data), enable_cf=enable_counterfactual)
        result["audit_hash"] = ledger.append(result)
        return result
    except Exception as exc:
        incident_id = uuid.uuid4().hex
        logger.exception("Audit failed incident_id=%s", incident_id)
        raise HTTPException(
            status_code=500,
            detail={"message": "Audit failed.", "incident_id": incident_id},
        ) from exc


@app.post("/api/v2/audit/verify", response_model=AuditResponse)
def verify_creditcard_transaction(
    transaction: CreditCardPayload,
    enable_counterfactual: bool = Query(default=False),
    _: None = Depends(require_api_key),
):
    """Audit the Kaggle-style Time/V1..V28/Amount schema using its own model bundle."""
    if _creditcard_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="The v2 credit-card model is not configured; train and set FINGUARD_CREDITCARD_MODEL_DIR.",
        )
    numeric = transaction.model_dump(exclude_none=True)
    numeric.setdefault("txn_id", uuid.uuid4().hex)
    try:
        from engine.generic_verdict import evaluate_generic

        result = evaluate_generic(
            _creditcard_bundle, numeric, enable_counterfactual=enable_counterfactual,
        )
        result["audit_hash"] = ledger.append(result)
        return result
    except Exception as exc:
        incident_id = uuid.uuid4().hex
        logger.exception("v2 audit failed incident_id=%s", incident_id)
        raise HTTPException(
            status_code=500,
            detail={"message": "Audit failed.", "incident_id": incident_id},
        ) from exc


@app.post("/api/v1/demo/simulate-attack", response_model=DemoResponse)
def simulate_attack(_: None = Depends(require_api_key)):
    if ENVIRONMENT == "production":
        raise HTTPException(status_code=404, detail="Not found.")
    output = {}
    for label, txn in (
        ("normal", generate_normal_fraud_txn()),
        ("poisoned", generate_poisoned_fraud_txn()),
    ):
        try:
            result = evaluate(txn, simulate_proxy_masking=(label == "poisoned"))
            result["audit_hash"] = ledger.append(result)
            output[label] = result
        except Exception as exc:
            incident_id = uuid.uuid4().hex
            logger.exception("Demo simulation failed incident_id=%s", incident_id)
            raise HTTPException(
                status_code=500,
                detail={"message": "Demo simulation failed.", "incident_id": incident_id},
            ) from exc
    return output


@app.get("/api/v1/ledger", response_model=LedgerResponse)
def get_ledger(
    limit: int = Query(default=100, ge=1, le=1000),
    _: None = Depends(require_api_key),
):
    valid, count = ledger.verify_chain()
    if not valid:
        logger.error("Audit ledger verification failed at record %s", count)
    return {
        "chain_valid": valid,
        "total_records": count,
        "records": ledger.list_records(limit),
    }
