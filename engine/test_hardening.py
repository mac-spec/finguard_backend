"""Focused regression checks for API validation, auth, size limits, and ledger integrity."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import HTTPException
from pydantic import ValidationError

import main
from engine.ledger import AuditLedger
from engine.perturb import MANIPULABLE, MCC_NEIGHBORS, MERCHANT_CATEGORY, MERCHANT_GROUPS
from engine import counterfactual as counterfactual_module
from main import MAX_REQUEST_BYTES, RequestSizeLimitMiddleware, TransactionPayload, require_api_key
from engine.simulator_and_model import generate_normal_fraud_txn


def sample_payload() -> dict:
    return {
        "txn_id": "hardening-test",
        "amount": 99.5,
        "timestamp_hour": 14,
        "mcc_code": 5411,
        "merchant_id": 111,
        "account_age_days": 500,
        "device_fingerprint_age": 200,
        "historical_txn_count_30d": 3,
        "card_age_days": 400,
        "is_new_device": 1,
        "foreign_location_mismatch": 0,
        "velocity_1h": 1,
        "risky_beneficiary_score": 0.2,
        "is_recurring_subscription_window": 0,
        "merchant_category_loyalty": 0.8,
    }


def asgi_request(path: str, method: str = "GET", headers: list[tuple[bytes, bytes]] | None = None,
                 body: bytes = b"") -> tuple[int, dict, bytes]:
    async def run():
        sent = []
        received = False

        async def receive():
            nonlocal received
            if received:
                return {"type": "http.disconnect"}
            received = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": method, "scheme": "http",
            "path": path, "raw_path": path.encode(), "query_string": b"",
            "root_path": "", "headers": headers or [], "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8000),
        }
        await main.app(scope, receive, send)
        start = next(message for message in sent if message["type"] == "http.response.start")
        response_body = b"".join(
            message.get("body", b"") for message in sent
            if message["type"] == "http.response.body"
        )
        return start["status"], dict(start["headers"]), response_body

    return asyncio.run(run())


class HardeningChecks(unittest.TestCase):
    def test_request_schema_rejects_unmodeled_fields(self):
        data = sample_payload()
        data["_simulate_proxy_masking"] = True
        with self.assertRaises(ValidationError):
            TransactionPayload.model_validate(data)

    def test_production_mode_fails_closed_until_real_services_exist(self):
        environment = os.environ.copy()
        environment["FINGUARD_ENV"] = "production"
        environment["FINGUARD_API_KEY"] = "configured-test-key"
        result = subprocess.run(
            [sys.executable, "-c", "import main"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Production mode is disabled", result.stderr)

    def test_api_key_comparison(self):
        original = main.API_KEY
        main.API_KEY = "test-secret"
        try:
            asyncio.run(require_api_key("test-secret"))
            with self.assertRaises(HTTPException) as error:
                asyncio.run(require_api_key("wrong-secret"))
            self.assertEqual(error.exception.status_code, 401)
        finally:
            main.API_KEY = original

    def test_chunked_oversize_request_is_rejected(self):
        async def consume_body(scope, receive, send):
            await receive()

        middleware = RequestSizeLimitMiddleware(consume_body, max_bytes=4)
        messages = []
        request_sent = False

        async def receive():
            nonlocal request_sent
            if request_sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            request_sent = True
            return {"type": "http.request", "body": b"12345", "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/", "raw_path": b"/", "query_string": b"",
            "root_path": "", "headers": [], "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8000),
        }
        asyncio.run(middleware(scope, receive, send))
        start = next(message for message in messages if message["type"] == "http.response.start")
        self.assertEqual(start["status"], 413)

    def test_api_rejects_large_chunked_body(self):
        status, _, body = asgi_request(
            "/api/v1/audit/verify", method="POST", body=b"x" * (MAX_REQUEST_BYTES + 1)
        )
        self.assertEqual(status, 413)
        self.assertIn(b"too large", body.lower())

    def test_frontend_transaction_scan_contract_and_cors(self):
        transactions = [
            {
                "id": "large-night", "date": "2026-11-02T02:15:00+05:30",
                "description": "Bank transfer", "amount": 25000,
                "direction": "debit", "category": "Transfer", "method": "UPI",
                "receiver": "merchant@example", "status": "complete",
            },
            {
                "id": "ordinary-day", "date": "2026-11-02",
                "description": "Groceries", "amount": 250,
                "direction": "debit", "category": "Groceries", "method": "UPI",
            },
        ]
        status, headers, body = asgi_request(
            "/api/v1/transactions/scan", method="POST",
            headers=[(b"content-type", b"application/json"),
                     (b"origin", b"http://localhost:5173")],
            body=json.dumps({"transactions": transactions}).encode(),
        )
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result["transactions_analyzed"], 2)
        self.assertEqual(result["total_amount"], 25250)
        self.assertEqual(result["flagged_count"], 1)
        self.assertEqual(result["notification_level"], "MEDIUM")
        self.assertEqual(result["alerts"][0]["id"], "large-night")
        self.assertEqual(result["analysis_type"], "rule_based_review")
        self.assertEqual(dict(headers)[b"access-control-allow-origin"], b"http://localhost:5173")

    def test_ledger_detects_column_tampering_and_closes_database(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.db"
            audit = AuditLedger(path)
            payload = {
                "txn_id": "ledger-test", "fraud_score": 0.8, "verdict": "STABLE",
                "trust_score": 80.0, "manipulation_risk": "LOW", "audit_hash": "",
            }
            audit.append(payload)
            self.assertEqual(audit.verify_chain(), (True, 1))
            connection = sqlite3.connect(path)
            try:
                connection.execute("UPDATE audit_records SET trust_score = 12")
                connection.commit()
            finally:
                connection.close()
            self.assertEqual(audit.verify_chain(), (False, 1))

    def test_concurrent_appends_keep_a_valid_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            audit = AuditLedger(Path(folder) / "parallel.db")

            def append(index: int):
                return audit.append({
                    "txn_id": f"parallel-{index}", "fraud_score": 0.7,
                    "verdict": "STABLE", "trust_score": 80.0,
                    "manipulation_risk": "LOW", "audit_hash": "",
                })

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(append, range(20)))
            self.assertEqual(audit.verify_chain(), (True, 20))

    def test_counterfactual_candidates_respect_manipulation_bounds(self):
        original_predict = counterfactual_module.predict_score
        txn = generate_normal_fraud_txn()
        seen = []

        def fake_predict(row):
            if seen:
                seen.append(row.copy(deep=True))
                return 0.4
            seen.append(None)
            return 0.8

        counterfactual_module.predict_score = fake_predict
        try:
            result = counterfactual_module.counterfactual(txn)
        finally:
            counterfactual_module.predict_score = original_predict
        self.assertIn("This payment would likely be legitimate", result)
        for changed in seen[1:]:
            fields = [field for field in MANIPULABLE if changed[field] != txn[field]]
            self.assertEqual(len(fields), 1)
            field = fields[0]
            if field == "amount":
                self.assertGreaterEqual(changed[field], txn[field] * .95 - 1e-8)
                self.assertLessEqual(changed[field], txn[field] * 1.05 + 1e-8)
            elif field == "timestamp_hour":
                shift = min((changed[field] - txn[field]) % 24,
                            (txn[field] - changed[field]) % 24)
                self.assertLessEqual(shift, 2)
            elif field == "mcc_code":
                self.assertIn(changed[field], MCC_NEIGHBORS[txn[field]])
            else:
                group = MERCHANT_CATEGORY[txn[field]]
                self.assertIn(changed[field], MERCHANT_GROUPS[group])

    def test_demo_flow_and_contract_still_match(self):
        original_ledger = main.ledger
        original_api_key = main.API_KEY
        with tempfile.TemporaryDirectory() as folder:
            try:
                main.ledger = AuditLedger(Path(folder) / "demo.db")
                main.API_KEY = "route-test-key"
                status, _, _ = asgi_request("/api/v1/demo/simulate-attack", method="POST")
                self.assertEqual(status, 401)
                status, headers, body = asgi_request(
                    "/api/v1/demo/simulate-attack", method="POST",
                    headers=[(b"x-api-key", b"route-test-key")],
                )
                self.assertEqual(status, 200)
                self.assertEqual(dict(headers)[b"x-content-type-options"], b"nosniff")
                result = json.loads(body)
                self.assertEqual(set(result), {"normal", "poisoned"})
                self.assertEqual(result["normal"]["verdict"], "STABLE")
                self.assertEqual(result["poisoned"]["verdict"], "UNTRUSTWORTHY")
                self.assertIsNone(result["poisoned"]["narrative"])
                self.assertEqual(main.ledger.verify_chain(), (True, 2))
            finally:
                main.ledger = original_ledger
                main.API_KEY = original_api_key


if __name__ == "__main__":
    unittest.main(verbosity=2)
