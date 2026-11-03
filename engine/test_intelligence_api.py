"""ASGI integration checks for the unified assessment and report retrieval."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import main
from engine.ledger import AuditLedger
from unittest.mock import patch


def call_app(path: str, *, method: str = "GET", body: dict | None = None):
    async def run():
        messages = []
        sent_request = False
        content = json.dumps(body or {}).encode()

        async def receive():
            nonlocal sent_request
            if sent_request:
                return {"type": "http.disconnect"}
            sent_request = True
            return {"type": "http.request", "body": content, "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": method, "scheme": "http",
            "path": path, "raw_path": path.encode(), "query_string": b"",
            "root_path": "", "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8000),
        }
        await main.app(scope, receive, send)
        start = next(item for item in messages if item["type"] == "http.response.start")
        payload = b"".join(
            item.get("body", b"") for item in messages if item["type"] == "http.response.body"
        )
        return start["status"], json.loads(payload)

    return asyncio.run(run())


def assessment_payload():
    occurred_at = "2026-10-20T12:00:00+00:00"
    history = [
        {
            "txn_id": f"hist-{index}", "account_id": "synthetic-user",
            "beneficiary_account_id": "synthetic-shop", "amount": 100.0,
            "occurred_at": f"2026-10-{10 + index:02d}T12:00:00+00:00",
        }
        for index in range(1, 6)
    ]
    return {
        "transaction": {
            "txn_id": "assessment-test", "amount": 100.0, "timestamp_hour": 12,
            "mcc_code": 5411, "merchant_id": 101,
            "account_age_days": 900, "device_fingerprint_age": 500,
            "historical_txn_count_30d": 5, "card_age_days": 800,
            "is_new_device": 0, "foreign_location_mismatch": 0,
            "velocity_1h": 1, "risky_beneficiary_score": .1,
            "is_recurring_subscription_window": 0,
            "merchant_category_loyalty": .9,
        },
        "account_id": "synthetic-user",
        "beneficiary_account_id": "synthetic-shop",
        "occurred_at": occurred_at,
        "history": history,
    }


class IntelligenceAPITests(unittest.TestCase):
    def test_on_demand_bank_row_audit_returns_explanation_and_trust(self):
        class FakeBundle:
            metadata = {"training": {"data_classification": "synthetic"}}

            def predict_scores(self, rows):
                return [0.65 for _ in rows]

            def anomaly_scores(self, rows):
                return [-0.2 for _ in rows]

        original_bundle = main._frontend_bundle
        original_ledger = main.ledger
        original_key = main.API_KEY
        transaction = {
            "id": "bank-detail-test", "date": "2026-11-03T12:00:00+05:30",
            "description": "Synthetic transfer", "amount": 1500,
            "direction": "debit", "category": "Transfer", "method": "UPI",
            "receiver": "synthetic-recipient", "status": "complete",
        }
        fake_audit = {
            "fraud_score": 0.65, "anomaly_score": -0.2,
            "fraud_risk_level": "MEDIUM", "top3_features": ["amount", "velocity_1h", "is_new_device"],
            "top3_values": [0.2, 0.1, 0.05], "explanation_verdict": "STABLE",
            "trust_score": 82.0, "trust_classification": "Trusted",
            "stability_score": 100.0, "faithfulness_score": 75.0,
            "consistency_score": 80.0, "flip_count": 0,
            "perturbed_runs": [{"run": 1, "top3": ["amount"]}],
            "true_hidden_drivers": [], "narrative": "Synthetic example.",
            "model_provenance": "synthetic",
        }
        with tempfile.TemporaryDirectory() as folder:
            try:
                main._frontend_bundle = FakeBundle()
                main.ledger = AuditLedger(Path(folder) / "audit.db")
                main.API_KEY = ""
                with patch("engine.frontend_ml.model_row", return_value={"amount": 1500}), patch(
                    "engine.frontend_ml.audit_frontend_row", return_value=fake_audit.copy()
                ):
                    status, report = call_app(
                        "/api/v1/transactions/audit", method="POST", body=transaction,
                    )
                self.assertEqual(status, 200, report)
                self.assertEqual(report["trust_score"], 82.0)
                self.assertEqual(report["top3_features"][0], "amount")
                self.assertTrue(report["audit_hash"])
                self.assertEqual(main.ledger.verify_chain(), (True, 1))
            finally:
                main._frontend_bundle = original_bundle
                main.ledger = original_ledger
                main.API_KEY = original_key

    def test_creditcard_bulk_scan_scores_pca_rows_without_label_input(self):
        class FakeBundle:
            metadata = {"training": {"data_classification": "synthetic"}}

            def predict_scores(self, rows):
                assert "Class" not in rows[0]
                return [0.8 for _ in rows]

            def anomaly_scores(self, rows):
                return [-0.1 for _ in rows]

        original_bundle = main._creditcard_bundle
        original_key = main.API_KEY
        row = {"Time": 42.0, "Amount": 5.5, **{f"V{i}": 0.0 for i in range(1, 29)}}
        try:
            main._creditcard_bundle = FakeBundle()
            main.API_KEY = ""
            status, report = call_app(
                "/api/v2/transactions/scan", method="POST",
                body={"transactions": [{**row, "Class": 1}]},
            )
            self.assertEqual(status, 422, report)
            status, report = call_app(
                "/api/v2/transactions/scan", method="POST",
                body={"transactions": [row]},
            )
            self.assertEqual(status, 200, report)
            self.assertEqual(report["transactions_analyzed"], 1)
            self.assertEqual(report["flagged_count"], 1)
            self.assertEqual(report["alerts"][0]["fraud_score"], 0.8)
            self.assertEqual(report["alerts"][0]["anomaly_score"], -0.1)
            self.assertNotIn("Class", report["alerts"][0])
        finally:
            main._creditcard_bundle = original_bundle
            main.API_KEY = original_key

    def test_fraud_twin_endpoint_returns_measured_graph_aware_scenarios(self):
        original_key = main.API_KEY
        main.API_KEY = ""
        try:
            status, report = call_app(
                "/api/v1/fraud-twin/simulate", method="POST",
                body={"seed": 42, "split_parts": 4},
            )
            self.assertEqual(status, 200, report)
            self.assertEqual(report["report_type"], "synthetic_fraud_twin")
            attacks = {attack["strategy"]: attack for attack in report["attacks"]}
            self.assertEqual(len(attacks), 4)
            coordinated = attacks["coordinated_account"]
            self.assertTrue(coordinated["graph_intelligence"]["modified"]["target_flagged"])
            self.assertIsInstance(coordinated["evasion_success"], bool)
        finally:
            main.API_KEY = original_key

    def test_full_assessment_is_persisted_and_report_can_be_retrieved(self):
        original_ledger = main.ledger
        original_key = main.API_KEY
        with tempfile.TemporaryDirectory() as folder:
            try:
                main.ledger = AuditLedger(Path(folder) / "assessment.db")
                main.API_KEY = ""
                status, assessment = call_app(
                    "/api/v1/intelligence/assess", method="POST", body=assessment_payload(),
                )
                self.assertEqual(status, 200, assessment)
                self.assertEqual(assessment["model_provenance"], "synthetic")
                self.assertIn("not validated", assessment["disclaimer"])
                self.assertGreaterEqual(assessment["fraud_probability"], 0)
                self.assertIn("anomaly_score", assessment)
                self.assertIn("indicators", assessment["graph_intelligence"])
                self.assertTrue(assessment["behavioural_intelligence"]["analysis_available"])
                self.assertIn("trust_score", assessment["trust_verification"])
                self.assertTrue(assessment["audit_hash"])
                self.assertTrue(main.ledger.verify_chain()[0])

                report_status, report = call_app(
                    f"/api/v1/forensic/{assessment['txn_id']}"
                )
                self.assertEqual(report_status, 200, report)
                self.assertEqual(report["record_hash"], assessment["audit_hash"])
                self.assertIn("graph_intelligence", report["report"])
                self.assertIn("behavioural_intelligence", report["report"])
            finally:
                main.ledger = original_ledger
                main.API_KEY = original_key

    def test_naive_current_timestamp_is_rejected(self):
        payload = assessment_payload()
        payload["occurred_at"] = "2026-10-20T12:00:00"
        status, response = call_app(
            "/api/v1/intelligence/assess", method="POST", body=payload,
        )
        self.assertEqual(status, 422)
        self.assertIn("detail", response)

    def test_dedicated_graph_and_behaviour_routes_validate_and_return_signals(self):
        original_key = main.API_KEY
        main.API_KEY = ""
        try:
            transfers = [
                {
                    "txn_id": f"edge-{index}", "account_id": f"account-{index}",
                    "beneficiary_account_id": "shared-receiver", "amount": 25.0,
                    "occurred_at": f"2026-10-{10 + index:02d}T12:00:00+00:00",
                }
                for index in range(1, 4)
            ]
            graph_status, graph = call_app(
                "/api/v1/intelligence/graph", method="POST",
                body={"transfers": transfers, "target_transaction_id": "edge-1"},
            )
            self.assertEqual(graph_status, 200, graph)
            self.assertTrue(any(item["indicator"] == "common_receiver" for item in graph["indicators"]))
            self.assertNotIn("shared-receiver", repr(graph))

            behaviour_status, behaviour = call_app(
                "/api/v1/intelligence/behaviour", method="POST",
                body={"current": transfers[0], "history": transfers[1:]},
            )
            self.assertEqual(behaviour_status, 200, behaviour)
            self.assertFalse(behaviour["analysis_available"])
        finally:
            main.API_KEY = original_key


    def test_unified_assessment_includes_quantum_trust_and_audits_it(self):
        original_ledger = main.ledger
        original_key = main.API_KEY
        expected_quantum = {
            "engine": "QuantumTrust Engine",
            "assessment_state": "ELEVATED_REVIEW",
            "quantum_hardware_used": False,
            "quantum_era_cryptographic_risk": {
                "assessment_status": "ASSESSED",
                "readiness": "MIGRATION_REQUIRED",
                "quantum_era_exposure_score": 100.0,
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            try:
                main.ledger = AuditLedger(Path(folder) / "quantum-audit.db")
                main.API_KEY = ""
                payload = assessment_payload()
                payload["security_context"] = {
                    "key_exchange_algorithm": "ECDHE",
                    "signature_algorithm": "RSA-2048",
                    "symmetric_algorithm": "AES-256",
                }
                with patch(
                    "engine.assessment.assess_quantum_trust",
                    return_value=expected_quantum,
                ) as quantum_assess:
                    status, report = call_app(
                        "/api/v1/intelligence/assess", method="POST", body=payload,
                    )
                self.assertEqual(status, 200, report)
                self.assertEqual(report["quantum_trust"], expected_quantum)
                quantum_assess.assert_called_once()
                self.assertEqual(
                    quantum_assess.call_args.args[2],
                    payload["security_context"],
                )
                self.assertTrue(report["audit_hash"])
                self.assertTrue(main.ledger.verify_chain()[0])
                forensic_status, forensic = call_app(
                    f"/api/v1/forensic/{report['txn_id']}"
                )
                self.assertEqual(forensic_status, 200, forensic)
                self.assertEqual(
                    forensic["report"]["quantum_trust"],
                    expected_quantum,
                )
            finally:
                main.ledger = original_ledger
                main.API_KEY = original_key


if __name__ == "__main__":
    unittest.main(verbosity=2)
