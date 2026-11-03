"""Unit checks for Fraud Twin's controlled synthetic attack strategies."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.fraud_twin import (
    behaviour_mimicry_attack,
    coordinated_account_attack,
    run_fraud_twin,
    transaction_splitting_attack,
    velocity_manipulation_attack,
)
from engine.schema import FEATURES, IMMUTABLE, MANIPULABLE


def sample_transaction() -> dict:
    return {
        "txn_id": "synthetic-sample",
        "amount": 2000.0,
        "timestamp_hour": 2,
        "mcc_code": 6011,
        "merchant_id": 129,
        "account_age_days": 900,
        "device_fingerprint_age": 700,
        "historical_txn_count_30d": 15,
        "card_age_days": 800,
        "is_new_device": 1,
        "foreign_location_mismatch": 1,
        "velocity_1h": 3.0,
        "risky_beneficiary_score": 0.88,
        "is_recurring_subscription_window": 0,
        "merchant_category_loyalty": 0.15,
    }


class FraudTwinStrategyTests(unittest.TestCase):
    def setUp(self):
        self.source = sample_transaction()

    def test_transaction_splitting_preserves_total_and_history(self):
        result = transaction_splitting_attack(self.source, parts=4)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(sum(row["amount"] for row in result), self.source["amount"])
        self.assertTrue(all(0 < row["amount"] < self.source["amount"] for row in result))
        for row in result:
            for feature in IMMUTABLE:
                self.assertEqual(row[feature], self.source[feature])
        self.assertEqual(self.source["amount"], 2000.0, "source record must remain unchanged")

    def test_splitting_rejects_invalid_parts_and_too_small_amount(self):
        with self.assertRaises(ValueError):
            transaction_splitting_attack(self.source, parts=1)
        tiny = {**self.source, "amount": 0.02}
        with self.assertRaises(ValueError):
            transaction_splitting_attack(tiny, parts=4)

    def test_behaviour_mimicry_only_changes_allowed_fields(self):
        normal = {
            "amount": 800.0, "timestamp_hour": 13,
            "mcc_code": 5411, "merchant_id": 101,
            "account_age_days": 10, "historical_txn_count_30d": 1,
        }
        result = behaviour_mimicry_attack(self.source, normal)
        self.assertEqual(
            [name for name in FEATURES if result[name] != self.source[name]],
            [name for name in MANIPULABLE if result[name] != self.source[name]],
        )
        for feature in IMMUTABLE:
            self.assertEqual(result[feature], self.source[feature])
        self.assertEqual(self.source["amount"], 2000.0)

    def test_velocity_attack_spreads_hours_and_keeps_immutable_history(self):
        burst = [{**self.source, "txn_id": f"burst-{index}"} for index in range(4)]
        result = velocity_manipulation_attack(burst, spacing_hours=2)
        self.assertEqual([row["timestamp_hour"] for row in result], [2, 4, 6, 8])
        for before, after in zip(burst, result):
            for feature in IMMUTABLE:
                self.assertEqual(after[feature], before[feature])
        self.assertTrue(all(row["amount"] == self.source["amount"] for row in result))
        with self.assertRaises(ValueError):
            velocity_manipulation_attack(burst * 7, spacing_hours=2)

    def test_coordinated_attack_builds_synthetic_ring_without_changing_model_features(self):
        result = coordinated_account_attack(self.source, account_count=4)
        self.assertEqual(len(result), 4)
        accounts = {row["account_id"] for row in result}
        self.assertEqual(len(accounts), 4)
        for index, row in enumerate(result):
            self.assertEqual(row["beneficiary_account_id"], result[(index + 1) % 4]["account_id"])
            for feature in FEATURES:
                self.assertEqual(row[feature], self.source[feature])
            for feature in IMMUTABLE:
                self.assertEqual(row[feature], self.source[feature])

    def test_evasion_success_depends_on_measured_model_output(self):
        # This fake scorer is test-only: it makes the decision rule easy to
        # verify. Production reports call the currently loaded model functions.
        def score(row):
            return 0.8 if row["amount"] in {500.0, 2000.0} else 0.1

        with patch("engine.fraud_twin.predict_score", side_effect=score), \
                patch("engine.fraud_twin.anomaly_score", return_value=1.0):
            report = run_fraud_twin(self.source, seed=17, split_parts=4)
        attacks = {item["strategy"]: item for item in report["attacks"]}
        self.assertTrue(attacks["transaction_splitting"]["original_detected"])
        self.assertFalse(attacks["transaction_splitting"]["evasion_success"])
        self.assertTrue(attacks["behaviour_mimicry"]["evasion_success"])
        self.assertFalse(attacks["velocity_manipulation"]["evasion_success"])
        self.assertFalse(attacks["coordinated_account"]["evasion_success"])
        self.assertTrue(
            attacks["coordinated_account"]["graph_intelligence"]["modified"]["target_flagged"]
        )
        self.assertEqual(report["seed"], 17)


if __name__ == "__main__":
    unittest.main(verbosity=2)
