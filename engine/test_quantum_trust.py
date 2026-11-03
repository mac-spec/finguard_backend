"""Tests for the classical QuantumTrust Engine signals."""
import unittest

from engine.quantum_trust import assess_quantum_trust, _crypto_posture


class QuantumTrustTests(unittest.TestCase):
    def test_missing_crypto_inventory_is_unknown_not_zero_risk(self):
        result = _crypto_posture(None)
        self.assertEqual(result["assessment_status"], "NOT_ASSESSED")
        self.assertIsNone(result["quantum_era_exposure_score"])

    def test_classical_public_key_inventory_requires_migration(self):
        result = _crypto_posture({
            "key_exchange_algorithm": "ECDHE",
            "signature_algorithm": "RSA-2048",
            "symmetric_algorithm": "AES-256",
        })
        self.assertEqual(result["readiness"], "MIGRATION_REQUIRED")
        self.assertEqual(result["quantum_era_exposure_score"], 66.67)

    def test_pqc_candidate_inventory_is_not_claimed_as_verified(self):
        result = _crypto_posture({
            "key_exchange_algorithm": "ML-KEM-768",
            "signature_algorithm": "ML-DSA-65",
            "symmetric_algorithm": "AES-256",
        })
        self.assertEqual(result["readiness"], "PQC_CANDIDATE_PRESENT")
        self.assertTrue(any("verify implementation" in item for item in result["findings"]))

    def test_invalid_ensemble_size_is_rejected(self):
        with self.assertRaises(ValueError):
            assess_quantum_trust({"txn_id": "test"}, 0.2, ensemble_runs=2)


if __name__ == "__main__":
    unittest.main()
