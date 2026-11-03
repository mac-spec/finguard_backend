"""Quantum-inspired transaction assurance; deliberately uses classical computation.

This module samples a bounded ensemble of plausible transaction perturbations.
It does not emulate quantum hardware, quantum mechanics, or a quantum attack.
Scores are heuristic review signals and require validation on representative data.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from engine.perturb import generate_perturbations
from engine.simulator_and_model import predict_score

ENGINE_VERSION = "1.0.0"
ENSEMBLE_RUNS = 24
PQC_KEX = ("ML-KEM", "KYBER")
PQC_SIGNATURES = ("ML-DSA", "DILITHIUM", "SLH-DSA", "SPHINCS")
CLASSICAL_KEX = ("RSA", "DH", "DIFFIE-HELLMAN", "ECDH", "ECDHE")
CLASSICAL_SIGNATURES = ("RSA", "ECDSA", "DSA", "ED25519", "ED448")


def _normalise_algorithm(value: str | None) -> str:
    return (value or "").strip().upper().replace("_", "-").replace(" ", "-")


def _crypto_posture(context: dict[str, Any] | None) -> dict[str, Any]:
    if not context:
        return {
            "assessment_status": "NOT_ASSESSED",
            "quantum_era_exposure_score": None,
            "readiness": "UNKNOWN",
            "findings": ["Cryptographic inventory was not supplied; no quantum-readiness claim can be made."],
            "disclaimer": "This is an inventory-based migration signal, not a prediction of when cryptographically relevant quantum computers will exist.",
        }

    kex = _normalise_algorithm(context.get("key_exchange_algorithm"))
    signature = _normalise_algorithm(context.get("signature_algorithm"))
    symmetric = _normalise_algorithm(context.get("symmetric_algorithm"))
    findings: list[str] = []
    score = 0.0
    assessed = 0

    if kex:
        assessed += 1
        if any(name in kex for name in PQC_KEX) or "HYBRID" in kex and any(name in kex for name in PQC_KEX):
            findings.append("Key exchange inventory identifies a post-quantum or hybrid candidate; verify implementation and standards conformance.")
        elif any(name in kex for name in CLASSICAL_KEX):
            score += 100
            findings.append("Classical public-key key exchange is exposed to a future sufficiently capable quantum adversary; plan a post-quantum migration.")
        else:
            score += 50
            findings.append("Key-exchange algorithm is unrecognised; manual cryptographic inventory is required.")
    if signature:
        assessed += 1
        if any(name in signature for name in PQC_SIGNATURES):
            findings.append("Signature inventory identifies a post-quantum candidate; verify implementation and standards conformance.")
        elif any(name in signature for name in CLASSICAL_SIGNATURES):
            score += 100
            findings.append("Classical public-key signatures may be vulnerable to future quantum-enabled forgery; plan migration and signature agility.")
        else:
            score += 50
            findings.append("Signature algorithm is unrecognised; manual cryptographic inventory is required.")
    if symmetric:
        assessed += 1
        if "AES-256" in symmetric or "AES256" in symmetric:
            findings.append("AES-256 is a comparatively strong symmetric baseline; maintain sound key management and monitor standards guidance.")
        elif "AES-128" in symmetric or "AES128" in symmetric or "3DES" in symmetric or "DES" == symmetric:
            score += 60
            findings.append("Symmetric algorithm/key size should be reviewed against long-term confidentiality requirements.")
        else:
            score += 35
            findings.append("Symmetric algorithm is not classified by this inventory heuristic; review its key size and lifecycle.")

    if assessed == 0:
        return {
            "assessment_status": "NOT_ASSESSED",
            "quantum_era_exposure_score": None,
            "readiness": "UNKNOWN",
            "findings": ["No recognised cryptographic inventory fields were supplied."],
            "disclaimer": "Inventory-only heuristic; this does not scan deployed cryptographic dependencies.",
        }

    exposure = round(min(100.0, score / assessed), 2)
    readiness = "PQC_CANDIDATE_PRESENT" if exposure == 0 else "MIGRATION_REQUIRED" if exposure >= 65 else "REVIEW_REQUIRED"
    return {
        "assessment_status": "ASSESSED",
        "quantum_era_exposure_score": exposure,
        "readiness": readiness,
        "inventory_coverage": assessed,
        "findings": findings,
        "disclaimer": "Algorithm-name heuristic only. It does not verify protocol negotiation, parameters, certificates, implementation quality, key custody, or deployed usage.",
    }


def assess_quantum_trust(
    transaction: dict[str, Any],
    baseline_probability: float,
    security_context: dict[str, Any] | None = None,
    *,
    ensemble_runs: int = ENSEMBLE_RUNS,
) -> dict[str, Any]:
    """Estimate model uncertainty and bounded evasion sensitivity using classical sampling."""
    if not 4 <= ensemble_runs <= 100:
        raise ValueError("ensemble_runs must be between 4 and 100.")
    baseline = float(np.clip(baseline_probability, 0.0, 1.0))
    row = pd.Series(transaction)
    seed_material = str(transaction.get("txn_id", "finguard-quantum-trust")).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
    candidates = generate_perturbations(row, n=ensemble_runs, seed=seed)
    scores = np.asarray(
        [float(np.clip(predict_score(candidate), 0.0, 1.0)) for candidate in candidates],
        dtype=float,
    )
    all_scores = np.concatenate(([baseline], scores))
    mean_score = float(np.mean(all_scores))
    variance = float(np.var(all_scores))
    std_dev = float(np.std(all_scores))
    p = float(np.clip(mean_score, 1e-9, 1 - 1e-9))
    binary_entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    max_delta = float(np.max(np.abs(scores - baseline))) if len(scores) else 0.0
    mean_delta = float(np.mean(np.abs(scores - baseline))) if len(scores) else 0.0
    score_flips = int(np.sum((scores >= 0.5) != (baseline >= 0.5)))
    flip_rate = score_flips / float(ensemble_runs)

    uncertainty = float(np.clip(100.0 * (0.55 * min(1.0, std_dev / 0.25) + 0.45 * binary_entropy), 0, 100))
    vulnerability = float(np.clip(
        100.0 * (0.45 * min(1.0, max_delta / 0.30)
                 + 0.30 * min(1.0, mean_delta / 0.15)
                 + 0.25 * flip_rate), 0, 100
    ))
    if uncertainty >= 65 or vulnerability >= 65:
        state = "HIGH_REVIEW"
    elif uncertainty >= 35 or vulnerability >= 35:
        state = "ELEVATED_REVIEW"
    else:
        state = "STABLE_UNDER_TESTED_PERTURBATIONS"

    crypto = _crypto_posture(security_context)
    return {
        "engine": "QuantumTrust Engine",
        "engine_version": ENGINE_VERSION,
        "computation": "classical_quantum_inspired_ensemble",
        "quantum_hardware_used": False,
        "assessment_state": state,
        "uncertainty": {
            "score": round(uncertainty, 2),
            "ensemble_mean_probability": round(mean_score, 6),
            "ensemble_standard_deviation": round(std_dev, 6),
            "ensemble_variance": round(variance, 8),
            "binary_decision_entropy_bits": round(binary_entropy, 6),
            "samples": ensemble_runs + 1,
        },
        "adversarial_vulnerability": {
            "score": round(vulnerability, 2),
            "max_probability_shift": round(max_delta, 6),
            "mean_absolute_probability_shift": round(mean_delta, 6),
            "decision_flip_count": score_flips,
            "decision_flip_rate": round(flip_rate, 4),
            "test_scope": "bounded perturbations of amount, hour, known-neighbour MCC and known merchant group",
        },
        "quantum_era_cryptographic_risk": crypto,
        "limitations": [
            "Quantum-inspired is a design metaphor for ensemble uncertainty and scenario sensitivity; no quantum algorithm or quantum computer is used.",
            "Perturbations are bounded heuristic tests, not a complete adversarial threat model or proof of robustness.",
            "The cryptographic readiness result is only as complete as the caller-supplied inventory.",
            "Scores are uncalibrated review indicators, not probabilities of an attack or quantum compromise.",
        ],
    }
