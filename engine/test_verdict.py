from engine.simulator_and_model import generate_normal_fraud_txn, generate_poisoned_fraud_txn
from engine.verdict import evaluate
from engine.schema import FEATURES, HIGH_RISK
from unittest.mock import patch

def main():
    outcomes = []
    for i in range(20):
        txn = generate_normal_fraud_txn() if i % 2 == 0 else generate_poisoned_fraud_txn()
        txn["txn_id"] = f"TEST-{i+1:02d}"
        result = evaluate(txn, simulate_proxy_masking=(i % 2 == 1))
        outcomes.append(result)
        print(f"{result['txn_id']:<12} | {result['fraud_score']:.4f} | "
              f"{result['flip_count']} | {result['trust_score']:.1f} | "
              f"{result['verdict']:<13} | {result['latency_ms']:.2f}")
    verdicts = {x["verdict"] for x in outcomes}
    assert "STABLE" in verdicts, "Expected at least one STABLE verdict."
    assert "UNTRUSTWORTHY" in verdicts, (
        "No UNTRUSTWORTHY example was produced. Review model/perturbation calibration."
    )
    expected_proxies = {
        "mcc_code", "is_recurring_subscription_window", "merchant_category_loyalty"
    }
    for result in outcomes[1::2]:
        assert set(result["original_top3"]) == expected_proxies
        assert result["flip_count"] > 1
        assert {"is_new_device", "foreign_location_mismatch"}.issubset(
            set(result["true_hidden_drivers"])
        )
        assert result["narrative"] is None
        assert result["trust_classification"] == "Manipulated"


def test_stable_but_manipulated_explanation_is_withheld():
    txn = generate_normal_fraud_txn()
    fake_explanation = {
        "txn_id": "withhold-test", "fraud_score": .8,
        "fraud_risk_level": "HIGH", "top3_features": FEATURES[:3],
        "top3_values": [0.0, 0.0, 0.0],
        "shap_values_full": {feature: 0.0 for feature in FEATURES},
    }

    def fake_ablation(_row, features, baseline=None):
        if list(features) == FEATURES[:2]:
            return {feature: 0.0 for feature in features}
        return {feature: (0.5 if feature == HIGH_RISK[0] else 0.0) for feature in features}

    with patch("engine.verdict.explain", return_value=fake_explanation), \
            patch("engine.verdict.generate_perturbations", return_value=[txn.copy() for _ in range(5)]), \
            patch("engine.verdict._ablation", side_effect=fake_ablation):
        result = evaluate(txn)
    assert result["verdict"] == "STABLE"
    assert result["trust_classification"] == "Manipulated"
    assert result["narrative"] is None

if __name__ == "__main__":
    print("txn_id       | fraud_score | flips | trust | verdict       | latency_ms")
    main()
    test_stable_but_manipulated_explanation_is_withheld()
