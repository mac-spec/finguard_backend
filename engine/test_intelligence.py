"""Synthetic unit tests for relationship and behavioural intelligence."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from engine.behaviour_intelligence import analyze_behaviour
from engine.graph_intelligence import analyze_graph


def event(txn_id, source, target, amount, time):
    return {
        "txn_id": txn_id, "account_id": source,
        "beneficiary_account_id": target, "amount": amount,
        "occurred_at": time,
    }


class GraphIntelligenceTests(unittest.TestCase):
    def test_normal_sparse_network_has_no_flags(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        transfers = [
            event("n1", "acct-A", "acct-B", 25, start),
            event("n2", "acct-B", "acct-C", 40, start + timedelta(days=2)),
        ]
        result = analyze_graph(transfers, target_transaction_id="n1")
        self.assertFalse(result["target_flagged"])
        self.assertEqual(result["indicators"], [])

    def test_cycle_repeated_pair_rapid_flow_and_common_receiver_are_explained(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        transfers = [
            event("ab1", "acct-A", "acct-B", 100, start),
            event("ab2", "acct-A", "acct-B", 100, start + timedelta(minutes=1)),
            event("ab3", "acct-A", "acct-B", 100, start + timedelta(minutes=2)),
            event("bc", "acct-B", "acct-C", 90, start + timedelta(minutes=3)),
            event("ca", "acct-C", "acct-A", 80, start + timedelta(minutes=4)),
            event("da", "acct-D", "acct-A", 20, start + timedelta(minutes=5)),
            event("ea", "acct-E", "acct-A", 20, start + timedelta(minutes=6)),
        ]
        result = analyze_graph(transfers, target_transaction_id="bc")
        kinds = {item["indicator"] for item in result["indicators"]}
        self.assertIn("repeated_account_pair", kinds)
        self.assertIn("circular_flow", kinds)
        self.assertIn("rapid_fund_movement", kinds)
        self.assertIn("common_receiver", kinds)
        self.assertTrue(result["target_flagged"])
        encoded = repr(result)
        self.assertNotIn("acct-A", encoded, "raw account IDs must not be returned")

    def test_dense_cluster_and_high_connectivity(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        star = [
            event("a", "hub", "a", 10, start),
            event("b", "hub", "b", 10, start),
            event("c", "hub", "c", 10, start),
            event("d", "hub", "d", 10, start),
            event("e", "hub", "e", 10, start),
        ]
        kinds = {item["indicator"] for item in analyze_graph(star)["indicators"]}
        self.assertIn("unusual_connectivity", kinds)

        triangle = [
            event("ab", "A", "B", 10, start),
            event("bc", "B", "C", 10, start),
            event("ca", "C", "A", 10, start),
        ]
        kinds = {item["indicator"] for item in analyze_graph(triangle)["indicators"]}
        self.assertIn("dense_account_cluster", kinds)


class BehaviourIntelligenceTests(unittest.TestCase):
    def test_normal_behaviour_is_not_flagged(self):
        start = datetime(2026, 2, 1, 12, tzinfo=timezone.utc)
        history = [
            event(f"h{i}", "user-1", "shop-1", 100 + (i % 2), start + timedelta(days=i))
            for i in range(12)
        ]
        current = event("current", "user-1", "shop-1", 101, start + timedelta(days=13))
        result = analyze_behaviour(current, history)
        self.assertTrue(result["analysis_available"])
        self.assertEqual(result["indicators"], [])

    def test_spike_is_explained_and_future_records_are_ignored(self):
        start = datetime(2026, 2, 1, 12, tzinfo=timezone.utc)
        history = [
            event(f"h{i}", "user-1", "shop-1", 100, start + timedelta(days=i))
            for i in range(11)
        ]
        current_time = start + timedelta(days=11, minutes=1)
        history.append(event("h11", "user-1", "shop-1", 100, current_time - timedelta(minutes=1)))
        history.extend([
            event(f"burst-{index}", "user-1", "shop-1", 100,
                  current_time - timedelta(minutes=4 - index))
            for index in range(1, 4)
        ])
        current = event("current", "user-1", "new-shop", 3000, current_time)
        future = event("future", "user-1", "future-shop", 9999, current_time + timedelta(days=1))
        result = analyze_behaviour(current, history + [future])
        kinds = {item["indicator"] for item in result["indicators"]}
        self.assertIn("unusual_amount", kinds)
        self.assertIn("unusually_short_interval", kinds)
        self.assertIn("frequency_burst", kinds)
        self.assertIn("new_recipient", kinds)
        self.assertIn("daily_volume_shift", kinds)
        self.assertEqual(result["history_rows_used"], 15)
        self.assertEqual(result["ignored_future_or_current_rows"], 1)

    def test_insufficient_history_disables_behaviour_flags(self):
        now = datetime(2026, 2, 14, tzinfo=timezone.utc)
        history = [event(f"h{i}", "user-1", "shop", 10, now - timedelta(days=i + 1)) for i in range(3)]
        current = event("current", "user-1", "new", 10000, now)
        result = analyze_behaviour(current, history)
        self.assertFalse(result["analysis_available"])
        self.assertEqual(result["indicators"], [])

    def test_unseen_hour_requires_ten_history_rows(self):
        start = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)
        history = [
            event(f"h{i}", "user-1", "shop", 100, start + timedelta(days=i))
            for i in range(10)
        ]
        current = event("current", "user-1", "shop", 100, start + timedelta(days=11, hours=2))
        result = analyze_behaviour(current, history)
        self.assertIn("unusual_transaction_hour", {item["indicator"] for item in result["indicators"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
