"""Explainable relationship indicators over supplied transfer history.

NetworkX is used only for graph structure. Indicators are review clues, never
fraud labels, and this module does not train a graph neural network.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import networkx as nx

MAX_CYCLE_LENGTH = 4
MAX_RAPID_MOVEMENT_MINUTES = 60
MIN_PAIR_FREQUENCY = 3
MIN_COMMON_SOURCES = 3
MIN_DENSE_CLUSTER_NODES = 3
MIN_DENSE_CLUSTER_DENSITY = 0.70
MIN_HIGH_CONNECTIVITY = 5
MAX_GRAPH_INDICATORS = 1000
MAX_CYCLE_SCANS = 10000
MAX_RAPID_PAIR_CHECKS = 20000
MAX_REFERENCES_PER_INDICATOR = 100

_pseudonym_secret = os.getenv("FINGUARD_GRAPH_PSEUDONYM_KEY", "").encode("utf-8")
if not _pseudonym_secret:
    # Ephemeral default prevents returning raw account IDs in local demos.
    _pseudonym_secret = os.urandom(32)


def _time(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError("transfer timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _account_ref(account_id: str) -> str:
    digest = hmac.new(_pseudonym_secret, account_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"acct-{digest[:12]}"


def analyze_graph(
    transfers: list[dict[str, Any]], *, target_transaction_id: str | None = None,
) -> dict[str, Any]:
    """Build a directed transfer graph and return transparent risk indicators.

    Each transfer needs `account_id`, `beneficiary_account_id`, `amount`,
    `occurred_at`, and `txn_id`. The result contains pseudonymous account refs,
    never the supplied account IDs.
    """
    graph = nx.MultiDiGraph()
    pair_edges: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    analysis_truncated = False
    for transfer in transfers:
        source = str(transfer["account_id"])
        target = str(transfer["beneficiary_account_id"])
        event = {
            "txn_id": str(transfer["txn_id"]),
            "source": source,
            "target": target,
            "amount": float(transfer["amount"]),
            "occurred_at": _time(transfer["occurred_at"]),
        }
        records.append(event)
        pair_edges[(source, target)].append(event)
        graph.add_edge(source, target, key=event["txn_id"], **event)

    indicators: list[dict[str, Any]] = []

    def add_indicator(kind: str, severity: str, explanation: str,
                      accounts: set[str], transaction_ids: set[str], **details: Any) -> None:
        nonlocal analysis_truncated
        if len(indicators) >= MAX_GRAPH_INDICATORS:
            analysis_truncated = True
            return
        ordered_transactions = sorted(transaction_ids)
        if target_transaction_id in transaction_ids:
            ordered_transactions.remove(target_transaction_id)
            ordered_transactions.insert(0, target_transaction_id)
        if len(ordered_transactions) > MAX_REFERENCES_PER_INDICATOR:
            details["transaction_refs_truncated"] = True
            ordered_transactions = ordered_transactions[:MAX_REFERENCES_PER_INDICATOR]
            analysis_truncated = True
        ordered_accounts = sorted(_account_ref(account) for account in accounts)
        if len(ordered_accounts) > MAX_REFERENCES_PER_INDICATOR:
            details["account_refs_truncated"] = True
            ordered_accounts = ordered_accounts[:MAX_REFERENCES_PER_INDICATOR]
            analysis_truncated = True
        indicators.append({
            "indicator": kind,
            "severity": severity,
            "explanation": explanation,
            "account_refs": ordered_accounts,
            "transaction_ids": ordered_transactions,
            "details": details,
        })

    # Repeated pair transfers: at least three in any rolling 24-hour period.
    for (source, target), events in pair_edges.items():
        ordered = sorted(events, key=lambda event: event["occurred_at"])
        left = 0
        for right, event in enumerate(ordered):
            while event["occurred_at"] - ordered[left]["occurred_at"] > timedelta(hours=24):
                left += 1
            window = ordered[left:right + 1]
            if len(window) >= MIN_PAIR_FREQUENCY:
                add_indicator(
                    "repeated_account_pair", "medium",
                    "The same directed account pair transferred funds at least three times within 24 hours.",
                    {source, target}, {item["txn_id"] for item in window},
                    transfer_count=len(window), window_hours=24,
                )
                break

    simple = nx.DiGraph()
    simple.add_nodes_from(graph.nodes)
    simple.add_edges_from((source, target) for source, target in pair_edges)
    cycles = nx.simple_cycles(simple, length_bound=MAX_CYCLE_LENGTH)
    cycle_scans = 0
    for cycle in cycles:
        cycle_scans += 1
        if cycle_scans > MAX_CYCLE_SCANS:
            analysis_truncated = True
            break
        if len(cycle) < 2:
            continue
        cycle_edges: set[str] = set()
        for index, source in enumerate(cycle):
            target = cycle[(index + 1) % len(cycle)]
            cycle_edges.update(event["txn_id"] for event in pair_edges[(source, target)])
        add_indicator(
            "circular_flow", "high",
            f"Transfers form a directed circular path involving {len(cycle)} accounts.",
            set(cycle), cycle_edges, cycle_length=len(cycle),
        )

    incoming: dict[str, set[str]] = defaultdict(set)
    incoming_txns: dict[str, set[str]] = defaultdict(set)
    for event in records:
        incoming[event["target"]].add(event["source"])
        incoming_txns[event["target"]].add(event["txn_id"])
    for receiver, sources in incoming.items():
        if len(sources) >= MIN_COMMON_SOURCES:
            add_indicator(
                "common_receiver", "high",
                "Several distinct accounts transferred to the same receiver.",
                {*sources, receiver}, incoming_txns[receiver],
                distinct_senders=len(sources),
            )

    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    undirected.add_edges_from((source, target) for source, target in pair_edges)
    for component in nx.connected_components(undirected):
        if len(component) < MIN_DENSE_CLUSTER_NODES:
            continue
        subgraph = undirected.subgraph(component)
        density = nx.density(subgraph)
        if density >= MIN_DENSE_CLUSTER_DENSITY:
            txn_ids = {
                event["txn_id"] for event in records
                if event["source"] in component and event["target"] in component
            }
            add_indicator(
                "dense_account_cluster", "medium",
                "A connected account group has unusually dense transfer links in this batch.",
                set(component), txn_ids, node_count=len(component), density=round(density, 3),
            )

    # Funds may move onward quickly; this is a timing pattern, not proof of layering.
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in records:
        outgoing[event["source"]].append(event)
    rapid_pair_checks = 0
    rapid_checks_exhausted = False
    for first in records:
        for second in outgoing.get(first["target"], []):
            rapid_pair_checks += 1
            if rapid_pair_checks > MAX_RAPID_PAIR_CHECKS:
                analysis_truncated = True
                rapid_checks_exhausted = True
                break
            minutes = (second["occurred_at"] - first["occurred_at"]).total_seconds() / 60
            if 0 <= minutes <= MAX_RAPID_MOVEMENT_MINUTES and second["amount"] <= first["amount"] * 1.05:
                add_indicator(
                    "rapid_fund_movement", "medium",
                    "A receiver passed on a similar or smaller amount within 60 minutes.",
                    {first["source"], first["target"], second["target"]},
                    {first["txn_id"], second["txn_id"]},
                    elapsed_minutes=round(minutes, 1),
                )
        if rapid_checks_exhausted:
            break

    distinct_degree = {
        node: len(set(graph.successors(node)) | set(graph.predecessors(node)))
        for node in graph.nodes
    }
    for account, degree in distinct_degree.items():
        if degree >= MIN_HIGH_CONNECTIVITY:
            txn_ids = {
                event["txn_id"] for event in records
                if event["source"] == account or event["target"] == account
            }
            add_indicator(
                "unusual_connectivity", "medium",
                "An account is connected to at least five distinct counterparties in this batch.",
                {account}, txn_ids, distinct_counterparties=degree,
            )

    target_indicators = [
        indicator for indicator in indicators
        if target_transaction_id and target_transaction_id in indicator["transaction_ids"]
    ]
    return {
        "analysis_available": len(records) >= 2,
        "transactions_analyzed": len(records),
        "accounts_analyzed": graph.number_of_nodes(),
        "analysis_truncated": analysis_truncated,
        "analysis_reason": (
            None if len(records) >= 2 else "At least two transfers are needed to examine relationships."
        ),
        "indicators": indicators,
        "target_transaction_id": target_transaction_id,
        "target_flagged": bool(target_indicators),
        "target_indicators": target_indicators,
        "interpretation": (
            "Graph indicators describe relationships for human review; they do not establish fraud."
        ),
    }
