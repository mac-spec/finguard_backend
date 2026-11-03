"""Temporal account-behaviour checks with strict past-only history selection."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any

MIN_HISTORY = 5
BASELINE_DAYS = 30
AMOUNT_ROBUST_Z_THRESHOLD = 3.5
BURST_PRIOR_TXNS_PER_HOUR = 3
SHORT_GAP_MINUTES = 5
UNUSUAL_HOUR_MIN_HISTORY = 10
DAILY_VOLUME_MULTIPLIER = 3.0


def _time(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError("transaction timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _robust_z(value: float, values: list[float]) -> float:
    center = median(values)
    mad = median([abs(item - center) for item in values])
    if mad > 1e-12:
        return 0.6745 * (value - center) / mad
    deviations = [abs(item - center) for item in values]
    scale = median(deviations) * 1.4826
    if scale > 1e-12:
        return (value - center) / scale
    # A constant baseline makes any material change stand out; avoid division by 0.
    return 0.0 if abs(value - center) <= 1e-9 else float("inf")


def analyze_behaviour(
    current: dict[str, Any], history: list[dict[str, Any]], *,
    min_history: int = MIN_HISTORY,
) -> dict[str, Any]:
    """Compare one account's current transaction with up to 30 days of prior data.

    Every historical row at or after the current timestamp is excluded, even if
    a caller accidentally includes it in `history`.
    """
    if min_history < 1:
        raise ValueError("min_history must be positive.")
    current_time = _time(current["occurred_at"])
    account = str(current["account_id"])
    start = current_time - timedelta(days=BASELINE_DAYS)
    eligible = []
    ignored_future = 0
    for row in history:
        if str(row["account_id"]) != account:
            continue
        event_time = _time(row["occurred_at"])
        if event_time >= current_time:
            ignored_future += 1
            continue
        if event_time >= start:
            eligible.append({**row, "_parsed_time": event_time, "amount": float(row["amount"])})
    eligible.sort(key=lambda row: row["_parsed_time"])
    amounts = [row["amount"] for row in eligible]
    recipient_ids = [str(row["beneficiary_account_id"]) for row in eligible]
    indicators: list[dict[str, Any]] = []

    def flag(name: str, severity: str, explanation: str, **details: Any) -> None:
        indicators.append({
            "indicator": name, "severity": severity,
            "explanation": explanation, "details": details,
        })

    available = len(eligible) >= min_history
    if available:
        amount = float(current["amount"])
        amount_z = _robust_z(amount, amounts)
        if abs(amount_z) >= AMOUNT_ROBUST_Z_THRESHOLD:
            flag(
                "unusual_amount", "high" if abs(amount_z) >= 6 else "medium",
                "The transaction amount is far from this account's recent median amount.",
                current_amount=round(amount, 2), median_amount=round(median(amounts), 2),
                robust_z=(round(amount_z, 2) if amount_z != float("inf") else "infinite"),
                threshold=AMOUNT_ROBUST_Z_THRESHOLD,
            )

        previous_hour = [
            row for row in eligible
            if timedelta(0) < current_time - row["_parsed_time"] <= timedelta(hours=1)
        ]
        span_days = max(
            (eligible[-1]["_parsed_time"] - eligible[0]["_parsed_time"]).total_seconds() / 86400,
            1.0,
        )
        expected_hourly = len(eligible) / (span_days * 24)
        burst_threshold = max(BURST_PRIOR_TXNS_PER_HOUR, int(expected_hourly * 4 + 0.999))
        if len(previous_hour) >= burst_threshold:
            flag(
                "frequency_burst", "high",
                "Several transactions occurred in the hour before this one, above the account's baseline rate.",
                prior_hour_count=len(previous_hour), expected_hourly_rate=round(expected_hourly, 3),
                threshold=burst_threshold,
            )

        if eligible:
            gap_minutes = (current_time - eligible[-1]["_parsed_time"]).total_seconds() / 60
            gaps = [
                (later["_parsed_time"] - earlier["_parsed_time"]).total_seconds() / 60
                for earlier, later in zip(eligible, eligible[1:])
                if later["_parsed_time"] > earlier["_parsed_time"]
            ]
            typical_gap = median(gaps) if gaps else None
            if gap_minutes < SHORT_GAP_MINUTES and (
                typical_gap is None or gap_minutes < typical_gap * 0.1
            ):
                flag(
                    "unusually_short_interval", "medium",
                    "The time since the previous payment is unusually short for this account.",
                    minutes_since_previous=round(gap_minutes, 2),
                    baseline_median_gap_minutes=(round(typical_gap, 2) if typical_gap is not None else None),
                )

        if len(eligible) >= UNUSUAL_HOUR_MIN_HISTORY:
            hour_counts = Counter(row["_parsed_time"].hour for row in eligible)
            if hour_counts[current_time.hour] == 0:
                flag(
                    "unusual_transaction_hour", "low",
                    "This hour has not appeared in the account's recent transaction history.",
                    current_hour=current_time.hour, history_rows=len(eligible),
                )

        receiver = str(current["beneficiary_account_id"])
        if receiver not in set(recipient_ids):
            flag(
                "new_recipient", "medium",
                "The recipient does not appear in the account's recent recipient history.",
                distinct_recipients=len(set(recipient_ids)),
            )

        daily_totals: dict[str, float] = defaultdict(float)
        for row in eligible:
            daily_totals[row["_parsed_time"].date().isoformat()] += row["amount"]
        if len(daily_totals) >= 5:
            typical_daily = median(list(daily_totals.values()))
            if typical_daily > 0 and amount >= typical_daily * DAILY_VOLUME_MULTIPLIER:
                flag(
                    "daily_volume_shift", "high",
                    "This payment alone is several times the account's typical daily transaction volume.",
                    amount=round(amount, 2), median_daily_volume=round(typical_daily, 2),
                    multiplier=DAILY_VOLUME_MULTIPLIER,
                )

    return {
        "analysis_available": available,
        "history_rows_used": len(eligible),
        "minimum_history_required": min_history,
        "ignored_future_or_current_rows": ignored_future,
        "baseline_window_days": BASELINE_DAYS,
        "baseline": ({
            "median_amount": round(median(amounts), 2),
            "mean_amount": round(sum(amounts) / len(amounts), 2),
            "distinct_recipients": len(set(recipient_ids)),
            "median_gap_minutes": round(median([
                (later["_parsed_time"] - earlier["_parsed_time"]).total_seconds() / 60
                for earlier, later in zip(eligible, eligible[1:])
                if later["_parsed_time"] > earlier["_parsed_time"]
            ]), 2) if len(eligible) > 1 and any(
                later["_parsed_time"] > earlier["_parsed_time"]
                for earlier, later in zip(eligible, eligible[1:])
            ) else None,
        } if eligible else None),
        "indicators": indicators,
        "interpretation": (
            "Behaviour indicators compare this transaction only with earlier same-account records; they are not fraud labels."
            if available else
            "Insufficient history: no behavioural risk indicators were emitted."
        ),
    }
