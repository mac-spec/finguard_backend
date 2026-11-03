"""Canonical transaction schema shared by models, validation, and audits."""

FEATURES = [
    "amount", "timestamp_hour", "mcc_code", "merchant_id",
    "account_age_days", "device_fingerprint_age",
    "historical_txn_count_30d", "card_age_days",
    "is_new_device", "foreign_location_mismatch", "velocity_1h",
    "risky_beneficiary_score", "is_recurring_subscription_window",
    "merchant_category_loyalty",
]
MANIPULABLE = ["amount", "timestamp_hour", "mcc_code", "merchant_id"]
IMMUTABLE = [
    "account_age_days", "device_fingerprint_age",
    "historical_txn_count_30d", "card_age_days",
]
HIGH_RISK = [
    "is_new_device", "foreign_location_mismatch",
    "velocity_1h", "risky_beneficiary_score",
]
PROXIES = ["is_recurring_subscription_window", "merchant_category_loyalty"]
