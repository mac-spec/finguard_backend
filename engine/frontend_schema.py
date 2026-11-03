"""Feature mapping for the browser's normalized transaction records."""

from engine.schema import FEATURES

FRONTEND_MODEL_FEATURES = [
    *FEATURES, "timestamp_known", "direction_code", "method_code",
]
FRONTEND_CATEGORICAL_FEATURES = [
    "mcc_code", "merchant_id", "direction_code", "method_code",
]

CATEGORY_CODES = {
    "food": 0, "groceries": 1, "travel": 2, "shopping": 3, "bills": 4,
    "health": 5, "education": 6, "entertainment": 7, "rent": 8,
    "salary": 9, "transfer": 10, "other": 11,
}
CATEGORY_MCC_CODES = {
    "food": 5812, "groceries": 5411, "travel": 4111, "shopping": 5999,
    "bills": 4900, "health": 5912, "education": 8211,
    "entertainment": 7832, "rent": 6513, "salary": 6011,
    "transfer": 4829, "other": 5999,
}
METHOD_CODES = {
    "google pay": 0, "phonepe": 1, "paytm": 2, "bhim upi": 3,
    "bank transfers": 4, "multiple payment apps": 5, "upi": 6, "unknown": 7,
}

SIMULATED_CONTEXT_FEATURES = [
    "account_age_days", "device_fingerprint_age", "historical_txn_count_30d",
    "card_age_days", "is_new_device", "foreign_location_mismatch",
    "velocity_1h", "risky_beneficiary_score",
    "is_recurring_subscription_window", "merchant_category_loyalty",
]
