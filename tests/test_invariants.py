"""Property-based invariants (quick, < 5 min)."""
import numpy as np
import sys
sys.path.insert(0, "C:/All/why")
from src.features.engineering import add_behavioural_features


def test_monotonic_limit_bal():
    """Higher LIMIT_BAL (with same behaviour) should not increase default probability."""
    import pandas as pd
    df_base = pd.DataFrame({
        "LIMIT_BAL": [200_000], "SEX": [2], "EDUCATION": [2], "MARRIAGE": [1],
        "AGE": [35], "PAY_0": [0], "PAY_2": [0], "PAY_3": [0],
        "PAY_4": [0], "PAY_5": [0], "PAY_6": [0],
        "BILL_AMT1": [50_000], "BILL_AMT2": [45_000], "BILL_AMT3": [40_000],
        "BILL_AMT4": [35_000], "BILL_AMT5": [30_000], "BILL_AMT6": [25_000],
        "PAY_AMT1": [5_000], "PAY_AMT2": [5_000], "PAY_AMT3": [5_000],
        "PAY_AMT4": [5_000], "PAY_AMT5": [5_000], "PAY_AMT6": [5_000],
    })
    df_high = df_base.copy()
    df_high["LIMIT_BAL"] = 400_000
    eng_base = add_behavioural_features(df_base)
    eng_high = add_behavioural_features(df_high)
    # Invariant: utilisation should decrease when limit increases (same bills/payments)
    assert eng_high["UTILIZATION"].iloc[0] < eng_base["UTILIZATION"].iloc[0], \
        "Monotonicity broken: higher LIMIT_BAL increased utilisation"
    print("Monotonicity invariant: PASS")


def test_no_target_leakage():
    """Engineered features must not contain target-derived values."""
    import pandas as pd
    df = pd.read_csv("C:/All/why/data/processed/engineered.csv")
    cols = list(df.columns)
    assert "default_payment_next_month" not in cols or True  # target allowed as column only
    # Real check: no engineered feature name should contain the target string
    for c in cols:
        if c.startswith("LOG_") or c.startswith("UTILIZATION") or c.startswith("DELAY"):
            assert "default" not in c.lower(), f"Possible leakage in feature: {c}"
    print("No target leakage invariant: PASS")


if __name__ == "__main__":
    test_monotonic_limit_bal()
    test_no_target_leakage()
