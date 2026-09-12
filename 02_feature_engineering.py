"""
ChurnGuard AI — Step 2: Feature Engineering
=============================================
WHY THIS STEP EXISTS:
Raw fields like "Contract type" or "tenure" are useful, but the real
predictive power comes from RATIOS and INTERACTIONS that capture
customer behavior patterns — not just single snapshots.
"""

import pandas as pd
from importlib import import_module

data_prep = import_module("01_data_prep")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Tenure buckets
    # WHY: churn risk isn't linear with tenure. New customers (0-6mo)
    # and customers right after a contract ends behave very differently
    # than someone who's been around 5 years. Bucketing lets tree models
    # split on these natural risk zones more easily.
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"],
    )

    # --- Average charge per month of tenure
    # WHY: a customer paying a lot relative to how long they've stayed
    # signals price sensitivity risk — different from raw MonthlyCharges.
    df["avg_charge_per_tenure"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # --- Services subscribed count
    # WHY: customers with more add-on services (security, backup, tech
    # support, streaming) are more "locked in" and historically churn
    # less. A single combined count is a stronger, less noisy signal
    # than 6 separate Yes/No columns.
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)

    # --- Contract risk flag
    # WHY: month-to-month contracts have dramatically higher churn than
    # 1-2 year contracts (no lock-in). This is one of the strongest
    # known churn signals in subscription businesses, so we make it
    # explicit rather than relying on the model to discover it from the
    # raw categorical.
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # --- High monthly charges flag (top 25%)
    # WHY: price is a top-3 churn driver in most subscription
    # businesses. Flagging the top quartile helps the model isolate
    # price-sensitive segments even after other features are controlled
    # for.
    threshold = df["MonthlyCharges"].quantile(0.75)
    df["is_high_charge"] = (df["MonthlyCharges"] >= threshold).astype(int)

    return df


def encode_for_model(df: pd.DataFrame):
    """One-hot encode categoricals. Returns (X, y)."""
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


if __name__ == "__main__":
    df = data_prep.load_and_clean()
    df = engineer_features(df)
    print("New columns added:")
    print([c for c in df.columns if c not in
           ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure"]][-5:])
    X, y = encode_for_model(df)
    print("\nFinal feature matrix shape:", X.shape)
