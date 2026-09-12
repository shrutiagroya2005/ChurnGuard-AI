"""
ChurnGuard AI — Step 1: Data Loading & Cleaning
=================================================
WHY THIS STEP EXISTS:
Raw data is never model-ready. Here we fix data types, handle missing
values, and encode the target label. Skipping this causes silent bugs
later (e.g., TotalCharges loaded as text means every downstream
calculation on it will fail or be wrong).
"""

import pandas as pd

def load_and_clean(path="data/telco_churn.csv"):
    df = pd.read_csv(path)

    # --- Fix TotalCharges: it's loaded as text because a few rows have
    # blank strings instead of numbers (new customers with 0 tenure).
    # WHY: pandas can't do math on text. We coerce to numeric and fill
    # the resulting NaNs with 0, since these are customers who haven't
    # been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --- Drop the customer ID column for modeling
    # WHY: an ID is a unique identifier with zero predictive value.
    # Leaving it in risks the model "memorizing" IDs instead of learning
    # real patterns (a subtle form of overfitting).
    df = df.drop(columns=["customerID"])

    # --- Encode the target label as 0/1
    # WHY: ML models need numeric targets, not "Yes"/"No" strings.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


if __name__ == "__main__":
    df = load_and_clean()
    print("Shape:", df.shape)
    print("\nChurn rate:", round(df["Churn"].mean() * 100, 2), "%")
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\nSample:\n", df.head(3))
