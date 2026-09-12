"""
ChurnGuard AI — Step 5: Score a New Customer
===============================================
WHY THIS STEP EXISTS:
This is what "using" ChurnGuard actually looks like in practice: a
new customer record comes in, and you get back a risk score + reasons
— the same logic a scheduled job or API endpoint would run daily.
"""

import joblib
import pandas as pd
import shap
from importlib import import_module

feat_eng = import_module("02_feature_engineering")


def score_customer(customer_dict: dict):
    """
    customer_dict: a single customer's raw fields, same schema as the
    training CSV (minus customerID and Churn).
    """
    xgb = joblib.load("models/churnguard_xgb.joblib")
    feature_cols = joblib.load("models/feature_columns.joblib")

    # Build a 1-row dataframe from the input
    df = pd.DataFrame([customer_dict])

    # TotalCharges needs the same cleaning as training
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df = feat_eng.engineer_features(df)
    X = pd.get_dummies(df, drop_first=True)

    # WHY reindex: a single new customer won't have every possible
    # one-hot category present. We align to the exact training columns
    # and fill anything missing with 0, or the model will error out.
    X = X.reindex(columns=feature_cols, fill_value=0)

    risk = xgb.predict_proba(X)[:, 1][0]

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X)
    sv = pd.Series(shap_values.values[0], index=X.columns)
    top_reasons = sv.sort_values(ascending=False).head(3)

    print(f"Churn risk: {risk*100:.1f}%")
    print("Top risk drivers:")
    for feat, impact in top_reasons.items():
        print(f"  - {feat} = {X.iloc[0][feat]}  (impact: {impact:+.3f})")

    return risk, top_reasons


if __name__ == "__main__":
    # Example: a new, month-to-month, high-paying, short-tenure customer
    example_customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5,
        "TotalCharges": "191.0",
    }
    score_customer(example_customer)
