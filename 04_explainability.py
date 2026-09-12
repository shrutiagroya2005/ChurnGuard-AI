"""
ChurnGuard AI — Step 4: Explainability (SHAP)
================================================
WHY THIS STEP EXISTS:
A retention team can't act on "the model says 82% risk." They need to
know WHY — e.g., "usage dropped" needs a different fix than "billing
complaints." SHAP breaks each prediction down into the contribution of
every feature, in the customer's actual units.
"""

import joblib
import shap
import pandas as pd
from importlib import import_module

data_prep = import_module("01_data_prep")
feat_eng = import_module("02_feature_engineering")


def explain_top_risk_customers(n=5):
    # --- Load saved model + rebuild the feature matrix
    xgb = joblib.load("models/churnguard_xgb.joblib")
    feature_cols = joblib.load("models/feature_columns.joblib")

    df = data_prep.load_and_clean()
    df = feat_eng.engineer_features(df)
    X, y = feat_eng.encode_for_model(df)

    # WHY: the saved model was trained on a specific column layout.
    # New/scored data must match exactly, or predictions silently break.
    X = X.reindex(columns=feature_cols, fill_value=0)

    # --- Score every customer
    probs = xgb.predict_proba(X)[:, 1]
    df["churn_risk"] = probs

    # --- SHAP explainer
    # WHY TreeExplainer specifically: it's exact (not approximated) and
    # fast for tree-based models like XGBoost, unlike the generic
    # model-agnostic SHAP explainer.
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X)

    # --- Show reasons for the top N highest-risk customers
    top_idx = df["churn_risk"].sort_values(ascending=False).head(n).index

    print(f"Top {n} highest-risk customers and WHY:\n")
    for idx in top_idx:
        row_pos = df.index.get_loc(idx)
        print(f"Customer #{idx} — Churn risk: {df.loc[idx, 'churn_risk']*100:.1f}%")

        # Top 3 features pushing this customer's risk UP
        sv = pd.Series(shap_values.values[row_pos], index=X.columns)
        top_reasons = sv.sort_values(ascending=False).head(3)
        for feat, impact in top_reasons.items():
            actual_value = X.iloc[row_pos][feat]
            print(f"    + {feat} = {actual_value}  (pushed risk up by {impact:.3f})")
        print()

    return df, shap_values


if __name__ == "__main__":
    explain_top_risk_customers(n=5)
