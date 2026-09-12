"""
ChurnGuard AI — Step 3: Model Training & Evaluation
======================================================
WHY THIS STEP EXISTS:
This trains two models (a simple baseline, then a stronger one) and
evaluates both with metrics that actually matter for churn — not
plain accuracy, which is misleading on imbalanced data.
"""

import pandas as pd
import numpy as np
import joblib
from importlib import import_module
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

data_prep = import_module("01_data_prep")
feat_eng = import_module("02_feature_engineering")


def main():
    # --- Load + engineer
    df = data_prep.load_and_clean()
    df = feat_eng.engineer_features(df)
    X, y = feat_eng.encode_for_model(df)

    # --- Train/test split
    # WHY 80/20 stratified: stratify keeps the same churn ratio in both
    # sets. This dataset has no timestamps, so a random split is the
    # right call here — but note: in a real production system with
    # dated records, you'd split by TIME instead (train on past,
    # test on future) to avoid leaking future patterns backward.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ============================================================
    # MODEL 1: Logistic Regression (interpretable baseline)
    # WHY: always start with a simple, explainable model. It gives you
    # a performance floor to compare against, and tells stakeholders
    # "here's what a basic linear model already catches."
    # ============================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    lr_probs = log_reg.predict_proba(X_test_scaled)[:, 1]

    print("=" * 60)
    print("MODEL 1: Logistic Regression (baseline)")
    print("=" * 60)
    print(f"ROC-AUC: {roc_auc_score(y_test, lr_probs):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_test, lr_probs):.3f}")

    # ============================================================
    # MODEL 2: XGBoost (production-grade model)
    # WHY class imbalance handling: churners are ~27% of customers.
    # scale_pos_weight tells the model to pay more attention to the
    # minority class instead of just predicting "no churn" for
    # everyone and still scoring high accuracy.
    # ============================================================
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("MODEL 2: XGBoost (production model)")
    print("=" * 60)
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb_probs):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_test, xgb_probs):.3f}")

    # --- Business-driven threshold instead of default 0.5
    # WHY: missing a real churner (false negative) usually costs far
    # more than a wasted retention offer (false positive). Lowering
    # the threshold to 0.35 catches more true churners at the cost of
    # a few extra false alarms — a trade-off that usually favors the
    # business, not the statistics textbook default.
    threshold = 0.35
    xgb_preds = (xgb_probs >= threshold).astype(int)

    print(f"\nClassification report at threshold={threshold}:")
    print(classification_report(y_test, xgb_preds, target_names=["Stayed", "Churned"]))

    cm = confusion_matrix(y_test, xgb_preds)
    print("Confusion Matrix:")
    print(f"                Predicted Stay   Predicted Churn")
    print(f"Actual Stay     {cm[0][0]:<16} {cm[0][1]}")
    print(f"Actual Churn    {cm[1][0]:<16} {cm[1][1]}")

    # --- Lift chart data: if we contact the riskiest 10% of customers,
    # what fraction of actual churners do we catch?
    # WHY this is the metric that matters to the business: it directly
    # answers "is this model worth acting on?"
    results = pd.DataFrame({"y_true": y_test.values, "y_prob": xgb_probs})
    results = results.sort_values("y_prob", ascending=False)
    top_10pct = int(len(results) * 0.10)
    caught = results.head(top_10pct)["y_true"].sum()
    total_churners = results["y_true"].sum()
    print(f"\nLift check: contacting the top 10% highest-risk customers "
          f"catches {caught}/{total_churners} "
          f"({caught/total_churners*100:.1f}%) of all actual churners.")

    # --- Save everything needed for scoring new customers later
    joblib.dump(xgb, "models/churnguard_xgb.joblib")
    joblib.dump(list(X.columns), "models/feature_columns.joblib")
    print("\nModel saved to models/churnguard_xgb.joblib")

    return xgb, X_test, y_test, xgb_probs


if __name__ == "__main__":
    main()
