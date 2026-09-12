# ChurnGuard AI

A machine learning system that predicts which telecom customers are likely to cancel their subscription (churn) — and explains *why* — so a retention team can step in before it's too late.

Built on the IBM Telco Customer Churn dataset (7,043 customers).

---

## Problem

Acquiring a new customer typically costs far more than retaining an existing one, yet most companies only find out a customer is unhappy after they've already left. ChurnGuard AI flips that — it scores every customer's churn risk in advance and surfaces the specific reasons behind that score, so retention efforts can be targeted instead of blanket.

## What It Does

1. **Cleans raw customer data** — fixes missing/incorrectly typed billing values
2. **Engineers behavioral features** — tenure buckets, service-usage counts, contract-risk flags, spend ratios
3. **Trains two models** — a Logistic Regression baseline and an XGBoost classifier
4. **Explains every prediction** — uses SHAP to show which factors pushed a customer's risk up or down
5. **Scores new customers on demand** — takes a single customer's record and returns a risk percentage with reasons

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.84 |
| PR-AUC | 0.65 |
| Recall (churn class) @ threshold 0.35 | 87% |

Contacting just the **top 10% highest-risk customers** catches roughly **29% of all actual churners** — a strong lift over contacting customers at random.

The decision threshold (0.35 instead of the default 0.5) was chosen deliberately: in this business context, missing a real churner is more costly than a wasted retention offer, so the model is tuned to catch more true churners at the expense of a few extra false alarms.

## Tech Stack

- **Python** — core language
- **Pandas** — data cleaning and feature engineering
- **Scikit-learn** — baseline Logistic Regression model, train/test splitting, metrics
- **XGBoost** — main gradient-boosted classifier
- **SHAP** — model explainability (per-prediction feature attribution)
- **Jupyter Notebook** — end-to-end interactive walkthrough
- **Matplotlib** — EDA and SHAP visualizations

## Project Structure

```
ChurnGuard-AI/
├── data/
│   └── telco_churn.csv              # Raw dataset
├── models/
│   ├── churnguard_xgb.joblib         # Trained XGBoost model
│   └── feature_columns.joblib        # Saved feature schema for scoring
├── 01_data_prep.py                   # Cleans raw data
├── 02_feature_engineering.py         # Builds derived features
├── 03_train_model.py                 # Trains + evaluates both models
├── 04_explainability.py              # SHAP explanations for top-risk customers
├── 05_score_new_customer.py          # Scores a single new customer
├── ChurnGuard_AI_Walkthrough.ipynb   # Full pipeline, interactive, with charts
└── README.md
```

## How to Run

```bash
# Set up environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Install dependencies
pip install pandas scikit-learn xgboost shap matplotlib jupyter nbformat

# Run the pipeline
python 01_data_prep.py
python 02_feature_engineering.py
python 03_train_model.py          # trains and saves the model
python 04_explainability.py       # explains top-risk customers
python 05_score_new_customer.py   # scores one example customer
```

Or open `ChurnGuard_AI_Walkthrough.ipynb` to run the entire pipeline interactively with EDA charts and SHAP visualizations.

## Sample Output

```
Churn risk: 82.3%
Top risk drivers:
  - tenure = 2                (impact: +0.613)
  - is_month_to_month = 1     (impact: +0.522)
  - avg_charge_per_tenure = 95.5  (impact: +0.372)
```

This tells the retention team not just *that* a customer is at risk, but *why* — a new, month-to-month, high-spending customer — which points directly to the right intervention (e.g., a loyalty discount or a longer-term contract offer).

## What This Is / Isn't

This is a complete, working prototype covering the full ML lifecycle — data cleaning through explainable predictions. It is **not** yet a production system. Moving toward production would involve:

- Replacing the static CSV with a live database/warehouse connection
- Automating scoring on a schedule (cron job, Airflow, or an API endpoint)
- Tuning the decision threshold against real business cost data
- Monitoring for data drift and retraining periodically
- Integrating risk scores into wherever the retention team actually works (CRM, dashboard, Slack alerts)

## Dataset

[IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) — 7,043 customers, 20 features, publicly available for learning and prototyping.
