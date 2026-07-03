# Loan Default Risk Predictor

**A binary classification project predicting whether a borrower will default on a loan, using demographic, financial, and loan-specific features — deployed as a live interactive Streamlit app.**

Dataset: [Kaggle — Loan Default](https://www.kaggle.com/datasets/nikhil1e9/loan-default) (255,347 records, 18 features)
Live app: *(add your Streamlit Cloud link here once deployed)*

---

## Why This Project Exists

Lenders need to estimate, before disbursing a loan, how likely a borrower is to default — a decision that directly affects pricing, approval, and risk exposure. This project builds and evaluates that prediction from first principles on a real, imbalanced dataset (only 11.6% of borrowers in the data actually defaulted), which forces honest engagement with a core classification problem: a model can look highly "accurate" while being practically useless if it never predicts the minority outcome.

---

## What's Inside

| File | Description |
|---|---|
| `loan_default_classifier.py` | Full analysis script — data exploration, cleaning, encoding, model training (Logistic Regression + Random Forest), evaluation, feature importance |
| `app.py` | Deployable Streamlit app — takes borrower inputs, returns default probability in real time |
| `loan_model.pkl`, `scaler.pkl`, `model_columns.pkl` | Saved model artifacts used by `app.py` |
| `model_evaluation.png` | Confusion matrix + ROC curve comparison |
| `feature_importance.png` | Top 10 predictors of default (Random Forest) |
| `requirements.txt` | Dependencies for local run / Streamlit Cloud deployment |

---

## Key Results

- **Class imbalance handled via `class_weight='balanced'`** rather than discarding data through undersampling — preserves all training information while still forcing the model to attend to the minority (default) class.
- **Two models compared, not one**: Logistic Regression (ROC-AUC 0.753) vs. Random Forest (ROC-AUC 0.754) — near-identical ranking ability, so the simpler, faster, more interpretable model was chosen for deployment.
- **Deliberate precision/recall trade-off**: the deployed model favors recall (catching ~70% of actual defaulters) over precision, since for a lender, missing a real defaulter is typically costlier than an extra manual review of a false alarm.
- **Top predictors**: Age, Interest Rate, Income, and Months Employed — with an explicit note that Age's strength here is a correlation the model found, not a claimed causal effect.

*Full detail — economic logic, STAR framework, challenges, and limitations — is in the [project narrative](./loan_default_project_narrative.pdf).*

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to Deploy (free, ~5 minutes)

1. Push all files in this repo (including the `.pkl` files) to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app", select this repo, set the main file to `app.py`.
4. Deploy — Streamlit Community Cloud builds and hosts it for free at a public URL.
5. Add that URL back into this README.

---

## Data Source

Public dataset sourced from Kaggle (originally associated with a Coursera loan-default modeling challenge). All data is anonymized/synthetic borrower-level data; no real personal information is used or exposed.

**Disclaimer**: This is a demonstration/portfolio model trained on a public dataset, not a production credit-decision system.
