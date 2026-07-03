"""
Loan Default Risk Predictor — Streamlit App
Trains the model at startup directly from Loan_default.csv (no .pkl files needed).
Run locally: streamlit run app.py
Deploy free: push to GitHub (with Loan_default.csv in the same repo), connect at share.streamlit.io
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💳", layout="centered")

# ── Train model once, cache it (so it doesn't retrain on every click) ──
@st.cache_resource
def train_model():
    df = pd.read_csv('Loan_default.csv')
    df = df.drop(columns=['LoanID'])
    X = df.drop(columns=['Default'])
    y = df['Default']
    X_encoded = pd.get_dummies(X, drop_first=True)
    model_columns = list(X_encoded.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, model_columns

with st.spinner("Training model on 255,347 records... (first load only, ~10 seconds)"):
    model, scaler, model_columns = train_model()

st.title("💳 Loan Default Risk Predictor")
st.caption("Logistic Regression model trained on 255,347 historical loan records. "
           "Estimates the probability a borrower defaults, given loan and borrower details.")

st.markdown("---")

# ── Input form ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 35)
    income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000, step=500)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    months_employed = st.slider("Months Employed", 0, 480, 60)
    num_credit_lines = st.slider("Number of Credit Lines", 0, 20, 3)

with col2:
    interest_rate = st.slider("Interest Rate (%)", 0.0, 30.0, 12.0, step=0.1)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35, step=0.01)
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

col3, col4 = st.columns(2)
with col3:
    has_mortgage = st.radio("Has Mortgage?", ["No", "Yes"], horizontal=True)
    has_dependents = st.radio("Has Dependents?", ["No", "Yes"], horizontal=True)
with col4:
    has_cosigner = st.radio("Has Co-Signer?", ["No", "Yes"], horizontal=True)
    loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])

st.markdown("---")

if st.button("Predict Default Risk", type="primary", use_container_width=True):
    raw = {
        'Age': age, 'Income': income, 'LoanAmount': loan_amount,
        'CreditScore': credit_score, 'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate,
        'LoanTerm': loan_term, 'DTIRatio': dti_ratio,
        'Education': education, 'EmploymentType': employment_type,
        'MaritalStatus': marital_status, 'HasMortgage': has_mortgage,
        'HasDependents': has_dependents, 'LoanPurpose': loan_purpose,
        'HasCoSigner': has_cosigner
    }
    input_df = pd.DataFrame([raw])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")
    st.metric("Estimated Default Probability", f"{prob:.1%}")

    if prob >= 0.5:
        st.error("⚠️ Higher risk profile — model flags this application for manual review.")
    elif prob >= 0.25:
        st.warning("⚡ Moderate risk — within a range where lenders often apply additional checks.")
    else:
        st.success("✅ Lower risk profile based on the historical data patterns.")

    st.caption(
        "Note: this model prioritizes catching real defaulters (recall) over minimizing "
        "false alarms (precision) — a deliberate trade-off, since missing an actual "
        "defaulter is typically costlier to a lender than an extra manual review. "
        "This is a demonstration model trained on a public dataset, not a production "
        "credit decision system."
    )

st.markdown("---")
st.caption("Built by Shivank Thakur · [GitHub](https://github.com/shivank-byte) · "
           "Model: Logistic Regression, class-balanced · ROC-AUC 0.753")
