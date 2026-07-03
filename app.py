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

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="📖", layout="centered")

# ── Theme: "Monsoon Ledger" — Indian financial-document aesthetic ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

.stApp {
    background-color: #f2ede0;
    background-image: repeating-linear-gradient(
        0deg, rgba(28,46,74,0.035) 0px, rgba(28,46,74,0.035) 1px,
        transparent 1px, transparent 32px
    );
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1c2e4a; }

h1 {
    font-family: 'Libre Baskerville', serif !important;
    color: #1c2e4a !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #1c2e4a;
    padding-bottom: 12px;
}
h2, h3 { font-family: 'Libre Baskerville', serif !important; color: #1c2e4a !important; }

.stCaption, p, label, .stMarkdown { color: #3c4d68 !important; }

/* Buttons - stamped ink look */
.stButton > button {
    background-color: #1c2e4a; color: #f2ede0; border: 1px solid #1c2e4a;
    font-family: 'Libre Baskerville', serif; font-weight: 700;
    border-radius: 2px; letter-spacing: 0.5px;
}
.stButton > button:hover { background-color: #b5442e; border-color: #b5442e; color: #f2ede0; }

hr { border-top: 1px solid #c9bfa5 !important; }

/* Passbook entry / stamp result box */
.ledger-entry {
    background-color: #fbf8f0;
    border: 1px solid #c9bfa5;
    border-left: 4px solid #1c2e4a;
    padding: 20px 24px;
    margin: 16px 0;
    font-family: 'IBM Plex Mono', monospace;
}
.ledger-entry-row {
    display: flex; justify-content: space-between;
    border-bottom: 1px dotted #c9bfa5; padding: 6px 0;
    font-size: 14px;
}
.ledger-label { color: #6b7a94; text-transform: uppercase; letter-spacing: 1px; font-size: 11px; }
.ledger-value { font-weight: 600; }

.stamp {
    display: inline-block;
    border: 3px double;
    padding: 10px 20px;
    font-family: 'Libre Baskerville', serif;
    font-weight: 700;
    font-size: 18px;
    letter-spacing: 2px;
    text-transform: uppercase;
    transform: rotate(-2deg);
    margin: 14px 0;
}
.stamp-low { color: #2e6b4f; border-color: #2e6b4f; }
.stamp-moderate { color: #b5442e; border-color: #b5442e; }
.stamp-high { color: #8c1f1f; border-color: #8c1f1f; background: rgba(140,31,31,0.06); }
</style>
""", unsafe_allow_html=True)

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

st.title("📖 Loan Default Risk Ledger")
st.caption("An entry-by-entry risk assessment, modeled on 255,347 historical loan records. "
           "Enter applicant details below to generate a risk entry.")

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
    dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35, step=0.01,
                          help="Entered manually — in this dataset, DTI has no measurable "
                               "relationship to income or loan amount (correlation ≈ 0), "
                               "so it can't be reliably auto-calculated from other fields.")
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

    import datetime
    entry_no = datetime.datetime.now().strftime("%y%m%d-%H%M")

    if prob >= 0.5:
        stamp_class, stamp_text = "stamp-high", "Flagged — Review"
    elif prob >= 0.25:
        stamp_class, stamp_text = "stamp-moderate", "Moderate Risk"
    else:
        stamp_class, stamp_text = "stamp-low", "Cleared — Low Risk"

    st.markdown(f"""
    <div class="ledger-entry">
        <div class="ledger-entry-row">
            <span class="ledger-label">Entry No.</span>
            <span class="ledger-value">{entry_no}</span>
        </div>
        <div class="ledger-entry-row">
            <span class="ledger-label">Applicant Age</span>
            <span class="ledger-value">{age}</span>
        </div>
        <div class="ledger-entry-row">
            <span class="ledger-label">Loan Amount</span>
            <span class="ledger-value">${loan_amount:,}</span>
        </div>
        <div class="ledger-entry-row">
            <span class="ledger-label">Estimated Default Probability</span>
            <span class="ledger-value">{prob:.1%}</span>
        </div>
        <div class="stamp {stamp_class}">{stamp_text}</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Note: this model prioritizes catching real defaulters (recall) over minimizing "
        "false alarms (precision) — a deliberate trade-off, since missing an actual "
        "defaulter is typically costlier to a lender than an extra manual review. "
        "This is a demonstration model trained on a public dataset, not a production "
        "credit decision system."
    )

    # ── Plain-language factor explanation ──
    # Each feature's contribution to THIS prediction = its coefficient x its scaled value.
    # Positive = pushed risk up. Negative = pushed risk down.
    coefs = model.coef_[0]
    contributions = coefs * input_scaled[0]
    contrib_series = pd.Series(contributions, index=model_columns).sort_values(key=abs, ascending=False)

    plain_names = {
        'Age': ("applicant's age", age),
        'Income': ("annual income", f"${income:,}"),
        'LoanAmount': ("loan amount", f"${loan_amount:,}"),
        'CreditScore': ("credit score", credit_score),
        'MonthsEmployed': ("length of employment", f"{months_employed} months"),
        'NumCreditLines': ("number of open credit lines", num_credit_lines),
        'InterestRate': ("interest rate", f"{interest_rate}%"),
        'LoanTerm': ("loan term", f"{loan_term} months"),
        'DTIRatio': ("debt-to-income ratio", f"{dti_ratio:.2f}"),
    }

    def describe(feature, value):
        direction = "increased" if value > 0 else "decreased"
        if feature in plain_names:
            label, shown_val = plain_names[feature]
            return f"**{label.capitalize()}** ({shown_val}) {direction} the estimated risk."
        # categorical dummy columns, e.g. "EmploymentType_Unemployed"
        label = feature.replace('_', ': ').replace('Type', ' type')
        return f"**{label}** {direction} the estimated risk."

    top_factors = contrib_series.head(4)

    st.markdown("#### What drove this result")
    for feat, val in top_factors.items():
        st.markdown(f"- {describe(feat, val)}")

    st.caption(
        "These are the factors this specific application weighed most heavily — shown "
        "because they had the largest effect on this prediction, not because they're "
        "necessarily the most important factors in general. This reflects a statistical "
        "pattern learned from historical data, not a causal explanation."
    )

st.markdown("---")
st.caption("Built by Shivank Thakur · [GitHub](https://github.com/shivank-byte) · "
           "Model: Logistic Regression, class-balanced · ROC-AUC 0.753")
