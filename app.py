"""
Loan Default Risk Predictor — Streamlit App (India-localized)
Trains the model at startup directly from Loan_default.csv (no .pkl files needed).
Run locally: streamlit run app.py
Deploy free: push to GitHub (with Loan_default.csv in the same repo), connect at share.streamlit.io
"""

import streamlit as st
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Currency assumption — documented, not hidden ──
# The underlying dataset's Income ($15K-$150K) and Loan Amount ($5K-$250K)
# reflect a US borrower population. Converting to INR at an approximate,
# fixed rate keeps the model mathematically valid (predictions still map to
# the range it was trained on) while making the interface usable for an
# Indian audience. This is a display/input convenience, not a claim that
# the model was trained on Indian income data.
USD_TO_INR = 83.0


def format_inr(n):
    """Format a number using Indian digit grouping, e.g. 5000000 -> ₹50,00,000"""
    n = int(round(n))
    sign = '-' if n < 0 else ''
    s = str(abs(n))
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        formatted = ','.join(parts) + ',' + last3
    return f"{sign}₹{formatted}"


st.set_page_config(page_title="Loan Risk Assessment Portal", page_icon="🏦", layout="centered")

# ── Theme: "Banking Portal" — clean net-banking / loan origination system look ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #eef2f7; }

/* Portal header bar */
.portal-header {
    background: linear-gradient(135deg, #0d2c54 0%, #123a6b 100%);
    margin: -1rem -1rem 1.5rem -1rem;
    padding: 22px 28px;
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 3px solid #1565c0;
}
.portal-header-title { color: #ffffff; font-size: 22px; font-weight: 800; margin: 0; }
.portal-header-sub { color: #9db8d8; font-size: 12.5px; margin: 2px 0 0 0; }
.portal-session-badge {
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.25);
    color: #cfe0f5; font-size: 11.5px; padding: 5px 12px; border-radius: 20px;
    white-space: nowrap;
}

h1, h2, h3, h4 { color: #0d2c54 !important; font-weight: 700 !important; }

.stCaption, p, label { color: #4a5a70 !important; }

/* Section cards */
.portal-card {
    background: #ffffff; border: 1px solid #d7dee6; border-radius: 8px;
    padding: 18px 20px; margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(13,44,84,0.06);
}

/* Buttons */
.stButton > button {
    background-color: #1565c0; color: #ffffff; border: none;
    font-weight: 600; border-radius: 6px; padding: 10px 0;
}
.stButton > button:hover { background-color: #0d2c54; color: #ffffff; }

hr { border-top: 1px solid #d7dee6 !important; }

.dti-note {
    background-color: #f2f6fb; border: 1px solid #d7dee6; border-left: 3px solid #1565c0;
    border-radius: 4px; padding: 10px 14px; font-size: 13px; color: #4a5a70; margin-top: 6px;
}

/* Application status card */
.status-card {
    background: #ffffff; border: 1px solid #d7dee6; border-radius: 10px;
    padding: 22px 24px; margin: 16px 0;
    box-shadow: 0 2px 6px rgba(13,44,84,0.08);
}
.status-row {
    display: flex; justify-content: space-between; padding: 7px 0;
    border-bottom: 1px solid #eef2f7; font-size: 14px;
}
.status-label { color: #6b7a8f; font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; }
.status-value { font-weight: 700; color: #0d2c54; }

.status-badge {
    display: inline-block; padding: 7px 18px; border-radius: 20px;
    font-weight: 700; font-size: 13px; letter-spacing: 0.4px; margin-top: 12px;
}
.badge-low { background: #e3f5e9; color: #1e7e34; }
.badge-moderate { background: #fdf1dc; color: #b8860b; }
.badge-high { background: #fbe4e4; color: #b02a2a; }

/* Risk meter */
.risk-meter-track {
    background: #eef2f7; border-radius: 6px; height: 12px; width: 100%;
    margin-top: 10px; overflow: hidden; position: relative;
}
.risk-meter-fill { height: 100%; border-radius: 6px; }
</style>

<div class="portal-header">
    <div>
        <p class="portal-header-title">🏦 National Credit Risk Portal</p>
        <p class="portal-header-sub">Loan Origination &amp; Risk Assessment System — Demo Environment</p>
    </div>
    <div class="portal-session-badge">🔒 Demo Session (not a live security system)</div>
</div>
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

st.subheader("New Loan Application — Risk Assessment")
st.caption("Enter applicant and loan details to generate an automated risk assessment. "
           "Amounts shown in ₹ (INR), converted from the underlying dataset at an approximate "
           "rate of ₹83 = $1 — see note at the bottom of this page.")

st.markdown("---")

# ── Input form ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 35)
    income_inr = st.number_input(
        "Annual Income (₹)", min_value=1200000, max_value=12500000,
        value=4000000, step=50000,
        help="Range reflects the training data's income distribution converted to ₹."
    )
    loan_amount_inr = st.number_input(
        "Loan Amount (₹)", min_value=400000, max_value=21000000,
        value=1500000, step=25000
    )
    credit_score = st.slider("Credit Score", 300, 850, 650)
    months_employed = st.slider("Months Employed", 0, 480, 60)
    num_credit_lines = st.slider("Number of Credit Lines", 0, 20, 3)

with col2:
    interest_rate = st.slider("Interest Rate (%)", 0.0, 30.0, 12.0, step=0.1)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    other_monthly_debt_inr = st.number_input(
        "Other Monthly Debt Payments (₹)", min_value=0, max_value=500000,
        value=15000, step=1000,
        help="Existing EMIs, credit card minimums, etc. — used to compute "
             "Debt-to-Income Ratio below, rather than asking you to guess it directly."
    )
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

# ── Compute DTI from the new loan's EMI + existing debt, instead of a guessed slider ──
monthly_rate = (interest_rate / 100) / 12
n_months = loan_term
if monthly_rate > 0:
    new_loan_emi = loan_amount_inr * monthly_rate * (1 + monthly_rate) ** n_months / \
                   ((1 + monthly_rate) ** n_months - 1)
else:
    new_loan_emi = loan_amount_inr / n_months

monthly_income_inr = income_inr / 12
dti_ratio_computed = (other_monthly_debt_inr + new_loan_emi) / monthly_income_inr
dti_ratio = min(dti_ratio_computed, 1.0)  # cap for model stability — see note below

st.markdown(
    f'<div class="dti-note">📐 <b>Computed Debt-to-Income Ratio: {dti_ratio_computed:.2f}</b> '
    f'— from this loan\'s estimated EMI ({format_inr(new_loan_emi)}/month) plus your other monthly debt '
    f'({format_inr(other_monthly_debt_inr)}), divided by monthly income ({format_inr(monthly_income_inr)}). '
    f'{"⚠️ Capped at 1.00 for the model input — an uncapped ratio this high falls outside what the model was trained on." if dti_ratio_computed > 1.0 else ""}'
    f'</div>', unsafe_allow_html=True
)

st.markdown("---")

if st.button("Predict Default Risk", type="primary", use_container_width=True):
    # Convert INR inputs back to USD-equivalent scale, since the model was
    # trained on the original dataset's USD figures — the ₹ conversion is
    # purely a display/input layer, not a retraining of the model.
    income_usd = income_inr / USD_TO_INR
    loan_amount_usd = loan_amount_inr / USD_TO_INR

    raw = {
        'Age': age, 'Income': income_usd, 'LoanAmount': loan_amount_usd,
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

    entry_no = "APP-" + datetime.datetime.now().strftime("%y%m%d%H%M")

    if prob >= 0.5:
        badge_class, badge_text, meter_color = "badge-high", "⚠ HIGH RISK — MANUAL UNDERWRITING REQUIRED", "#b02a2a"
    elif prob >= 0.25:
        badge_class, badge_text, meter_color = "badge-moderate", "◐ MODERATE RISK — ADDITIONAL REVIEW SUGGESTED", "#b8860b"
    else:
        badge_class, badge_text, meter_color = "badge-low", "✓ LOW RISK — ELIGIBLE FOR STANDARD PROCESSING", "#1e7e34"

    fill_pct = min(prob * 100, 100)

    st.markdown(f"""
    <div class="status-card">
        <div class="status-row">
            <span class="status-label">Application ID</span>
            <span class="status-value">{entry_no}</span>
        </div>
        <div class="status-row">
            <span class="status-label">Applicant Age</span>
            <span class="status-value">{age}</span>
        </div>
        <div class="status-row">
            <span class="status-label">Loan Amount</span>
            <span class="status-value">{format_inr(loan_amount_inr)}</span>
        </div>
        <div class="status-row">
            <span class="status-label">Debt-to-Income Ratio</span>
            <span class="status-value">{dti_ratio:.2f}</span>
        </div>
        <div class="status-row" style="border-bottom:none;">
            <span class="status-label">Estimated Default Probability</span>
            <span class="status-value" style="font-size:20px;">{prob:.1%}</span>
        </div>
        <div class="risk-meter-track">
            <div class="risk-meter-fill" style="width:{fill_pct}%; background-color:{meter_color};"></div>
        </div>
        <div class="status-badge {badge_class}">{badge_text}</div>
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
    coefs = model.coef_[0]
    contributions = coefs * input_scaled[0]
    contrib_series = pd.Series(contributions, index=model_columns).sort_values(key=abs, ascending=False)

    plain_names = {
        'Age': ("applicant's age", age),
        'Income': ("annual income", format_inr(income_inr)),
        'LoanAmount': ("loan amount", format_inr(loan_amount_inr)),
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
with st.expander("ℹ️ About the ₹ conversion and DTI calculation"):
    st.markdown(
        "- **Currency**: the underlying model was trained on a US dataset (income "
        "$15K–$150K, loan amount $5K–$250K). Figures here are converted to ₹ at a fixed "
        "approximate rate (₹83 = $1) purely so the interface is usable for an Indian "
        "audience — it does not mean the model was trained on Indian financial data, "
        "and the ₹ ranges above will look elevated for that reason.\n"
        "- **Debt-to-Income Ratio**: earlier versions of this app asked users to enter DTI "
        "directly, which is hard to estimate accurately. Testing showed DTI has "
        "essentially no correlation with any other field in the training data — so instead "
        "of guessing, this version computes it properly from the new loan's estimated EMI "
        "plus your reported existing monthly debt, divided by monthly income — the same "
        "logic a real lender uses."
    )

st.caption("National Credit Risk Portal (demo) · Built by Shivank Thakur · "
           "[GitHub](https://github.com/shivank-byte) · Model: Logistic Regression, class-balanced · ROC-AUC 0.753")
