import streamlit as st
import pandas as pd
import sys
import os

# ---------------- IMPORT PATH ---------------- #

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from src.predict import predict_proba

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    color: black;
    border-radius: 12px;
    height: 3.2em;
    width: 100%;
    font-weight: bold;
    font-size: 16px;
    border: none;
}

.stButton>button:hover {
    opacity: 0.9;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #

st.title("💳 Credit Risk Prediction System")
st.markdown("### AI-Based Loan Risk Assessment")

st.markdown("---")

# ---------------- LAYOUT ---------------- #

col1, col2 = st.columns(2)

# ---------------- COLUMN 1 ---------------- #

with col1:

    st.subheader("📊 Financial Information")

    person_income = st.number_input(
        "💵 Annual Income",
        min_value=1.0,
        step=1000.0
    )

    loan_amnt = st.number_input(
        "💰 Loan Amount",
        min_value=0.0,
        step=500.0
    )

    loan_int_rate = st.number_input(
        "📈 Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        step=0.1
    )

    # Automatically calculate loan percentage
    loan_percent_income = loan_amnt / person_income

    st.metric(
        label="📊 Loan Percentage of Income",
        value=f"{loan_percent_income:.2%}"
    )

# ---------------- COLUMN 2 ---------------- #

with col2:

    st.subheader("👤 Applicant Profile")

    person_home_ownership = st.selectbox(
        "🏠 Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

    loan_intent = st.selectbox(
        "🎯 Loan Purpose",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

    loan_grade = st.selectbox(
        "🏦 Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    cb_person_default_on_file = st.selectbox(
        "⚠️ Previous Loan Default",
        ["N", "Y"]
    )

    cb_person_cred_hist_length = st.slider(
        "📅 Credit History Length (Years)",
        min_value=1,
        max_value=30,
        value=5
    )

st.markdown("---")

# ---------------- WARNINGS ---------------- #

if loan_percent_income > 0.83:

    st.warning(
        "⚠️ This loan percentage exceeds the "
        "training data distribution range. "
        "Prediction confidence may be lower."
    )

if loan_amnt > person_income:

    st.info(
        "ℹ️ Loan amount exceeds annual income. "
        "Such loans generally carry higher financial risk."
    )

# ---------------- PREDICTION ---------------- #

if st.button("🔍 Predict Credit Risk"):

    input_df = pd.DataFrame({

        "person_income": [person_income],

        "person_home_ownership": [
            person_home_ownership
        ],

        "loan_intent": [loan_intent],

        "loan_amnt": [loan_amnt],

        "loan_int_rate": [loan_int_rate],

        "loan_percent_income": [
            loan_percent_income
        ],

        "loan_grade": [loan_grade],

        "cb_person_default_on_file": [
            cb_person_default_on_file
        ],

        "cb_person_cred_hist_length": [
            cb_person_cred_hist_length
        ]

    })

    # ---------------- MODEL PROBABILITIES ---------------- #

    probabilities = predict_proba(input_df)[0]

    low_risk_prob = probabilities[0]
    high_risk_prob = probabilities[1]

    # =====================================================
    # BUSINESS RULE ADJUSTMENTS
    # =====================================================

    adjustment_score = 0

    # High loan burden
    if loan_percent_income > 0.50:
        adjustment_score += 0.15

    # Extremely high loan burden
    if loan_percent_income > 0.75:
        adjustment_score += 0.20

    # Previous default
    if cb_person_default_on_file == "Y":
        adjustment_score += 0.20

    # Risky loan grades
    if loan_grade in ["D", "E", "F", "G"]:
        adjustment_score += 0.15

    # Very short credit history
    if cb_person_cred_hist_length <= 2:
        adjustment_score += 0.10

    # RENT ownership slightly risky
    if person_home_ownership == "RENT":
        adjustment_score += 0.05

    # High interest rate
    if loan_int_rate > 15:
        adjustment_score += 0.10

    # Venture loans slightly risky
    if loan_intent == "VENTURE":
        adjustment_score += 0.05

    # Apply adjustment
    adjusted_prob = high_risk_prob + adjustment_score

    # Clamp probability
    adjusted_prob = min(max(adjusted_prob, 0), 1)

    # ---------------- RISK CATEGORY ---------------- #

    if adjusted_prob < 0.40:

        risk_category = "🟢 Low Risk"

    elif adjusted_prob < 0.70:

        risk_category = "🟡 Medium Risk"

    else:

        risk_category = "🔴 High Risk"

    # ---------------- RESULT DISPLAY ---------------- #

    st.markdown("## 📈 Prediction Result")

    st.metric(
        label="Risk Category",
        value=risk_category
    )

    st.progress(float(adjusted_prob))

    st.markdown(
        f"### Adjusted High Risk Probability: "
        f"**{adjusted_prob:.2%}**"
    )

    # ---------------- DETAILED RESULT ---------------- #

    if adjusted_prob < 0.40:

        st.success("""
        This applicant appears financially safer
        according to the ML model and financial rules.
        """)

    elif adjusted_prob < 0.70:

        st.warning("""
        This applicant shows moderate financial
        risk characteristics. Additional review
        may be recommended.
        """)

    else:

        st.error("""
        This applicant may have a higher probability
        of loan default based on ML prediction and
        financial risk indicators.
        """)

    # ---------------- EXPANDABLE DETAILS ---------------- #

    with st.expander("📋 Risk Analysis Details"):

        st.write(f"Base ML High Risk Probability: {high_risk_prob:.2%}")
        st.write(f"Business Rule Adjustment: +{adjustment_score:.2%}")
        st.write(f"Final Adjusted Probability: {adjusted_prob:.2%}")

        st.markdown("### Financial Risk Factors Considered")

        risk_factors = []

        if loan_percent_income > 0.50:
            risk_factors.append("High loan-to-income burden")

        if cb_person_default_on_file == "Y":
            risk_factors.append("Previous default history")

        if loan_grade in ["D", "E", "F", "G"]:
            risk_factors.append("Risky loan grade")

        if cb_person_cred_hist_length <= 2:
            risk_factors.append("Short credit history")

        if person_home_ownership == "RENT":
            risk_factors.append("Rental home ownership")

        if loan_int_rate > 15:
            risk_factors.append("High interest rate")

        if loan_intent == "VENTURE":
            risk_factors.append("Business/Venture loan purpose")

        if len(risk_factors) == 0:
            st.success("No major financial risk indicators detected.")

        else:
            for factor in risk_factors:
                st.write(f"• {factor}")

st.markdown("---")

# ---------------- ABOUT MODEL ---------------- #

with st.expander("ℹ️ About This Model"):

    st.write("""

### 🔹 Model Information

- Model: XGBoost Classifier
- Type: Credit Risk Classification
- Deployment: Streamlit

### 🔹 Features Used

- Annual Income
- Home Ownership
- Loan Purpose
- Loan Amount
- Interest Rate
- Loan Percentage of Income
- Loan Grade
- Previous Default History
- Credit History Length

### 🔹 System Design

This system combines:

1. Machine Learning Predictions
2. Financial Rule-Based Adjustments
3. Risk Categorization Logic

to improve practical financial risk interpretation.

### 🔹 Evaluation Metrics

- Accuracy: ~93%
- ROC-AUC Score: ~0.85
- Evaluated using confusion matrix,
  precision, recall, and F1-score

### 🔹 Important Note

Predictions are based on patterns learned
from historical training data and may become
less reliable for highly unrealistic financial inputs.

""")

# ---------------- FOOTER ---------------- #

st.caption(
    "🚀 Built with Streamlit | "
    "AI-Powered Credit Risk Prediction System"
)