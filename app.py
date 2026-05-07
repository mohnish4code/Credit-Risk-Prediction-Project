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
    layout="centered"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3, h4 {
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

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #

st.title("💳 Credit Risk Prediction")
st.markdown(
    "AI-powered loan risk assessment system"
)

st.markdown("---")

# ---------------- INPUT SECTION ---------------- #

st.subheader("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:

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

    cb_person_cred_hist_length = st.slider(
        "📅 Credit History Length",
        min_value=1,
        max_value=30,
        value=5
    )

with col2:

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
        "⚠️ Previous Default",
        ["N", "Y"]
    )

# ---------------- DERIVED FEATURE ---------------- #

loan_percent_income = loan_amnt / person_income

st.markdown("---")

st.metric(
    label="📊 Loan Percentage of Income",
    value=f"{loan_percent_income:.2%}"
)

# ---------------- LIGHT WARNINGS ---------------- #

if loan_percent_income > 0.83:

    st.warning(
        "This loan ratio exceeds most training data examples. "
        "Prediction confidence may be lower."
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

    # ---------------- MODEL PREDICTION ---------------- #

    probabilities = predict_proba(input_df)[0]

    high_risk_prob = probabilities[1]

    # ---------------- BUSINESS RULE ADJUSTMENT ---------------- #

    adjustment_score = 0

    if loan_percent_income > 0.50:
        adjustment_score += 0.15

    if loan_percent_income > 0.75:
        adjustment_score += 0.20

    if cb_person_default_on_file == "Y":
        adjustment_score += 0.20

    if loan_grade in ["D", "E", "F", "G"]:
        adjustment_score += 0.15

    if cb_person_cred_hist_length <= 2:
        adjustment_score += 0.10

    if person_home_ownership == "RENT":
        adjustment_score += 0.05

    if loan_int_rate > 15:
        adjustment_score += 0.10

    adjusted_prob = high_risk_prob + adjustment_score

    adjusted_prob = min(max(adjusted_prob, 0), 1)

    # ---------------- RISK CATEGORY ---------------- #

    if adjusted_prob < 0.40:

        risk_category = "🟢 Low Risk"

        message = (
            "This applicant appears financially "
            "stable according to the risk model."
        )

    elif adjusted_prob < 0.70:

        risk_category = "🟡 Medium Risk"

        message = (
            "This applicant shows moderate "
            "financial risk characteristics."
        )

    else:

        risk_category = "🔴 High Risk"

        message = (
            "This applicant may have a higher "
            "probability of loan default."
        )

    # ---------------- RESULT DISPLAY ---------------- #

    st.markdown("---")

    st.subheader("📈 Prediction Result")

    st.metric(
        label="Risk Category",
        value=risk_category
    )

    st.progress(float(adjusted_prob))

    st.markdown(
        f"### Risk Confidence: "
        f"**{adjusted_prob:.2%}**"
    )

    st.info(message)

# ---------------- ABOUT SECTION ---------------- #

st.markdown("---")

with st.expander("ℹ️ About This Model"):

    st.write("""

### Model Information

- Model: XGBoost Classifier
- Task: Credit Risk Classification
- Deployment: Streamlit

### Features Used

- Annual Income
- Loan Amount
- Interest Rate
- Home Ownership
- Loan Purpose
- Loan Grade
- Previous Default History
- Credit History Length
- Loan Percentage of Income

### Model Performance

- Accuracy: ~93%
- ROC-AUC Score: ~0.85

### Note

This system combines:
- Machine Learning predictions
- Financial risk adjustment rules

to improve practical credit risk assessment.

""")

# ---------------- FOOTER ---------------- #

st.caption(
    "Built with Streamlit | AI-Based Credit Risk Prediction System"
)