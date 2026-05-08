import streamlit as st
import pandas as pd
import sys, os

# ---------------- IMPORT ---------------- #

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_proba

# ---------------- CONFIG ---------------- #

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)

# ---------------- STYLE ---------------- #

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
    height: 3em;
    width: 100%;
    font-weight: bold;
    font-size: 16px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}

.low {
    background-color: #133a2a;
}

.medium {
    background-color: #4a3f0b;
}

.high {
    background-color: #4a1f1f;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #

st.title("💳 Credit Risk Prediction")
st.caption("AI-powered loan risk assessment system")

st.markdown("---")

# ---------------- INPUT ---------------- #

st.subheader("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    person_income = st.number_input("💵 Annual Income", min_value=1.0, step=1000.0)
    loan_amnt = st.number_input("💰 Loan Amount", min_value=0.0, step=500.0)
    loan_int_rate = st.number_input("📈 Interest Rate (%)", min_value=0.0, step=0.1)
    cb_person_cred_hist_length = st.slider("📅 Credit History Length", 1, 30, 5)

with col2:
    person_home_ownership = st.selectbox("🏠 Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_intent = st.selectbox("🎯 Loan Purpose",
        ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"]
    )
    loan_grade = st.selectbox("🏦 Loan Grade", ["A","B","C","D","E","F","G"])
    cb_person_default_on_file = st.selectbox("⚠️ Previous Default", ["N","Y"])

# ---------------- DERIVED ---------------- #

loan_percent_income = loan_amnt / person_income

st.markdown("---")

st.metric("📊 Loan % of Income", f"{loan_percent_income:.2%}")

if loan_percent_income > 0.83:
    st.warning("⚠️ This value is outside most training data range")

# ---------------- PREDICT ---------------- #

if st.button("🔍 Predict Credit Risk"):

    input_df = pd.DataFrame({
        "person_income": [person_income],
        "person_home_ownership": [person_home_ownership],
        "loan_intent": [loan_intent],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "loan_grade": [loan_grade],
        "cb_person_default_on_file": [cb_person_default_on_file],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length]
    })

    # -------- ML PROBABILITY -------- #

    prob = predict_proba(input_df)[0][1]

    # -------- SOFT ADJUSTMENT -------- #

    adjustment = 0

    if loan_percent_income > 0.50:
        adjustment += 0.05
    if loan_percent_income > 0.75:
        adjustment += 0.08
    if cb_person_default_on_file == "Y":
        adjustment += 0.07
    if loan_grade in ["D","E","F","G"]:
        adjustment += 0.05

    adjusted_prob = min(prob + adjustment, 0.95)

    # -------- HARD RULE OVERRIDE -------- #

    if loan_percent_income > 0.75 and cb_person_default_on_file == "Y":
        risk = "🔴 High Risk"
        confidence = "Strong"
        css_class = "high"

    elif loan_percent_income > 0.80:
        risk = "🔴 High Risk"
        confidence = "High"
        css_class = "high"

    elif cb_person_default_on_file == "Y" and loan_grade in ["D","E","F","G"]:
        risk = "🔴 High Risk"
        confidence = "High"
        css_class = "high"

    elif adjusted_prob < 0.40:
        risk = "🟢 Low Risk"
        confidence = "Moderate"
        css_class = "low"

    elif adjusted_prob < 0.70:
        risk = "🟡 Medium Risk"
        confidence = "High"
        css_class = "medium"

    else:
        risk = "🔴 High Risk"
        confidence = "Strong"
        css_class = "high"

    # -------- DISPLAY -------- #

    st.markdown("---")
    st.subheader("📈 Prediction Result")

    st.markdown(f"""
    <div class="result-box {css_class}">
        <h2>{risk}</h2>
        <h4>Confidence: {confidence}</h4>
    </div>
    """, unsafe_allow_html=True)

    # -------- EXPLANATION -------- #

    reasons = []

    if loan_percent_income > 0.75:
        reasons.append("Very high loan compared to income")

    if cb_person_default_on_file == "Y":
        reasons.append("Previous loan default")

    if loan_grade in ["D","E","F","G"]:
        reasons.append("Poor loan grade")

    if loan_int_rate > 15:
        reasons.append("High interest rate")

    if cb_person_cred_hist_length <= 2:
        reasons.append("Short credit history")

    if reasons:
        st.markdown("### ⚠️ Key Risk Factors")
        for r in reasons:
            st.write(f"- {r}")

# ---------------- ABOUT ---------------- #

st.markdown("---")

with st.expander("ℹ️ About This Model"):
    st.write("""
    - Model: XGBoost Classifier  
    - Accuracy: ~93%  
    - ROC-AUC: ~0.85  

    This system combines:
    - Machine Learning predictions  
    - Financial risk rules  

    to improve real-world reliability.
    """)

# ---------------- FOOTER ---------------- #

st.caption("Built with Streamlit | Credit Risk ML System")