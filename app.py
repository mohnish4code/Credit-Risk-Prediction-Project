import streamlit as st
import pandas as pd
import sys, os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict, predict_proba

# Page config
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# Custom CSS (clean dark fintech style)
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.stButton>button {
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("💳 Credit Risk Prediction System")
st.markdown("### Loan Risk Assessment")

st.markdown("---")

# Layout (2 columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Loan Details")
    loan = st.number_input("💰 Loan Amount", min_value=0.0)
    loan_int_rate = st.number_input("📈 Interest Rate (%)", min_value=0.0)
    loan_percent_income = st.slider(
        "📊 Loan % of Income",
        min_value=0.0,
        max_value=1.0,
        step=0.01
    )

with col2:
    st.subheader("👤 Customer Profile")
    home = st.selectbox(
        "🏠 Home Ownership",
        ["RENT", "OWN", "MORTGAGE"]
    )
    intent = st.selectbox(
        "🎯 Loan Purpose",
        ["PERSONAL", "EDUCATION", "VENTURE"]
    )

st.markdown("---")

# Prediction
if st.button("🔍 Predict Risk"):

    input_df = pd.DataFrame({
        "loan_intent": [intent],
        "person_home_ownership": [home],
        "loan_amnt": [loan],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income]
    })

    result = predict(input_df)[0]
    prob = predict_proba(input_df)[0][1]

    st.markdown("## 📈 Prediction Result")

    if result == 1:
        st.error(f"⚠️ High Risk Customer")
        st.progress(float(prob))
        st.write(f"### Risk Probability: **{prob:.2%}**")
    else:
        st.success(f"✅ Low Risk Customer")
        st.progress(float(prob))
        st.write(f"### Risk Probability: **{prob:.2%}**")

st.markdown("---")

# Info section
with st.expander("ℹ️ About Model"):
    st.write("""
    - Model: XGBoost Classifier  
    - Features Used:
        - Loan Intent  
        - Home Ownership  
        - Loan Amount  
        - Interest Rate  
        - Loan % of Income  
    - Pipeline includes preprocessing + encoding + scaling
    """)

# Footer
st.caption("Built with using Streamlit | Credit Risk ML System")