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

from src.predict import predict, predict_proba

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

    # Automatically calculate loan percentage income
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

    prediction = predict(input_df)[0]

    probability = predict_proba(input_df)[0][1]

    st.markdown("## 📈 Prediction Result")

    # ---------------- HIGH RISK ---------------- #

    if prediction == 1:

        st.error("⚠️ High Risk Customer")

        st.progress(float(probability))

        st.markdown(
            f"### Risk Probability: "
            f"**{probability:.2%}**"
        )

        st.warning("""
        This applicant may have a higher
        probability of loan default based on
        financial and credit characteristics.
        """)

    # ---------------- LOW RISK ---------------- #

    else:

        st.success("✅ Low Risk Customer")

        st.progress(float(probability))

        st.markdown(
            f"### Risk Probability: "
            f"**{probability:.2%}**"
        )

        st.info("""
        This applicant appears financially safer
        according to the trained ML model.
        """)

st.markdown("---")

# ---------------- ABOUT MODEL ---------------- #

with st.expander("ℹ️ About This Model"):

    st.write("""

### 🔹 Model Information

- Model: XGBoost Classifier
- Type: Binary Classification
- Purpose: Credit Risk Prediction

### 🔹 Features Used

- Annual Income
- Home Ownership
- Loan Purpose
- Loan Amount
- Interest Rate
- Loan Percentage of Income
- Loan Grade
- Previous Loan Default
- Credit History Length

### 🔹 Pipeline Includes

- Data preprocessing
- Feature encoding
- Scaling
- Model prediction pipeline

### 🔹 Performance

- Accuracy: ~93%
- ROC-AUC Score: ~0.85

""")

# ---------------- FOOTER ---------------- #

st.caption(
    "Built with Streamlit | "
    "Machine Learning Credit Risk Prediction System"
)