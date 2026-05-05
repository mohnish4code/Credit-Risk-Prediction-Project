# import streamlit as st
# import pandas as pd
# import sys, os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.predict import predict, predict_proba

# st.title("💳 Credit Risk Prediction System")

# st.write("Enter Customer Details")

# income = st.number_input("Income")
# loan = st.number_input("Loan Amount")
# age = st.number_input("Age")

# home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
# intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "VENTURE"])

# if st.button("Predict"):
#     input_df = pd.DataFrame({
#         "income": [income],
#         "loan_amnt": [loan],
#         "age": [age],
#         "person_home_ownership": [home],
#         "loan_intent": [intent]
#     })

#     result = predict(input_df)[0]
#     prob = predict_proba(input_df)[0][1]

#     if result == 1:
#         st.error(f"High Risk ❌ (Risk Score: {prob:.2f})")
#     else:
#         st.success(f"Low Risk ✅ (Risk Score: {prob:.2f})")