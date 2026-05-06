# 💳 Credit Risk Prediction System

An end-to-end **Machine Learning project** that predicts whether a loan applicant is **high-risk or low-risk** using financial and personal attributes. The project is deployed as an interactive web application using **Streamlit**.

---

## 📌 Problem Statement

Financial institutions need to assess the risk of loan applicants efficiently.
This project builds a machine learning system that predicts **credit risk** based on key applicant features.

---

## 🧠 Solution Approach

The project follows a complete ML pipeline:

1. **Data Analysis (EDA)**
2. **Data Preprocessing**

   * Handling missing values (SimpleImputer)
   * Encoding categorical features (OneHotEncoder)
   * Scaling numerical features (StandardScaler)
3. **Feature Selection**
4. **Model Training**
5. **Hyperparameter Tuning**
6. **Pipeline Creation**
7. **Deployment using Streamlit**

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit

---

## 📊 Features Used

* Loan Intent
* Home Ownership
* Loan Amount
* Interest Rate
* Loan Percentage of Income

---

## 🤖 Model Details

* Model: **XGBoost Classifier**
* Hyperparameters tuned using grid/random search
* Integrated with **Pipeline** to ensure consistent preprocessing during training and inference

---

## 📈 Workflow

```
User Input → DataFrame → Pipeline (Preprocessing + Model) → Prediction
```

---

## ⚠️ Key Learnings

* Importance of **Pipeline for production ML systems**
* Handling **feature mismatch between training and inference**
* Structuring ML projects for **deployment**
* Building **interactive ML applications**

---

## 🔥 Future Improvements

* Add batch prediction (CSV upload)
* Integrate SHAP for explainability
* Add FastAPI backend
* Improve UI/UX further

---

## 👨‍💻 Author

**Mohnish Shandilya**

---

## ⭐ If you found this useful

Give this repo a ⭐ on GitHub!


