import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
from src.predict import predict_churn, risk_level, business_recommendations

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
st.title("Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to churn using the best trained model.")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=float(monthly_charges * max(tenure, 1)))

input_data = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

if st.button("Predict Churn"):
    try:
        pred, prob = predict_churn(input_data)
        st.subheader("Prediction Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", "Churn" if pred == 1 else "No Churn")
        c2.metric("Churn Probability", f"{prob:.2%}")
        c3.metric("Risk Level", risk_level(prob))
        st.subheader("Business Recommendations")
        for rec in business_recommendations(input_data, prob):
            st.write(f"- {rec}")
    except FileNotFoundError:
        st.error("Trained model not found. Please run: python src/train.py")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Models compared: Logistic Regression, Decision Tree, Random Forest, and XGBoost. Final model is selected using F1 score on the test set.")
