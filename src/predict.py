import joblib
import pandas as pd
from .config import MODELS_DIR
from .data_preprocessing import add_features


def load_model(model_path=None):
    model_path = model_path or MODELS_DIR / "best_model.pkl"
    return joblib.load(model_path)


def predict_churn(input_dict):
    model = load_model()
    df = pd.DataFrame([input_dict])
    # add_features expects TotalCharges, tenure, MonthlyCharges
    df = add_features(df)
    probability = model.predict_proba(df)[0, 1]
    prediction = int(probability >= 0.5)
    return prediction, probability


def risk_level(probability):
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def business_recommendations(input_dict, probability):
    recs = []
    if input_dict.get("Contract") == "Month-to-month":
        recs.append("Offer a discount for switching to a one-year or two-year contract.")
    if input_dict.get("tenure", 0) <= 12:
        recs.append("Provide onboarding support and early loyalty benefits for new customers.")
    if input_dict.get("MonthlyCharges", 0) > 70:
        recs.append("Review the monthly bill and offer a personalized retention plan.")
    if input_dict.get("TechSupport") == "No":
        recs.append("Offer free or discounted technical support for a limited period.")
    if probability >= 0.7:
        recs.append("Assign this customer to a retention team for priority follow-up.")
    if not recs:
        recs.append("Maintain regular engagement and monitor the customer periodically.")
    return recs
