import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap

# Set page title
st.set_page_config(page_title="GDM Risk Prediction Tool", layout="centered")
st.title("🤰 GDM Gestational Diabetes Mellitus Risk Prediction Tool")
st.markdown("This tool uses a machine learning model (XGBoost) to predict whether an individual has GDM.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("best_model_xgboost.pkl")

model = load_model()

# Feature names
feature_names = ['BMI', 'As', 'Cd', 'LDL', 'PA', 'LY%', 'ChE', 'Glucose', 'Age']

# Input interface
st.subheader("📋 Enter 9 Clinical Features (Standardized Values)")

def get_input():
    inputs = []
    for name in feature_names:
        val = st.number_input(name, value=0.0, step=0.1, format="%.2f")
        inputs.append(val)
    return np.array([inputs])

user_input = get_input()

# Prediction logic
if st.button("🔍 Predict GDM Risk"):
    try:
        # Predict probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_input)[0][1]
        else:
            st.error("❌ The model does not support predict_proba().")
            st.stop()

        threshold = 0.3
        pred = "Positive (GDM)" if proba >= threshold else "Negative (Non-GDM)"

        st.markdown(f"### 🧪 Prediction Result: **{proba*100:.2f}%** Probability of GDM")
        st.markdown(f"### 🩺 Diagnosis: **{pred}** (Threshold: 0.3)")

        # SHAP force plot
        st.markdown("---")
        st.markdown("🎯 **Feature Contribution Plot (SHAP Force Plot)**")

        # Extract classifier
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            base_model = model.named_steps["clf"]
            input_data = model.named_steps["scaler"].transform(user_input)
        else:
            base_model = model
            input_data = user_input

        explainer = shap.Explainer(base_model)
        shap_values = explainer(input_data)

        # Generate force plot and adjust width
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values.values[0],
            input_data[0],
            feature_names=feature_names,
            matplotlib=False
        )

        st.components.v1.html(
            shap.getjs() + force_plot.html(),
            height=300,  # Height
            width=700    # Width close to the input box width
        )

        st.markdown("🔴 Red features push the prediction towards GDM, 🔵 Blue features push the prediction towards Non-GDM.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

st.markdown("---")
st.markdown("🔬 This tool is intended for research/assistance purposes only and is not a substitute for medical diagnosis.")
