# app.py
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient details below to predict the likelihood of diabetes:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)
risk = st.number_input("Diabetes Risk Score", min_value=0.0, format="%.2f")

if st.button("🔍 Predict"):
    features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, risk]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("⚠️ The model predicts **Diabetes Positive**.")
    else:
        st.success("✅ The model predicts **No Diabetes**.")
