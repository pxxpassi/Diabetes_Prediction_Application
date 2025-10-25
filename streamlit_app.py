# app.py
import streamlit as st
import numpy as np
import joblib

# Load model
try:
    model = joblib.load("diabetes_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'diabetes_model.pkl' exists.")
    st.stop()

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient details below to predict the likelihood of diabetes:")

# Use columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)

with col2:
    skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.2f")

with col3:
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    risk = st.number_input("Diabetes Risk Score", min_value=0.0, max_value=100.0, value=20.0, format="%.2f")

if st.button("🔍 Predict"):
    try:
        features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, risk]])
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error("⚠️ The model predicts **Diabetes Positive**.")
        else:
            st.success("✅ The model predicts **No Diabetes**.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
