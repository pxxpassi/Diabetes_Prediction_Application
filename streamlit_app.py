import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('diabetes_model.pkl')

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient details below to predict the likelihood of diabetes:")

# Input fields (adjust names & order to match your dataset)
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0, step=1)

if st.button("🔍 Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ The model predicts **Diabetes Positive**.")
    else:
        st.success("✅ The model predicts **No Diabetes**.")
