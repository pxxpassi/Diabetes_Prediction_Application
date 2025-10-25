import streamlit as st
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import Row

# Start Spark session
spark = SparkSession.builder.appName("StreamlitDiabetes").getOrCreate()

# Load PySpark model
model = PipelineModel.load("spark_diabetes_model")

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient details below to predict the likelihood of diabetes:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
risk = st.number_input("Diabetes Risk Score", min_value=0.0, max_value=100.0, value=20.0, format="%.2f")

if st.button("🔍 Predict"):
    # Create PySpark DataFrame from user input
    input_df = spark.createDataFrame([Row(
        pregnancies=int(pregnancies),
        glucose_imp=float(glucose),
        blood_pressure_imp=float(bp),
        skin_thickness_imp=float(skin),
        insulin_imp=float(insulin),
        bmi=float(bmi),
        diabetes_pedigree=float(dpf),
        age=int(age),
        diabetes_risk_score=float(risk)
    )])

    # Make prediction
    prediction = model.transform(input_df).collect()[0]['prediction']

    if prediction == 1.0:
        st.error("⚠️ The model predicts **Diabetes Positive**.")
    else:
        st.success("✅ The model predicts **No Diabetes**.")
