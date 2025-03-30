import streamlit as st
import numpy as np
import pickle
import joblib

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = joblib.load('scalar.pkl')

# Streamlit UI
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("Diabetes Prediction App")
st.markdown("Enter the details below to predict whether the patient has diabetes or not.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=30)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    # Scale the input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    # Display result
    if prediction[0] == 1:
        st.error("The model predicts that the patient **has diabetes**.")
    else:
        st.success("The model predicts that the patient **does not have diabetes**.")
