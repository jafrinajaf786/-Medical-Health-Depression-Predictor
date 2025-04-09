import streamlit as st
import pickle
import numpy as np

# ------------------- Load Model -------------------
try:
    with open("D://Users//admin//pyt//depression_model.pkl", 'rb') as file:
        model_dict = pickle.load(file)
        model = model_dict["model"]  # Make sure this key is correct
except FileNotFoundError:
    st.error("❌ Model file not found. Please check the path.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ------------------- App Config -------------------
st.set_page_config(page_title="Depression Predictor", layout="centered")
st.title("🧠 Mental Health Depression Predictor")

st.markdown("""
Fill in the information below to predict the likelihood of depression.  
**Disclaimer:** This tool is for educational/demo purposes only and not a substitute for professional medical advice.""")

# ------------------- Input Form (Text + Number Inputs) -------------------
gender = st.text_input("Gender (Male/Female)").strip().capitalize()
age = st.number_input("Age", min_value=10, max_value=100, step=1)
status = st.text_input("Are you a Student or Working Professional?").strip().capitalize()
academic_pressure = st.number_input("Academic Pressure (0-10)", min_value=0, max_value=10)
work_pressure = st.number_input("Work Pressure (0-10)", min_value=0, max_value=10)
study_satisfaction = st.number_input("Study Satisfaction (0-10)", min_value=0, max_value=10)
job_satisfaction = st.number_input("Job Satisfaction (0-10)", min_value=0, max_value=10)
sleep_duration = st.number_input("Sleep Duration (in hours)", min_value=0, max_value=24)
diet = st.text_input("Diet (Healthy/Unhealthy)").strip().capitalize()
suicidal_thoughts = st.text_input("Have you had suicidal thoughts? (Yes/No)").strip().capitalize()
work_hours = st.number_input("Work/Study Hours per Day", min_value=0, max_value=24)
financial_stress = st.number_input("Financial Stress (0-10)", min_value=0, max_value=10)
family_history = st.text_input("Family History of Mental Illness (Yes/No)").strip().capitalize()

# ------------------- Encode Categorical Inputs -------------------
try:
    gender_val = 1 if gender == "Male" else 0
    status_val = 1 if status == "Working professional" else 0
    diet_val = 1 if diet == "Healthy" else 0
    suicidal_val = 1 if suicidal_thoughts == "Yes" else 0
    family_val = 1 if family_history == "Yes" else 0

    # ------------------- Input Vector -------------------
    input_data = np.array([[gender_val, age, status_val, academic_pressure,
                            work_pressure, study_satisfaction, job_satisfaction,
                            sleep_duration, diet_val, suicidal_val, work_hours,
                            financial_stress, family_val]])
except Exception as e:
    st.error(f"❌ Input processing error: {e}")
    st.stop()

# ------------------- Predict Button -------------------
if st.button("Predict"):
    try:
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.error(f"🧠 Likely Depressed (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ Not Depressed (Confidence: {1 - prob:.2f})")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
