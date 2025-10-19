import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the model and scaler
model = joblib.load('student_grade_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Student Grade Prediction")

# 2. Collect user inputs
age = st.number_input("Age", min_value=10, max_value=25, value=17)
studytime = st.selectbox("Weekly Study Time", [1,2,3,4])
internet = st.radio("Internet Access at Home?", ["Yes","No"])
romantic = st.radio("In a Romantic Relationship?", ["Yes","No"])
G1 = st.number_input("First Period Grade (G1)", 0, 20, 10)
G2 = st.number_input("Second Period Grade (G2)", 0, 20, 10)
sex = st.radio("Gender", ["Female","Male"])

# 3. Preprocess inputs
# Binary encoding
internet = 1 if internet=="Yes" else 0
romantic = 1 if romantic=="Yes" else 0
sex_M = 1 if sex=="Male" else 0

# 4. Create DataFrame for model
input_df = pd.DataFrame({
    'age': [age],
    'studytime': [studytime],
    'internet': [internet],
    'romantic': [romantic],
    'G1': [G1],
    'G2': [G2],
    'sex_M': [sex_M],
    # 'attendance_percentage' was removed because the scaler was trained without it
})

# 5. Scale features
input_scaled = scaler.transform(input_df)

# 6. Make prediction
prediction = model.predict(input_scaled)[0]

# 7. Display result
st.subheader("Predicted Final Grade (G3):")
st.write(f"{prediction:.2f}")
