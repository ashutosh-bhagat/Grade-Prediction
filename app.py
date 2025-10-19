import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the model and scaler
model = joblib.load('student_grade_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page config for better appearance
st.set_page_config(page_title='Student Grade Predictor', layout='wide')

# Header
st.markdown("<h1 style='font-size:34px; margin-bottom:0'>Student Grade Prediction</h1>", unsafe_allow_html=True)
st.markdown("<div style='color: #6c757d; margin-top:0'>Enter student information on the left and get a predicted re-mid score on the right.</div>", unsafe_allow_html=True)
st.write('')

# Sidebar for inputs (keeps UI clean)
st.sidebar.header('Input features')
age = st.sidebar.number_input("Age", min_value=10, max_value=30, value=17)
studytime = st.sidebar.selectbox("Weekly Study Time (Hours)", [1,2,3,4])
internet = st.sidebar.radio("Internet Access at Home?", ["Yes","No"] )
romantic = st.sidebar.radio("In a Romantic Relationship?", ["Yes","No"] )
G1 = st.sidebar.number_input("Mid 1 Marks (0-20)", 0, 20, 10)
G2 = st.sidebar.number_input("Mid 2 Marks (0-20)", 0, 20, 10)
sex = st.sidebar.radio("Gender", ["Female","Male"] )

# Optional: show a compact summary in the sidebar
with st.sidebar.expander('Preview input'):
    st.write({'age': age, 'studytime': studytime, 'internet': internet, 'romantic': romantic, 'G1': G1, 'G2': G2, 'sex': sex})

# 3. Preprocess inputs (binary encoding)
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
})

# 5. Scale features and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

# Layout: show prediction prominently
left, right = st.columns([1, 1])

with left:
    st.markdown("<h3 style='color:#333'>Input summary</h3>", unsafe_allow_html=True)
    st.table(input_df.T.rename(columns={0: 'value'}))

with right:
    st.markdown("<h3 style='color:#333'>Predicted Re-Mid Marks</h3>", unsafe_allow_html=True)
    st.metric(label="Predicted Score (G3)", value=f"{prediction:.2f}")
    st.markdown("<div style='margin-top:8px; color:#6c757d'>This prediction is produced by a saved regression model and a StandardScaler. If you change features, retrain the model to include them.</div>", unsafe_allow_html=True)
