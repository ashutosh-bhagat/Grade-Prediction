import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the model and scaler
model = joblib.load('student_grade_model.pkl')
scaler = joblib.load('scaler.pkl')

# Use a centered layout which tends to work better on mobile
st.set_page_config(page_title='Student Grade Predictor', layout='centered')

# Small CSS adjustments for touch targets and typography
st.markdown(
    """
    <style>
      .big-title {font-size:28px; font-weight:700; margin-bottom:4px}
      .subtitle {color:#6c757d; margin-top:0; margin-bottom:12px}
      .pred-card {background:#f8f9fa; padding:12px; border-radius:8px}
      .pred-value {font-size:48px; font-weight:700; color:#0b5ed7}
      .stButton>button {height:48px; font-size:16px}
      .input-label {font-size:15px}
      @media (max-width:600px) {
        .pred-value {font-size:40px}
        .big-title {font-size:24px}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='big-title'>Student Grade Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter student info below — the form is optimized for mobile screens.</div>", unsafe_allow_html=True)

# Put inputs into a form so mobile users can fill then tap Predict (better UX)
with st.form('input_form'):
    # Responsive columns: on narrow screens Streamlit stacks them automatically
    c1, c2 = st.columns([1, 1])
    with c1:
        age = st.number_input("Age", min_value=10, max_value=30, value=17)
        studytime = st.selectbox("Daily Study Time (Hours)", [1,2,3,4])
        internet = st.selectbox("Internet Access at Home?", ["Yes","No"])
        romantic = st.selectbox("In a Romantic Relationship?", ["Yes","No"])
    with c2:
        G1 = st.number_input("Mid 1 Marks (0-20)", 0, 20, 10)
        G2 = st.number_input("Mid 2 Marks (0-20)", 0, 20, 10)
        sex = st.selectbox("Gender", ["Female","Male"]) 
        # small spacer for alignment on desktop
        st.write('')

    submit = st.form_submit_button('Predict')

    # Optional input preview for mobile users
    if st.checkbox('Show input preview'):
        st.write({'age': age, 'studytime': studytime, 'internet': internet, 'romantic': romantic, 'G1': G1, 'G2': G2, 'sex': sex})

# Only predict when the user taps Predict to improve mobile performance
prediction = None
if submit:
    # Preprocess inputs
    internet_val = 1 if internet == "Yes" else 0
    romantic_val = 1 if romantic == "Yes" else 0
    sex_M = 1 if sex == "Male" else 0

    input_df = pd.DataFrame({
        'age': [age],
        'studytime': [studytime],
        'internet': [internet_val],
        'romantic': [romantic_val],
        'G1': [G1],
        'G2': [G2],
        'sex_M': [sex_M],
    })

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

# Display prediction area
st.write('')
col_main, col_pred = st.columns([1, 1])

with col_main:
    st.markdown("<h4>About</h4>", unsafe_allow_html=True)
    st.markdown("This lightweight app predicts a student's re-mid score (G3) from basic features. Tap Predict after filling the form.")

with col_pred:
    if prediction is not None:
        st.markdown("<div class='pred-card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex; align-items:center; justify-content:space-between'><div><div style='font-size:14px; color:#6c757d'>Predicted Re-Mid Marks</div><div class='pred-value'>{prediction:.2f}</div></div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info('Fill the form and tap Predict to see the prediction.')
