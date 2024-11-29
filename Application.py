import os
import sys

# Ensure joblib is installed
try:
    import joblib
except ImportError:
    os.system(f"{sys.executable} -m pip install joblib")
    import joblib


import streamlit as st
import numpy as np
import joblib

#Load the trained model
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

import streamlit as st

# Application title
st.markdown("<h1 style='text-align: left; font-size: 55px;'>Heart Disease Prediction</h1>", unsafe_allow_html=True) 

# Increasing text size for feature titles using Markdown
st.markdown("## Input Features")  # Use '##' for a larger subtitle size

# Input fields
st.markdown("### Age")
a = st.slider("Select Age", min_value=0, max_value=120, step=1)

st.markdown("### Sex")
b = st.radio("Select Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", horizontal=True)

st.markdown("### Chest Pain Type")
c = st.radio(
    "Select Chest Pain Type", 
    options=[0, 1, 2, 3], 
    format_func=lambda x: ["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"][x], 
    horizontal=True)

st.markdown("### RestingBP")
d = st.slider("Select RestingBP", min_value=0, max_value=200)


st.markdown("### Serum Cholesterol")
e = st.slider("CHOL: Serum Cholesterol (0 - 603)", min_value=0, max_value=603)

st.markdown("### Fasting Blood Sugar")
f = st.radio(
    "Select Fasting Blood Sugar", 
    options=[0, 1], 
    format_func=lambda x: "Otherwise" if x == 0 else "Fasting Blood Sugar >120 mg/dl", 
    horizontal=True
)

st.markdown("### Resting ECG Results")
g = st.radio(
    "Select Resting ECG Results", 
    options=[0, 1, 2], 
    format_func=lambda x: ["Normal", "LVH (Left Ventricular Hypertrophy)", "ST (ST-T Wave Abnormality)"][x], 
    horizontal=True
)

st.markdown("### Maximum Heart Rate Achieved")
h = st.slider("MAXHR: Maximum Heart Rate Achieved (0 - 202)", min_value=0, max_value=202)

st.markdown("### Exercise Induced Angina")
i = st.radio(
    "Select Exercise Induced Angina", 
    options=[0, 1], 
    format_func=lambda x: "No" if x == 0 else "Yes", 
    horizontal=True)

st.markdown("### ST Depression Induced by Exercise")
j = st.slider("OLDPEAK: ST Depression Induced by Exercise Relative to Rest (0 - 6)", min_value=0.0, max_value=6.0)

st.markdown("### Slope of the Peak Exercise ST Segment")
k = st.radio(
    "Select Slope of the Peak Exercise ST Segment", 
    options=[0, 1, 2], 
    format_func=lambda x: ["Downsloping", "Flat", "Upsloping"][x], 
    horizontal=True
)

# Submit button
btn = st.button("Predict")

if btn:
    # Combine inputs into a NumPy array and reshape
    inputs = np.array([[a, b, c, d, e, f, g, h, i, j, k]])

    # Apply the same scaling transformation to the input data
    input_data_scaled = scaler.transform(inputs)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    # Example: Display the first prediction result
    if prediction[0] == 1:
        st.write("Heart disease predicted.")
    else:
        st.write("No heart disease predicted.")










#C:/Users/hp/AppData/Roaming/Python/Python310/Scripts/streamlit.exe run Application.py --server.enableXsrfProtection false
