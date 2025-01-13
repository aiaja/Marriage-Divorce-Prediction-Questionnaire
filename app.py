import streamlit as st
import numpy as np
import h5py
import json

# Load model
with h5py.File('reduced_model.h5', 'r') as h5file:
    coefficients = h5file['coefficients'][:]
    intercept = h5file['intercept'][:]
    components = h5file['components'][:]
    mean = h5file['mean'][:]

# Function to preprocess input
def preprocess_input(user_inputs, components, mean):
    standardized = np.array(user_inputs) - mean
    return np.dot(standardized, components.T)

# Load questions
with open('questions.json', 'r') as file:
    questions = json.load(file)

# Streamlit app
st.title("Divorce Prediction Questionnaire")
st.write("Rate each question on a scale of 0 (Strongly Disagree) to 4 (Strongly Agree).")

# Collect user inputs
user_inputs = [st.slider(q, 0, 4, 2) for q in questions.values()]

# Predict
if st.button("Predict"):
    # Preprocess user inputs
    reduced_input = preprocess_input(user_inputs, components, mean)
    # Calculate probability
    probability = 1 / (1 + np.exp(-(np.dot(reduced_input, coefficients.T) + intercept)))
    divorce_probability = float(probability[0]) * 100  # Convert to percentage

    # Display prediction result
    st.write("### Prediction Result")
    if divorce_probability >= 50:
        st.error(f"The model predicts: **Divorced** (Probability: {divorce_probability:.2f}%)")
    else:
        st.success(f"The model predicts: **Not Divorced** (Probability: {divorce_probability:.2f}%)")
