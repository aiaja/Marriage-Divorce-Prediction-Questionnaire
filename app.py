import streamlit as st
import numpy as np
import h5py
import json
from streamlit_option_menu import option_menu

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

# Split questions into pages
questions_per_page = 9
pages = [list(questions.values())[i:i+questions_per_page] for i in range(0, len(questions), questions_per_page)]

# Session state to track page number and user inputs
if 'page_number' not in st.session_state:
    st.session_state['page_number'] = 0

if 'user_inputs' not in st.session_state:
    st.session_state['user_inputs'] = [2] * len(questions)

# Navbar
selected = option_menu(
    menu_title=None,
    options=["About", "How to Use"],
    icons=["info-circle", "question-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#363636"},
        "icon": {"color": "white", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "color": "white",
            "--hover-color": "#48494b",
        },
        "nav-link-selected": {"background-color": "#f01e2c"},
    },
)

# Page navigation
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.title("Divorce Prediction Questionnaire")

current_page = st.session_state['page_number']
questions_on_page = pages[current_page]

# Display questions for the current page
for i, question in enumerate(questions_on_page):
    index = current_page * questions_per_page + i
    st.session_state['user_inputs'][index] = st.slider(question, 0, 4, st.session_state['user_inputs'][index])

# Navigation buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Back") and current_page > 0:
        st.session_state['page_number'] -= 1
with col3:
    if st.button("Next") and current_page < len(pages) - 1:
        st.session_state['page_number'] += 1

# Predict button on the last page
if current_page == len(pages) - 1:
    if st.button("Predict"):
        # Preprocess user inputs
        reduced_input = preprocess_input(st.session_state['user_inputs'], components, mean)
        # Calculate probability
        probability = 1 / (1 + np.exp(-(np.dot(reduced_input, coefficients.T) + intercept)))
        divorce_probability = float(probability[0]) * 100

        # Display prediction result
        st.write("### Prediction Result")
        if divorce_probability >= 50:
            st.error(f"The model predicts: **There will be a Divorce** (Divorce Probability: {divorce_probability:.2f}%)")
        else:
            st.success(f"The model predicts: **There will be no Divorce** (Divorce Probability: {divorce_probability:.2f}%)")
