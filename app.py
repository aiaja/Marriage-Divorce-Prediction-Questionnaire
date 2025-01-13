import streamlit as st
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def load_model_and_pca(model_file):
    try:
        with h5py.File(model_file, 'r') as h5file:
            coefficients = h5file['coefficients'][...]
            intercept = h5file['intercept'][...]
            explained_variance_ratio = h5file['explained_variance_ratio'][...]
            components = h5file['components'][...]
            mean = h5file['mean'][...]  # Load the mean
            
            # Optional: Include explained_variance_
            explained_variance = h5file['explained_variance'][...] if 'explained_variance' in h5file else None

        # Reconstruct PCA
        pca = PCA(n_components=components.shape[0])
        pca.components_ = components
        pca.explained_variance_ratio_ = explained_variance_ratio
        pca.mean_ = mean  # Set the mean
        if explained_variance is not None:
            pca.explained_variance_ = explained_variance

        # Reconstruct Logistic Regression Model
        model = LogisticRegression()
        model.coef_ = coefficients
        model.intercept_ = intercept

        return model, pca
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# Path to the saved model file
model_file = "reduced_model.h5"

# Load the model and PCA
model, pca = load_model_and_pca(model_file)

# Streamlit app layout
st.title("Marriage and Divorce Prediction")
st.markdown("This app predicts whether a person is likely to stay married or get divorced based on their answers to 54 questions.")

# Collect user inputs for Q1-Q54
st.header("Answer the following questions:")
questions = [f"Q{i}" for i in range(1, 55)]
responses = []

for question in questions:
    response = st.slider(f"{question}", 0, 4, 2)  # Assume a scale of 0 to 4 for answers
    responses.append(response)

# Make prediction
if st.button("Predict"):
    if model is None or pca is None:
        st.error("Model or PCA is not properly loaded. Please check the model file.")
    else:
        # Convert responses to a NumPy array
        user_data = np.array(responses).reshape(1, -1)

        # Apply PCA transformation
        user_data_reduced = pca.transform(user_data)

        # Make prediction
        prediction = model.predict(user_data_reduced)[0]
        probability = model.predict_proba(user_data_reduced)[0]

        # Display results
        if prediction == 0:
            st.success(f"The prediction is: Married with a probability of {probability[0] * 100:.2f}%.")
        else:
            st.warning(f"The prediction is: Divorced with a probability of {probability[1] * 100:.2f}%.")

# Display the app footer
st.markdown("---")
st.markdown("Powered by Streamlit and Logistic Regression with PCA.")
