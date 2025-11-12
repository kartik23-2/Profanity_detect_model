# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 22:50:13 2025

@author: Lenovo
"""

import streamlit as st
import pickle
import sklearn # Required to unpickle the scikit-learn model

# --- 1. Load the Trained Profanity Model ---
@st.cache_resource
def load_model():
    """Loads the saved profanity detection pipeline from a pickle file."""
    try:
        # Open the file in 'read-binary' ('rb') mode
        with open('profanity_model.pkl', 'rb') as file:
            pipeline = pickle.load(file)
            return pipeline
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'profanity_model.pkl' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

profanity_pipeline = load_model()

# --- 2. Create the Streamlit App UI ---
st.set_page_config(page_title="Profanity Filter", page_icon="🤬")
st.title("Profanity Filter")
st.write("Enter a comment below to check if it contains profanity or toxic language.")

# Add a text area for user input
user_input = st.text_area("Your Comment:", height=150)

# Add a button to classify
if st.button("Check Comment"):
    # Ensure the model is loaded and input is provided
    if profanity_pipeline is not None and user_input.strip() != "":
        # --- 3. Make Prediction ---
        prediction = profanity_pipeline.predict([user_input])
        probability = profanity_pipeline.predict_proba([user_input])
        
        # Get the probability of the 'Profane' class (which is class 1)
        profane_prob = probability[0][1] # probability of class '1'

        # --- 4. Display the Result ---
        if prediction[0] == 1:
            st.error(f"This comment is flagged as Profane/Toxic. (Confidence: {profane_prob*100:.2f}%)")
        else:
            st.success(f"This comment seems fine. (Confidence: {(1-profane_prob)*100:.2f}%)")
            
    elif user_input.strip() == "":
        st.warning("Please enter a comment to check.")