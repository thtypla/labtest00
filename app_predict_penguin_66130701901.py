import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and encoders (ensure the model file path is correct)
model_file = 'model_penguin_xxx (1).pkl'
try:
    with open(model_file, 'rb') as file:
        model, species_encoder, island_encoder, sex_encoder = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file '{model_file}' not found. Please ensure it is uploaded to the app directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.stop()

# Streamlit app
st.title("Penguin Species Prediction")

# Input fields for the user
island = st.selectbox("Select Island", options=list(island_encoder.classes_))
bill_length_mm = st.number_input("Bill Length (mm)", value=37.0, min_value=0.0, step=0.1)
bill_depth_mm = st.number_input("Bill Depth (mm)", value=19.3, min_value=0.0, step=0.1)
flipper_length_mm = st.number_input("Flipper Length (mm)", value=192.3, min_value=0.0, step=0.1)
body_mass_g = st.number_input("Body Mass (g)", value=3750.0, min_value=0.0, step=1.0)
sex = st.selectbox("Select Sex", options=list(sex_encoder.classes_))

# Create a DataFrame with the inputs
x_new = pd.DataFrame({
    'island': [island],
    'bill_length_mm': [bill_length_mm],
    'bill_depth_mm': [bill_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex],
    'year': [2023]  # You can adjust the year or remove if not required
})

# Check if the island and sex categories exist in the encoder
try:
    x_new['island'] = island_encoder.transform(x_new['island'])
    x_new['sex'] = sex_encoder.transform(x_new['sex'])
except ValueError as e:
    st.error(f"Error with encoding input values: {str(e)}")
    st.stop()

# Prediction and display results
if st.button("Predict"):
    try:
        y_pred_new = model.predict(x_new)
        result = species_encoder.inverse_transform(y_pred_new)
        st.write(f"Predicted Species: {result[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
