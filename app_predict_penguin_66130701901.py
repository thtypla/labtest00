
!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the model and encoders
with open('model_penguin_xxx.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Streamlit app
st.title("Penguin Species Prediction")

# Input fields
island = st.selectbox("Island", options=list(island_encoder.classes_))
bill_length_mm = st.number_input("Bill Length (mm)", value=37.0)
bill_depth_mm = st.number_input("Bill Depth (mm)", value=19.3)
flipper_length_mm = st.number_input("Flipper Length (mm)", value=192.3)
body_mass_g = st.number_input("Body Mass (g)", value=3750)
sex = st.selectbox("Sex", options=list(sex_encoder.classes_))


# Create input DataFrame
x_new = pd.DataFrame({
    'island': [island],
    'bill_length_mm': [bill_length_mm],
    'bill_depth_mm': [bill_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex],
    'year': [2023]  # Assuming you have a default year value to use
})

# Transform categorical features
x_new['island'] = island_encoder.transform(x_new['island'])
x_new['sex'] = sex_encoder.transform(x_new['sex'])

# Predict and display results
if st.button("Predict"):
    y_pred_new = model.predict(x_new)
    result = species_encoder.inverse_transform(y_pred_new)
    st.write(f"Predicted Specie: {result[0]}")
    
