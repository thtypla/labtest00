

import streamlit as st
import pandas as pd
import pickle

# Load the model and encoders (make sure the model file path is correct)
with open('model_penguin_66130701901.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Streamlit app title
st.title("Penguin Species Prediction")

# Input fields for the user
island = st.selectbox("Select Island", options=list(island_encoder.classes_))
culmen_length_mm = st.number_input("Culmen Length (mm)", value=37.0, min_value=0.0, step=0.1)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", value=19.3, min_value=0.0, step=0.1)
flipper_length_mm = st.number_input("Flipper Length (mm)", value=192.3, min_value=0.0, step=0.1)
body_mass_g = st.number_input("Body Mass (g)", value=3750, min_value=0, step=1)
sex = st.selectbox("Select Sex", options=list(sex_encoder.classes_))

# Create DataFrame from input values
x_new = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

# Transform categorical features
x_new['island'] = island_encoder.transform(x_new['island'])
x_new['sex'] = sex_encoder.transform(x_new['sex'])

# Prediction and display result
if st.button("Predict"):
    try:
        y_pred_new = model.predict(x_new)
        result = species_encoder.inverse_transform(y_pred_new)
        st.write(f"Predicted Species: {result[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

