
pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Initialize your encoders (assuming you have already fitted these somewhere in your existing code)
island_encoder = LabelEncoder()
sex_encoder = LabelEncoder()

# Add your existing code for training the encoders, if necessary
# Example: island_encoder.fit(data['island'])
# Example: sex_encoder.fit(data['sex'])

# Streamlit App layout
st.title("Penguin Classification")

# Input fields for features
culmen_length = st.number_input("Culmen Length (mm)", min_value=0.0, max_value=100.0)
culmen_depth = st.number_input("Culmen Depth (mm)", min_value=0.0, max_value=100.0)
island = st.selectbox("Island", options=['Torgersen', 'Dream', 'Biscoe']) # Modify with actual islands
sex = st.selectbox("Sex", options=['Male', 'Female']) # Modify with actual sexes
year = st.number_input("Year", min_value=1900, max_value=2100, value=2023)  # Default year value

# Create a new DataFrame based on user input
x_new = pd.DataFrame({
    'culmen_length_mm': [culmen_length],
    'culmen_depth_mm': [culmen_depth],
    'island': [island],
    'sex': [sex],
    'year': [year]
})

# Rename columns in x_new to match the expected column names by the model
x_new = x_new.rename(columns={
    'culmen_length_mm': 'bill_length_mm',
    'culmen_depth_mm': 'bill_depth_mm'
})

# Processing 'island'
x_new['island'] = x_new['island'].astype(str)
known_island_categories = list(island_encoder.classes_)
x_new['island'] = x_new['island'].apply(lambda x: x if x in known_island_categories else known_island_categories[0])
x_new['island'] = island_encoder.transform(x_new['island'])

# Processing 'sex'
x_new['sex'] = x_new['sex'].astype(str)
known_sex_categories = list(sex_encoder.classes_)
x_new['sex'] = x_new['sex'].apply(lambda x: x if x in known_sex_categories else known_sex_categories[0])
x_new['sex'] = sex_encoder.transform(x_new['sex'])

# Output the processed DataFrame
st.write("Processed Input Data:")
st.dataframe(x_new)

# Prediction button (assuming you have a trained model)
if st.button("Predict"):
    # Replace 'model' with your actual trained model variable
    # prediction = model.predict(x_new)
    # For demonstration, we'll use a placeholder
    prediction = "Your Prediction Here"
    st.write(f"Prediction: {prediction}")
    streamlit run app.py
