import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Load the preprocessed dataset
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Load the trained neural network
model = tf.keras.models.load_model('diabetes_model.h5')

# Define the Streamlit app
st.title('Diabetes Predictor')
st.write('Enter your information below to predict your risk of diabetes.')

# Add input fields for user data
age = st.slider('Age', 1, 100, 25)
bmi = st.slider('BMI', 10.0, 50.0, 25.0)
pregnancies = st.slider('Pregnancies', 0, 10, 0)
glucose = st.slider('Glucose', 0, 200, 100)
blood_pressure = st.slider('Blood Pressure', 0, 150, 70)
skin_thickness = st.slider('Skin Thickness', 0, 100, 20)
insulin = st.slider('Insulin', 0, 300, 0)
dpf = st.slider('Diabetes Pedigree Function', 0.0, 2.0, 0.5)

# Make a prediction based on user input
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
prediction = model.predict(input_data)

# Display the prediction to the user
if prediction[0] >= 0.5:
    st.write('You are at high risk for diabetes.')
else:
    st.write('You are at low risk for diabetes.')
