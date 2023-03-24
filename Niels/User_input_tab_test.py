import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

model_folder = Path('./Models/')
model_name = '0.80(acc)_0.51(loss)_epochs_20_L1(10)_L2(5)_OutputLayer(1)_Scale_Yes_Sampling_RandomOverSampler.h5'

model_path = model_folder / model_name

# Load the trained neural network
model = tf.keras.models.load_model(model_path)



# Use Selected Model to make predictions
st.title('Diabetes Predictor')
st.write('Enter your information below to predict your risk of diabetes.')

# Create User Input sheet
age = st.empty()
age.text_input("input your age", value="", key="1")

bmi = st.empty()
bmi.text_input("Insert your BMI", value="", key="2")

glucose = st.empty()
glucose.text_input("Insert your Glucose Levels", value="", key="3")

pregnancies = st.slider('Pregnansies',0,20,0)

blood_pressure = st.empty()
blood_pressure.text_input("Insert your bloodpressure", value="", key="4")

skin_thickness = st.empty()
skin_thickness.text_input("Insert your thinkness", value="", key="5")


skin_thickness = st.slider('Skin Thickness', 0, 100, 20)
insulin = st.slider('Insulin', 0, 300, 0)
diabetespedigreefunction = st.slider('Diabetes Pedigree Function', 0.0, 2.0, 0.5)

# Use model and input data to make prediction
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age]])
prediction = model.predict(input_data)

# Display the prediction to the user
if prediction[0] >= 0.5:
    st.write('You are at high risk for diabetes.')
else:
    st.write('You are at low risk for diabetes.')
