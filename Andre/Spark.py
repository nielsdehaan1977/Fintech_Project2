
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the selected model
best_model = load_model('best_model.h5')

# Define the input fields
age = st.slider('Age', min_value=18, max_value=100, value=25, step=1)
bmi = st.slider('BMI', min_value=10, max_value=50, value=25, step=0.1)
glucose = st.slider('Glucose Levels', min_value=50, max_value=200, value=100, step=1)
bp = st.slider('Blood Pressure', min_value=50, max_value=200, value=100, step=1)
skin_thickness = st.slider('Skin Thickness', min_value=0, max_value=100, value=20, step=1)
dpf = st.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.0, value=0.5, step=0.01)
insulin = st.slider('Insulin', min_value=0, max_value=846, value=79, step=1)

# Make predictions on the user inputs
X_input = np.array([age, bmi, glucose, bp, skin_thickness, dpf, insulin]).reshape(1, -1)
y_pred = best_model.predict(X_input)[0]

# Display the prediction results
if y_pred == 0:
    st.write('You are at a low risk of developing diabetes.')
else:
    st.write('You are at a high risk of developing diabetes.')


