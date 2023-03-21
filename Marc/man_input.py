import pandas as pd
from pathlib import Path
import streamlit as st

col1, col2, col3, col4, col5, col6,col7 = st.columns(7)
with col1:
    with st.form('Form1'):
        st.slider(label='BMI', min_value=0, max_value=100, key=1)
        submitted1 = st.form_submit_button('Submit 1')

with col2:
    with st.form('Form2'):
        st.slider(label='Glucose', min_value=0, max_value=200, key=2)
        submitted2 = st.form_submit_button('Submit 2')
with col3:
    with st.form('Form3'):
        st.slider(label='BloodPressure', min_value=0, max_value=200, key=3)
        submitted3 = st.form_submit_button('Submit 3')
with col4:
    with st.form('Form4'):
        st.slider(label='SkinThickness',  min_value=0, max_value=100, key=4)
        submitted4 = st.form_submit_button('Submit 4')
with col5:
    with st.form('for5'):
        st.slider(label='DiabetesPedigreeFunction', min_value=0.0, max_value=8.0, key=5)
        submitted5 = st.form_submit_button('Submit 5')
with col6:
    with st.form('Form6'):
        st.slider(label='Insulin', min_value=0, max_value=200, key=6)
        submitted6 = st.form_submit_button('Submit 6')
with col7:
    with st.form('Form7'):
       st.slider(label='Age', min_value=0, max_value=100, key=7)
       submitted7 = st.form_submit_button('Submit 7')
