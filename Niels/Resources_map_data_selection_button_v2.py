import pandas as pd
from pathlib import Path
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt
import os

data_source = 'https://www.kaggle.com/datasets/mathchi/diabetes-data-set'

# create tab as container
tab = st.container()

#create tabs for display in steamlit
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Project Objective','Data Selection','Original Data','Data Preparation','Machine Learning Model', 'Predictions'])

with tab2:
    
    with st.form(key='data_form_1'):

            # Set the folder path for outputted Models
        folder_path_data = Path('../Resources/')

        # Get a list of all Models in the folder
        data_list = os.listdir(folder_path_data)

        # Select CSV files only
        csv_list = list(filter(lambda f: f.endswith('.csv'), data_list))

         # Create a dropdown menu with the file names
        #selected_model = st.selectbox('Select a file', data_list)

        selected_model = st.selectbox('Select a file', csv_list)

        data_path = os.path.join(folder_path_data, selected_model)

        submit_button_data = st.form_submit_button(label='Load File')
            
        st.write('Data Selected:', data_path)

        #df = pd.read_csv(selected_model)
        df = pd.read_csv(data_path)
        st.write(df.head())


    # if select_input_data:

    #     # Get the file path for the selected file
        

    #     # read selected data
    #     df = pd.read_csv(data_path)

    #     # Print the selected file contents
        

    #     st.write(df.head())

with tab6:
                # Set the folder path for outputted Models
    with st.form(key='my_form1'):
    
        folder_path_models = Path('../Models/')

        # Get a list of all Models in the folder
        model_list = os.listdir(folder_path_models)

        # Create a dropdown menu with the file names
        selected_model = st.selectbox('Select a file', model_list)

        #select_model_data = st.button('Correct Model Selected?')

        model_path = os.path.join(folder_path_models, selected_model)
        
        submit_button_model = st.form_submit_button(label='Load Model')

        st.write('Model Loaded:', model_path)
    # if select_input_data:

    #     # Get the file path for the selected file
    #     model_path = os.path.join(folder_path_models, selected_model)

    #     # read selected data
    #     #df = pd.read_csv(model_path)

        

    #     # Print the selected file contents
    #     st.write('Model Selected:', model_path)

    #     st.write(df.head())