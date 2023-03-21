import os
import streamlit as st

# set folder path for stored models
folder_path = Path('./Models/')

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Create a dropdown menu with the file names
selected_file = st.selectbox('Select a file', file_list)

# Print the selected file name
st.write('You selected:', selected_file)
