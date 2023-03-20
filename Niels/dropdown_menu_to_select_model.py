import os
import streamlit as st

# Set the folder path
folder_path = 'C:/Users/Username/Folder/'

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Create a dropdown menu with the file names
selected_file = st.selectbox('Select a file', file_list)

# Print the selected file name
st.write('You selected:', selected_file)
