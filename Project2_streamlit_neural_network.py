# Imports
import pandas as pd
from pathlib import Path
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder

# create tab as container
tab = st.container()

#create tabs for display in steamlit
tab1, tab2, tab3, tab4 = st.tabs(['Project Objective','Data','Data cleaning and analysis','Machine Learning Model'])

# Import Data
# set data path
path_csv = Path('./Resources/diabetes_data_kaggle.csv')
# read in data
df = pd.read_csv(path_csv)

# check if there are categorial variables in the data yes/no if so use onehotencoder to encode variable int new dataframe
categorical_variables = list(df.dtypes[df.dtypes == "object"].index)
# make a serie of the categorical variables
my_cat_variables_serie = pd.Series(categorical_variables)

# check if there are any categorial variables if so us OneHotEncoder to make numbers out of categorical columns, else pass
if not my_cat_variables_serie.empty:
    # Create a OneHotEncoder instance
    enc = OneHotEncoder(sparse=False)

    # Encode the categorcal variables using OneHotEncoder
    encoded_data = enc.fit_transform(df[categorical_variables])
    
    # Create a DataFrame with the encoded variables
    encoded_df = pd.DataFrame(
    encoded_data,
    columns = enc.get_feature_names_out(categorical_variables)
    )
    df = encoded_df
else:
    pass


# Create list of columns names to select as X features and y label
columns_names_all = df.columns.tolist()

# Create sidebar with user input for neural network
with st.sidebar:
    st.header('Selection Options')
    # create slider for user to indicate how much training data to select:
    training_data_depth = st.slider('How large should the testing set be?',min_value=10, max_value=100,value=50,step=10) 
    
    # Define the number of neurons in the output layer
    output_neurons = st.slider('How many output Neurons?',min_value=1,max_value=10,value=1,step=1)

    # Define the number of hidden nodes for the first hidden layer
    hidden_nodes_layer_1 = st.slider('How many hidden nodes in layer 1?',min_value=1,max_value=10,value=2,step=1)

    # Define the number of hidden nodes for the second hidden layer
    hidden_nodes_layer_2 = st.slider('How many hidden nodes in layer 2?',min_value=0,max_value=5,value=1,step=1)

    # create drop down to select the amount of n_estimators
    n_estimators = st.selectbox('How many estimators',[100,200,300,'no limit'],index = 0)
    
    # create drop down to select how many epochs to run
    n_epochs = st.selectbox('How many epochs would you like to run',[20,50,100,200,500,1000],index=0)
    
    # Select y column from list of column names
    label_select = st.selectbox("Select Y column", options=columns_names_all)

    # Remove selected y column from X (features)
    column_names_x=df.drop(columns=label_select).columns.tolist()

    # Create a multi select box for users to select as features
    features_select = st.multiselect("Select X columns",options=column_names_x, default=column_names_x)
    

    #st.write(list_of_columns)

# INTRODUCTION TAB PROJECT OBJECTIVE
with tab1:

    st.title('Machine Learning Diabetes Predictions')

    st.header('Project Objective:')

    st.subheader('Diabetes Predictions')

    st.text('In this project we try to predict if a person has diabetes using machine learning')

    st.image('Images/Machine_Learning.jpg',use_column_width=True)


    
    #usd_amount_2 = st.number_input('How much money would you like to invest (in USD)?', min_value=500, value=500, step=500)

# DATA TAB
with tab2:
    # type header of the tab
    st.header('Data used for Project')
    # type where you find the dataset
    st.text('I found this dataset on kaggle on below link')
    # create link to dataset used
    st.write('Please click here for dataset [link](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)')

    # Display dataset in tab
    st.subheader('Display first 5 rows of Data')
    st.write(df.head())

    #display datatypes of dataset
    st.subheader('Dipslay datatypes of the data')
    st.write(df.dtypes)

    # Count and display amount of different values in y (label)
    st.header('Distribution of selected y label')
    column_y_values = df[label_select].value_counts()
    st.write(column_y_values)

    # Create a data distribution
    BMI_dist = pd.DataFrame(df['BMI'].value_counts())
    
    # create a barchart of the distribution
    st.subheader('Distribution Chart')
    st.bar_chart(BMI_dist)

# DATA CLEANING AND ANALYSIS TAB
with tab3:
    st.header('Data Cleaning and Analysis')

    st.markdown('* **My first feature **')

# MACHINE LEARNING MODEL TAB
with tab4:
    st.header('Train Model')
    st.text('Use the following model')

    # create 2 columns in model training tab
    sel_col,disp_col = st.columns(2)

    # Create the labels set (y) from the “Outcome” column, and then create the features (X) DataFrame from the remaining columns.
    X = df[features_select]
    y = df[label_select]

    # Check if label is properly balanced
    data_org = y.value_counts()
    st.write(data_org)

    # Assign a random_state of 1 to the function
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1) 

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Fit the scaler to the features training dataset
    X_scaler = scaler.fit(X_train)

    # Fit the scaler to the features training dataset
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Define the the number of inputs (features) to the model
    number_input_features = len(X_train.iloc[0])

    # Review the number of features
    number_input_features
    st.subheader('Number of input features is:')
    st.write(number_input_features)

    # Define the number of neurons in the output layer
    number_output_neurons = 1

    # Select Model
    model = LogisticRegression(random_state=1)
    
    # Fit Model using training data
    lr_original_model = model.fit(X_train,y_train)

    # Make a prediction using the testing data
    y_original_pred = lr_original_model.predict(X_test)

    conf_matrix = confusion_matrix(y_test,y_original_pred)
    st.write(conf_matrix)

    class_rep = classification_report_imbalanced(y_test,y_original_pred)
    st.write(class_rep)


    
