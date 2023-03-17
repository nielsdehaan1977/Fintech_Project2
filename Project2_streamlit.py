import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from imblearn.metrics import classification_report_imbalanced

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


#header = st.container()
#dataset = st.container()
#features = st.container()
#model_training = st.container()
tab = st.container()

path_csv = Path('./Resources/diabetes_data_kaggle.csv')
df = pd.read_csv(path_csv)

tab1, tab2, tab3, tab4 = st.tabs(['Project Objective','Data','Data cleaning and analysis','Machine Learning Model'])


with st.sidebar:
    st.header('Sidebar')
    training_data_depth = st.slider('How large should the testing set be?',min_value=10, max_value=100,value=50,step=10) 
    n_estimators = st.selectbox('How many estimators',[100,200,300,'no limit'],index = 0)
    #input_feature = st.text_input()


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

    # display datatypes of dataset
    #st.write(df.dtypes)
    
    # Display dataset in tab
    st.write(df.head())

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
    X = df.drop(columns='Outcome')
    y = df['Outcome']

    data_org = y.value_counts()
    st.write(data_org)

    # Assign a random_state of 1 to the function
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1) 

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


    
