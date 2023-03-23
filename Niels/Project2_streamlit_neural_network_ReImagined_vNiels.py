# Import Required Libraries
import pandas as pd
import numpy as np
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

### NIELS -- Streamlit development --- INTRODUCE CACHING of DATAFRAME
# create tab as container
tab = st.container()

#create tabs for display in steamlit
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(['Project','Data Selection','Original Data','Data Preparation','Setup ML Model','Model Performance','Predictions','Recommendations'])

# Set Paths for different folders 
images_path = Path('../Images/')
data_path = Path('../Resources/')
model_path = Path('../Models/')

# create selectbox for sampling oversampling, undersampling or no sampling (use dataset as is)
sampling_dataset = ['None','RandomOverSampler','RandomUnderSampler']

# create variables that are inputs into the models, activations, optimizers and loss variables
activations_layers = ['relu','selu','deserialize','elu','exponential','gelu','get','hard_sigmoid','linear','serialize','sigmoid','softmax','softplus','softsign','swish','tanh']
activations_output = ['sigmoid','relu','selu','deserialize','elu','exponential','gelu','get','hard_sigmoid','linear','serialize','softmax','softplus','softsign','swish','tanh']
compile_optimizer = ['adam','sgd','adagrad','adadelta','rmsprop','optimizer','nadam','ftrl','adamax']
compile_loss_probalistic = ['binary_crossentropy','categorical_crossentropy','sparse_categorical_crossentropy','poisson','KLDivergence','kl_divergence']
compile_loss_regression = ['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error','mean_squared_logarithmic_error','cosine_similarity','huber','log_cosh']
compile_loss_hinge = ['hinge','squared_hinge','categorical_hinge']
compile_metrics = ['accuracy','binary_accuracy','binary_crossentropy','categorical_accuracy','categorical_crossentropy','categorical_hinge','cosine_similarity','f1_score','fbeta_score','hinge','kullback_leibler_divergence','logcosh','mean','mean_absolute_error','mean_absolute_percentage_error','mean_squared_error','mean_squared_logarithmic_error','mean_tensor','poisson','r2_score','root_mean_squared_error','sparse_categorical_accuracy','sparse_categorical_crossentropy','sparse_top_k_categorical_accuracy','squared_hinge','sum','top_k_categorical_accuracy','']

# create selectbox for which loss class to use probabilistic, regression or hinge
output_goal = ['predict probability distribution', 'predict continues numerical value','predict classification']

### JASON --- Introduction to why this app is useful for insurance companies... 
# INTRODUCTION TAB PROJECT OBJECTIVE
with tab1:

    st.title('Neural Network Predictions')

    st.header('Project Objective:')

    st.subheader('Diabetes Predictions')

    st.text('In this project we try to predict if a person has diabetes using machine learning')
    
    st.image(os.path.join(images_path,'Neural_Networks_2.jpg'),use_column_width=True)

@st.cache_data
def load_selected_data(data_file_path):
    df = pd.read_csv(data_file_path)
    # drop empty columns instantly before going any further
    df = df.dropna(axis='columns',how='all')
    return df

### NIELS Streamlit app development
# Create a TAB where user can select any csv file from set folder
with tab2:
    
    with st.form(key='data_form_1'):

        st.header('Please select datafile:')

        # Set the folder path for outputted Models
        folder_path_data = data_path

        # Get a list of all Models in the folder
        data_list = os.listdir(folder_path_data)

        # Select CSV files only
        csv_list = list(filter(lambda f: f.endswith('.csv'), data_list))

        # Select a model from the list to Analyze (CSV ONLY)
        selected_model = st.selectbox('Select a file', csv_list)

        # Data path and which file was selected
        data_file_path = os.path.join(folder_path_data, selected_model)

        # Set a button that submits the selected file for further processing
        submit_button_data = st.form_submit_button(label='Load File')
        
        # provide a sentence that tells you which file was selected and used 
        st.write('Data Selected:', data_file_path)

        # Read selected file with read_csv
        df = load_selected_data(data_file_path)




### NIELS --- streamlit appp fixing if other dataset is used program seems to run into run time error 

# Create a TAB where user can select Y column for machine Learning Model.       
with tab3:

    # Provide a overview of the first 5 rows of the data file
    
    st.subheader('Overview of Original Data Selected in Data Selection tab')
    st.write(df.head())

    with st.form(key='feature_label_select_1'):
        
        # Create Header to Display what is required from user
        st.header('Selection Your y column')
        
        # Create list of columns names to select as X features and y label
        columns_names_all = df.columns.tolist()

        # Select y column from list of column names
        label_select = st.selectbox("Select y column", options=columns_names_all)

        # Remove selected y column from X (features)
        column_names_x=df.drop(columns=label_select).columns.tolist()

        # Create a button to select the Y column wihthn the form
        submit_button_y = st.form_submit_button(label='Select y')

        # Create the labels set (y) after selection
        y = df[label_select]

        # Display which Y column is currently selected
        st.subheader(f'Current Selected y Column is: {label_select}')



### MARC data cleaning _- NIELS to fix steamlit ---> all dataclearning features under 1 buttons
#Create a TAB wehere user can enhance the dataset
with tab4:

    # Start Preparation of DATA for Mahchine Learning Model
    st.subheader('Prepare Data for Machine Learning Model')
    st.subheader(f'Current Selected y Column is: {label_select}')
    st.write('If you want to change y Column, please go back to tab ORIGINAL DATA')
    st.write(df.head())     


    
    # Display the correlation matrix between X features and y Label:
    # Create a correlation matrix for the dataframe before any adjustments
    corr_matrix = df[column_names_x + [label_select]].corr()

    # Create a heatmap of the correlation matrix
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix,annot=True,cmap='inferno',ax=ax)

    #display correlation matrix heatmap in streamlit
    st.subheader('Correlation matrix heatmap Original Data')
    st.pyplot(fig) 

# Create a Form to make changes to the original data
    with st.form(key='Data_select_1'):
        st.header('Data Enhancements')

        # Remove any of the columns that "seem" insignificant don't have much correlation with Y column
        st.subheader('1. Would you like to remove any feature Columns (X)')

        # Create a multi select box for users to select as features
        features_select = st.multiselect("Select X columns",options=column_names_x, default=column_names_x)

        # Create a button to select the Y column wihthn the form
        submit_button_x_select = st.form_submit_button(label='Select Features')

        # Create the features set (X) from original DATAFRAME
        X = df[features_select]
        X_remaining_features = X.copy()
        st.write(X.head())

    # check if there are categorial variables in the data yes/no if so use onehotencoder to encode variable in tje X dataframe
    categorical_variables = list(X.dtypes[X.dtypes == "object"].index)
    # make a serie of the categorical variables
    my_cat_variables_serie = pd.Series(categorical_variables)

    # check if there are any categorial variables if so us OneHotEncoder to make numbers out of categorical columns, else pass
    st.header('Categorical Variables Handling (If any)')
    if not my_cat_variables_serie.empty:
        with st.form(key='Data_select_2'):
            # If there are any objects in the dataset would you like to use OnehotEnhancer?
            st.subheader('Are there any columns that require OneHotEnhancer based on dtypes overview?')
            # create a yes or no button to apply onehotenhancer to the remaining dataset

            # indicate to use standard scaler yes/no
            onehotenhancer_select = st.radio('Use One Hot Enhancer (No/Yes)?',('No','Yes'))

            # Create a button to select the Y column wihthn the form
            submit_button_ohe_select = st.form_submit_button(label='Apply')

            if onehotenhancer_select == 'No':
                exit()

            # Check if onehotenhancer need to applied according to user
            else:
                  
                # Create a OneHotEncoder instance
                enc = OneHotEncoder(sparse=False)

                # Encode the categorcal variables using OneHotEncoder
                encoded_data = enc.fit_transform(X[categorical_variables])
        
                # Create a DataFrame with the encoded variables
                encoded_df = pd.DataFrame(
                encoded_data,
                columns = enc.get_feature_names_out(categorical_variables))

                # concatenate both old X and new categorical df(encode_df)
                X = pd.concat([X.drop(columns = categorical_variables),encoded_df],axis=1)

                # show adjusted X features data
                st.subheader('Adjusted X features')
                st.write(X.head())

                # check if there are categorial variables in the data yes/no if so use onehotencoder to encode variable in tje X dataframe
                categorical_variables_check2 = list(X.dtypes[X.dtypes == "object"].index)
                # make a serie of the categorical variables
                my_cat_variables_serie_check2 = pd.Series(categorical_variables_check2)

                if not my_cat_variables_serie_check2.empty:
                    st.write('Dataset still has categorical features, please use OneHotEnhancer to remove')
                    exit()
                else:
                    pass

    # if there are nocategorial variables pass and print that there are no categorical variables available to use onehotenahncer on
    else:
        st.write('No categorical variables in Dataset')
        pass


    # Check dtypes of data
    st.subheader('Dipslay datatypes of Remaining X Features')
    x_remaining_dtypes = X.dtypes
    st.write(x_remaining_dtypes)

    #display if there are any categorical variables in dataset
    st.header('Categorical Variables in adjusted Dataset')
    
    # See if the remaining data set still has categorical variable yes/no if not pass, otherwise print the categorical variables
    try:
        st.write(my_cat_variables_serie_check2)
    except:
        pass
        
    # make a form in streamlit if you want to apply standard scaler to the dataset no/yes        
    with st.form(key='Data_select_3'):
        st.subheader('Would you like to apply Standard Scaler to the Dataset?')
        
        # indicate to use standard scaler yes/no
        standard_scaler_select = st.radio('Apply Standard Scaler (No/Yes)',('No','Yes'))

        # Create a button to select to apply standard scaler
        #submit_button_standard_scaler_select = st.form_submit_button(label='Apply')       

        
        # Create a StandardScaler instance if requested by user
        if standard_scaler_select == 'Yes':
            scaler = StandardScaler()

            # Fit the scaler to the features training dataset
            X_scaled = scaler.fit_transform(X)

            # Display to be used X dataframe indicate if standard scaler is used yes/no
            st.subheader(f'Features DataFrame (Standard Scaler Used? {standard_scaler_select})')
            st.write(X_scaled)
        else:
            # If scaler should not be applied just name X --> X-scaler (to make further processing possible with one variable)
            st.write('No changes were made to the dataset, Standard Scaler not applied to dataset')
            X_scaled = X
            st.write(X_scaled)


    #with st.form(key='Data_select_4'):
        
        # Select the percentage of training data you want to use:
        st.subheader('Select the Percentage Amount of data to be used as test data')
        test_data_select = st.slider('What percentage of data should be test data?',min_value=0.1,max_value=0.99,value=0.2,step=0.1)

        # Create a button to select to % to apply on testing dta
        submit_button_test_data_select = st.form_submit_button(label='Apply')       

        # Split up data using train test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=test_data_select,random_state=1) 

        # check the split up data set
        xtrain_count = X_train.shape[0]
        xtest_count = X_test.shape[0]
        ytrain_count = y_train.shape[0]
        ytest_count = y_test.shape[0]

        st.subheader('Data allocation after train test split')
        st.text(f'X_train = {xtrain_count}\nX_test = {xtest_count}\ny_train = {ytrain_count}\ny_test = {ytest_count}')



    # create a form to indicate if any sampling needs to be applied to database
    with st.form(key='Data_select_5'):
        st.subheader('Based on the Distribution of chosen column y, please indicate if you would like to use OverSampling/Undersampling or None')

        # Count and display amount of different values in y (label)
        st.header('Distribution of y column original data')
        column_y_values = y.value_counts()
        st.write(column_y_values)

        # Indicate to use random oversampling or undersampling or none
        sampling_dataset_select = st.selectbox('Select Sampling of DataSet',sampling_dataset)

        # Create a button to select to apply standard scaler
        submit_button_sampling_select = st.form_submit_button(label='Apply') 

        # apply the selected sampling method to the dataset
        if sampling_dataset_select == 'RandomOverSampler':
            sampler = RandomOverSampler(random_state=1)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        elif sampling_dataset_select == 'RandomUnderSampler':
            sampler = RandomUnderSampler(random_state=1)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train
            st.text('No Sampling Applied')


        # Count the distinct values of the resampled labels data
        y_resampled_count = y_resampled.value_counts()
        st.subheader(f'y label to be used after train test split, (resampling method used: {sampling_dataset_select})')
        st.write(y_resampled_count)

    st.text('PLEASE PROCEED TO MACHINE LEARNING TAB')

#### NIELS will double check what going on here
with tab5:
    st.header('Neural Network')

    # Define the the number of inputs (features) to the model
    number_input_features = len(X.iloc[0])


    with st.form(key='neural_net_feat_1'):
        
        # select what kind of outpot the model is used for
        compile_loss_select = st.selectbox('Model Loss Selection Model',output_goal)

        # Change compile loss based on compile loss select
        if compile_loss_select == 'predict probability distribution':
            compile_loss_select_used = st.selectbox('Model Loss Selection Options',compile_loss_probalistic)
        elif compile_loss_select == 'predict continues numerical value':
            compile_loss_select_used = st.selectbox('Model Loss Selection Options',compile_loss_regression)
        else:
            compile_loss_select_used = st.selectbox('Model Loss Selection Options',compile_loss_hinge)

        # select what kind of optimizer model should use in compile
        compile_optimizer_select = st.selectbox('Model optimizer select',compile_optimizer)

        # select what kind of metric model should use in compile
        compile_metric_select = st.selectbox('Model metric select',compile_metrics)

        # Define the number of neurons in the output layer
        output_neurons = st.slider('How many output Neurons?',min_value=1,max_value=10,value=1,step=1)
    
        # Define activation for output layer
        output_activation = st.selectbox('Output Layer Activation',activations_output)

        # define number of hidden nodes for first and second layer based upon provided best practices in NNs
        hidden_nodes_layer_1_calc = (number_input_features + output_neurons)//2
        hidden_nodes_layer_2_calc = (hidden_nodes_layer_1_calc + output_neurons)//2

        st.write(f'Number of input features into NN are: {number_input_features}')
        st.write(f'Number of standard Layer 1 nodes (Best Practices NN): {hidden_nodes_layer_1_calc}')
        st.write(f'Number of standard Layer 2 nodes (Best Practices NN): {hidden_nodes_layer_2_calc}')


        # Define the number of hidden nodes for the first hidden layer
        hidden_nodes_layer_1 = st.slider('How many hidden nodes in layer 1?',min_value=1,max_value=50,value=hidden_nodes_layer_1_calc,step=1)

        # Define activation for hidden layer1
        layer_1_activation = st.selectbox('Layer 1 Activation',activations_layers)

        # Define the number of hidden nodes for the second hidden layer
        hidden_nodes_layer_2 = st.slider('How many hidden nodes in layer 2?',min_value=0,max_value=50,value=hidden_nodes_layer_2_calc,step=1)

        # Define activation for hidden layer2
        layer_2_activation = st.selectbox('Layer 2 Activation',activations_layers)

        # create drop down to select how many epochs to run
        n_epochs = st.selectbox('How many epochs would you like to run',[20,50,100,200,500,1000],index=0)

        # Create a button to select to nn specifics
        submit_button_nn_specifics = st.form_submit_button(label='Apply and Run') 

        # Create overview of features of Neural Network 
        features_nn = ['Input Features', 'Amount of Hidden Layers Layer 1','Layer 1 Optimizer', 'Amount of Hidden Layers Layer 2','Layer 2 Optimizer', 'Amount of Output Layer(s)', 'Output Layer Activation','Standard Scaler', 'Sampling','Epochs','Model Optimizer']
        inputs_nn = [number_input_features, hidden_nodes_layer_1,layer_1_activation, hidden_nodes_layer_2,layer_2_activation, output_neurons,output_activation, standard_scaler_select, sampling_dataset_select,n_epochs,compile_optimizer_select ]
        data_nn = {'Features_nn':features_nn,'Input nn':inputs_nn}
        nn_characteristics = pd.DataFrame(data_nn)
    
        # Select Model
        nn = Sequential()
        
        # Add the first hidden layer
        nn.add(Dense(units=hidden_nodes_layer_1, input_dim=number_input_features, activation=layer_1_activation))

        # Add the second hidden layer
        nn.add(Dense(units=hidden_nodes_layer_2, activation=layer_2_activation))

        # Add the output layer to the model specifying the number of output neurons and activation function
        nn.add(Dense(units=output_neurons, activation=output_activation))

        # Compile the Sequential model_original
        nn.compile(loss=compile_loss_select_used, optimizer=compile_optimizer_select, metrics=compile_metric_select)
        
        # Fit the model using n_epochs variable epochs and the training data
        fit_model = nn.fit(X_resampled, y_resampled, epochs=n_epochs)
    
        # Evaluate the model loss and accuracy metrics using the evaluate method and the test data
        model_loss, model_accuracy = nn.evaluate(X_test, y_test, verbose=2)

        st.subheader('Neural Network Loss and Model Accuracy')
        st.write(f'Loss: {model_loss:.3f}, Accuracy: {model_accuracy:.3f}')

        
        file_path = Path(f'../Models/{model_accuracy:.2f}(acc)_{model_loss:.2f}(loss)_epochs_{n_epochs}_L1({hidden_nodes_layer_1})_L2({hidden_nodes_layer_2})_OutputLayer({output_neurons})_Scale_{standard_scaler_select}_Sampling_{sampling_dataset_select}.h5')
        nn.save(file_path)
        st.write(f'Model save as: {file_path}')


### ANDRE
# Model Performace TAB
with tab6:
    st.header('Performance of  Model:')
    
    # Display the model loss and accuracy results
    #loss_accuracy = print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
    st.subheader('Neural Network Loss and Model Accuracy')
    st.write(f'Loss: {model_loss:.3f}, Accuracy: {model_accuracy:.3f}')

    # Create Loss and accuracy graph per Epoch passed
    fig,ax =plt.subplots()
    ax.plot(fit_model.history['loss'],label='Training Loss')
    ax.plot(fit_model.history['accuracy'],label='Training Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss/Accuracy')
    ax.legend()
    st.subheader('Loss and Accuracy graph per Epoch')
    st.pyplot(fig)


    #show neural network characteristics in streamlit
    st.subheader('This is based on a Neural Network with the following Characteristics')
    st.write(nn_characteristics)   




#### MARC
# Create Tab with Form fo ruser to put in patient values and make a prediction
with tab7:
    st.header('Predictions')
    # Check which features are left after removing uncorrelated ones for example
    #st.write(X_remaining_features)

    # Load selected Model into tf.kera.models.load_model
    model_loaded = tf.keras.models.load_model(file_path)   
    st.write(f'Model Loaded: {file_path}')



    st.subheader('Input Form for User Input (only used X_Features in Model)')
    # Define function to create form fields based on column data type
    def form_input_field(X_feature, dtype):
        if dtype == "object":
            return st.text_input(X_feature)
        elif dtype == "int64":
            return st.number_input(X_feature)
        elif dtype == "float64":
            return st.number_input(X_feature, format="%f")
        else:
            return st.text_input(X_feature)

    # Define function to create form based on column headers and data types
    def create_form(remaining_columns):
        for i in remaining_columns.columns:
            dtype = remaining_columns[i].dtype
            form_field = form_input_field(i, dtype)
            st.write(form_field)


    with st.form(key='input_form_1'):
        # Create the flexible form in this form by running functions above
        create_form(X_remaining_features)

        # Create a button to select to apply standard scaler
        submit_button_save_user_input = st.form_submit_button(label='Save/Apply User Input') 
        


        # # Reset index of new to build np array    
        # x_remaining_array = x_remaining_dtypes.reset_index(drop=False)

        # # rename column names
        # x_remaining_array.rename(columns={x_remaining_array.columns[0]:'X_Features',x_remaining_array.columns[1]:'dtype'},inplace=True)

        
        # # # create nuympy array from X_remaining_features column names
        # x_remaining_arr = x_remaining_array[['X_Features']].to_numpy()

        # st.write(x_remaining_arr)

#### JASON
with tab8:
    st.header('Recommendations') 
    ### If high risk ---> whats the cost per individual? 
    ### what kind of things can we suggest 
    ### If BMI is 30 --> exercise more --> suggest walking more
