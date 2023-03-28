import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt
import os
#import hydralit_components as hc

### NIELS -- Streamlit development --- INTRODUCE CACHING of DATAFRAME

st.set_page_config(page_title="NN Diabetes Predictions", layout='wide')

#menu_data = [
 #   {'label': 'Project'}, {'label': 'Data Selection'}, {'label': 'Original Data'}, {'label':'Data Preparation'}, {'label': 'User_Input'}, {'label':'Setup ML Model'}, {'label', 'Model Performance'}, {'label': 'Predictions'}, {'label': 'Recommendations'}]

#menu_id = hc.nav_bar(menu_definition=menu_data, sticky_nav=True, sticky_mode='pinned')

# create tab as container
tab = st.container()

#create tabs for display in steamlit
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(['Project','Data Selection','Original Data','Data Preparation','User_Input','Setup ML Model','Model Performance','Predictions','Recommendations'])

# Set Paths for different folders 
images_path = Path('./Images/')
data_path = Path('./Resources/')
model_path = Path('./Models/')

# create variables that are inputs into the models, activations, optimizers and loss variables

# To accomodate for this project we only need Probalistic prediction so we can skip regression to make it less complex. For future you can include in below list
#output_goal = ['Probalistic', 'Regression']
output_goal = ['Probalistic']

# Make selection of probabilistic loss options available: (all options work )
compile_loss_probalistic = ['binary_crossentropy','categorical_crossentropy','poisson','KLDivergence','kl_divergence']

# Make Selection for Regression compile loss options
compile_loss_regression = ['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error','mean_squared_logarithmic_error']

# Make selection of model optimizers options available: (all options work)
compile_optimizer = ['adam','sgd','adagrad','adadelta','rmsprop','nadam','ftrl','adamax']

# Make selection of activation output layer Probability options available: (all options work)
activations_output_probalistic = ['sigmoid','relu','selu','elu','exponential','gelu','hard_sigmoid','linear','softmax','softplus','softsign','swish','tanh']

# Make selection of activation output layer Regression options available:
activations_output_regression = ['linear','sigmoid','relu','selu','elu','exponential','gelu','hard_sigmoid','softmax','softplus','softsign','swish','tanh']

# Make selection of activation layer options available:
activations_layers = ['relu','selu','elu','exponential','gelu','hard_sigmoid','linear','sigmoid','softmax','softplus','softsign','swish','tanh']

# create selectbox for sampling oversampling, undersampling or no sampling (use dataset as is)
sampling_dataset = ['None','RandomOverSampler','RandomUnderSampler']

### JASON --- Introduction to why this app is useful for insurance companies... 
# INTRODUCTION TAB PROJECT OBJECTIVE
with tab1:

    st.title('Neural Network Predictions')

    st.header('Project Objective:')

    st.subheader('Diabetes Predictions')

    st.text('In this project we try to predict if a person has diabetes using machine learning')
    
    st.image(os.path.join(images_path,'Neural_Networks_2.jpg'),use_column_width=True)

### NIELS -- Streamlit development --- INTRODUCE CACHING of DATAFRAME
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

### NIELS 
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

### MARC data cleaning _- 
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
    cat_var = my_cat_variables_serie.value_counts()

    # check if there are any categorial variables if so us OneHotEncoder to make numbers out of categorical columns, else pass
    st.header(f'Categorical Variables Handling (If any): {cat_var}')
    st.subheader('Categorical Variables in Dataset:')
    st.write(my_cat_variables_serie)
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
                cat_var2 = my_cat_variables_serie_check2.value_counts()

                # Check if categorical varibales is now empty
                if not my_cat_variables_serie_check2.empty:
                    st.write(f'Dataset still has categorical features {cat_var2}, please use OneHotEnhancer to remove')
                    exit()
                elif len(cat_var2) == 0:
                    st.subheader('No categorical variables left in "Enhanced" Dataset')
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


        # Count the distinct values of the resampled labels data if a random sampler is selected and applied
        if sampling_dataset_select == 'None':
            pass
        else: 
            y_resampled_count = y_resampled.value_counts()
            st.subheader(f'y label to be used after train test split, (resampling method used: {sampling_dataset_select})')
            st.write(y_resampled_count)

    st.text('PLEASE PROCEED TO SETUP ML MODEL TAB')

# Create flexible User Input Tab (adjusted to selected Xfeatures (otherwise it cannot be used by the model)
with tab5:
    
    # only request questions to user for features that were used in the current model:
    features_used_in_model = X_remaining_features.columns
    # count how many features used in model 
    number_of_features_used_in_model = len(features_used_in_model)

    #st.write(number_of_features_used_in_model)

    # Create input form for user/patient to input his/her data
    st.title('Diabetes input form')
    st.subheader('Units for reference: Glucose in mg/dL, BP in mm Hg, Insulin in uU/mL, BMI in kg/m^2, SkinThickness in mm')
    # make a counter for all the remainign features in this specific model
    model_feature_number = 0
    # make eampty list with answers:    
    answers =[]

    # Create a form that matches the remaining X features    
    with st.form('flex_form'):
        
        # make a for loop throught features used in model 
        for model_feature in features_used_in_model:
            # start counting features
            model_feature_number += 1

        # create input slider for gluvose bloodpressure and insuline
            if model_feature == 'Glucose' or model_feature == 'BloodPressure':
                submitted = str('submitted') + str(model_feature_number)
                submitted = st.slider(label=model_feature,min_value=0,max_value=200,value=100,step=1,key=model_feature_number)
                #st.write(submitted)
                answers.append(submitted)
            elif model_feature == 'Insulin':
                submitted = str('submitted') + str(model_feature_number)
                submitted = st.slider(label=model_feature,min_value=0,max_value=700,value=100,step=1,key=model_feature_number)
                #st.write(submitted)
                answers.append(submitted)
            # create input slider from BMI skinthinkness and age
            elif model_feature == 'BMI' or model_feature =='SkinThickness' or model_feature == 'Age':
                submitted = str('submitted') + str(model_feature_number)
                submitted = st.slider(label=model_feature,min_value=0,max_value=100,value=50,step=1,key=model_feature_number)
                #st.write(submitted)
                answers.append(submitted)
            elif model_feature == 'DiabetesPedigreeFunction':
                submitted = str('submitted') + str(model_feature_number)
                submitted = st.slider(label=model_feature,min_value=float(0),max_value=float(1),value=0.001,step=float(0.001),key=model_feature_number)
                #st.write(submitted)
                answers.append(submitted)
            # create input values for everything else
            else:
                submitted = str('submitted') + str(model_feature_number)
                submitted = st.slider(label=model_feature,min_value=int(0),max_value=int(1),value=int(0),step=int(1),key=model_feature_number)
                #st.write(submitted)
                answers.append(submitted)

        #create submit button for all above input values    
        submit_button_save_user_input = st.form_submit_button(label='Save/Apply User Input')

        #if user data is not saved (no need to run streamlit app totally if no data is inputted)
        if len(answers) == 0:
            exit()
        else:
            #st.write(answers)
            pass


#### NIELS Setup of Neural Network and option to save NN
with tab6:
    st.header('Neural Network')

    # Define the the number of inputs (features) to the model
    number_input_features = len(X.iloc[0])


    with st.form(key='neural_net_feat_1'):
        
        # select what kind of outpot the model is used for Probability (%) or Regerssion (number)
        # For This project we only need Probalisitc model, if you want to add regression you only need to add Regression to output_goal variable
        compile_loss_select = st.selectbox('Purpose of Model',output_goal)
        submit_button_nn_purpose = st.form_submit_button(label='Model Purpose Select') 
        
    # FOR FUTURE YOU CAN CHANGE variable output_goal ton include Regression, but for this project we only need Proballistic and % change for getting Diabetes. 
    with st.form(key='nn_feat_2'):
        if compile_loss_select == 'Probalistic':
            compile_loss_select_used = st.selectbox(f'Model loss selection Probalisitc:',compile_loss_probalistic)
            compile_optimizer_select = st.selectbox(f'Model optimizer selection Probalistic', compile_optimizer)
            compile_metric_select = 'accuracy'
            activations_output = activations_output_probalistic
            st.write('Model Metric used: ', compile_metric_select)
        elif compile_loss_select == 'Regression':
            compile_loss_select_used = st.selectbox(f'Model loss selection Regression:',compile_loss_regression)
            compile_optimizer_select = st.selectbox(f'Model optimizer selection Regression', compile_optimizer)
            compile_metric_select = 'mse'
            activations_output = activations_output_regression
            st.write('Model Metric used: ', compile_metric_select)
        else:
            pass

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
        n_epochs = st.selectbox('How many epochs would you like to run',[1,20,50,100,200,500,1000],index=0)

        # Create a button to select to nn specifics
        submit_button_nn_specifics = st.form_submit_button(label='Apply and Run') 

        # Create overview of features of Neural Network 
        features_nn = ['Model Purpose','Input Features', 'Amount of Hidden Layers Layer 1','Layer 1 Optimizer', 'Amount of Hidden Layers Layer 2','Layer 2 Optimizer', 'Amount of Output Layer(s)', 'Output Layer Activation','Standard Scaler', 'Sampling','Epochs','Model Optimizer']
        inputs_nn = [compile_loss_select,number_input_features, hidden_nodes_layer_1,layer_1_activation, hidden_nodes_layer_2,layer_2_activation, output_neurons,output_activation, standard_scaler_select, sampling_dataset_select,n_epochs,compile_optimizer_select ]
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
    
        # Evaluate the model loss and accuracy metrics using the evaluate method and the test data for probalistic 
        if compile_loss_select == 'Probalistic':
            model_loss, model_accuracy = nn.evaluate(X_test, y_test, verbose=2)
            # display loss and accuracy for probalistic model
            st.subheader('Neural Network Loss and Model Accuracy')
            st.write(f'Loss: {model_loss:.3f}, Accuracy: {model_accuracy:.3f}')
            # Set file path for Regression model:
            file_path_model = Path(f'{model_path}/{model_accuracy:.3f}(acc)_{model_loss:.3f}(loss)_{compile_loss_select}_epochs_{n_epochs}_L1({hidden_nodes_layer_1})_L2({hidden_nodes_layer_2})_OutputLayer({output_neurons})_Scale_{standard_scaler_select}_Sampling_{sampling_dataset_select}.json')
            file_path_weights = Path(f'{model_path}/{model_accuracy:.3f}(acc)_{model_loss:.3f}(loss)_Probalisitic_epochs_{n_epochs}_L1({hidden_nodes_layer_1})_L2({hidden_nodes_layer_2})_OutputLayer({output_neurons})_Scale_{standard_scaler_select}_Sampling_{sampling_dataset_select}.h5')
        
        # Evaluate the model loss and accuracy metrics using the evaluate method and the test data for regression
        elif compile_loss_select == 'Regression':
            loss, mse = nn.evaluate(X_test, y_test, verbose=2)
            st.subheader('Neural Network Loss and Model MSE')
            st.write(f'Loss: {loss:.3f}, Accuracy: {mse:.3f}')
            # Set file path for Regression model:
            file_path_model = Path(f'{model_path}/{mse:.3f}(mse)_{loss:.3f}(loss)_{compile_loss_select}_epochs_{n_epochs}_L1({hidden_nodes_layer_1})_L2({hidden_nodes_layer_2})_OutputLayer({output_neurons})_Scale_{standard_scaler_select}_Sampling_{sampling_dataset_select}.json')
            file_path_weights = Path(f'{model_path}/{mse:.3f}(mse)_{loss:.3f}(loss)_Regression_epochs_{n_epochs}_L1({hidden_nodes_layer_1})_L2({hidden_nodes_layer_2})_OutputLayer({output_neurons})_Scale_{standard_scaler_select}_Sampling_{sampling_dataset_select}.h5')
        else:
            pass
        
        # SAVE THE MODEL to specifiec locations a per above:
        # Save Model in JSON Format
        nn_json = nn.to_json()

        # Write the model to the the file 
        with open(file_path_model,'w') as json_file:
            json_file.write(nn_json)
        # Save the weights to the file path
        nn.save_weights(file_path_weights)
        
        st.write(f'Model save as: {file_path_model}')
        st.write(f'Model weights saved as: {file_path_weights}')


### ANDRE
# Model Performace TAB
with tab7:
    st.header('Performance of Model:')

    # Display the model loss and accuracy results
    # HERE AGAIN WE ALREADY INCLUDED THE OTPION TO USE REGRESSION TO UNLOCK ADD REGRESSION TO OUTPUT_GOAL VARIABLE
    #loss_accuracy = print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
    if compile_loss_select == 'Probalistic':
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

    elif compile_loss_select == 'Regression':
        st.subheader('Regression Neural Network Loss mse')
        st.write(f'Loss:{loss:.3f}, mse: {mse:.3f}')

        # Create Loss and accuracy graph per Epoch passed
        fig,ax =plt.subplots()
        ax.plot(fit_model.history['loss'],label='Loss_function - Training')
        #ax.plot(fit_model.history['mse'],label='Loss Function training')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('loss')
        ax.legend()
        st.subheader('Loss Function Training per Epoch')
        st.pyplot(fig)

        #show neural network characteristics in streamlit
        st.subheader('This is based on a Neural Network with the following Characteristics')
        st.write(nn_characteristics)   
    else:
        pass


#### MARC
# Load Model and create Tab with Form fo ruser to put in patient values and make a prediction
with tab8:
    st.header('Predictions')

    # Load Model
    with open(file_path_model,"r") as json_file:
        model_json = json_file.read()
    loaded_model = model_from_json(model_json)

    st.write(f'Model Loaded: {file_path_model}')

    loaded_model.load_weights(file_path_weights)
    
    # USE LOADED MODEL TO PREDICT POINTS FOR THE TEST DATA
    y_pred = loaded_model.predict(X_test)

    #st.write('Model Predictions based on Test Data')
    #results_model_test = y_pred[:2,:]
    #st.write(results_model_test)

    # check if stardard scaler was applied to model input data
    if standard_scaler_select == 'Yes':
                
        #make np array from answers:
        input_answers = np.array(answers)
        # reshape np array to be able to fit run thoruhg the model
        input_answers = input_answers.reshape(1,-1)
        # write input Answers
        #st.write(input_answers)
           
        # Apply standard scaler to input answers
        answers_scaled = scaler.transform(input_answers)
        #st.write(answers_scaled)
                
        user_input_pred = loaded_model.predict(answers_scaled)
        #st.write(f'prediction is, {user_input_pred}')
        st.write(f'Diabetes Prediction based on scaled input data is, {user_input_pred[0]}')

    else:

        #make np array from answers:
        input_answers = np.array(answers)
        # reshape np array to be able to fit run thoruhg the model
        input_answers = input_answers.reshape(1,-1)
                
        # Input raw input into model
        user_input_pred = loaded_model.predict(input_answers)
        #st.write(f'prediction is, {user_input_pred}')
        st.write(f'Diabetes Prediction based on raw input data is, {user_input_pred[0]}')


    # create header for prediction results
    st.header('PREDICTIONS FOR DIABETES BASED ON MODEL')
    # create subhedaer to highlight its based on the new input data
    st.subheader('Diabetes prediction based on new input data:')

    # create simple high/medium low diabetes risk result
    if user_input_pred[0] >= 0.50:
        st.subheader('RESULT: High Risk of Diabetes')
        st.write('Based on below input variables')
        input_answers = pd.DataFrame(input_answers, columns=features_used_in_model)
        st.write(input_answers)
    elif user_input_pred[0]>0.25 <0.50:
        st.subheader('RESULT: Medium Risk of Diabetes')
        st.write('Based on below input variables')
        input_answers = pd.DataFrame(input_answers,columns=features_used_in_model)
        st.write(input_answers)
    else:
        st.subheader('RESULT: Low Risk of Diabetes')
        st.write('Based on below input variables')
        input_answers = pd.DataFrame(input_answers,columns=features_used_in_model)
        st.write(input_answers)




#### JASON Recommendations tab
with tab9:
    st.header('Recommendations')
    #make recommendations based on user input predictions
    BMI_expander = st.expander(label="Recommendations for your BMI")
    with BMI_expander:
        if answers[4] <= 18.5:
            st.write("You are underweight. Consider increasing calorie intake to acheive a healthier BMI.")
        elif 18.5<answers[4]<24.9:
            st.write('Excellent job! You are in a healthy weight range!')
        elif 25.0<=answers[4]<29.9:
            st.write('You are overweight. Consider increasing physical activity level and/or decreasing caloric intake. You are at increased risk for diabetes and may benefit from a consultation with a physician and/or nutritionist.')
        else:
            st.write('You fall into the obese weight range. You are at high risk for developing diabetes. Please consult a physician to determine your next steps towards obtaining a healthy BMI.')
    glucose_expander = st.expander(label = 'Recommendations for your glucose level')
    with glucose_expander:
        if answers[0] <=99:
            st.write('Glucose levels normal. Please consult other recommendations.')
        elif 100<=answers[0]<=125:
            st.write('Glucose level is in the prediabetic range. You are at significant risk for developing diabetes without lifestyle modifications. Please consult a physician and nutritionist.')
        else:
            st.write('You are diabetic. Please seek physician attention immediately to avoid further complications.')
#data claims 2 hr oral glucose tolerance test used. however, these results would indicate none of the patients have diabetes, though some would fall into the prediabetic category. data is relatively consistent with a fasting plasma glucose tolerance test, which is more common. for a 2 hr oral glucose tolerence test, diabetes is only diagnosed at 200 mg/dL or higher. No patients in this dataset were above 200 mg/dL.
#sources: https://diabetes.org/diabetes/a1c/diagnosis#:~:text=Oral%20Glucose%20Tolerance%20Test%20(OGTT,how%20your%20body%20processes%20sugar, https://www.mayoclinic.org/tests-procedures/glucose-tolerance-test/about/pac-20394296#:~:text=A%20normal%20fasting%20blood%20glucose,(8.6%20mmol%2FL).
    bp_expander = st.expander(label = 'Recommendations for your diastolic blood pressure level')
    with bp_expander:
        if answers[1] <80:
            st.write('You have normal blood pressure.')
        elif 80<=answers[1]<=89:
            st.write('You have Stage 1 Hypertension. Though this result is minimally related to diabetes, please seek physician attention to determine changes in diet, exercise, and medication.')
        elif 90<=answers[1]<=120:
            st.write('You have Stage 2 Hypertension. Though this result is minimally related to diabetes, please seek immediate physician attention to determine changes in diet, exercise, and medication to avoid end organ damage.')
        else:
            st.write('You are in hypertensive crisis. Go to the emergency room immediately to avoid heart attack, stroke, cerebral hemorrhage or other life threatening complications.') 
    pedigree_expander = st.expander(label = 'Recommendations for your diabetes pedigree function')
    with pedigree_expander:
        if answers[5] <= 0.448:
            st.write('You are not at increased risk for diabetes based on your family history.')
        else:
            st.write('You are at increased risk for developing diabetes based on your family history.')
    insulin_expander = st.expander(label = 'Recommendations for your insulin level')
    with insulin_expander:
        if answers[3] < 16:
            st.write('Insulin level low. Please consult a physician immediately to prevent further complications.')
        elif 16<=answers[3]<=166:
            st.write('Normal insulin level reported.')
        else:
            st.write('Insulin level high. Please consult a physician immediately to prevent further complications.')
   # insurance_cost_expander = st.expander(label = 'New cost to insurance estimate')
    #with insurance_cost_expander:
     #   monthly_insurance_bill = insur_cost * 2.3
      #  st.write('Your new estimated insurance bill will be $',monthly_insurance_bill)
    overall_risk_expander = st.expander(label = 'Recommendations for your overall risk for diabetes')
    with overall_risk_expander:
        if user_input_pred[0] >0.50:
            st.write('Based on your cumulative inputs, you are at high risk for diabetes. Please consult a physician.')
        elif 0.25<user_input_pred[0]<=0.50: 
            st.write('Based on your cumulative inputs, you are at moderate risk for diabetes. Please consider consulting a physician.')
        else:
            st.write('You are not at increased risk for developing diabetes based on your cumulative inputs.')
   # insurance_expander = st.expander(label = 'Input current medical expenses')
    #with insurance_expander:
     #   if user_input_pred[0] >0.50:
      #      st.write('Given your high risk of diabetes, please input current annual medical expenses to determine additional cost due to diabetes.')
            #st.number_input(label='In USD')
       #     int_val = st.number_input('USD', min_value=1, max_value=10000, value=300, step=50)
         #   int_new= float(st.button(int_val))
        #    st.write('new cost is', int_new*2.3)
with tab9:
    st.header('Medical Expense')            
    with st.form('Insurance'):
        insur_cost=st.slider(label='Current monthly medical expenses', min_value=0, max_value=1500, value= 500,step=1,key=10)
        submitted8 = float(st.form_submit_button('Submit'))
            
    insurance_cost_expander = st.expander(label = 'Medical expenses with diabetes diagnosis')
    with insurance_cost_expander:
        monthly_insurance_bill = insur_cost * 2.3
        st.write('Your new estimated medical expenses will be $',round(monthly_insurance_bill, 2))