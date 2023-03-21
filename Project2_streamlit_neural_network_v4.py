# Imports
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

@st.cache_data # add the caching decorator
def read_data(filename_data):
    path_csv = Path('./Resources/')
    file_path = path_csv / filename_data
    df = pd.read_csv(file_path)
    return df

# create tab as container
tab = st.container()

#create tabs for display in steamlit
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Project Objective','Original Data','Data Preparation','Machine Learning Model', 'Predictions'])

# # read in data
df = read_data('diabetes_data_kaggle.csv')

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

# create selectbox for sampling oversampling, undersampling or no sampling (use dataset as is)
sampling_dataset = ['None','RandomOverSampler','RandomUnderSampler']

# Create list of columns names to select as X features and y label
columns_names_all = df.columns.tolist()

# Create sidebar with user input for neural network
with st.sidebar:
    st.header('Selection Options')
    
    # Select y column from list of column names
    label_select = st.selectbox("Select Y column", options=columns_names_all)

    # Remove selected y column from X (features)
    column_names_x=df.drop(columns=label_select).columns.tolist()

    # Create a multi select box for users to select as features
    features_select = st.multiselect("Select X columns",options=column_names_x, default=column_names_x)
    
    # Select the percentage of training data you want to use:
    test_data_select = st.slider('What percentage of data should be test data?',min_value=0.1,max_value=1.0,value=0.2,step=0.1)

    # select what kind of outpot the model is used for
    compile_loss_select = st.selectbox('Model Loss Selection Model',output_goal)

    # Change compile loss based on compile loss select
    if compile_loss_select == 'predict probability distribution':
        compile_loss_select_used = st.selectbox('Model Loss Selection Options',compile_loss_probalistic)
    elif compile_loss_select == 'predict continues numerical value':
        compile_loss_select_used = st.selectbox('Model Loss Selection Options',compile_loss_regression)
    else:
        compile_loss_select_used = st.selectbox('Model Loss Selection Options',compile_loss_hinge)

    # indicate to use standard scaler yes/no
    standard_scaler_select = st.radio('Use Standard Scaler (Yes/No)',('Yes','No'))

    # Indicate to use random oversampling or undersampling or none
    sampling_dataset_select = st.selectbox('Select Sampling of DataSet',sampling_dataset)

    # select what kind of optimizer model should use in compile
    compile_optimizer_select = st.selectbox('Model optimizer select',compile_optimizer)

    # select what kind of metric model should use in compile
    compile_metric_select = st.selectbox('Model metric select',compile_metrics)

    # Define the number of neurons in the output layer
    output_neurons = st.slider('How many output Neurons?',min_value=1,max_value=10,value=1,step=1)
    
    # Define activation for output layer
    output_activation = st.selectbox('Output Layer Activation',activations_output)

    # Define the number of hidden nodes for the first hidden layer
    hidden_nodes_layer_1 = st.slider('How many hidden nodes in layer 1?',min_value=1,max_value=50,value=2,step=1)

    # Define activation for hidden layer1
    layer_1_activation = st.selectbox('Layer 1 Activation',activations_layers)

    # Define the number of hidden nodes for the second hidden layer
    hidden_nodes_layer_2 = st.slider('How many hidden nodes in layer 2?',min_value=0,max_value=50,value=1,step=1)

    # Define activation for hidden layer2
    layer_2_activation = st.selectbox('Layer 2 Activation',activations_layers)

    # create drop down to select how many epochs to run
    n_epochs = st.selectbox('How many epochs would you like to run',[20,50,100,200,500,1000],index=0)
    

# INTRODUCTION TAB PROJECT OBJECTIVE
with tab1:

    st.title('Neural Network Predictions')

    st.header('Project Objective:')

    st.subheader('Diabetes Predictions')

    st.text('In this project we try to predict if a person has diabetes using machine learning')

    st.image('Images/Neural_Networks_2.jpg',use_column_width=True)


    
    #usd_amount_2 = st.number_input('How much money would you like to invest (in USD)?', min_value=500, value=500, step=500)

# DATA TAB
with tab2:
    # type header of the tab
    st.header('Data used for Project')
    # type where you find the dataset
    st.text('Dataset origin: Data found on kaggle on below link')
    # create link to dataset used
    st.write(f'Please click here for dataset [link]({data_source})')

    # Display dataset in tab
    st.subheader('Display first 5 rows of Data')
    st.write(df.head())

    #display datatypes of dataset
    st.subheader('Dipslay datatypes of the data')
    st.write(df.dtypes)

    # Count and display amount of different values in y (label)
    st.header('Distribution of y label original data')
    column_y_values = df[label_select].value_counts()
    st.write(column_y_values)

    # Create a data distribution
    BMI_dist = pd.DataFrame(df['BMI'].value_counts())
    
    # Create a correlation matrix for the dataframe:
    corr_matrix = df[features_select + [label_select]].corr()
    
    # Create a heatmap of the correlation matrix
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix,annot=True,cmap='inferno',ax=ax)
    
    #display correlation matrix heatmap in streamlit
    st.subheader('Correlation matrix heatmap Original Data')
    st.pyplot(fig)

# DATA Preparation
with tab3:
    st.header('Data Preparation')

    # Create the labels set (y) from the “Outcome” column, and then create the features (X) DataFrame from the remaining columns.
    X = df[features_select]
    y = df[label_select]

    # Create a StandardScaler instance if requested by user
    if standard_scaler_select == 'Yes':
        scaler = StandardScaler()

        # Fit the scaler to the features training dataset
        X_scaled = scaler.fit_transform(X)
    else:
        # If scaler should not be applied just name X --> X-scaler (to make further processing possible with one variable)
        X_scaled = X

    # Display to be used X dataframe indicate if standard scaler is used yes/no
    st.subheader(f'Features DataFrame (Standard Scaler Used? {standard_scaler_select})')
    st.write(X_scaled)

    # Split up data using train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=test_data_select,random_state=1) 

    # check the split up data set
    xtrain_count = X_train.shape[0]
    xtest_count = X_test.shape[0]
    ytrain_count = y_train.shape[0]
    ytest_count = y_test.shape[0]

    st.subheader('data after train test split')
    st.write(f'xtrain = {xtrain_count} xtest = {xtest_count} Ytrain = {ytrain_count} ytest_count = {ytest_count}')

    # apply the selected sampling method to the dataset
    if sampling_dataset_select == 'RandomOverSampler':
        sampler = RandomOverSampler(random_state=1)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    elif sampling_dataset_select == 'RandomUnderSampler':
        sampler = RandomUnderSampler(random_state=1)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    else:
        X_resampled, y_resampled = X_train, y_train


    # Count the distinct values of the resampled labels data
    y_resampled_count = y_resampled.value_counts()
    st.subheader(f'y label to be used (resampling method used: {sampling_dataset_select})')
    st.write(y_resampled_count)

# MACHINE LEARNING MODEL TAB
with tab4:
    st.header('Neural Network')
    
    # Define the the number of inputs (features) to the model
    number_input_features = len(X.iloc[0])

    # Create overview of features of Neural Network 
    features_nn = ['Input Features', 'Amount of Hidden Layers Layer 1','Layer 1 Optimizer', 'Amount of Hidden Layers Layer 2','Layer 2 Optimizer', 'Amount of Output Layer(s)', 'Output Layer Activation','Standard Scaler', 'Sampling','Epochs']
    inputs_nn = [number_input_features, hidden_nodes_layer_1,layer_1_activation, hidden_nodes_layer_2,layer_2_activation, output_neurons,output_activation, standard_scaler_select, sampling_dataset_select,n_epochs]
    data_nn = {'Features_nn':features_nn,'Input nn':inputs_nn}
    nn_characteristics = pd.DataFrame(data_nn)
    
    #show neural network characteristics in streamlit
    st.subheader('Neural Network Characteristics')
    st.write(nn_characteristics)

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

    # Set the model's file path
    file_path = Path(f'./Models/{model_accuracy:.2f}(acc)_{model_loss:.2f}(loss)_epochs_{n_epochs}_L1({hidden_nodes_layer_1})_L2({hidden_nodes_layer_2})_OutputLayer({output_neurons})_Scale_{standard_scaler_select}_Sampling_{sampling_dataset_select}.h5')

    # Export your model to a HDF5 file
    nn.save(file_path)

with tab5:
    st.header('Predictions')
    st.subheader('Model Used')


# # Add Caching decorator for loading model


#     # set folder path for stored models
#     folder_path = Path('./Models/')
    
#     @st.cache_resource 
#     def select_model(model_path):
#         model_list = os.listdir(folder_path)


#     @st.cache_resource 
#     def load_model(model_path):
#        # Load and process the model
#         model_selected = tf.keras.models.load_model(model_path)
#         return model_selected

#     # Get a list of all files in the folder
#     model_list = os.listdir(folder_path)

#     # Create a dropdown menu with the file names
#     selected_model = st.selectbox('Select a file', model_list)

#     # Get the file path for the selected file
#     model_path = os.path.join(folder_path, selected_model)

#     # Call the select_model function to load the model
#     loaded_model = load_model(file_path)

#     # # Print the selected file contents
#     # st.write('Model Selected:', loaded_model)
