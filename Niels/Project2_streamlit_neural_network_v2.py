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

# create tab as container
tab = st.container()

#create tabs for display in steamlit
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Project Objective','Data','Data cleaning and analysis','Machine Learning Model', 'Predictions'])

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
sampling_dataset = ['None',RandomOverSampler,RandomUnderSampler]

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
    test_data_select = st.slider('What percentage of data should be test data?',min_value=0.1,max_value=1.0,value=0.3,step=0.1)

    # select what kind of outpot the model is used for
    compile_loss_select = st.selectbox('Compile Loss Selection Model',output_goal)

    # Change compile loss based on compile loss select
    if compile_loss_select == 'predict probability distribution':
        compile_loss_select_used = st.selectbox('Compile Loss Selection Options',compile_loss_probalistic)
    elif compile_loss_select == 'predict continues numerical value':
        compile_loss_select_used = st.selectbox('Compile Loss Selection Options',compile_loss_regression)
    else:
        compile_loss_select_used = st.selectbox('Compile Loss Selection Options',compile_loss_hinge)

    # Indicate to use random oversampling or undersampling or none
    sampling_dataset_select = st.selectbox('Select Sampling of DataSet',sampling_dataset)


    # select what kind of optimizer model should use in compile
    compile_optimizer_select = st.selectbox('Compile optimizer select',compile_optimizer)

    # select what kind of metric model should use in compile
    compile_metric_select = st.selectbox('Compile metric select',compile_metrics)

    # Define the number of neurons in the output layer
    output_neurons = st.slider('How many output Neurons?',min_value=1,max_value=10,value=1,step=1)
    
    # Define activation for output layer
    output_activation = st.selectbox('Output Layer Activation',activations_output)

    # Define the number of hidden nodes for the first hidden layer
    hidden_nodes_layer_1 = st.slider('How many hidden nodes in layer 1?',min_value=1,max_value=10,value=2,step=1)

    # Define activation for hidden layer1
    layer_1_activation = st.selectbox('Layer 1 Activation',activations_layers)

    # Define the number of hidden nodes for the second hidden layer
    hidden_nodes_layer_2 = st.slider('How many hidden nodes in layer 2?',min_value=0,max_value=10,value=1,step=1)

    # Define activation for hidden layer2
    layer_2_activation = st.selectbox('Layer 2 Activation',activations_layers)

    # create drop down to select how many epochs to run
    n_epochs = st.selectbox('How many epochs would you like to run',[20,50,100,200,500],index=0)
    

    

    #st.write(list_of_columns)

# INTRODUCTION TAB PROJECT OBJECTIVE
with tab1:

    st.title('Machine Learning Diabetes Predictions')

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
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_data_select,random_state=1) 

    # If required by input from user implement sampling requested (standard is NONE)
    if sampling_dataset_select !='None':
        random_sampler = sampling_dataset_select(random_state=1)

        # Fit the original training data to the random_oversampler model
        X, y = random_sampler.fit_resample(X_train,y_train)
    else:
        pass



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
    nn = Sequential()
    
    # Add the first hidden layer
    nn.add(Dense(units=hidden_nodes_layer_1, input_dim=number_input_features, activation=layer_1_activation))

    # Add the second hidden layer
    nn.add(Dense(units=hidden_nodes_layer_2, activation=layer_2_activation))

    # Add the output layer to the model specifying the number of output neurons and activation function
    nn.add(Dense(units=number_output_neurons, activation=output_activation))

    #nn_summary = nn.summary()
    #st.subheader('Neural Network Summary')
    #st.markdown(nn_summary)

    # Compile the Sequential model
    nn.compile(loss=compile_loss_select_used, optimizer=compile_optimizer_select, metrics=compile_metric_select)

    # Fit the model using 50 epochs and the training data
    fit_model = nn.fit(X_train_scaled, y_train, epochs=n_epochs)

    # Evaluate the model loss and accuracy metrics using the evaluate method and the test data
    model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)

    # Display the model loss and accuracy results
    #loss_accuracy = print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
    st.subheader('Neural Network Loss and Model Accuracy')
    st.write(f'Loss: {model_loss:.3f}, Accuracy: {model_accuracy:.3f}')


    # Set the model's file path
    file_path = Path(f'./Models/MODEL_output_neurons{output_neurons}_layer1_nodes_{hidden_nodes_layer_1}_layer2_nodes_{hidden_nodes_layer_2}_epochs_{n_epochs}_model_loss{model_loss:.2f}_model_accuracy{model_accuracy:.2f}.h5')

    # Export your model to a HDF5 file
    nn.save(file_path)

with tab5:
    st.header('Predictions')
    st.subheader('Model Used')

    
