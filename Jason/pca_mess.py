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
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA




st.set_page_config(page_title="NN Diabetes Predictions", layout='wide')  
    
st.title('Principal Component Analysis')
    
df=pd.read_csv('Diabetnew.csv')
    
if 'df' in locals():
    # Remove any rows with missing values
    df.dropna(inplace=True)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Perform PCA
    pca = PCA()
    pca.fit(scaled_data)
    
    # Extract the principal components
    principal_components = pca.transform(scaled_data)
    
    # Create a dataframe of the principal components
    pc_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
        
if 'df' in locals():
    # Display the original data
    st.subheader('Original Data')
    st.write(df)
    
    # Display the standardized data
    st.subheader('Standardized Data')
    st.write(pd.DataFrame(data = scaled_data, columns = df.columns))
    
    # Display the explained variance ratio
    st.subheader('Explained Variance Ratio')
    st.write(pca.explained_variance_ratio_)
    
    # Display a scatter plot of the principal components
    st.subheader('Principal Components')
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('2 component PCA')
    ax.scatter(pc_df['PC1'], pc_df['PC2'])
    st.pyplot(fig) 
    
    #'''st.subheader('Principal Component Analysis')
    
    #st.subheader('Input Data')
    
    #dm_data = X_scaled()

   # st.write(X_scaled)

    #st.subheader('Explore the original data')

   # xvar = st.selectbox('Select x-axis:', X_scaled.columns[:-1])
    #yvar = df['Outcome']

    #st.write(px.scatter(X_scaled, x=xvar, y=yvar, color='Outcome'))

    #X_scaled = StandardScaler().fit_transform(X, axis=1))
    #pca = PCA()
    #X_transformed = pca.fit_transform(X_scaled)

    #columns = column_names_x

   # X_transformed_df = pd.DataFrame(X_transformed, columns=columns)
   # X_transformed_df = pd.concat([X_transformed_df, df['Outcome']], axis=1)

   # with st.form(key='feature_label_select_1'):
        
        # Create Header to Display what is required from user
    #    st.header('Selection Your y column')
        
        # Create list of columns names to select as X features and y label
     #   columns_names_all = df.columns.tolist()

        # Select y column from list of column names
      #  label_select = st.selectbox("Select y column", options=columns_names_all)

        # Remove selected y column from X (features)
       # column_names_x=df.drop(columns=label_select).columns.tolist()

        # Create a button to select the Y column wihthn the form
        #submit_button_y = st.form_submit_button(label='Select y')

        # Create the labels set (y) after selection
        #y = df[label_select]

        # Display which Y column is currently selected
        #st.subheader(f'Current Selected y Column is: {label_select}')                               
                                  
                                  
    #st.subheader('Transformed data')

    #st.write(X_transformed_df)

    #st.subheader('Evaluation of Principal Components')

    #xvar = st.selectbox('Select x-axis:', X_transformed_df.columns[:-1])
    #yvar = df['Outcome']

   # fig= px.scatter(X_transformed_df, x=xvar, y=yvar, color='Outcome')                             
                                  
    #st.write(fig)

    #st.subheader('Explore loadings')

    #loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    #loadings_df = pd.DataFrame(loadings, columns=columns)
    #loadings_df = pd.concat([loadings_df, 
                         #pd.Series(X_scaled.columns[0:4], name='var')], 
                         #axis=1)

    #component = st.selectbox('Select component:', loadings_df.columns[0:4])

    #pca_scatter = px.scatter(loadings_df[['var', component]].sort_values(component), 
        #           y='var', 
       #            x=component, 
      #             orientation='h',
     #              range_x=[-1,1])


    #st.write(pca_scatter)

   # st.write(loadings_df)'''
    
    #pca=PCA(n_components=2)
    #pca_data=pca.fit_transform(df)
    #st.subheader('The explained variance ratio is {pca.explained_variance_ratio}')
    #df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2']

    # Initialize the K-Means model
    #model = KMeans(n_clusters=2)

    # Fit the model
    #model.fit(df_pca)

    # Predict clusters
    #pt_segments = model.predict(df_pca)

    # Create a copy of the original DataFrame
    #df_pca_predictions = df_pca.copy()

    # Create a new column in the DataFrame with the predicted clusters
    #df_pca_predictions["Outcome"] = pt_segments

    #plot clusters
    #df_pca_predictions.hvplot.scatter(x='PC1', y='PC2', by='Outcome')
    
# Create flexible User Input Tab (adjusted to selected Xfeatures (otherwise it cannot be used by the model)
             