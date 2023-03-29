# Fintech_Project 2 GROUP 2

![NN_Machine_Learning.jpg](https://github.com/nielsdehaan1977/Fintech_Project2/blob/main/Images/Machine_Learning.jpg)
---
# Neural Networks and Diabetes Predictions. 
---
## This python code can be utilized as an app by running Streamlit. The app utilizes a Neural Network to predict the level of risk an individual runs based on mutliple input values like glucose levels, age, bloodpressure and BMI. Based on the input values the model will generate a prediction of the level of risk an individual runs of becoming a Diabetic. The code is run as a streamlit app to provide a user friendly interface. 


![Neural_Networks_2.jpg](https://github.com/nielsdehaan1977/Fintech_Project2/blob/main/Images/Neural_Networks_2.jpg)

---
## Project2_Group2_Final_Draft.py
---
### This application was build to provide both a flexible way to create a Neural Network model and to use that same model to provide predictions on the level of risk an individual runs to becoming diabetic. The general idea of the application is to provide a hlepful tool for insurance companies or healthcare organizations to identify patients that have a high risk of becoming diabetic. The tool could allow insurance companies or healthcare organizations to provide preventative care which would save the insurance company a lot of costs and could relieve pressure on the workload in healthcare organizations. 

---
The tool can help predict based on user input what level of risk an individual is running to become diabetic and also gives an estimate on how much your healthcare insurance premium could increase if no preventative action is taken. 

* The tool goes through on the following tabs/steps (Steps are described in more detail in Usage passage of Readme): 
1. Description of the project and tool
2. Data Selection
3. Overview of Original data
4. Data Preparation
5. User Input
6. Setup of Neural Network Model
7. Model Performance
8. Predictions
9. Recommendations
---
## Table of Content

- [Tech](#technologies)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Contributor(s)](#contributor(s))
- [License(s)](#license(s))

---
## Tech

This project leverages python 3.9 with the following packages:
```
`Python 3.9`
```
* [Path](https://docs.python.org/3/library/pathlib.html) - This module offers classes representing filesystem paths with semantics appropriate for different operating systems.

* [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) - Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

* [numpy](https://numpy.org/doc/stable/index.html) - NumPy is the fundamental package for scientific computing in Python.

* [streamlit](https://streamlit.io/) - Streamlit is an open-source Python library that makes it easy to create and share, custom web apps for machine learning and data science.

* [tensorflow](https://www.tensorflow.org/) - tensorflow library provides end-to-end machine learning solutions

* [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) - Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).

* [Sequential](https://www.tensorflow.org/guide/keras/sequential_model) - A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

* [modelfromjson](https://www.tensorflow.org/api_docs/python/tf/keras/models/model_from_json) - Parses a JSON model configuration string and returns a model instance.

* [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) - Split arrays or matrices into random train and test subsets.

* [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) - Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

* [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) - The input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features. This creates a binary column for each category and returns a sparse matrix or dense array (depending on the sparse_output parameter)

* [Randomoversampler](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html) - Object to over-sample the minority class(es) by picking samples at random with replacement. The bootstrap can be generated in a smoothed manner.

* [Randomundersampler](https://imbalanced-learn.org/stable/references/under_sampling.html) - The imblearn.under_sampling provides methods to under-sample a dataset.

* [seaborn](https://seaborn.pydata.org/examples/many_pairwise_correlations.html) - Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

* [matplolib](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) - matplotlib.pyplot is a state-based interface to matplotlib. It provides an implicit, MATLAB-like, way of plotting. It also opens figures on your screen, and acts as the figure GUI manager.

---

## Installation Guide

### Before running the application first install the following dependencies in either Gitbash or Terminal. (If not already installed)

#### Step1: Activate dev environment in Gitbash or Terminal to do so type:
```python
    conda activate dev
```
#### Step2: install the following libraries (if not installed yet) by typing:
```python
    pip install pandas
    pip install numpy
    pip install --upgrade tensorflow
    pip install -U scikit-learn
 ```
#### Step3: Start Jupyter Lab
Jupyter Lab can be started by:
1. Activate your developer environment in Terminal or Git Bash (already done in step 1)
2. Type "jupyter lab --ContentsManager.allow_hidden=True" press enter (This will open Jupyter Lab in a mode where you can also see hidden files)

![JupyterLab](https://github.com/nielsdehaan1977/Fintech_Module13/blob/main/Images/JupyterLab.PNG)


## Usage

 (Currently we only allow for 1 data set to be picked up, but the ultimate goal of the tool is that it can process any dataset that you'd like to run through a Neural Network). 


To use the venture funding with deep learning jupyter lab notebook, simply clone the full repository and open the **venture_funding_with_deep_learning.ipynb** file in Jupyter Lab. 

The tool will go through the following steps:

### Prepare the data for use on a neural network model.
* Import of data to analyze
* Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define features and target variables.
* Encode the dataset’s categorical variables using OneHotEncoder, and then place the encoded variables into a new DataFrame. 
* Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.
* Using the preprocessed data, create the features (X) and target (y) datasets.
* Split the features and target sets into training and testing datasets.
* Use scikit-learn's StandardScaler to scale the features data.

### Compile and evaluate a binary classification model using a neural network.
* Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.
* Compile and fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric.
* Evaluate the model using the test data to determine the model’s loss and accuracy.
* Save and export your model to an HDF5 file

### Optimize the neural network model.
* Define three new deep neural network models (resulting in the original plus 3 optimization attempts). With each, try to improve on your first model’s predictive accuracy.
* Excecuted model with removing 1 of the columns every run, to see if there are any columns that have limited impact on the model
* Add an additional hidden layer and run the model with different amounts of hidden nodes and evaluate the results
* Change activation functions for the hidden layers and change the number of epoch in the training regimen and evaluate the results. 

## Contributor(s)

This project was created by Niels de Haan (nlsdhn@gmail.com)

---

## License(s)

MIT
