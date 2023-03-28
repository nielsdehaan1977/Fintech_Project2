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
The tool can help to predict based on user input what level of risk an individual is running to become diabetic and will give that individual an estimate on how much their healthcare insurance premium would go up if they don't take preventative action. 
* The tool goes through on the following steps: 
1. Prepare the data for use on a neural network model.
2. Compile and evaluate a binary classification model using a neural network.
3. Optimize the neural network model.
---
## Table of Contents

- [Tech](#technologies)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Contributor(s)](#contributor(s))
- [License(s)](#license(s))

---
## Tech

This project leverages python 3.9 and Jupyter Lab with the following packages:

* `Python 3.9`
* `Jupyter lab`

* [JupyterLab](https://jupyter.org/) - Jupyter Lab is the latest web-based interactive development environment for notebooks, code, and data.

* [Path](https://docs.python.org/3/library/pathlib.html) - This module offers classes representing filesystem paths with semantics appropriate for different operating systems.

* [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) - Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

* [concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html) - Concatenate pandas objects along a particular axis

* [numpy](https://numpy.org/doc/stable/index.html) - NumPy is the fundamental package for scientific computing in Python.

* [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) - Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

* [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) - The input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features. This creates a binary column for each category and returns a sparse matrix or dense array (depending on the sparse_output parameter)

* [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) - Split arrays or matrices into random train and test subsets.

* [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) - Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).

* [Sequential](https://www.tensorflow.org/guide/keras/sequential_model) - A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

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

## Recommendations
* Recommendations are specific to patient/user input data. 
* BMI separated into underweight, healthy weight, overweight and obese. 
* Glucose level recommendations (normal, prediabetic, diabetic) are based on fasting plasma glucose levels. Though the dataset claims that a 2 hr oral glucose tolerance test (OGTT) was used, these glucose levels would indicate no patients are diabetic. However, these datapoints are relatively consistent with what would be seen with fasting plasma glucose (FPG) tests.
* Blood pressure (BP) is specific to diastolic BP to remain consistant with ML model. Recommendations determined based on if patient has: normal/elevated BP, Stage 1 hypertension (HTN), Stage 2 HTN, or is in hypertensive crisis (emergent condition).
* Diabetes Pedigree Function is a measure of how likely a patient is to develop diabetes based on family history. Patients separated into high or low risk based on if they have an above average DiabetesPedigreeFunction.
* Insulin level recommendations are interesting because low and high insulin levels can both lead to complications. 

## Sources
* Healthcare costs: https://www.cdc.gov/chronicdisease/programs-impact/pop/diabetes.htm#:~:text=%241%20out%20of%20every%20%244,caring%20for%20people%20with%20diabetes.&text=%24237%20billion%E2%80%A1(a)%20is,(a)%20on%20reduced%20productivity.&text=61%25%20of%20diabetes%20costs%20are,is%20mainly%20paid%20by%20Medicare.
* Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
* BMI: https://www.cdc.gov/healthyweight/assessing/index.html
* Glucose:https://www.cdc.gov/diabetes/basics/getting-tested.html#:~:text=A%20fasting%20blood%20sugar%20level,higher%20indicates%20you%20have%20diabetes.
* OGTT vs FPG: https://diabetes.org/diabetes/a1c/diagnosis#:~:text=Oral%20Glucose%20Tolerance%20Test%20(OGTT,how%20your%20body%20processes%20sugar, https://www.mayoclinic.org/tests-procedures/glucose-tolerance-test/about/pac-20394296#:~:text=A%20normal%20fasting%20blood%20glucose,(8.6%20mmol%2FL).
* Diastolic BP: https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings
* Diabetes Pedigree Function: https://assets.researchsquare.com/files/rs-1753046/v1_covered.pdf?c=1655141180
* Insulin: https://emedicine.medscape.com/article/2089224-overview
## Contributor(s)

This project was created by Niels de Haan (nlsdhn@gmail.com)

---

## License(s)

MIT
