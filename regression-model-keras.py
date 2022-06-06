## Downloading and cleaning data

import pandas as pd
import numpy as np

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

concrete_data.shape

concrete_data.describe()
concrete_data.isnull().sum()

## Splitting data into predictors and target

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

## predictors and the target dataframes
predictors.head()
target.head()

## normalizing the data by substracting the mean and dividing by the standard deviation

predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

n_cols = predictors_norm.shape[1] # number of predictors

## Import Keras
import keras

## import packages from the Keras library needed to build regressoin model
from keras.models import Sequential
from keras.layers import Dense

## Build a Neural Network
### define regression model
def regression_model():
    ### create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    ### compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
  
## Train and Test the Network
### build the model
model = regression_model()

## train and test the model at the same time using the fit method
### leaving out 30% of the data for validation and train the model for 100 epochs
### fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)


