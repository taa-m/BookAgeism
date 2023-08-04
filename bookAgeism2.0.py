# -*- coding: utf-8 -*-
"""
Created on Tue May 31 08:45:43 2022

IMDb Sentiment Analysis Example

@author: Taahirah Mangera
"""

#%% Import packages

import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import sequence
from keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow as tf

#%% Load in the data
vocabulary_size = 50000
predictorsdf = pd.read_csv('bookData_predictors.csv', dtype='int64')
responsedf = pd.read_csv('bookData_response.csv')
#%%Load in training and test data

xTrainValid, xTest , yTrainValid, yTest = train_test_split(predictorsdf, responsedf, test_size=0.15, random_state=42, stratify = responsedf)

#verify class split
yTrainValid.value_counts().sort_index().plot.bar(x='Target Value', y='Number of Occurrences')
yTest.value_counts().sort_index().plot.bar(x='Target Value', y='Number of Occurrences')

xTrainValid = xTrainValid.to_numpy(dtype = 'int64')
xTest = xTest.to_numpy(dtype = 'int64')
yTrainValid = yTrainValid.to_numpy(dtype = 'int64')
yTest = yTest.to_numpy(dtype = 'int64')

max_words = 1062

#%%Define RNN model

embedding_size=32
model= tf.keras.Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(3, activation='softmax'))
print(model.summary())



#%% Train and evaluate the model

model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

batch_size = 64
num_epochs = 3
xValid, yValid = xTrainValid[:batch_size], yTrainValid[:batch_size]
xTrain, yTrain =  xTrainValid[batch_size:], yTrainValid[batch_size:]
model.fit(xTrain, yTrain, validation_data=(xValid, yValid), batch_size=batch_size, epochs=num_epochs)

#%% Test the model

scores = model.evaluate(xTest, yTest, verbose=0)
print('Test accuracy:', scores[1])