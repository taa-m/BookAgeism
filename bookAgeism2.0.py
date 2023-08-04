# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:52:52 2022

@author: Staff

Getting vocab size

"""

import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec as w2v
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from keras.preprocessing import sequence
from keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow as tf 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import layers
import keras_tuner
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from sklearn.metrics import ConfusionMatrixDisplay
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from keras.utils.vis_utils import plot_model
import pydot
#%%Import the data

bookData = pd.read_csv('./inout.csv')

#%%Clean the data

bookData = bookData.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
bookData['synops'] = bookData['synops'].apply(clean_text)

#%%Tokenize and create matrix of weights from Glove

# The maximum number of words to be used. (most frequent)
vocabSize = 50000 #set large enough so that every word has a unique vector
# Max number of words in each synopsis
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100 #create vectors of length 100
tokenizer = Tokenizer(num_words=vocabSize, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(bookData['synops'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# size_of_vocabulary=len(tokenizer.word_index) + 1 #+1 for padding
# print(size_of_vocabulary)

# embeddings_index = dict()
# f = open('./glove.42B.300d/glove.42B.300d.txt', encoding='utf8')

# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs

# f.close()
# print('Loaded %s word vectors.' % len(embeddings_index))

# # create a weight matrix for words in training docs
# embedding_matrix = np.zeros((size_of_vocabulary, 300))

# for word, i in tokenizer.word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector


#%%Truncate and pad and convert categoricals

X = tokenizer.texts_to_sequences(bookData['synops'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(bookData['age']).values
print(Y.shape)


#%%Split the data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42, stratify = Y)
print(X_train.shape,Y_train.shape[1])
print(X_test.shape,Y_test.shape[1])

print(X_train.shape[1])

#%% Define the RNN model
model=keras.Sequential([keras.layers.Embedding(vocabSize, EMBEDDING_DIM, input_length=X_train.shape[1]),
                 keras.layers.SpatialDropout1D(0.2),
                 keras.layers.LSTM(100,dropout=0.2, recurrent_dropout=0.2),
                 keras.layers.Dense(3,activation='softmax')])

tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True) 
                
#Add loss function, metrics, optimizera LSTM A
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=["acc"]) 

#Adding callbacks
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
mc= tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True,verbose=1)  

#Print summary of model
print(model.summary())

epochs = 1
batch_size = 32

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[es,mc])
tf.keras.models.save_model(model,'./ageismmodel', overwrite = True, include_optimizer=True,save_format=None,signatures=None,options=None)
#Plot training curve
##Not the most useful for this model as the model is basically converged after the first epoch
print(history.history.keys())
fig = plt.figure(figsize=(8,6))
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.legend()
plt.savefig('ageism.png', dpi = 300, transparent = 'True')

#%% Tuning the model with Keras Tuner


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocabSize, EMBEDDING_DIM, input_length=X_train.shape[1]))
    model.add(tf.keras.layers.SpatialDropout1D(0.2))
    model.add(tf.keras.layers.LSTM(100,dropout=0.2, recurrent_dropout=0.2))
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 15)):
        model.add(
            tf.keras.layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

build_model(keras_tuner.HyperParameters()) #check that the model builds

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
)

tuner.search_space_summary()

tuner.search(X_train,Y_train, epochs=2, validation_split=0.1)
#%%
print(model.evaluate(X_test,Y_test))
## Create confusion matrix
Y_pred = model.predict(X_test) #ouputs predictions in a range from 0-1
y_pred = Y_pred>0.5 #convert ouptuts to only 0s or 1s
print('Confusion Matrix')
print(confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1)))
cm = confusion_matrix(Y_test, y_pred)
cm = confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
#Create classification report
print('Classification Report')
target_names = ['Beginner', 'Intermediate', 'YA']
print(classification_report(Y_test, y_pred, target_names=target_names)) 

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Beginner','Intermediate','YA'])
disp.plot()
plt.savefig('ageism.png', dpi = 300, transparent = 'True')
plt.show()