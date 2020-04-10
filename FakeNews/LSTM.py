from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM, RNN, SpatialDropout1D
from DataPreprocess import max_features, X, y, test_text
import os
import re
import numpy as np
from string import punctuation
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# LSTM Neural Network
lstm_model = Sequential(name='lstm_nn_model')
lstm_model.add(layer=Embedding(input_dim=max_features, output_dim=120, name='1st_layer'))
lstm_model.add(layer=LSTM(units=120, dropout=0.2, recurrent_dropout=0.2, name='2nd_layer'))
lstm_model.add(layer=Dropout(rate=0.5, name='3rd_layer'))
lstm_model.add(layer=Dense(units=120, activation='relu', name='4th_layer'))
lstm_model.add(layer=Dropout(rate=0.5, name='5th_layer'))
lstm_model.add(layer=Dense(units=len(set(y)), activation='sigmoid', name='output_layer'))
# compiling the model
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

lstm_model_fit = lstm_model.fit(X_train, y_train, epochs=1)

# GRU neural Network
gru_model = Sequential(name='gru_nn_model')
gru_model.add(layer=Embedding(input_dim=max_features, output_dim=120, name='1st_layer'))
gru_model.add(layer=GRU(units=120, dropout=0.2,
                        recurrent_dropout=0.2, recurrent_activation='relu',
                        activation='relu', name='2nd_layer'))
gru_model.add(layer=Dropout(rate=0.4, name='3rd_layer'))
gru_model.add(layer=Dense(units=120, activation='relu', name='4th_layer'))
gru_model.add(layer=Dropout(rate=0.2, name='5th_layer'))
gru_model.add(layer=Dense(units=len(set(y)), activation='softmax', name='output_layer'))
# compiling the model
gru_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
gru_model.summary()
gru_model_fit = gru_model.fit(X_train, y_train, epochs=1)
lstm_prediction = lstm_model.predict_classes(test_text)