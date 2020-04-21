from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from DataPreprocess import max_features, y, test_text, test, X_train, X_test, y_train, y_test
from sklearn import metrics


def create_model():
    # Build a neural network :
    rf_model = Sequential()
    rf_model.add(Dense(128, input_dim=max_features, activation='relu'))
    rf_model.add(Dense(256, activation='relu'))
    rf_model.add(Dense(256, activation='relu'))
    rf_model.add(Dense(256, activation='relu'))
    rf_model.add(Dense(1, activation='sigmoid'))
    return rf_model


def train():

    if not os.path.exists("rf_model.h5"):
        rf_model = create_model()

    else:
        print("Loading and evaluating RF model")
        rf_model = load_model('rf_model.h5')
        rf_model.summary()

    rf_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    rf_model_fit = rf_model.fit(X_train, y_train, epochs=150, batch_size=64, validation_split=0.2)
    rf_model.save('rf_model.h5')
    print("Saved RF model to disk")
    train_acc = rf_model.evaluate(X_train, y_train, verbose=0)
    test_acc = rf_model.evaluate(X_test, y_test, verbose=0)
    print('Train:', train_acc, 'Test:', test_acc)

    # plot for loss
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(rf_model_fit.history['loss'], label="train loss")
    plt.plot(rf_model_fit.history['val_loss'], label="test loss")
    plt.legend()

    plt.subplot(211)
    plt.title('Accuracy')
    plt.plot(rf_model_fit.history['acc'], label="train accuracy")
    plt.plot(rf_model_fit.history['val_acc'], label="test accuracy")
    plt.legend()

    plt.show()

    return rf_model_fit


def predict():

    rf_model = load_model('rf_model.h5')
    rf_prediction = rf_model.predict_classes(test_text)
    rf_prediction = rf_prediction.flatten()
    return rf_prediction


if __name__ == "__main__":
    print("\t1.Train")
    print("\t2.Predict\n")
    choice = input("Write the number of your choice: ")
    choice = int(choice)

    if choice == 1:
        rf_model_fit = train()
    elif choice == 2:
        prediction = predict()
        print(prediction)
        submission = pd.DataFrame({'id': test.index, 'rf_prediction': prediction})
        submission.to_csv('Kaggle/rf_pred.csv', index=False)
    else:
        print("Wrong. Enter 1 or 2")
