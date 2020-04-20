from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from DataPreprocess import max_features, X, y, test_text, test


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    if not os.path.exists("rf_model.h5"):
        rf_model = create_model()
        rf_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        rf_model_fit = rf_model.fit(X_train, y_train, epochs=150, batch_size=64, validation_split=0.2)
        rf_model.save('lstm_model.h5')
        print("Saved RF model to disk")
    else:
        print("Loading and evaluating RF model")
        rf_model = load_model('rf_model.h5')
        rf_model.summary()
        rf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        rf_model_fit = rf_model.fit(X_train, y_train, epochs=1)
        # evaluate the model
        scores = rf_model.evaluate(X, y, verbose=0)
        print("%s: %.2f%%" % (rf_model.metrics_names[1], scores[1] * 100))

    # plot for loss
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(rf_model_fit.history['loss'], label="train")
    plt.plot(rf_model_fit.history['val_loss'], label="test")
    plt.legend()

    plt.subplot(211)
    plt.title('Accuracy')
    plt.plot(rf_model_fit.history['acc'], label="train")
    plt.plot(rf_model_fit.history['val_acc'], label="test")
    plt.legend()

    plt.show()

    return rf_model_fit


def predict():
    # After the training, we use the json files to make predictions

    rf_model = load_model('rf_model.h5')
    rf_prediction = rf_model.predict(test_text)
    # round predictions
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
