import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from DataPreprocess import max_features, X, y, test_text, test
import tensorflow as tf
from tensorflow.keras.models import load_model


def create_model():
    # build LSTM Neural Network
    lstm_model = Sequential(name='lstm_nn_model')
    lstm_model.add(layer=Embedding(input_dim=max_features, output_dim=120, name='1st_layer'))
    lstm_model.add(layer=LSTM(units=120, dropout=0.2, recurrent_dropout=0.2, name='2nd_layer'))
    lstm_model.add(layer=Dropout(rate=0.5, name='3rd_layer'))
    lstm_model.add(layer=Dense(units=120, activation='relu', name='4th_layer'))
    lstm_model.add(layer=Dropout(rate=0.5, name='5th_layer'))
    lstm_model.add(layer=Dense(units=len(set(y)), activation='sigmoid', name='output_layer'))
    return lstm_model

def train():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    if not os.path.exists("lstm_model.h5"):
        print("Creating new LSTM model")
        lstm_model = create_model()
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm_model_fit = lstm_model.fit(X_train, y_train, epochs=1, batch_size=128)
        # evaluate the model
        scores = lstm_model.evaluate(X, y, verbose=0, batch_size=128)
        print("%s: %.2f%%" % (lstm_model.metrics_names[1], scores[1] * 100))
        # X_train.astype('int32')
        # X_test.astype('int32')
        lstm_model.save('lstm_model.h5')
        print("Saved LSTM model to disk")
    # if model exists, load it
    else:
        print("Loading and evaluating LSTM model")
        lstm_model = load_model('lstm_model.h5')
        lstm_model.summary()
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm_model_fit = lstm_model.fit(X_train, y_train, epochs=1)
        # evaluate the model
        scores = lstm_model.evaluate(X, y, verbose=0, batch_size=256)
        print("%s: %.2f%%" % (lstm_model.metrics_names[1], scores[1] * 100))

    print("Saved lstm model to disk")
    # plot for loss
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(lstm_model_fit.history['loss'], label="train")
    plt.plot(lstm_model_fit.history['val_loss'], label="train")
    plt.legend()

    plt.subplot(211)
    plt.title('Accuracy')
    plt.plot(lstm_model_fit.history['acc'], label="train")
    plt.plot(lstm_model_fit.history['val_acc'], label="train")
    plt.legend()

    plt.show()

    return lstm_model_fit


def predict():

    lstm_model = load_model('lstm_model.h5')
    lstm_prediction = lstm_model.predict_classes(test_text)
    return lstm_prediction


if __name__ == "__main__":
    print("\t1.Train")
    print("\t2.Predict\n")
    choice = input("Write the number of your choice: ")
    choice = int(choice)

    if choice == 1:
        lstm_model_fit = train()
    elif choice == 2:
        prediction = predict()
        print(prediction)
        submission = pd.DataFrame({'id': test.index, 'lstm_prediction': prediction})
        submission.to_csv('Kaggle/lstm_pred.csv', index=False)
    else:
        print("Wrong. Enter 1 or 2")
