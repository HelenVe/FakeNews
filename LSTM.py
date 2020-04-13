from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM
from tensorflow.keras.models import Sequential
import tensorflow as tf
from DataPreprocess import max_features, X, y, test_text
import os
from tensorflow.keras.models import model_from_json, load_model


def train():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    if not os.path.exists("lstm_model.json") and not os.path.exists("lstm_model.h5"):
        # build LSTM Neural Network
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
        # evaluate the model
        scores = lstm_model.evaluate(X, y, verbose=0)
        print("%s: %.2f%%" % (lstm_model.metrics_names[1], scores[1] * 100))

        # if model exists, load it
    else:
        with open('lstm_model.json', 'r') as f:
            lstm_model = tf.keras.models.model_from_json(f.read())

        lstm_model.load_weights('lstm_model.h5')
        lstm_model = Sequential(name='lstm_nn_model')
        lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        lstm_model_fit = lstm_model.fit(X_train, y_train, epochs=1)
        # evaluate the model
        scores = lstm_model.evaluate(X, y, verbose=0)
        print("%s: %.2f%%" % (lstm_model.metrics_names[1], scores[1] * 100))

    #  serialize model to JSON

    with open("lstm_model.json", "w") as json_file:
        json_file.write(lstm_model.to_json())
    lstm_model.save_weights("lstm_model.h5")
    print("Saved lstm model to disk")


def predict():
    # After the training, we use the json files to make predictions
    with open('lstm_model.json', 'r') as f:
        lstm_model = tf.keras.models.model_from_json(f.read())

    lstm_model.load_weights('lstm_model.h5')
    lstm_prediction = lstm_model.predict_classes(test_text)
    return lstm_prediction


if __name__ == "__main__":
    print("\t1.Train")
    print("\t2.Predict\n")
    choice = input("Write the number of your choice: ")
    choice = int(choice)

    if choice == 1:
        train()
    elif choice == 2:
        predict()
    else:
        print("Wrong. Enter 1 or 2")
