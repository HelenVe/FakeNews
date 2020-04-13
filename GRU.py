from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM
from tensorflow.keras.models import Sequential
import tensorflow as tf
from DataPreprocess import max_features, X, y, test_text
import os
from tensorflow.keras.models import model_from_json


def train():

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    if not os.path.exists("gru_model.json") and not os.path.exists("gru_model.h5"):

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
        # evaluate the model
        scores = gru_model.evaluate(X, y, verbose=0)
        print("%s: %.2f%%" % (gru_model.metrics_names[1], scores[1] * 100))
    else:

        with open('gru_model.json', 'r') as f:
            gru_model = tf.keras.models.model_from_json(f.read())

        gru_model.load_weights('gru_model.h5')
        gru_model = Sequential(name='gru_nn_model')
        gru_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        gru_model_fit = gru_model.fit(X_train, y_train, epochs=1)
        # evaluate the model
        scores = gru_model.evaluate(X, y, verbose=0)
        print("%s: %.2f%%" % (gru_model.metrics_names[1], scores[1] * 100))

    with open("gru_model.json", "w") as json_file:
        json_file.write(gru_model.to_json())
    gru_model.save_weights("gru_model.h5")
    print("Saved lstm model to disk")


def predict():
    # After the training, we use the json files to make predictions
    with open('gru_model.json', 'r') as f:
        gru_model = tf.keras.models.model_from_json(f.read())

    gru_model.load_weights('gru_model.h5')
    gru_prediction = gru_model.predict_classes(test_text)
    return gru_prediction


if __name__ == "__main__":
    print("GRU Model \n")
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
