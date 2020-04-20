from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU
from tensorflow.keras.models import Sequential
import tensorflow as tf
import pandas as pd
from DataPreprocess import max_features, X, y, test_text, test_data, test
import os
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def create_model():
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
    return gru_model


def train():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    if not os.path.exists("gru_model.h5"):
        gru_model = create_model()
        gru_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        gru_model_fit = gru_model.fit(X_train, y_train, validation=(X_train, y_train), epochs=1)
        gru_model.save('gru_model.h5')
        print("Saved GRU model to disk")

    else:
        gru_model = tf.keras.models.load_model('gru_model.h5')
        gru_model.summary()
        gru_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        gru_model_fit = gru_model.fit(X_train, y_train, epochs=1)
        # evaluate the model
        scores = gru_model.evaluate(X, y, verbose=0, batch_size=128)
        print("%s: %.2f%%" % (gru_model.metrics_names[1], scores[1] * 100))

    # plot for loss
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(gru_model_fit.history['loss'], label="train")
    plt.plot(gru_model_fit.history['val_loss'], label="test")
    plt.legend()

    plt.subplot(211)
    plt.title('Accuracy')
    plt.plot(gru_model_fit.history['accuracy'], label="train")
    plt.plot(gru_model_fit.history['val_accuracy'], label="test")
    plt.legend()

    plt.show()


def predict():
    # After the training, we use the json files to make prediction

    gru_model= tf.keras.models.load_model('gru_model.h5')
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
        prediction = predict()
        print(prediction)
        submission = pd.DataFrame({'id': test.index, 'gru_prediction': prediction})
        submission.to_csv('prediction.csv', index=False, mode='a')
        report = classification_report(test['label'], prediction)
        print(report)
    else:
        print("Wrong. Enter 1 or 2")
