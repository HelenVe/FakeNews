import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from DataPreprocess import test_text, test, X_train, X_test, y_train, y_test, vocab_size, embeddings_matrix


def create_model():
    # Build the architecture of the model

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=100, weights=[embeddings_matrix], trainable=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.LSTM(20),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def train():

    if not os.path.exists("model.h5"):
        model = create_model()
    else:
        print("Loading and evaluating  model")
        model = load_model('model.h5')

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
    model.save('model.h5')
    print("Saved  model to disk")
    train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train:', train_acc, 'Test:', test_acc)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

    return history


def predict():

    model = load_model('model.h5')
    prediction = model.predict(test_text)
    prediction = np.reshape(prediction, -1)
    prediction = np.round(prediction)
    return prediction


if __name__ == "__main__":
    print("\t1.Train")
    print("\t2.Predict\n")
    choice = input("Write the number of your choice: ")
    choice = int(choice)

    if choice == 1:
        model_fit = train()
    elif choice == 2:
        prediction = predict()
        print(prediction)
        pd_pred = pd.DataFrame({'id': test.index, 'prediction': prediction})
        pd_pred.to_csv('Kaggle/rf_pred.csv', index=False)
    else:
        print("Wrong. Enter 1 or 2")
