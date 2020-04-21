from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU
from tensorflow.keras.models import Sequential
import tensorflow as tf
import pandas as pd
from DataPreprocess import max_features, y, test_text, test, X_train, X_test, y_train, y_test
import os
from sklearn import metrics
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

    if not os.path.exists("gru_model.h5"):
        gru_model = create_model()
    else:
        gru_model = tf.keras.models.load_model('gru_model.h5')
        gru_model.summary()

    gru_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    gru_model_fit = gru_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)
    # evaluate the model
    train_acc = gru_model.evaluate(X_train, y_train, verbose=0, batch_size=128)
    test_acc = gru_model.evaluate(X_test, y_test, verbose=0, batch_size=128)
    print('Train: %.3f, Test: %.3f', train_acc, test_acc)

    # plot for loss
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(gru_model_fit.history['loss'], label="train loss")
    plt.plot(gru_model_fit.history['val_loss'], label="test loss")
    plt.legend()

    plt.subplot(211)
    plt.title('Accuracy')
    plt.plot(gru_model_fit.history['acc'], label="train accuracy")
    plt.plot(gru_model_fit.history['val_acc'], label="test accuracy")
    plt.legend()

    plt.show()


def predict():
    # After the training, we use the json files to make prediction

    gru_model = tf.keras.models.load_model('gru_model.h5')
    gru_prediction = gru_model.predict_classes(test_text)
    gru_prediction = gru_prediction.flatten()
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
        submission = pd.DataFrame({'id': test['id'], 'gru_prediction': prediction})
        submission.to_csv('Kaggle/gru_pred.csv', index=True)

    else:
        print("Wrong. Enter 1 or 2")
