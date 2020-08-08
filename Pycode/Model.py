import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,12)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.layers import LSTM, Softmax, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model


class LSTM_Model:
    def __init__(self, X_train, y_train, X_test, y_test, lr, n_epochs, activation, dr):
        self.lr = lr
        self.n_epochs = n_epochs
        self.activation = activation
        self.dr = dr

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test



    def build_model(self):

        # Network Architecture
        lstm = Sequential()


        lstm.add(LSTM(200, activation=activation, stateful=True,
                      batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]),
                      return_sequences=True, dropout=dr, recurrent_dropout=dr))

        lstm.add(LSTM(200, activation=activation, stateful=True,
                      batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]),
                      return_sequences=True, dropout=dr, recurrent_dropout=dr))

        lstm.add(LSTM(100, activation=activation, stateful=True,
                      batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]),
                      return_sequences=True, dropout=dr, recurrent_dropout=dr))

        lstm.add(LSTM(40, activation=activation, stateful=True,
                      batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]),
                      return_sequences=False, dropout=dr, recurrent_dropout=dr))

        # Output layer
        lstm.add(Dense(units=1, activation='sigmoid'))


        print(lstm.summary())

    def train(self):

        # Fitting across epochs
        callbacks = [ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10,
                                       mode="auto", min_delta=0.0001, min_lr=1e-9)]

        # Compiling Optimizer
        rmsprop = RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
        lstm.compile(loss=BinaryCrossentropy(), optimizer=rmsprop,
                     metrics=['accuracy'])

        # loss arrays
        train_loss_list = []
        train_acc_list = []

        test_loss_list = []
        test_acc_list = []

        counter = 1

        for n in range(n_epochs):
            print('** epoch {}/{} **'.format(counter, n_epochs))

            history = lstm.fit(x=X_train, y=y_train, epochs=1, validation_data=(X_test, y_test),
                               batch_size=batch_size, shuffle=False, callbacks=callbacks,
                               class_weight=class_weights,
                               use_multiprocessing=True)

            # Append the list to follow the evolution of loss on train and test sets
            train_loss_list.append(history.history['loss'][0])
            train_acc_list.append(history.history['accuracy'][0])

            test_loss_list.append(history.history['val_loss'][0])
            test_acc_list.append(history.history['val_accuracy'][0])

            # Reset the cell states (required for stateful)
            lstm.reset_states()

            counter += 1

            # Plot the errors
            fig, ax = plt.subplots(nrows=4)
            _ = ax[0].plot(range(1, counter), train_loss_list)
            _ = ax[0].set_title('Binary Crossentropy Loss on train set')

            _ = ax[1].plot(range(1, counter), train_acc_list)
            _ = ax[1].set_title('Accuracy Loss on train set')

            _ = ax[2].plot(range(1, counter), test_loss_list)
            _ = ax[2].set_title('Binary Crossentropy Loss on test set')

            _ = ax[3].plot(range(1, counter), test_acc_list)
            _ = ax[3].set_title('Accuracy Loss on test set')

            plt.show()


