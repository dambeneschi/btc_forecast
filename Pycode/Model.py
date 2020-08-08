import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,12)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#TF 1.5 with Keras 2.2.5 for CUDA 9.0
from keras.layers import LSTM, Dense, TimeDistributed, SpatialDropout1D
from keras.optimizers import RMSprop
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model


class LSTM_Model:
    '''
    Builds an LSTM unidirectional model
    one LSTM layer of 300 units, one SpatialDropout
    and Dense with Softmax for propability like output for binary classification
    with RMSProp optimizer,
    and trains it across n_epochs
    '''

    def __init__(self, X_train, y_train, X_test, y_test, lr, n_epochs, activation, batch_size, dr, class_weights):
        self.lr = lr
        self.n_epochs = n_epochs
        self.activation = activation
        self.batch_size = batch_size
        self.dr = dr

        self.class_weights = class_weights
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.trained_path = os.path.join(os.getcwd(), 'Trained/')



    def build_model(self):

        # Network Architecture
        lstm = Sequential()


        lstm.add(LSTM(300, activation=self.activation, stateful=True,
                      batch_input_shape=(self.batch_size, self.X_train.shape[1], self.X_train.shape[2]),
                      return_sequences=True, dropout=0, recurrent_dropout=0))

        # Dropout for overfitting - constant mask
        lstm.add(SpatialDropout1D(self.dr))

        # Output layer
        lstm.add(TimeDistributed(Dense(units=1, activation='sigmoid')))

        print(lstm.summary())
        self.lstm = lstm



    def train(self):

        # Fitting across epochs
        callbacks = [ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10,
                                       mode="auto", min_delta=0.0001, min_lr=1e-9)]

        # Compiling Optimizer
        rmsprop = RMSprop(lr=self.lr, rho=0.9, epsilon=None, decay=0.0)
        self.lstm.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

        # loss arrays
        train_loss_list = []
        train_acc_list = []

        test_loss_list = []
        test_acc_list = []

        counter = 1

        for n in range(self.n_epochs):
            print('** epoch {}/{} **'.format(counter, self.n_epochs))

            history = self.lstm.fit(x=self.X_train, y=self.y_train, epochs=1,
                                    validation_data=(self.X_test, self.y_test),
                                    batch_size=self.batch_size, class_weight=self.class_weights,
                                    shuffle=False, callbacks=callbacks, use_multiprocessing=True)

            # Append the list to follow the evolution of loss on train and test sets
            train_loss_list.append(history.history['loss'][0])
            train_acc_list.append(history.history['acc'][0])

            test_loss_list.append(history.history['val_loss'][0])
            test_acc_list.append(history.history['val_acc'][0])

            # Reset the cell states (required for stateful)
            self.lstm.reset_states()

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



    def save_trained(self):
        self.lstm.save_model(os.path.join(self.trained_path, 'BTC_{}ep.h5'.format(self.n_epochs)))
