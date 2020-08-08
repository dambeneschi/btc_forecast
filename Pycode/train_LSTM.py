import pickle

from .Data_Aquisition import get_crypto_data
from .Features_Engineering import *
from .Model import LSTM_Model



class Training_Pipeline():

    #def __init__(self):



    def data_pipeline(self, n_batch_obs,  currency, exchange, n_best, look_back, batch_size, test_batches, val_batches):
        ''' Data Pipeline:
        - Get data from the CryptoCompare API
        - Engineer the featires based on the BTC prices data
        - Engineer the target for the classification problem
        - Keep only the k best features for classificaiton problem
        - Format into 3D matrix with lagged series for LSTM
        - Split into Train/Test/Validation sets

        :param n_batch_obs: number of observations batches (1 batch = 2000 obs) retrieved from the CryptoAPI
        :param currency: currency fo the BTC price (USD, EUR)
        :param exchange: Exchange platform to get the prices for (Coinbase, Brittrex)
        :param n_best: K Best features to keep for the classification problem
        :param look_back: number of lags to use for the LSTM (3rd dimension input array)
        :param batch_size: batch_size argument for the neural network
        :param test_batches: number of batches for testing (n * batch_size)
        :param val_batches: number of batches for validation (n * batch_size)
        '''
        print('* RUNNING THE DATA PIPELINE')

        # Useful for the model training
        self.batch_size = batch_size

        # Get data from API
        df = get_crypto_data(n_batch_obs=n_batch_obs,  currency=currency, exchange=exchange)

        # Create the features dataset
        feats_df = features_engineering(df)

        # Create the target series
        target, self.class_weights = target_engineering(df)

        # Select Kbest features
        kbest_selector, kbest_df = features_selection(feats_df, target, n_best)

        # Format the data for LSTM (3D array)
        third_df = format_3D_LSTM(kbest_df, target, look_back)

        # Split data for Training / Testing / Validation
        X_scaler, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val = split_data(third_df,
                                                                                                                 target,
                                                                                                                 batch_size,
                                                                                                                 test_batches,
                                                                                                                 val_batches,
                                                                                                                 look_back,
                                                                                                                 n_best)

        # Pickle the fitted objects to /Trained
        for obj, name in zip([kbest_selector, X_scaler], ['kbest_selector.pkl', 'X_scaler.pkl']):
            pkl_path = os.path.join(os.getcwd(), 'Trained/{}'.format(name))

            with open(pkl_path, 'wb') as pickled:
                pickle.dump(obj, pickled)






    def train_LSTM(self):
        '''
        Train the LSTM across epochs and save the instance as .h5
        :return:
        '''
        print('* TRAINING THE LSTM MODEL')
        lstm = LSTM_Model(self.X_train, self.y_train, self.X_test, self.y_test,
                          lr=1e-5, n_epochs=2000, activation='tanh',
                          batc_size=self.batch_size, dr=0.1, class_weights=self.class_weights)






