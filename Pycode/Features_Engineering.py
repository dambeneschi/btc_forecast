import numpy as np
import pandas as pd
import pandas_ta as ta
import os

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,12)

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif



def rolling_tscores(series, window):
    '''
    Compute the T-Score on the previous values from a rolling window
    in order to not calculate a t-score based on a distribution containing future values

    return: time series of the t-score based on previous values window sample
    '''

    # Get the rolling window
    roll_series = series.rolling(window)

    # Get the mean & std of the sample of previous records (distribution)
    m = roll_series.mean().shift(1)
    s = roll_series.std(ddof=0).shift(1)

    tscores = (series - m) / s

    return tscores



def features_engineering(df, _lags=False):

    # original columns to keep
    orig_cols = df.columns.tolist()[-2:]

    # ----------------------
    # Movements and pct return (time step is hour)
    periods = [1, 6, 12, 24]
    movs_list = [df.diff(i) for i in periods]  # + [df.pct_change(i) for i in periods]
    movs_labels = []

    for i in periods:
        # Labels for the time derivatives
        movs_labels += [col + '_mov{}H'.format(i) for col in df.columns]

    # Concatenate
    feats_df = pd.concat([df.loc[:, orig_cols]] + movs_list, axis=1)
    feats_df.columns = orig_cols + movs_labels

    print(feats_df.info())

    # ----------------------
    # Rolling Statistics & T-Score
    for col in feats_df.columns:
        # Get the series
        series = feats_df.loc[:, col]

        for i in periods[1:]:
            for l, stat in zip(['Average', 'Min', 'Max', 'Std', 'Sum'], [np.mean, np.min, np.max, np.std, np.sum]):

                # Apply rolling statistic and get its movement
                feats_df[col + '.M{}{}H'.format(l, i)] = series.rolling("{}H".format(i)).apply(stat)
                feats_df[col + '.M{}{}H.diff'.format(l, i)] = series.rolling("{}H".format(i)).apply(stat).diff()

            # T-Score on rolling 1 month & 6 months sample (tscore is zscore on a sample, not on whole distribution)
            feats_df[col + '.TScore6M'] = rolling_tscores(series=series, window='4400H')
            feats_df[col + '.TScore1M'] = rolling_tscores(series=series, window='720H')

    print(feats_df.info())

    # squared features
    for col in feats_df.columns:
        feats_df[col + '.Squared'] = feats_df[col] * feats_df[col]

    feats_df.info()


    # Time Wise Features
    # Add the day of week and month of year features
    time_feats = []
    time_feats.append(pd.DataFrame(pd.get_dummies(feats_df.index.dayofweek).values,
                                   columns=['Day_of_Week_{}'.format(i) for i in range(1, 8)],
                                   index=feats_df.index))

    time_feats.append(pd.DataFrame(pd.get_dummies(feats_df.index.hour).values,
                                   columns=['Hour_{}'.format(i) for i in range(1, 25)],
                                   index=feats_df.index))

    feats_df = pd.concat([feats_df] + time_feats, axis=1)



    ## TA INDICATORS
    # Prepare dataset for Technical Analysis Indicators
    ohcl_cols = ['close', 'open', 'high', 'low', 'volumefrom']
    df_ta = df.loc[:, ohcl_cols].dropna()
    df_ta.columns = df_ta.columns[:-1].tolist() + ['volume']
    df_ta.index.name = 'datetime'

    # Compute the Indicators
    df_ta.ta.strategy(name='all')

    # Drop the original OHCL columns
    df_ta.drop(columns=df_ta.columns[:6], inplace=True)
    df_ta.columns = ['BTC Price.' + col for col in df_ta.columns]

    # Drop totally empty columns & those with high number of missing values
    df_ta.dropna(how='all', axis='columns', inplace=True)

    print(df_ta.isnull().sum().sort_values(ascending=False).iloc[:20])
    df_ta.drop(columns=df_ta.isnull().sum().sort_values(ascending=False).index[:6], inplace=True)

    # Get the movements of the indicators
    df_ta_diff = df_ta.select_dtypes('number').diff()
    df_ta_diff.columns = [col + '.Mov' for col in df_ta_diff.columns]

    df_ta.info()



    #TODO: Add difference from min/max,

    #--------------------
    # Put all together
    df_all_feats = pd.concat([feats_df, df_ta, df_ta_diff], axis=1)
    print(df_all_feats.isnull().sum().sum())

    df_all_feats.dropna(how='all', axis='columns', inplace=True)
    print(df_all_feats.isnull().sum().sum())
    print(df_all_feats.isnull().sum().sort_values(ascending=False).iloc[:20])

    df_all_feats.dropna(inplace=True)
    print(df_all_feats.isnull().sum().sum())

    df_all_feats.info()



    # LAGGED VALUES
    if _lags:
        # Add lagged values
        for col in df_all_feats.columns:
            for l in range(1, 7):
                df_all_feats[col + '.lag{}'.format(l)] = df_all_feats[col].shift(l)

    df_all_feats.dropna(inplace=True)

    print('{} NaNs in the features'.format(df_all_feats.isnull().sum().sum()))
    print('{} inf values in the features'.format(df_all_feats.isin([np.inf, -np.inf]).sum().sum()))



    return df_all_feats




def target_engineering(df):
    # PLOT OF THE BTC PRICE
    fig, ax = plt.subplots(nrows=2)

    _ = df['close'].plot(ax=ax[0], title='BTC price')
    _ = df['close'].diff().plot(ax=ax[1], title='BTC movement')
    _ = plt.tight_layout()

    plt.show()

    # 2 labels: up/down
    price_mov = df['close'].diff().dropna()
    price_mov_class = price_mov.apply(lambda x: 'up' if x >= 0 else 'down')

    fig, ax = plt.subplots(ncols=2)
    _ = price_mov_class.value_counts().plot(kind='bar', title='Up/Down directions count', ax=ax[0])

    # Same/Different trend as previous one
    price_trend = (price_mov_class == price_mov_class.shift(1)).dropna()
    price_trend = price_trend.map({True: 0, False: 1})

    # Shift the target to have the forecasting objective trend +1h
    price_trend = price_trend.shift(-1)


    # Plot
    _ = price_trend.value_counts().plot(kind='bar', title='Keep(0)/Change(1) Trend direction count', ax=ax[1])
    plt.show()

    target = price_trend
    class_weights = price_trend.value_counts().to_dict()
    print(target.tail(20))
    print(class_weights)

    return target, class_weights


def features_selection(feats_df, target, n_best):

    # Align the features & Target indexes
    feats_df['target'] = target
    feats_df.dropna(inplace=True)
    target = feats_df['target']
    feats_df.drop(columns=['target'], inplace=True)

    # Keep the 400 most relevant features
    kbest_selector = SelectKBest(f_classif, n_best)

    # Fit on data
    kbest_values = kbest_selector.fit_transform(feats_df, target)
    kbest_df = pd.DataFrame(kbest_values,
                            index=feats_df.index,
                            columns=feats_df.loc[:, kbest_selector.get_support().tolist()].columns)

    kbest_df.info()



    return kbest_selector, kbest_df




def format_3D_LSTM(kbest_df, target, look_back):
    # Add the 3rd dimension for the lags
    lagged = []
    labels = []

    # Add Lagged Features (from last to newest)
    for l in range(look_back, 0, -1):
        print('- compute lag {}'.format(l))
        lagged.append(kbest_df.shift(l))
        labels += ['{}(t{})'.format(col, l) for col in kbest_df.columns]

    # Add actual features dataframe (no lag)
    lagged.append(kbest_df)
    labels += kbest_df.columns.tolist()

    # Put together into DF
    third_df = pd.concat(lagged, axis=1)
    third_df.columns = labels

    # Align the features & Target indexes
    third_df['target'] = target
    third_df.dropna(inplace=True)
    target = third_df['target']
    third_df.drop(columns=['target'], inplace=True)

    third_df.info()


    return third_df



def split_data(third_df, target, batch_size, test_batches, val_batches, look_back, n_best):
    # Split the data - validation on 1 week
    n_test = test_batches * batch_size
    n_val = val_batches * batch_size

    # Number of obs to keep in train set to have a round number of batches
    n_train = int((len(third_df) - n_test - n_val) / batch_size) * batch_size
    print(len(third_df) - n_test - n_val)
    print(int((len(third_df) - n_test - n_val) / batch_size))

    print('Train: {}, Test:{}, Validation:{}'.format(n_train, n_test, n_val))

    # Features splitting & Scaling
    X_all = third_df.values
    X_train = X_all[-n_train - n_test - n_val:-n_test - n_val, :]
    X_test = X_all[-n_test - n_val:-n_val, :]
    X_val = X_all[-n_val:, :]

    print('- Features arrays shapes: {} {} {} {}'.format(X_all.shape, X_train.shape, X_test.shape, X_val.shape))

    # Scale the values between [-1,1]
    X_scaler = MinMaxScaler((-1, 1))

    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    X_val = X_scaler.transform(X_val)

    # Reshape
    X_train = X_train.reshape(n_train, look_back + 1, n_best)
    X_test = X_test.reshape(n_test, look_back + 1, n_best)
    X_val = X_val.reshape(n_val, look_back + 1, n_best)

    print('- Features arrays shapes: {} {} {}'.format(X_train.shape, X_test.shape, X_val.shape))

    # Targets
    y_all = target.values
    y_train = y_all[-n_train - n_test - n_val:-n_test - n_val]
    y_test = y_all[-n_test - n_val:-n_val]
    y_val = y_all[-n_val:]

    print('- Target arrays shapes: {} {} {}'.format(y_train.shape, y_test.shape, y_val.shape))

    return X_scaler, X_train, y_train, X_test, y_test, X_val, y_val





