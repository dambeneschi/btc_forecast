import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler


def XGB_Data_Preparation(kbest_df, target, n_val=60):
    ''''''

    # Put together the features & Trading signal
    kbest_df['Target'] = target
    kbest_df.dropna(inplace=True)

    print('- {} NaNs in the dataset (features & target)'.format(kbest_df.isnull().sum().sum()))

    # Split again target & features
    target = kbest_df['Target']
    kbest_df.drop(columns='Target', inplace=True)
    print(kbest_df.shape, target.shape)

    print(kbest_df.info())

    # Validation data: most recent 2 weeks
    df_train = kbest_df.iloc[:-n_val, :]
    df_val = kbest_df.iloc[-n_val:, :]
    print(kbest_df.shape, df_train.shape, df_val.shape)

    # Split target
    target_train = target.iloc[:-n_val] / 100
    target_val = target.iloc[-n_val:] / 100

    # Scale
    X_scaler = RobustScaler()
    df_train_scaled = pd.DataFrame(X_scaler.fit_transform(df_train.values),
                                   columns=df_train.columns,
                                   index=df_train.index)

    df_val_scaled = pd.DataFrame(X_scaler.transform(df_val.values),
                                 columns=df_val.columns,
                                 index=df_val.index)

    return df_train_scaled, df_val_scaled, target_train, target_val





def xgb_ng_loss(yp, dmt):
    '''
    Custom loss function to be passed as feval argument in XGB train instance
    PENALIZED RMSE || TO BE MINIMIZED ON TRAIN SET

    - penalty factor is proportional to the number of wrong prediction
    - yp (predictions) is type np.array
    - dmt (real values) is type xgb.DMatrix -> yt should be extracted as array

    Checks that the resuls are:
           - in the right range (+/- threshold)  with a penalty factor to make this part more important
           - in the same direction

    At each boosting round, the loss is computed on the TRAIN set.
    yp & yt arrays have shapes (n_obs_train, )

    '''

    # Extract the y array from the true values DMatrix
    yt = dmt.get_label()

    penalty = 10 * np.sum((yt * yp) < 0)

    # return penalized RMSE
    return 'ng_loss', penalty * (np.mean((yt - yp) ** 2) ** 0.5)


def XGB_Tuning_Pipeline(df_train_kbest, df_val_kbest, target_train, target_val, n_test=200, n_tune=200):
    ''''''

    xgb_evals = {}

    # Slice trianing data for CV
    X_train = df_train_kbest.iloc[:-n_test, :].values
    X_test = df_train_kbest.iloc[-n_test:, :].values

    y_train = target_train.iloc[:-n_test].values
    y_test = target_train.iloc[-n_test:].values

    X_val = df_val_kbest.values
    y_val = target_val.values

    # XGB Matrix datasets
    dm_train = xgb.DMatrix(data=X_train, label=y_train,
                           feature_names=df_train_kbest.columns,
                           nthread=os.cpu_count())

    dm_test = xgb.DMatrix(data=X_test, label=y_test,
                          feature_names=df_train_kbest.columns,
                          nthread=os.cpu_count())

    dm_val = xgb.DMatrix(data=X_val, label=y_val,
                         feature_names=df_val_kbest.columns,
                         nthread=os.cpu_count())

    # Tuning results
    params_list = []
    mse_list = []
    dir_score_list = []

    # Tune over the subsample parameters
    subsample_space = np.linspace(0.01, 0.8, 20)
    colsample_bytree_space = np.linspace(0.01, 0.8, 20)
    colsample_bylevel_space = np.linspace(0.01, 0.8, 20)

    counter = 1
    np.random.seed(4788)

    for i in range(n_tune):
        # Random choice in the spaces
        subsample = round(np.random.choice(subsample_space, 1)[0], 4)
        colsample_bytree = round(np.random.choice(colsample_bytree_space, 1)[0], 4)
        colsample_bylevel = round(np.random.choice(colsample_bylevel_space, 1)[0], 4)

        params_draw = (subsample, colsample_bytree, colsample_bylevel)
        params_list.append(params_draw)
        print('* {}/{}: {}'.format(counter, n_tune, params_draw))

        # Fit model
        n_rounds = 1400
        xgb_reg = xgb.train(params={'objective': "reg:linear", 'booster': 'gbtree', 'disable_default_eval_metric': 1,
                                    'tree_method': 'hist', 'grow_policy': 'lossguide',

                                    'silent': True, 'n_jobs': os.cpu_count(), 'random_state': 123,

                                    "learning_rate": 0.01, "gamma": 0, "max_depth": 4,
                                    "reg_alpha": 0.01, "reg_lambda": 0,

                                    "subsample": subsample,
                                    "colsample_bytree": colsample_bytree,
                                    "colsample_bylevel": colsample_bylevel

                                    },

                            dtrain=dm_train, num_boost_round=n_rounds,
                            callbacks=[xgb.callback.early_stop(stopping_rounds=200, maximize=False, verbose=True)],

                            # Custom scoring
                            feval=xgb_ng_loss, evals=[(dm_train, 'train'), (dm_test, 'test')],
                            verbose_eval=0, evals_result=xgb_evals)

        # Score on Validation Set
        y_preds = xgb_reg.predict(dm_val)
        mse = mean_squared_error(y_val, y_preds)
        dir_score = np.mean((y_preds * y_val) > 0)

        dir_score_list.append(dir_score)
        mse_list.append(mse)

        counter += 1

    # Put results together
    tuning_df = pd.DataFrame({'Params': params_list,
                              'Dir_Score': dir_score_list,
                              "MSE": mse_list}).sort_values(['Dir_Score', 'MSE'],
                                                            ascending=[False, True])

    return tuning_df


def XGB_Eval_Pipeline(params, df, target_name, df_train_kbest, df_val_kbest, target_train, target_val, n_test=200,
                      _plot=False):
    ''''''

    xgb_evals = {}

    # Slice trianing data for CV
    X_train = df_train_kbest.iloc[:-n_test, :].values
    X_test = df_train_kbest.iloc[-n_test:, :].values

    y_train = target_train.iloc[:-n_test].values
    y_test = target_train.iloc[-n_test:].values

    X_val = df_val_kbest.values
    y_val = target_val.values

    # XGB Matrix datasets
    dm_train = xgb.DMatrix(data=X_train, label=y_train,
                           feature_names=df_train_kbest.columns,
                           nthread=os.cpu_count())

    dm_test = xgb.DMatrix(data=X_test, label=y_test,
                          feature_names=df_train_kbest.columns,
                          nthread=os.cpu_count())

    dm_val = xgb.DMatrix(data=X_val, label=y_val,
                         feature_names=df_val_kbest.columns,
                         nthread=os.cpu_count())

    # Fit model
    n_rounds = 1400
    xgb_reg = xgb.train(params={'objective': "reg:linear", 'booster': 'gbtree', 'disable_default_eval_metric': 1,
                                'tree_method': 'hist', 'grow_policy': 'lossguide',

                                'silent': True, 'n_jobs': os.cpu_count(), 'random_state': 123,

                                "learning_rate": 0.01, "gamma": 0, "max_depth": 4,
                                "reg_alpha": 0.01, "reg_lambda": 0,

                                "subsample": params[0],
                                "colsample_bytree": params[1],
                                "colsample_bylevel": params[2]

                                },

                        dtrain=dm_train, num_boost_round=n_rounds,
                        callbacks=[xgb.callback.early_stop(stopping_rounds=200, maximize=False, verbose=True)],

                        # Custom scoring
                        feval=xgb_ng_loss, evals=[(dm_train, 'train'), (dm_test, 'test')],
                        verbose_eval=0, evals_result=xgb_evals)

    # Plots
    if _plot:
        fig, ax = plt.subplots(nrows=2)

        # Train & Test
        _ = ax[0].plot(range(len(xgb_evals['train']['ng_loss'])), xgb_evals['train']['ng_loss'])
        _ = ax[1].plot(range(len(xgb_evals['train']['ng_loss'])), xgb_evals['test']['ng_loss'])
        plt.show()

    # Score on Validation Set
    y_preds = xgb_reg.predict(dm_val)

    dir_score = np.mean((y_preds * y_val) > 0)
    mse = mean_squared_error(y_val, y_preds)

    print('\n=> XGB Score on Validation set ({} obs): {}\t{}'.format(len(y_val),
                                                                     round(mse, 4),
                                                                     str(100 * dir_score) + '%'))

    # Get the average difference between preditced vs real value for correct direction
    results_diff = ((y_preds * y_val) > 0) * np.abs(y_preds - y_val) * 100
    av_diff = np.mean(results_diff)
    min_diff = np.min(results_diff)
    max_diff = np.max(results_diff)
    print('- Difference for Predicted Signal with correct sign: \n\t- average: {} \n\t- min: {} \n\t- max; {}'.format(
        av_diff,
        min_diff,
        max_diff))

    print('{}/{}'.format(np.sum((y_preds * y_val) > 0), len(y_val)))

    # Format test values as dataframe
    df_test = pd.DataFrame({'Real Signal': y_test * 100,
                            'Predicted Signal': xgb_reg.predict(dm_test) * 100},
                           index=target_train.iloc[-n_test:].index).round(0)

    # Format the forecasts for Trading Backtesting
    df_backtest = pd.DataFrame({'Real Signal': y_val * 100,
                                'Predicted Signal': y_preds * 100}, index=target_val.index).round(0)

    # Add copper price
    df_backtest['Copper Close Price'] = df['LME Copper_Close']
    df_backtest['Copper Open Price'] = df['LME Copper_Open']

    # Save as csv
    df_backtest.index.name = 'DateTime'

    # Replace values above the limit by the limits
    df_backtest['Real Signal'] = df_backtest['Real Signal'].apply(lambda x: 100 if x > 100 else x)
    df_backtest['Real Signal'] = df_backtest['Real Signal'].apply(lambda x: -100 if x < -100 else x)

    df_backtest['Predicted Signal'] = df_backtest['Predicted Signal'].apply(lambda x: 100 if x > 100 else x)
    df_backtest['Predicted Signal'] = df_backtest['Predicted Signal'].apply(lambda x: -100 if x < -100 else x)

    # To CSV
    df_backtest.to_csv('Backtest_XGBTradingSignal_{}.csv'.format(target_name))

    # Plots
    fig, ax = plt.subplots(nrows=2)

    _ = df_test.loc[:, ['Real Signal', 'Predicted Signal']].iloc[-40:, :].plot(kind='bar', ax=ax[0],
                                                                               title='Real vs Predicted {} Trading Signal on Test set'.format(
                                                                                   target_name))

    _ = ax[0].set_xticklabels([str(idx).split(' ')[0] for idx in
                               df_test.loc[:, ['Real Signal', 'Predicted Signal']].iloc[-40:, :].index.tolist()])

    _ = df_backtest.loc[:, ['Real Signal', 'Predicted Signal']].plot(kind='bar', ax=ax[1],
                                                                     title='Real vs Predicted {} Trading Signal on Validation set'.format(
                                                                         target_name))

    _ = ax[1].set_xticklabels(
        [str(idx).split(' ')[0] for idx in df_backtest.loc[:, ['Real Signal', 'Predicted Signal']].index.tolist()])

    _ = plt.tight_layout()
    plt.show()

    # Features Importance
    ftype_list = ['weight', 'total_gain']

    plt.rcParams['figure.figsize'] = (16, 6 * len(ftype_list))
    fig, ax = plt.subplots(nrows=len(ftype_list))

    for i, ftype in enumerate(ftype_list):
        # Sorted importance
        drivers_weight = pd.Series(xgb_reg.get_score(fmap='', importance_type=ftype)).sort_values(ascending=False)

        # Subplot
        _ = drivers_weight.iloc[:20].plot(kind='bar', color='b', ax=ax[i],
                                          title='Features Importance - {} criteria'.format(ftype))

    _ = plt.tight_layout()
    plt.show()

    return df_backtest
