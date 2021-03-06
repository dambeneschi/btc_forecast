import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb


def XGB_Data_Preparation(kbest_df, target, n_val=60):
    '''
    Pre processing pipeling for XGBoost that:
    - aligns the datetime indexes for the fetaures & target sets
    - Split between Train & Validation sets
    - Scales the data using RobustScaler
    '''

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
    target_train = target.iloc[:-n_val]
    target_val = target.iloc[-n_val:]

    # Scale
    X_scaler = RobustScaler()
    df_train_scaled = pd.DataFrame(X_scaler.fit_transform(df_train.values),
                                   columns=df_train.columns,
                                   index=df_train.index)

    df_val_scaled = pd.DataFrame(X_scaler.transform(df_val.values),
                                 columns=df_val.columns,
                                 index=df_val.index)


    return X_scaler, df_train_scaled, df_val_scaled, target_train, target_val






def XGB_Tuning_Pipeline(df_train_kbest, df_val_kbest, target_train, target_val, n_test=200, n_tune=200):
    '''
    Tune the XGBoost Classifier for Binary Logistic regression problem
    by selecting n_tune random combination of the subsample parameters.
    It returns the best parmeters based on the  AUC & Accuracy scores on the validation set
    '''

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
    auc_list = []
    acc_list = []

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
        n_rounds = 2000
        xgb_reg = xgb.train(params={'objective': "binary:logistic", 'booster': 'gbtree',
                                    'tree_method': 'hist', 'grow_policy': 'lossguide',
                                    'eval_metric': 'auc',

                                    'silent': True, 'n_jobs': os.cpu_count(), 'random_state': 123,

                                    "learning_rate": 0.005, "gamma": 0, "max_depth": 5,
                                    "reg_alpha": 0.01, "reg_lambda": 0,

                                    "subsample": subsample,
                                    "colsample_bytree": colsample_bytree,
                                    "colsample_bylevel": colsample_bylevel

                                    },

                            dtrain=dm_train, num_boost_round=n_rounds,
                            callbacks=[xgb.callback.early_stop(stopping_rounds=400, maximize=False, verbose=True)],

                            # Training scoring
                            evals=[(dm_train, 'train'), (dm_test, 'test')],
                            verbose_eval=0, evals_result=xgb_evals)

        # Score on Validation Set
        y_probas = xgb_reg.predict(dm_val)
        y_preds = 1 * (y_probas > 0.5)

        auc = roc_auc_score(y_val, y_probas)
        acc = np.mean(y_preds == y_val)

        auc_list.append(auc)
        acc_list.append(acc)
        counter += 1

    # Put results together
    tuning_df = pd.DataFrame({'Params': params_list,
                              'Accuracy': acc_list,
                              "AUC": auc_list}).sort_values(['AUC', 'Accuracy'],
                                                            ascending=[False, False])

    return tuning_df


def XGB_Eval_Pipeline(params, df_train_kbest, df_val_kbest, target_train, target_val, n_test=200):
    '''
    Evaluates the XGBoost classifier for binary logistic regressiomn problem
    for the given subsample parameters obtained from tuning
    and computes the Accuracy & AUC scores on test & validations sets, returned
    The predictions on the test & validation sets are returned as dataframe
    '''

    xgb_evals = {}

    # Slice training data for CV
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
    n_rounds = 2000
    xgb_reg = xgb.train(params={'objective': "binary:logistic", 'booster': 'gbtree',
                                'tree_method': 'hist', 'grow_policy': 'lossguide',
                                'eval_metric': 'auc',

                                'silent': True, 'n_jobs': os.cpu_count(), 'random_state': 123,

                                "learning_rate": 0.005, "gamma": 0, "max_depth": 5,
                                "reg_alpha": 0.01, "reg_lambda": 0,

                                "subsample": params[0],
                                "colsample_bytree": params[1],
                                "colsample_bylevel": params[2]

                                },

                        dtrain=dm_train, num_boost_round=n_rounds,
                        callbacks=[xgb.callback.early_stop(stopping_rounds=400, maximize=False, verbose=True)],

                        # Training scoring
                        evals=[(dm_train, 'train'), (dm_test, 'test')],
                        verbose_eval=20, evals_result=xgb_evals)


    # Score on Validation Set
    y_probas = xgb_reg.predict(dm_val)
    y_preds = 1 * (y_probas > 0.5)

    auc = roc_auc_score(y_val, y_probas)
    acc = np.mean(y_preds == y_val)


    print('\n=> XGB Score on Validation set ({} obs): AUC: {}\tAccuracy: {}'.format(len(y_val),
                                                                                    round(auc, 2),
                                                                                    str(100 * acc) + '%'))


    # Format test values as dataframe
    df_test = pd.DataFrame({'Real Label': y_test,
                            'Predicted Label': xgb_reg.predict(dm_test)},
                           index=target_train.iloc[-n_test:].index).round(0)

    # Format the forecasts for Trading Backtesting
    df_backtest = pd.DataFrame({'Real Label': y_val,
                                'Predicted Label': y_preds}, index=target_val.index).round(0)


    # Add the Labels Keep(0)/Change(1)
    df_test['Predicted Trend'] = df_test['Predicted Label'].map({1:'Change',  0: 'Keep'})
    df_backtest['Predicted Trend'] = df_backtest['Predicted Label'].map({1: 'Change', 0: 'Keep'})


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


    return xgb_reg, df_test, df_backtest
