from BTC_repository.Pycode.train_LSTM import Training_Pipeline

if __name__ == '__main__':

    # Run the Training Pipeline
    ಠ_ಠ = Training_Pipeline()
    ಠ_ಠ.data_pipeline(n_batch_obs=8,  currency='EUR', exchange='Coinbase',
                      n_best=400, look_back=6,
                      batch_size=32, test_batches=20, val_batches=10)

