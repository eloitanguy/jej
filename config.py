DATASET_CONFIG = {
    'train_percent': 0.9,  # the rest will be used for validation
    'csv_relative_path': 'data/train.csv',
    'test_csv_relative_path': 'data/evaluation.csv',
    'remove_zero': False,  # Impacts the XGB data preparation as well!
    'vocab_relative_path': 'data/vocab.json'
}

TRAIN_CONFIG = {
    'batch_size': 256,
    'learning_rate': 1e-2,
    'weight_decay': 1e-3,
    'workers': 6,
    'experiment_name': 'AE_6',
    'epochs': 1,
    'checkpoint_every': 200,
    'val_every': 200,
    'val_batches': 1000
}

RNN_CONFIG = {
    'hidden_size': 128,
    'layers': 5,
    'emb_dim': 300,
    'numeric_data_size': 5,
    'dropout': 0.5,
    'linear_hidden_1': 15,
    'linear_hidden_2': 20,
    'classifier': False,
    'use_AE': True,
    'vocab_using_n_tweets': 50000,
    'AE_vocab_size': 10000  # excludes the 'unknown token' which is at 0
}

XGBOOST_CONFIG = {
    'train_file': 'data/xgboost_dataset_train.npy',
    'val_file': 'data/xgboost_dataset_val.npy',
    'embedder': 'checkpoints/embedder/epoch_1.pth',
    'colsample_bytree': 0.5,  # percentage of features used per tree.
    'n_estimators': 200,  # number of trees
    'max_depth': 10,  # max tree depth
    'learning_rate': 0.5,  # in ]0,1]
    'alpha': 0,  # L1 regularisation
    'reg_lambda': 200,  # L2 regularisation
    'experiment_name': 'xgb_nonzero',
    'log': True,
    'remove_zero': False,  # /!\ this affects only the exportation and not the dataset preparation
    'remove_zero_threshold': 0.5
}

EXPORT_CONFIG = {
    'log': True,
    'threshold': None
}
