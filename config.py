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
    'weight_decay': 1e-4,
    'workers': 6,
    'experiment_name': 'AE_9',
    'epochs': 1,
    'checkpoint_every': 200,
    'val_every': 200,
    'val_batches': 1000
}

RNN_CONFIG = {
    'hidden_size': 32,
    'layers': 5,
    'emb_dim': 20,
    'numeric_data_size': 8,
    'dropout': 0.1,
    'linear_hidden_1': 15,
    'linear_hidden_2': 20,
    'classifier': False,
    'use_AE': True,
    'vocab_using_n_tweets': 50000,
    'AE_vocab_size': 10000,  # excludes the 'unknown token' which is at 0
    'keep_all_hidden_activations': False  # whether h entirely of only the last hidden activation before linear blocks
}

XGBOOST_CONFIG = {
    'train_file': 'data/xgboost_dataset_train.npy',
    'val_file': 'data/xgboost_dataset_val.npy',
    'test_file': 'data/xgboost_dataset_test.npy',
    'embedder': 'checkpoints/AE_9/epoch_0.pth',
    'embedding_use_hidden': False,  # use an included embedding layer: 'False', use the last hidden layer: 'True'
    'embedding_size': 20,
    'colsample_bytree': 0.5,  # percentage of features used per tree.
    'n_estimators': 200,  # number of trees
    'max_depth': 10,  # max tree depth
    'learning_rate': 0.5,  # in ]0,1]
    'alpha': 0,  # L1 regularisation
    'reg_lambda': 200,  # L2 regularisation
    'experiment_name': 'xgb_wae_6',
    'log': True,
    'numeric_data_size': 8
}

EXPORT_CONFIG = {
    'log': True,
    'threshold': None
}
