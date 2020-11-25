DATASET_CONFIG = {
    'train_percent': 0.9,  # the rest will be used for validation
    'csv_relative_path': 'data/train.csv',
    'test_csv_relative_path': 'data/evaluation.csv',
    'remove_zero': False
}

TRAIN_CONFIG = {
    'batch_size': 512,
    'learning_rate': 1e-3,
    'weight_decay': 0,
    'workers': 6,
    'experiment_name': 'latest_classifier_1',
    'epochs': 2,
    'checkpoint_every': 100,
    'val_every': 200,
    'val_batches': 1000
}

RNN_CONFIG = {
    'hidden_size': 64,
    'layers': 3,
    'emb_dim': 300,
    'numeric_data_size': 5,
    'dropout': 0.1,
    'linear_hidden_1': 15,
    'linear_hidden_2': 20,
    'classifier': True
}

XGBOOST_CONFIG = {
    'train_file': 'data/xgboost_dataset_train.npy',
    'val_file': 'data/xgboost_dataset_val.npy',
    'embedder': 'checkpoints/embedder/epoch_1.pth',
    'colsample_bytree': 0.5,  # percentage of features used per tree.
    'n_estimators': 200,  # number of trees
    'max_depth': 5,  # max tree depth
    'learning_rate': 0.5,  # in ]0,1]
    'alpha': 0,  # L1 regularisation
    'reg_lambda': 200,  # L2 regularisation
    'experiment_name': 'xgb_14',
    'log': True
}
