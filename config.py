DATASET_CONFIG = {
    'train_percent': 0.9,  # the rest will be used for validation
    'csv_relative_path': 'data/train.csv',
    'test_csv_relative_path': 'data/evaluation.csv',
    'remove_zero': False,  # Whether or not to remove entries with 0 RTs. Impacts the XGB data preparation as well!
    'vocab_relative_path': 'data/vocab.json'  # For AE models: the path to the vocabulary file
}

TRAIN_CONFIG = {
    'batch_size': 256,
    'learning_rate': 1e-2,
    'weight_decay': 1e-4,
    'workers': 6,
    'experiment_name': 'AE_reproduction',
    'epochs': 3,
    'checkpoint_every': 200,  # Save a checkpoint every n batches
    'val_every': 200,  # Run the validation routine every n batches
    'val_batches': 1000  # Run the validation on only n batches
}

RNN_CONFIG = {
    'hidden_size': 128,  # The hidden size of the LSTM model
    'layers': 5,  # The layers of the LSTM model
    'emb_dim': 300,  # The embedding dimension (either for the input) or for the integrated Embedding layer
    'numeric_data_size': 8,  # The number of numeric variables that are added as input to the linear layers
    'dropout': 0.1,  # The dropout probability between LSTM layers
    'linear_hidden_1': 25,
    'linear_hidden_2': 30,
    'classifier': False,  # Transforms the RNN into a classifier that outputs the probability that a Tweet has RTs > 0
    'use_AE': True,  # Whether or not to use an integrated embedding layer instead of pretrained embeddings
    'vocab_using_n_tweets': 50000,
    'AE_vocab_size': 10000  # excludes the 'unknown token' which is at 0
}

XGBOOST_CONFIG = {
    'train_file': 'data/xgboost_dataset_train.npy',
    'val_file': 'data/xgboost_dataset_val.npy',
    'test_file': 'data/xgboost_dataset_test.npy',
    'embedder': 'checkpoints/AE_reproduction/best.pth',
    'embedding_use_hidden': False,  # use the last hidden layer of the NN if True
    'embedding_use_output': True,  # use the output of the NN if True (if neither are True, use an Embedding layer)
    'embedding_size': 1,  # size of the embedding data given as input
    'colsample_bytree': 0.5,  # percentage of features used per tree.
    'n_estimators': 325,  # number of trees
    'max_depth': 12,  # max tree depth
    'learning_rate': 0.5,  # in ]0,1]
    'alpha': 0,  # L1 regularisation
    'reg_lambda': 200,  # L2 regularisation
    'subsample': 1,  # use this proportion of the train set at every step
    'experiment_name': 'XGB_reproduction',
    'log': True,  # Whether or not to estimate log(1+RT) instead of RT directly
    'numeric_data_size': 8  # The number of numeric variables that are added as input
}

EXPORT_CONFIG = {
    'log': True,
    'threshold': None
}
