DATASET_CONFIG = {
    'train_percent': 0.9,  # the rest will be used for validation
    'csv_relative_path': 'data/train.csv'
}

TRAIN_CONFIG = {
    'batch_size': 1000,
    'learning_rate': 1e-5,
    'weight_decay': 0,
    'workers': 6,
    'experiment_name': 'rnn_MAE_classifier',
    'epochs': 10,
    'checkpoint_every': 100,
    'val_every': 200,
    'val_batches': 10,
}

RNN_CONFIG = {
    'hidden_size': 64,
    'layers': 3,
    'emb_dim': 300,
    'numeric_data_size': 5,
    'dropout': 0.1,
    'linear_hidden_size': 5,
    'classifier': True
}
