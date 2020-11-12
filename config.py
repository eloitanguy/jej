DATASET_CONFIG = {
    'train_percent': 0.9,  # the rest will be used for validation
    'csv_relative_path': 'data/train.csv'
}

TRAIN_CONFIG = {
    'batch_size': 1000,
    'learning_rate': 1e-6,
    'weight_decay': 0,
    'workers': 6,
    'experiment_name': 'test',
    'epochs': 20,
    'checkpoint_every': 100,
    'val_every': 200,
    'val_batches': 10,
    'RNN_hidden_units': 100
}
