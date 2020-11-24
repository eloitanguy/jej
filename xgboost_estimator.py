import numpy as np
from config import DATASET_CONFIG
import csv
from dataset import COLUMN_NAME_TO_IDX, TweetDataset, collate_function, DATASET_SPLIT
from modules import printProgressBar
from torch.utils.data import DataLoader
import torch
from models import RNN
from config import XGBOOST_CONFIG, TRAIN_CONFIG
import xgboost as xgb
import argparse
import os


def prepare_datasets():
    checkpoint = torch.load(XGBOOST_CONFIG['embedder'])
    embed = RNN(config=checkpoint['net_config']).eval()
    embed.load_state_dict(checkpoint['model'])
    embed = embed.cuda()

    with open(DATASET_CONFIG['csv_relative_path'], newline='') as csvfile:
        data_list = list(csv.reader(csvfile))[1:]

    data = np.zeros((len(data_list), 5 + 20 + 1))  # 20 is rnn embedding, 1 for answer
    n = len(data_list)

    for i, entry in enumerate(data_list):
        printProgressBar(i, n, prefix='numeric data')
        data[i, 0] = int(entry[COLUMN_NAME_TO_IDX['user_verified']] == 'True')
        data[i, 1] = int(entry[COLUMN_NAME_TO_IDX['timestamp']]) / 1000 % (3600 * 24) / (3600 * 24 / 2) - 1
        data[i, 2] = int(entry[COLUMN_NAME_TO_IDX['user_statuses_count']])
        data[i, 3] = int(entry[COLUMN_NAME_TO_IDX['user_followers_count']])
        data[i, 4] = int(entry[COLUMN_NAME_TO_IDX['user_friends_count']])
        data[i, -1] = int(int(entry[COLUMN_NAME_TO_IDX['retweet_count']]))

    # add RNN embeddings here to train_data
    dataset = TweetDataset('all')
    loader = DataLoader(dataset, batch_size=TRAIN_CONFIG['batch_size'], num_workers=TRAIN_CONFIG['workers'],
                        collate_fn=collate_function, shuffle=False)
    current_idx = 0
    n = len(loader)
    print('')
    for batch_index, batch in enumerate(loader):
        printProgressBar(batch_index, n, prefix='embedding data')
        batch_size = batch['numeric'].shape[0]

        numeric = batch['numeric'].cuda()
        text = batch['embedding'].cuda()
        _, embedding = embed(text, numeric)

        for idx_in_batch in range(batch_size):
            data[current_idx + idx_in_batch, 5:-1] = embedding[idx_in_batch].detach().cpu().numpy()

        current_idx += batch_size

    np.save(XGBOOST_CONFIG['train_file'], data[1:DATASET_SPLIT])
    np.save(XGBOOST_CONFIG['val_file'], data[DATASET_SPLIT:])


def train():
    train_set = np.load(XGBOOST_CONFIG['train_file'])
    X, Y = train_set[:, :-1], np.log(1+train_set[:, -1]) if XGBOOST_CONFIG['log'] else train_set[:, -1]
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror',
                              colsample_bytree=XGBOOST_CONFIG['colsample_bytree'],
                              learning_rate=XGBOOST_CONFIG['learning_rate'],
                              max_depth=XGBOOST_CONFIG['max_depth'],
                              alpha=XGBOOST_CONFIG['alpha'],
                              reg_lambda=XGBOOST_CONFIG['reg_lambda'],
                              n_estimators=XGBOOST_CONFIG['n_estimators'],
                              verbosity=0)
    xg_reg.fit(X, Y)

    print('Computing train MAE ...')
    train_preds = xg_reg.predict(X)
    train_mae = np.mean(np.abs(np.exp(train_preds) - np.exp(Y))) if XGBOOST_CONFIG['log'] \
        else np.mean(np.abs(train_preds - Y))
    print('Train MAE: {}'.format(train_mae))

    print('Computing val MAE ...')
    val_set = np.load(XGBOOST_CONFIG['val_file'])
    X_val, Y_val = val_set[:, :-1], np.log(1+val_set[:, -1]) if XGBOOST_CONFIG['log'] else val_set[:, -1]
    val_preds = xg_reg.predict(X_val)
    val_mae = np.mean(np.abs(np.exp(val_preds) - np.exp(Y_val))) if XGBOOST_CONFIG['log'] \
        else np.mean(np.abs(val_preds - Y_val))
    print('Val MAE: {}'.format(val_mae))

    checkpoint_folder = 'checkpoints/{}/'.format(XGBOOST_CONFIG['experiment_name'])
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    xg_reg.save_model('{}/checkpoint.model'.format(checkpoint_folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XGBoost retweet estimator')
    parser.add_argument('--prepare', action='store_true',
                        help='Prepare the train and val datasets for the XGBoost regressor')
    parser.add_argument('--train', action='store_true',
                        help='Train the XGBoost regressor using the configuration in the config.py file')
    args = parser.parse_args()

    if args.prepare:
        prepare_datasets()
    if args.train:
        train()
