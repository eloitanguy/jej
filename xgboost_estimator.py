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
import json


def prepare_datasets():
    checkpoint = torch.load(XGBOOST_CONFIG['embedder'])
    embed = RNN(config=checkpoint['net_config']).eval()
    embed.load_state_dict(checkpoint['model'])
    embed = embed.cuda()

    dataset = TweetDataset('all')
    N = len(dataset)
    data = np.zeros((N, 5 + 20 + 1))  # 20 is rnn embedding, 1 for answer
    loader = DataLoader(dataset, batch_size=TRAIN_CONFIG['batch_size'], num_workers=TRAIN_CONFIG['workers'],
                        collate_fn=collate_function, shuffle=False)
    current_idx = 0
    n = len(loader)
    print('')
    for batch_index, batch in enumerate(loader):
        printProgressBar(batch_index, n)
        batch_size = batch['numeric'].shape[0]

        numeric = batch['numeric'].cuda()
        text = batch['embedding'].cuda()
        _, embedding = embed(text, numeric)

        data[current_idx:current_idx+batch_size, 5:-1] = embedding.detach().cpu().numpy()
        data[current_idx:current_idx+batch_size, :5] = numeric.detach().cpu().numpy()
        data[current_idx:current_idx+batch_size, -1] = batch['target'].numpy()

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
    with open('{}/model_params.json'.format(checkpoint_folder), 'w') as f:
        json.dump(XGBOOST_CONFIG, f, indent=4)


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
