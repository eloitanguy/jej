import numpy as np
from dataset import TweetDataset, collate_function, DATASET_SPLIT
from modules import printProgressBar
from torch.utils.data import DataLoader
import torch
from models import RNN
from config import XGBOOST_CONFIG, TRAIN_CONFIG, DATASET_CONFIG
import xgboost as xgb
import argparse
import os
import json
import time
from datetime import timedelta
import csv


def prepare_datasets():
    checkpoint = torch.load(XGBOOST_CONFIG['embedder'])
    embed = RNN(config=checkpoint['net_config']).eval()
    embed.load_state_dict(checkpoint['model'])
    embed = embed.cuda()

    annotated_dataset = TweetDataset(dataset_type='all')
    test_dataset = TweetDataset(dataset_type='test')

    def get_data(dataset):
        N = len(dataset)
        data = np.zeros((N, XGBOOST_CONFIG['numeric_data_size'] + XGBOOST_CONFIG['embedding_size'] + 1))  # 1 for answer
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

            if XGBOOST_CONFIG['embedding_use_hidden']:
                embedding = embed(text, numeric[:, :checkpoint['net_config']['numeric_data_size']])[1]
            elif XGBOOST_CONFIG['embedding_use_output']:
                embedding = torch.exp(embed(text, numeric[:, :checkpoint['net_config']['numeric_data_size']])[0]) - 1
            else:  # expecting a built-in embedding layer -> taking the mean of the embeddings
                embedding = embed.emb(text).mean(axis=1)

            data[current_idx:current_idx+batch_size, XGBOOST_CONFIG['numeric_data_size']:-1] = \
                embedding.detach().cpu().numpy()
            data[current_idx:current_idx+batch_size, :XGBOOST_CONFIG['numeric_data_size']] = \
                numeric.detach().cpu().numpy()
            data[current_idx:current_idx+batch_size, -1] = batch['target'].numpy()

            current_idx += batch_size

        return data

    annotated_data = get_data(annotated_dataset)
    split = int(len(annotated_dataset) * DATASET_CONFIG['train_percent'])
    np.save(XGBOOST_CONFIG['train_file'], annotated_data[1:split])
    np.save(XGBOOST_CONFIG['val_file'], annotated_data[split:])

    test_data = get_data(test_dataset)
    with open(DATASET_CONFIG['test_csv_relative_path'], newline='') as csvfile:
        ids = [line[0] for line in list(csv.reader(csvfile))[1:]]

    ids = np.array(ids).reshape(np.shape(ids)[0], 1)
    prepared_test_data = np.concatenate((test_data, ids), axis=1)
    np.save(XGBOOST_CONFIG['test_file'], prepared_test_data)


def train():
    print('Training {} ...'.format(XGBOOST_CONFIG['experiment_name']))
    train_set = np.load(XGBOOST_CONFIG['train_file'])
    X, Y = train_set[:, :-1], np.log(1+train_set[:, -1]) if XGBOOST_CONFIG['log'] else train_set[:, -1]
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror',
                              colsample_bytree=XGBOOST_CONFIG['colsample_bytree'],
                              learning_rate=XGBOOST_CONFIG['learning_rate'],
                              max_depth=XGBOOST_CONFIG['max_depth'],
                              alpha=XGBOOST_CONFIG['alpha'],
                              reg_lambda=XGBOOST_CONFIG['reg_lambda'],
                              n_estimators=XGBOOST_CONFIG['n_estimators'],
                              subsample=XGBOOST_CONFIG['subsample'],
                              verbosity=0)

    t0 = time.time()
    xg_reg.fit(X, Y)
    print('Training time: {}'.format(str(timedelta(seconds=time.time()-t0))))

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
