from models import RNN
import torch
import csv
from dataset import TweetDataset, collate_function, COLUMN_NAME_TO_IDX
from modules import printProgressBar
from torch.utils.data import DataLoader
from config import TRAIN_CONFIG, DATASET_CONFIG
import xgboost as xgb
import json
import numpy as np
import argparse


def export_RNN_regressor(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = RNN(checkpoint['net_config'])
    model.load_state_dict(checkpoint['model'])
    model = model.eval().cuda()

    test_dataset = TweetDataset(dataset_type='test')
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], num_workers=TRAIN_CONFIG['workers'],
                             collate_fn=collate_function, shuffle=False, pin_memory=True)

    with open(DATASET_CONFIG['test_csv_relative_path'], newline='') as csvfile:
        test_data = list(csv.reader(csvfile))[1:]

    ids = [datum[0] for datum in test_data]
    n = len(test_loader)

    with open("predictions.txt", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["TweetID", "NoRetweets"])
        current_idx = 0
        for batch_index, batch in enumerate(test_loader):
            printProgressBar(batch_index, n)
            batch_size = batch['numeric'].shape[0]

            numeric = batch['numeric'].cuda()
            text = batch['embedding'].cuda()
            prediction = model(text, numeric)

            for idx_in_batch in range(batch_size):
                writer.writerow([str(ids[current_idx + idx_in_batch]), str(int(prediction[idx_in_batch].item()))])

            current_idx += batch_size

    print("Exportation done! :)")


def export_xgb_regressor(experiment_name):
    with open('checkpoints/{}/model_params.json'.format(experiment_name), 'r') as f:
        config = json.load(f)

    checkpoint = torch.load(config['embedder'])
    embed = RNN(config=checkpoint['net_config']).eval()
    embed.load_state_dict(checkpoint['model'])
    embed = embed.cuda()

    xg_reg = xgb.XGBRegressor(objective='reg:squarederror',
                              colsample_bytree=config['colsample_bytree'],
                              learning_rate=config['learning_rate'],
                              max_depth=config['max_depth'],
                              alpha=config['alpha'],
                              reg_lambda=config['reg_lambda'],
                              n_estimators=config['n_estimators'],
                              verbosity=0)

    xg_reg.load_model('checkpoints/{}/checkpoint.model'.format(experiment_name))
    test_dataset = TweetDataset(dataset_type='test')
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], num_workers=TRAIN_CONFIG['workers'],
                             collate_fn=collate_function, shuffle=False, pin_memory=True)

    with open(DATASET_CONFIG['test_csv_relative_path'], newline='') as csvfile:
        test_data = list(csv.reader(csvfile))[1:]

    xgb_data = np.zeros((len(test_data), 5))  # 5 numeric entries
    n = len(test_data)

    for i, entry in enumerate(test_data):
        printProgressBar(i, n, prefix='preparing xgb numeric data')
        xgb_data[i, 0] = int(entry[COLUMN_NAME_TO_IDX['user_verified'] - 1] == 'True')
        xgb_data[i, 1] = int(entry[COLUMN_NAME_TO_IDX['timestamp']]) / 1000 % (3600 * 24) / (3600 * 24 / 2) - 1
        xgb_data[i, 2] = int(entry[COLUMN_NAME_TO_IDX['user_statuses_count'] - 1])
        xgb_data[i, 3] = int(entry[COLUMN_NAME_TO_IDX['user_followers_count'] - 1])
        xgb_data[i, 4] = int(entry[COLUMN_NAME_TO_IDX['user_friends_count'] - 1])
    print('')

    ids = [datum[0] for datum in test_data]
    n = len(test_loader)

    with open("predictions.txt", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["TweetID", "NoRetweets"])
        current_idx = 0
        for batch_index, batch in enumerate(test_loader):
            printProgressBar(batch_index, n, prefix='outputting result')
            batch_size = batch['numeric'].shape[0]

            numeric = batch['numeric'].cuda()
            text = batch['embedding'].cuda()
            _, embedding = embed(text, numeric)
            text_data = embedding.detach().cpu().numpy()  # (batch_size, emb_size)
            numeric_data = xgb_data[current_idx:current_idx+batch_size, :]  # (batch_size, emb_size)
            xgb_in = np.concatenate([numeric_data, text_data], axis=1)
            prediction = np.exp(xg_reg.predict(xgb_in))-1

            for idx_in_batch in range(batch_size):
                overall_idx = current_idx + idx_in_batch

                writer.writerow([str(ids[overall_idx]), str(int(prediction[idx_in_batch]))])

            current_idx += batch_size

    print("Exportation done! :)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retweet estimator exportation for kaggle submission')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to a torch checkpoint')
    parser.add_argument('--xgb-name', type=str, default='',
                        help='Path to a the name of an XGBoost experiment')
    args = parser.parse_args()

    if args.checkpoint != '':
        export_RNN_regressor(args.checkpoint)
    elif args.xgb_name != '':
        export_xgb_regressor(args.xgb_name)
    else:
        print('The provided inputs {} and {} are invalid'.format(args.checkpoint, args.xgb_name))
