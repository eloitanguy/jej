from models import RNN
import torch
import csv
from dataset import TweetDataset, collate_function
from modules import printProgressBar
from torch.utils.data import DataLoader
from config import TRAIN_CONFIG, DATASET_CONFIG, XGBOOST_CONFIG, EXPORT_CONFIG
import xgboost as xgb
import json
import numpy as np
import argparse


def export_RNN_regressor(checkpoint_path):
    """
    :param checkpoint_path: relative path to a PyTorch .pth checkpoint
    :return: None, dumps a prediction text file in the model's training folder
    """
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

    with open("checkpoints/{}/predictions.txt".format(checkpoint['train_config']['experiment_name']), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["TweetID", "NoRetweets"])
        current_idx = 0
        for batch_index, batch in enumerate(test_loader):
            printProgressBar(batch_index, n)
            batch_size = batch['numeric'].shape[0]

            numeric = batch['numeric'].cuda()
            text = batch['embedding'].cuda()
            prediction = torch.exp(model(text, numeric)) - 1 if EXPORT_CONFIG['log'] else model(text, numeric)

            if EXPORT_CONFIG['threshold']:
                prediction[prediction > EXPORT_CONFIG['threshold']] = EXPORT_CONFIG['threshold']

            for idx_in_batch in range(batch_size):
                writer.writerow([str(ids[current_idx + idx_in_batch]), str(int(prediction[idx_in_batch].item()))])

            current_idx += batch_size

    print("Exportation done! :)")


def export_xgb_regressor(experiment_name):
    """
    :param experiment_name: name of a finished XGBoost training (XGBOOST_CONFIG['experiment_name'])
    :return: None, dumps a prediction text file in the model's training folder
    """
    with open('checkpoints/{}/model_params.json'.format(experiment_name), 'r') as f:
        config = json.load(f)

    xg_reg = xgb.XGBRegressor(objective='reg:squarederror',
                              colsample_bytree=config['colsample_bytree'],
                              learning_rate=config['learning_rate'],
                              max_depth=config['max_depth'],
                              alpha=config['alpha'],
                              reg_lambda=config['reg_lambda'],
                              n_estimators=config['n_estimators'],
                              verbosity=0)

    xg_reg.load_model('checkpoints/{}/checkpoint.model'.format(experiment_name))

    test_data = np.load(XGBOOST_CONFIG['test_file'])
    X, ids = test_data[:, :-2], test_data[:, -1]
    prediction = np.exp(xg_reg.predict(X)) - 1 if EXPORT_CONFIG['log'] else xg_reg.predict(X)

    with open("checkpoints/{}/predictions.txt".format(experiment_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["TweetID", "NoRetweets"])
        for idx in range(prediction.shape[0]):
            writer.writerow([str(ids[idx]), str(int(prediction[idx]))])

    print("Exportation done! :)")


def export_average(prediction_list):
    """
    :param prediction_list: list of paths to valid .txt prediction files
    :return: None: dumps a predictions.txt prediction file in the main repository
    """
    data = []

    for file in prediction_list:
        with open(file, newline='') as csvfile:
            data.append(list(csv.reader(csvfile))[1:])

    with open("predictions.txt", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["TweetID", "NoRetweets"])
        for data_entry_idx in range(len(data[0])):
            avg = 0
            for prediction_idx in range(len(prediction_list)):
                prediction = data[prediction_idx][data_entry_idx][1]
                avg += int(prediction) / len(prediction_list)
            entry_id = data[0][data_entry_idx][0]
            writer.writerow([entry_id, avg])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retweet estimator exportation for kaggle submission')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to a torch checkpoint')
    parser.add_argument('--xgb-name', type=str, default='',
                        help='Path to a the name of an XGBoost experiment')
    parser.add_argument("--average", nargs="+", default=[])
    args = parser.parse_args()

    if args.checkpoint != '':
        export_RNN_regressor(args.checkpoint)
    elif args.xgb_name != '':
        export_xgb_regressor(args.xgb_name)
    elif args.average:
        export_average(args.average)
    else:
        print('The provided inputs {} and {} are invalid'.format(args.checkpoint, args.xgb_name))
