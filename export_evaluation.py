from models import RNN
import torch
import csv
from dataset import TweetDataset, collate_function
from modules import printProgressBar
from torch.utils.data import DataLoader
from config import TRAIN_CONFIG, DATASET_CONFIG


if __name__ == '__main__':
    checkpoint = torch.load('checkpoints/rnn_MSE_log_bigger4/epoch_0.pth')
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
                writer.writerow([str(ids[current_idx+idx_in_batch]), str(int(prediction[idx_in_batch].item()))])

            current_idx += batch_size

    print("Exportation done! :)")
