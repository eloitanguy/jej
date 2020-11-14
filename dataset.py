import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import csv
import torchtext
from config import DATASET_CONFIG

# ---------------------
# ----- Constants -----
# ---------------------


COLUMN_IDX_TO_NAME = ['id', 'timestamp', 'retweet_count', 'user_verified', 'user_statuses_count',
                      'user_followers_count', 'user_friends_count', 'user_mentions', 'urls', 'hashtags', 'text']

COLUMN_NAME_TO_IDX = {COLUMN_IDX_TO_NAME[idx]: idx for idx, name in enumerate(COLUMN_IDX_TO_NAME)}

MEANS = {'user_statuses_count': 41662.484375,
         'user_followers_count': 232645.125,
         'user_friends_count': 2737.07421875}

STDS = {'user_statuses_count': 98392.1584,
        'user_followers_count': 2438640.574379095,
        'user_friends_count': 17252.172964586}

GLOVE = torchtext.vocab.GloVe(dim=300)  # embedding dimension is emb_dim = 300 here

cfg = DATASET_CONFIG
DATASET_SIZE = 665777
DATASET_SPLIT = int(DATASET_SIZE * cfg['train_percent'])


# ---------------------------
# ----- Dataset objects -----
# ---------------------------


class TweetDataset(Dataset):
    """
    Defines a torch "Dataset" object for loading the training and validation data. \n
    dataset_type indicates what to build: \n
    * 'train' ->the training set (examples in [0, DATASET_SPLIT[) \n
    * 'val' or 'validation' -> the validation set (examples in [DATASET_SPLIT, ...[
    """

    def __init__(self, dataset_type='train'):
        assert dataset_type in ['train', 'val', 'validation'], "Please provide a valid "
        with open(cfg['csv_relative_path'], newline='') as csvfile:
            if dataset_type == 'train':
                self.data = list(csv.reader(csvfile))[1:DATASET_SPLIT]
            else:
                self.data = list(csv.reader(csvfile))[DATASET_SPLIT:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]  # the first line is the column names

        # normalising all values between -1 and 1
        numeric_data = torch.Tensor([int(line[COLUMN_NAME_TO_IDX['timestamp']]) % (3600 * 24) / (3600 * 24 / 2) - 1,
                                     int(line[COLUMN_NAME_TO_IDX['user_verified']] == 'True') * 2 - 1,
                                     (int(line[COLUMN_NAME_TO_IDX['user_statuses_count']]) -
                                      MEANS['user_statuses_count'])/STDS['user_statuses_count'],
                                     (int(line[COLUMN_NAME_TO_IDX['user_followers_count']]) -
                                      MEANS['user_followers_count'])/STDS['user_followers_count'],
                                     (int(line[COLUMN_NAME_TO_IDX['user_friends_count']]) -
                                      MEANS['user_friends_count'])/STDS['user_friends_count']
                                     ])

        text_str = line[COLUMN_NAME_TO_IDX['text']]
        text_word_list = text_str.split(' ')
        text_embedding = GLOVE.get_vecs_by_tokens(text_word_list, lower_case_backup=True)  # shape (text_size, emb_dim)

        return {'numeric': numeric_data, 'text_emb': text_embedding,
                'target': int(line[COLUMN_NAME_TO_IDX['retweet_count']])}


def collate_function(data):
    batch_size = len(data)
    numeric = torch.stack([data[idx]['numeric'] for idx in range(batch_size)])
    target = torch.stack([torch.tensor(data[idx]['target']) for idx in range(batch_size)])

    # We need to pad them to the same size in order to give a batch input to the RNN
    embeddings = pad_sequence([data[idx]['text_emb'] for idx in range(batch_size)],
                              batch_first=True)

    return {'numeric': numeric, 'target': target, 'embedding': embeddings}


if __name__ == '__main__':
    train_set = TweetDataset(dataset_type='val')
    print(len(train_set))
