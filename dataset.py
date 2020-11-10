import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import csv
import torchtext

column_idx_to_name = ['id', 'timestamp', 'retweet_count', 'user_verified', 'user_statuses_count',
                      'user_followers_count', 'user_friends_count', 'user_mentions', 'urls', 'hashtags', 'text']

column_name_to_idx = {column_idx_to_name[idx]: idx for idx, name in enumerate(column_idx_to_name)}

glove = torchtext.vocab.GloVe(dim=300)  # embedding dimension is emb_dim = 300 here


class TweetDataset(Dataset):
    def __init__(self, train_csv):
        with open(train_csv, newline='') as csvfile:
            self.data = list(csv.reader(csvfile))

    def __len__(self):
        return len(self.data)-1

    def __getitem__(self, idx):
        line = self.data[idx+1]  # the first line is the column names

        numeric_data = torch.Tensor([int(line[column_name_to_idx['timestamp']]),
                                     int(line[column_name_to_idx['retweet_count']]),
                                     int(line[column_name_to_idx['user_verified']] == 'True'),
                                     int(line[column_name_to_idx['user_statuses_count']]),
                                     int(line[column_name_to_idx['user_followers_count']]),
                                     int(line[column_name_to_idx['user_friends_count']])
                                     ])
        text_str = line[column_name_to_idx['text']]
        text_word_list = text_str.split(' ')
        text_embedding = glove.get_vecs_by_tokens(text_word_list, lower_case_backup=True)  # shape (text_size, emb_dim)

        return {'numeric': numeric_data, 'text_emb': text_embedding,
                'target': int(line[column_name_to_idx['retweet_count']])}


def collate_function(data):
    batch_size = len(data)
    numeric = torch.stack([data[idx]['numeric'] for idx in range(batch_size)])
    target = torch.stack([torch.tensor(data[idx]['target']) for idx in range(batch_size)])

    # We need to pad them to the same size in order to give a batch input to the RNN
    embeddings = pad_sequence([data[idx]['text_emb'] for idx in range(batch_size)],
                              batch_first=True)

    return {'numeric': numeric, 'target': target, 'embeddings': embeddings}


if __name__ == '__main__':
    data = TweetDataset("data/train.csv")
    print(data[3])
