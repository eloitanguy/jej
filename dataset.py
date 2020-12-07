import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import csv
import torchtext
from config import DATASET_CONFIG
from modules import printProgressBar
from config import RNN_CONFIG
import json
from nltk.stem import PorterStemmer
from _collections import OrderedDict

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

if not RNN_CONFIG['use_AE']:
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
    * 'train' ->the training set (examples in [0, DATASET_SPLIT[)
    * 'val' or 'validation' -> the validation set (examples in [DATASET_SPLIT, ...[
    * 'test' -> the Kaggle test set (with no RT entry)
    * 'all' -> the entire train.csv file
    This constructor relies on the DATASET_CONFIG dictionary in config.py. \n
    This also relies on RNN_CONFIG['use_AE']: \n
    * if True, we want to train an AutoEncoder and therefore the input should be indices
    * If False, we want to use pretrained word embeddings: in this cas we use GloVe vectors.
    """

    def __init__(self, dataset_type='train'):
        assert dataset_type in ['train', 'val', 'validation', 'test', 'all'], "Please provide a valid dataset type"

        if dataset_type != 'test':
            with open(cfg['csv_relative_path'], newline='') as csvfile:
                if dataset_type == 'train':
                    self.data = list(csv.reader(csvfile))[1:DATASET_SPLIT]
                elif dataset_type == 'all':
                    self.data = list(csv.reader(csvfile))[1:]
                else:
                    self.data = list(csv.reader(csvfile))[DATASET_SPLIT:]
            self.test = False
        else:
            with open(cfg['test_csv_relative_path'], newline='') as csvfile:
                self.data = list(csv.reader(csvfile))[1:]
                self.test = True

        if cfg['remove_zero'] and not self.test:
            print('Removing tweets with 0 RT')
            self.data = [line for line in self.data if int(line[COLUMN_NAME_TO_IDX['retweet_count']]) != 0]

        if RNN_CONFIG['use_AE']:  # in this case we need to use the vocabulary in order to convert the words to indices
            with open(cfg['vocab_relative_path'], 'r') as f:
                self.vocab = json.load(f)
                self.ps = PorterStemmer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]

        # normalising all values between -1 and 1
        offset = 1 if self.test else 0  # in the test file there is no retweet column -> offsetting the indices
        numeric_data = torch.Tensor([int(line[COLUMN_NAME_TO_IDX['timestamp']])  # minute in the day
                                     / 1000 % (3600 * 24) / (3600 * 24 / 2) - 1,
                                     int(line[COLUMN_NAME_TO_IDX['user_verified'] - offset] == 'True') * 2 - 1,
                                     (int(line[COLUMN_NAME_TO_IDX['user_statuses_count'] - offset]) -
                                      MEANS['user_statuses_count']) / STDS['user_statuses_count'],
                                     (int(line[COLUMN_NAME_TO_IDX['user_followers_count'] - offset]) -
                                      MEANS['user_followers_count']) / STDS['user_followers_count'],
                                     (int(line[COLUMN_NAME_TO_IDX['user_friends_count'] - offset]) -
                                      MEANS['user_friends_count']) / STDS['user_friends_count'],
                                     (len(line[COLUMN_NAME_TO_IDX['user_mentions'] - offset].split(' '))),
                                     (len(line[COLUMN_NAME_TO_IDX['urls'] - offset].split(' '))),
                                     (len(line[COLUMN_NAME_TO_IDX['hashtags'] - offset].split(' '))),
                                     ])

        text_str = line[COLUMN_NAME_TO_IDX['text'] - offset]
        text_word_list = text_str.lower().split(' ')

        if not RNN_CONFIG['use_AE']:  # the text data is a GloVe embedding of shape (text_size, emb_dim)
            text_embedding = GLOVE.get_vecs_by_tokens(text_word_list, lower_case_backup=True)
        else:  # the text data is an array of word indices fed to the auto-encoder, shape (text_size)
            text_embedding = torch.tensor([get_word_index(self.ps.stem(w), self.vocab) for w in text_word_list])

        return {'numeric': numeric_data, 'text_emb': text_embedding,
                'target': int(line[COLUMN_NAME_TO_IDX['retweet_count']]) if not self.test else 0
                }


def collate_function(data):
    """
    Defines the batch aggregation routine for the DataLoader \n
    :param data: a list of elements from the Dataset object
    :return: a prepared batch made from these entries
    """
    batch_size = len(data)
    numeric = torch.stack([data[idx]['numeric'] for idx in range(batch_size)])
    target = torch.stack([torch.tensor(data[idx]['target']) for idx in range(batch_size)])

    # We need to pad them to the same size in order to give a batch input to the RNN
    # notice that padding with zeros will work both for GloVe vector embeddings and indices
    embeddings = pad_sequence([data[idx]['text_emb'] for idx in range(batch_size)],
                              batch_first=True)

    return {'numeric': numeric, 'target': target, 'embedding': embeddings}


def create_vocabulary():
    """
    Using RNN_CONFIG['vocab_using_n_tweets'] tweets from the train.csv dataset, \n
    creates a vocabulary with RNN_CONFIG['AE_vocab_size']] words.\n
    The vocabulary is an ordered dictionary: the keys are the word radicals and the keys each word's index.\n
    :return: None, dumps the vocabulary as a .json file at data/vocab.json
    """
    with open(cfg['csv_relative_path'], newline='') as csvfile:
        data = list(csv.reader(csvfile))[1:]

    vocab = {}
    ps = PorterStemmer()

    for idx, line in enumerate(data[:RNN_CONFIG['vocab_using_n_tweets']]):
        printProgressBar(idx, RNN_CONFIG['vocab_using_n_tweets'], 'creating dictionary')
        for word in line[COLUMN_NAME_TO_IDX['text']].lower().split(' '):
            w = ps.stem(word)
            if w in vocab:
                vocab[w] += 1
            else:
                vocab[w] = 1

    # sort the vocabulary by descending occurrences
    vocab = OrderedDict(
        [(k, idx) for idx, (k, _) in
         enumerate(sorted(vocab.items(), key=lambda item: item[1], reverse=True)[:RNN_CONFIG['AE_vocab_size']])
         ]
    )

    with open('data/vocab.json', 'w') as f:
        json.dump(vocab, f, indent=4)


def get_word_index(word, vocabulary):
    """
    :param word: a string
    :param vocabulary: a dictionary (key: word, item: index)
    :return: the index of the word in the vocabulary if
    """

    if word in vocabulary:
        return vocabulary[word] + 1  # the index 0 is for unknown words
    else:
        return 0


if __name__ == '__main__':
    print('Creating vocabulary and exporting it as data/vocab.json ...')
    create_vocabulary()
