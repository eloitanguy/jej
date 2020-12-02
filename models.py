from torch.nn import Module, Embedding
from torch import nn
import torch
from config import RNN_CONFIG
from torch import relu


class NumericModel(Module):
    """
    A simple linear regression taking the numeric data and outputting an estimated retweet count \n
    in_dim represents the size of the input vector: [batch_size, in_dim]
    """
    def __init__(self, in_dim):
        super(NumericModel, self).__init__()
        self.lin1 = nn.Linear(in_features=in_dim, out_features=1)

    def forward(self, x):
        return self.lin1(x)


class RNN(Module):
    def __init__(self, config=None):
        super(RNN, self).__init__()
        if not config:
            config = RNN_CONFIG
        self.config = config

        if config['use_AE']:  # creating an embedding layer since we want an auto-encoder (+1 for the unknown word)
            self.emb = Embedding(num_embeddings=config['AE_vocab_size'] + 1, embedding_dim=config['emb_dim'])

        self.GRU = nn.GRU(input_size=config['emb_dim'], hidden_size=config['hidden_size'],
                          num_layers=config['layers'], batch_first=True, bidirectional=True,
                          dropout=config['dropout'])
        gru_out_size = 2 * config['hidden_size'] * config['layers']

        # takes the activated gru output and the numeric data as input
        self.prelu1 = nn.PReLU(num_parameters=gru_out_size+config['numeric_data_size'])
        self.linear1 = nn.Linear(in_features=gru_out_size+config['numeric_data_size'],
                                 out_features=config['linear_hidden_1'])

        self.prelu2 = nn.PReLU(num_parameters=config['linear_hidden_1'])
        self.linear2 = nn.Linear(in_features=config['linear_hidden_1'],
                                 out_features=config['linear_hidden_2'])

        self.prelu3 = nn.PReLU(num_parameters=config['linear_hidden_2'])
        self.linear3 = nn.Linear(in_features=config['linear_hidden_2'],
                                 out_features=1)

    def forward(self, x, numeric_data):

        if self.config['use_AE']:  # if the model has an AE, expects word indices of shape (batch, sequence)
            text_embedding = self.emb(x)
        else:  # otherwise they are expected to be GloVe embeddings, expects x of shape (batch, sequence, features)
            text_embedding = x

        batch, seq, _ = text_embedding.shape

        _, rnn_out = self.GRU(text_embedding)  # shape (layers*2, batch, hidden_size)
        # to shape (batch, layers*2*hidden_size)
        rnn_out = rnn_out.transpose(0, 1).reshape(batch, self.config['layers']*2*self.config['hidden_size'])

        # concatenating the numeric data
        lin_in = torch.cat((relu(rnn_out), numeric_data), dim=1)
        out = self.linear1(self.prelu1(lin_in))
        out = self.linear2(self.prelu2(out))
        hidden = self.prelu3(out)
        out = self.linear3(hidden)

        if self.config['classifier']:
            return out, hidden
        else:
            return relu(out)


class Classifier(Module):
    """
    A logistic regression that tries to classify whether a tweet is viral or not
    """
    def __init__(self, in_dim=5):
        super(Classifier, self).__init__()
        self.lin1 = nn.Linear(in_features=in_dim, out_features=1)

    def forward(self, x):
        return torch.sigmoid(self.lin1(x))
