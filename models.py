from torch.nn import Module
from torch import nn
import torch
from config import RNN_CONFIG


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
    """
    TODO
    A simple recurrent neural network (GRU + linear for output) taking the tweet text and giving a retweet count \n
    * emb_dim: embedding dimension, where the input tensor of of shape [batch, sequence, emb_dim] \n
    * hidden_size: number of output features of the GRU (it's bidirectional so this will be doubled)
    """
    def __init__(self, config=None):
        super(RNN, self).__init__()
        if not config:
            config = RNN_CONFIG
        self.config = config

        self.GRU = nn.GRU(input_size=config['emb_dim'], hidden_size=config['hidden_size'],
                          num_layers=config['layers'], batch_first=True, bidirectional=True,
                          dropout=config['dropout'])
        gru_out_size = 2*config['hidden_size']*config['layers']
        self.prelu1 = nn.PReLU(num_parameters=gru_out_size)
        # takes the activated gru output and the numeric data as input
        self.linear1 = nn.Linear(in_features=gru_out_size+config['numeric_data_size'],
                                 out_features=config['linear_hidden_size'])
        self.prelu2 = nn.PReLU(num_parameters=config['linear_hidden_size'])
        self.linear2 = nn.Linear(in_features=config['linear_hidden_size'], out_features=1)

    def forward(self, text_embedding, numeric_data):
        # expects x of shape (batch, sequence, features)
        batch, seq, _ = text_embedding.shape
        _, rnn_out = self.GRU(text_embedding)  # shape (layers*2, batch, hidden_size)
        # to shape (batch, layers*2*hidden_size)
        rnn_out = rnn_out.transpose(0, 1).reshape(batch, self.config['layers']*2*self.config['hidden_size'])

        # concatenating the numeric data
        lin_in = torch.cat((self.prelu1(rnn_out), numeric_data), dim=1)
        out = self.linear1(lin_in)
        out = self.linear2(self.prelu2(out))

        return torch.exp(out) if not self.config['classifier'] else torch.sigmoid(out)


class Classifier(Module):
    """
    A logistic regression that tries to classify whether a tweet is viral or not
    """
    def __init__(self, in_dim=5):
        super(Classifier, self).__init__()
        self.lin1 = nn.Linear(in_features=in_dim, out_features=1)

    def forward(self, x):
        return torch.sigmoid(self.lin1(x))
