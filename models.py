from torch.nn import Module, Embedding
from torch import nn
import torch
from config import RNN_CONFIG, CNN_CONFIG
from torch import relu
import torch.nn.functional as F


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


class CNN(nn.Module):
    def __init__(self, config=None):
        super(CNN, self).__init__()
        if not config:
            config = CNN_CONFIG
        self.config = config

        self.embed = nn.Embedding(num_embeddings=10000 + 1, embedding_dim=300)

        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=2)

        self.lin1 = nn.Linear(in_features=3*128, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=32)
        self.lin3 = nn.Linear(in_features=32+8, out_features=32)  # +8 for numeric
        self.lin4 = nn.Linear(in_features=32, out_features=32)
        self.lin5 = nn.Linear(in_features=32, out_features=16)
        self.lin6 = nn.Linear(in_features=16, out_features=1)

    def forward(self, text, numeric):
        batch_size, time = text.shape
        if time > 128:
            text = text[:, :128]
        else:
            text = torch.cat((text, torch.zeros(batch_size, 128-time, device='cuda')), dim=1)

        out = self.embed(text.long())
        out = self.conv1(out)
        out = self.pool1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.pool2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.pool3(out)
        out = F.relu(out)

        batch, out_time, hidden = out.shape
        out = out.reshape(batch, out_time*hidden)
        out = F.relu(self.lin1(out))
        out = F.relu(self.lin2(out))
        out = torch.cat((out, numeric), dim=1)  # adding numeric data
        out = F.relu(self.lin3(out))
        out = F.relu(self.lin4(out))
        out = F.relu(self.lin5(out))
        out = F.relu(self.lin6(out))

        return out
