from torch.nn import Module
from torch import nn
import torch


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
    def __init__(self, emb_dim=300, hidden_size=10, n_layers=1, numeric_data_size=5):
        super(RNN, self).__init__()
        self.GRU = nn.GRU(input_size=emb_dim, hidden_size=hidden_size,
                          num_layers=n_layers, batch_first=True, bidirectional=True)
        self.prelu1 = nn.PReLU(num_parameters=2*hidden_size)
        # 2 for beginning and end of bidirectional
        self.linear1 = nn.Linear(in_features=2*hidden_size+numeric_data_size, out_features=5)
        self.prelu2 = nn.PReLU(num_parameters=5)
        self.linear2 = nn.Linear(in_features=5, out_features=1)
        self.hidden_size = hidden_size

    def forward(self, text_embedding, numeric_data):
        # expects x of shape (batch, sequence, features)
        batch, seq, _ = text_embedding.shape
        sequence_out, _ = self.GRU(text_embedding)  # shape (batch, seq, 2*hidden_size)

        # averaging the passes of the bi-GRU
        rnn_out_avg = torch.mean(sequence_out, dim=1)  # size (batch, 2*hidden_size)
        # concatenating the numeric data
        lin_in = torch.cat((self.prelu1(rnn_out_avg), numeric_data), dim=1)
        out = self.linear1(lin_in)
        out = self.linear2(self.prelu2(out))

        return out


if __name__ == '__main__':
    from dataset import TweetDataset, collate_function
    from torch.utils.data import DataLoader
    from modules import MAE

    data = TweetDataset("val")
    loader = DataLoader(data, batch_size=3, num_workers=4,
                        collate_fn=collate_function, pin_memory=True)
    model_rnn = RNN().cuda()

    for batch in loader:
        out_rnn = model_rnn(batch['embedding'].cuda())
        print(out_rnn)
        target = batch['target'].cuda()
        print('\n\n', target)
        print(MAE(target, out_rnn))
        break


class Classifier(Module):
    """
    A logistic regression that tries to classify whether a tweet is viral or not
    """
    def __init__(self, in_dim=5):
        super(Classifier, self).__init__()
        self.lin1 = nn.Linear(in_features=in_dim, out_features=1)

    def forward(self, x):
        return torch.sigmoid(self.lin1(x))
