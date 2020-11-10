from torch.nn import Module
from torch import nn
import torch


class NumericModel(Module):
    def __init__(self, in_dim):
        super(NumericModel, self).__init__()
        self.lin1 = nn.Linear(in_features=in_dim, out_features=1)

    def forward(self, x):
        return self.lin1(x)


class RNN(Module):
    def __init__(self, emb_dim=300, hidden_size=10, n_layers=1):
        super(RNN, self).__init__()
        self.GRU = nn.GRU(input_size=emb_dim, hidden_size=hidden_size,
                          num_layers=n_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_features=2*hidden_size, out_features=1)  # 2 for beginning and end of bidirectional
        self.hidden_size = hidden_size

    def forward(self, x):
        # expects x of shape (batch, sequence, features)
        print("x", x.shape)
        batch, seq, _ = x.shape
        sequence_out, _ = self.GRU(x)  # shape (batch, seq, 2*hidden_size)
        print("seq b4", sequence_out.shape)
        print('to', (batch, seq, 2, self.hidden_size))
        # to shape (batch, seq, 2, hidden_size)
        sequence_out = sequence_out.view(batch, seq, 2, self.hidden_size)
        print(sequence_out.shape)

        # putting together the outputs at the end of the forward and backward passes of the bi-GRU
        forward_outs = sequence_out[:, -1, 0, :]
        backward_outs = sequence_out[:, -1, 1, :]
        # to shape (batch, 2*hidden_size)
        lin_in = torch.cat((forward_outs, backward_outs), dim=1)

        return self.linear(lin_in)


if __name__ == '__main__':
    from dataset import TweetDataset, collate_function
    from torch.utils.data import DataLoader
    data = TweetDataset("data/train.csv")
    loader = DataLoader(data, batch_size=3, num_workers=4,
                        collate_fn=collate_function, pin_memory=True)
    model_rnn = RNN().cuda()

    for batch in loader:
        out_rnn = model_rnn(batch['embeddings'].cuda())
        print(out_rnn)
        break
