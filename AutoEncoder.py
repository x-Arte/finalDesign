import torch.nn as nn
import torch


class AutoEncoder(nn.Module):
    def __init__(self, input_size, target_size, hidden_size, num_layer, seq_len, dropout=0.25):
        super().__init__()
        self.rnn1 = nn.LSTM(input_size, hidden_size, num_layer, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_size*seq_len, target_size)
        self.fc2 = nn.Linear(target_size, hidden_size*seq_len)
        self.rnn2 = nn.LSTM(hidden_size, input_size, num_layer,  dropout=dropout, batch_first=True)

    def forward(self, x):
        unbatched = len(x.shape) == 2
        batch_size = x.size(0)
        seq_len = x.size(0) if unbatched else x.size(1)
        x, _ = self.rnn1(x)
        x = self.fc1(x.view(-1) if unbatched else x.reshape(batch_size, -1))
        y = self.fc2(x)
        y, _ = self.rnn2(y.view(seq_len, -1) if unbatched else y.reshape(batch_size, seq_len, -1))
        return y

    def encode(self, x):
        unbatched = len(x.shape) == 2
        batch_size = x.size(0)
        seq_len = x.size(0) if unbatched else x.size(1)
        x, _ = self.rnn1(x)
        x = self.fc1(x.view(-1) if unbatched else x.reshape(batch_size, -1))
        return x
