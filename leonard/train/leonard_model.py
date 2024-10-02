import torch.nn as nn
import torch.nn.functional as F


class LeonardLstmMulti(nn.Module):
    def __init__(self, n_vocab):
        """
        Leonard使用的LSTM_multi
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, 32)
        self.lstm = nn.LSTM(32, 32, 2, batch_first=True)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeonardBiLstm(nn.Module):
    def __init__(self, n_vocab):
        """
        Leonard使用的biLSTM
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, 32)
        self.lstm = nn.LSTM(32, 32, 2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeonardLstmMultiBN(nn.Module):
    def __init__(self, n_vocab):
        """
        Leonard使用的LSTM_multi_bn
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, 32)
        self.lstm = nn.LSTM(32, 32, 2, batch_first=True)
        self.bn = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeonardLstmMultiBig(nn.Module):
    def __init__(self, n_vocab):
        """
        Leonard使用的LSTM_multi_big
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, 64)
        self.lstm = nn.LSTM(64, 64, 2, batch_first=True)
        self.fc1 = nn.Linear(64, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return x


class LeonardGruMulti(nn.Module):
    def __init__(self, n_vocab):
        """
        Leonard使用的LSTM_multi_bn
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, 32)
        self.gru = nn.GRU(32, 32, 2, batch_first=True)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeonardGruMultiBig(nn.Module):
    def __init__(self, n_vocab):
        """
        Leonard使用的LSTM_multi_bn
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, 32)
        self.gru = nn.GRU(32, 128, 2, batch_first=True)
        self.fc1 = nn.Linear(128, 64)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return x


class LeonardBiGru(nn.Module):
    def __init__(self, n_vocab):
        """
        Leonard使用的biGRU
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, 32)
        self.gru = nn.GRU(32, 32, 2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeonardBiGruBig(nn.Module):
    def __init__(self, n_vocab):
        """
        Leonard使用的biGRU_big
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, 32)
        self.gru = nn.GRU(32, 128, 2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(256, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return x


class LeonardFc(nn.Module):
    def __init__(self, n_vocab):
        """
        Leonard使用的FC_4layer
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, 32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
