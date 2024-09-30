import torch.nn as nn
import torch.nn.functional as F


class LeonardLstm(nn.Module):
    def __init__(self, n_vocab):
        """
        Leonard使用的LSTM_multi
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, 32)
        self.lstm = nn.LSTM(32, 32, 2, batch_first=True)
        # self.fc1 = nn.Linear(32, 64)
        # self.fc2 = nn.Linear(64, n_vocab)
        self.fc1 = nn.Linear(32, n_vocab) # My Modification

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc1(x) # My Modification
        return x

class MyGRU(nn.Module):
    def __init__(self, n_vocab):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, 32)
        self.gru = nn.GRU(32, 32, 2, batch_first=True)
        self.fc1 = nn.Linear(32, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return x

class MyBiGRU(nn.Module):
    def __init__(self, n_vocab):
        super(MyBiGRU, self).__init__()
        self.embedding = nn.Embedding(n_vocab, 32)
        self.bi_gru = nn.GRU(32, 32, 2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(32 * 2, n_vocab)  # 注意，由于是双向，所以hidden_size需要乘以2

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.bi_gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return x


class AttentionGRUModel(nn.Module):
    def __init__(self, n_vocab):
        super(AttentionGRUModel, self).__init__()
        self.embedding = nn.Embedding(n_vocab, 32)
        self.gru = nn.GRU(32, 32, 2, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=32,num_heads=2,batch_first=True)
        self.fc1 = nn.Linear(32, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x, _ = self.attention(x, x, x)  # 使用相同的输入进行多头注意力计算
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc1(x)
        return x
    
class CNNClassifier(nn.Module):
    def __init__(self, n_vocab):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(n_vocab, 13)
        self.conv1d = nn.Conv1d(in_channels=13, out_channels=26, kernel_size=3)
        # self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(208, n_vocab)
        # self.fc2 = nn.Linear(32, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # 调整维度使得卷积操作在最后一个维度上进行
        x = self.conv1d(x)
        # x = torch.relu(x)
        # x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # 将张量展平为一维向量
        x = self.fc1(x)
        # x = torch.relu(x)
        # x = self.fc2(x)
        return x