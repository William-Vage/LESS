from leonard.train.utils import EarlyStopping
from leonard.train.leonard_model import *
from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import time
import torch
import json
import os
import leonard.config as config


# 模型及训练参数
model = LeonardLstmMulti
n_vocab = None
batch_size = 2048
patience = 1000
seq_len = 10
epochs = 10
lr = 0.001
if config.use_gpu:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
# 训练数据路径
params_path = os.path.join(config.project_root, 'data/encode/params.json')
data_path = os.path.join(config.project_root, 'data/encode/tensor.npy')
model_dir = os.path.join(config.project_root, 'data/model')
name = 'leonard_lstm.pt'


class LeonardTrainer:
    def __init__(self, save_dir: str, model_name: str):
        """
        Leonard模型训练类
        :param save_dir: 模型存储路径
        :param model_name: 模型文件名
        """
        # 1.读取编码参数
        with open(params_path, 'r') as f:
            p = json.load(f)
            self.n_vocab = len(p['id2char_dict']) + 2
        # 2.训练数据
        self.data: np.array = np.load(data_path)
        # 3.构造avoid数组
        tmp = np.where(self.data == 1)[0]
        self.avoid = set()
        # for i in tmp:
        #     self.avoid.update(list(range(i, i + seq_len)))
        # 4.构建模型
        self.model = model(self.n_vocab).to(device)
        # 5.定义优化器
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        # 6.定义损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        # 7.早停机制
        self.early_stopping = EarlyStopping(patience=patience)
        # 8.存储路径
        self.save_dir = save_dir
        self.model_name = model_name

    def train(self):
        """
        训练Leonard模型
        :return: None
        """
        pos = np.array([i for i in range(seq_len, len(self.data)) if i not in self.avoid])
        for i in range(epochs):
            print(f'epoch: {i}/{epochs}')
            # 1. 随机选取数据
            np.random.shuffle(pos)
            # 2. 构造输入及目标序列
            n_batch = len(pos) // batch_size
            batch = pos[:n_batch * batch_size].reshape(-1, batch_size)
            bar = tqdm(range(n_batch), desc=f'loss: inf, accuracy: 0', total=n_batch, leave=True)
            for j in bar:
                data = []
                for k in range(batch_size):
                    data.append(self.data[batch[j][k] - seq_len: batch[j][k] + 1])
                data = np.array(data)
                x = torch.tensor(data[:, :-1], device=device)
                y = torch.tensor(data[:, -1], dtype=torch.long, device=device)
                # 3. 训练模型
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                acc = torch.eq(y, torch.argmax(y_pred, dim=-1)).float().mean()
                bar.set_description(f'loss: {loss.cpu()}, accuracy: {acc.cpu()}')
                self.optimizer.step()
                # 4.停止训练
                self.early_stopping(float(loss), self.model, self.save_dir, self.model_name)
                if self.early_stopping.early_stop:
                    break

    def save(self):
        """
        保存模型
        :return: None
        """
        torch.save(self.model, os.path.join(self.save_dir, self.model_name))


# if __name__ == '__main__':
def leonard_train_func():
    t_start = time.time()
    trainer = LeonardTrainer(save_dir=model_dir, model_name=name)
    trainer.train()
    t_end = time.time()
    print(f'Train model cost: {t_end - t_start}')
