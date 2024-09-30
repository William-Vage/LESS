from pipeline.edge.utils import EarlyStopping
from config import device, deep_model_info
from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import config
import os


# 模型及训练参数
n_vocab = 13  #0-9:正常数字，10:每个字段分隔符，11:每条记录分隔符，12:全部记录结束符
batch_size = 4096
patience = 100
seq_len = 10
epochs = 3
lr = 0.001


class Trainer:
    def __init__(self, model: nn.Module, save_dir: str, model_name: str, data_path: str):
        """
        模型训练类
        :param save_dir: 模型存储路径
        :param model_name: 模型文件名
        :param data_path: 数据路径
        """
        # 1.词典大小
        self.n_vocab = n_vocab
        # 2.训练数据
        self.data: np.array = np.load(data_path)
        # 3.构建模型
        self.model = model(self.n_vocab).to(device)
        # 4.定义优化器
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        # 5.定义损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        # 6.早停机制
        self.early_stopping = EarlyStopping(patience=patience)
        # 7.存储路径
        self.save_dir = save_dir
        self.model_name = model_name

    def train(self):
        """
        训练模型
        :return: None
        """
        for i in range(epochs):
            # print(f'epoch: {i}/{epochs}')
            n_batch = len(self.data) // batch_size
            bar = tqdm(range(0, len(self.data), batch_size), desc=f'loss: inf, accuracy: 0', total=n_batch, leave=True)
            # bar = range(0, len(self.data), batch_size)
            loss_sum = 0
            acc_sum = 0
            cnt = 0
            for j in bar:
                # 1.按滑动窗口顺序提取数据
                data = np.lib.stride_tricks.sliding_window_view(self.data[j: j + batch_size],
                                                                window_shape=seq_len + 1, writeable=False)
                # 2. 构造输入及目标序列
                x = torch.tensor(data[:, :-1], device=device)
                y = torch.tensor(data[:, -1], dtype=torch.long, device=device)
                # 3. 训练模型                
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                acc = torch.eq(y, torch.argmax(y_pred, dim=-1)).float().mean()
                acc_sum += acc.cpu()
                loss_sum += loss.cpu()
                cnt += 1
                bar.set_description(f'epoch: {i}, loss: {loss_sum / cnt}, accuracy: {acc_sum / cnt}')
                self.optimizer.step()
            # 4.停止训练
            mean_loss = loss_sum / cnt
            self.early_stopping(mean_loss, self.model, self.save_dir, self.model_name)
            if self.early_stopping.early_stop:
                break

    def save(self):
        """
        保存模型
        :return: None
        """
        torch.save(self.model, os.path.join(self.save_dir, self.model_name))


def train_deep_model(method_name='lstm'):
    if method_name in deep_model_info:
        model, model_store_name = deep_model_info[method_name]
    else:
        print("Function not found for deep learning method:", method_name)
        return

    # 训练数据路径
    data_path = os.path.join(config.project_root, 'data/encode/edge_encode.npy')
    model_dir = os.path.join(config.project_root, 'data/model')
    trainer = Trainer(model=model, save_dir=model_dir, model_name=model_store_name, data_path=data_path)
    trainer.train()


if __name__ == '__main__':
    train_deep_model()
