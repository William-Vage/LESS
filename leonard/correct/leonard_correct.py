from tqdm import tqdm
import numpy as np
import torch
import time
import json
import os
import leonard.config as config


# 模型参数
batch_size = 1024
seq_len = 10
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
# 训练数据及模型路径
params_path = os.path.join(config.project_root, 'data/encode/params.json')
data_path = os.path.join(config.project_root, 'data/encode/tensor.npy')
model_path = os.path.join(config.project_root, 'data/model/leonard_lstm.pt')
calibration_table_dir = os.path.join(config.project_root, 'data/correct')


class LeonardCorrector:
    def __init__(self, model_path):
        """
        Leonard纠错表构造类
        """
        # 1.读取编码参数
        with open(params_path, 'r') as f:
            p = json.load(f)
            self.id2char_dict = p['id2char_dict']
            self.re_values = p['re_values_dict']
            self.n_vocab = len(self.id2char_dict) + 2
        # 2.读取编码数据
        data: np.array = np.load(data_path)
        # 3.将数据按1切开
        pos = np.where(data == 1)[0] + 1
        self.data = np.split(data, pos)
        # 4.加载模型
        self.model = torch.load(model_path, device)
        # self.model.lstm.flatten_parameters()
        # 5.校准表
        self.table = {}

    def model_predict_and_correct(self, x: np.ndarray, y: np.ndarray, sep_pos: list[int]):
        """
        模型预测
        :param x: 输入数据
        :param y: 预测目标
        :param sep_pos: 序列分割点
        :return: 预测结果
        """
        # 1.模型预测
        x_ = torch.tensor(x, device=device)
        y_ = torch.tensor(y, dtype=torch.long, device=device).reshape(-1)
        y_pred = self.model(x_)
        y_pred = torch.argmax(y_pred, dim=-1)
        # 2.纠错
        res = []
        diff = torch.not_equal(y_, y_pred)
        start = 0
        for i in range(len(sep_pos)):
            tmp = []
            for j in range(start, start + sep_pos[i]):
                if diff[j]:
                    tmp.append([str(j - start), str(int(y_[j]))])
            start += sep_pos[i]
            res.append(tmp)
        return res

    def get_calibration_table(self):
        """
        生成校准表
        :return: None
        """
        # 1.计算需要几个batch
        n_batch = int(np.ceil(len(self.data) / batch_size))
        # 2.记录编号
        for i in tqdm(range(n_batch)):
            x, y, timestamp, sep_pos = [], [], [], []
            for j in range(i * batch_size, min((i + 1) * batch_size, len(self.data))):
                if len(self.data[j]) == 0:
                    continue
                # 3.提取节点或边编号
                if self.data[j][0] == 3:
                    tmp = 'e:'
                else:
                    tmp = 'v:'
                zero_pos = 0
                for k in range(8, len(self.data[j])):
                    if self.data[j][k] != 0:
                        tmp += self.id2char_dict[str(self.data[j][k])]
                    else:
                        zero_pos = k
                        break
                timestamp.append(tmp)
                # 4.按滑动窗口提取数据
                slide_window = np.lib.stride_tricks.sliding_window_view(self.data[j][zero_pos + 1 - seq_len:],
                                                                        window_shape=seq_len + 1, writeable=False)
                sep_pos.append(len(slide_window))
                x.append(slide_window[:, :-1])
                y.append(slide_window[:, -1:])
            # 5.运行模型
            x = np.concatenate(x, axis=0)
            y = np.concatenate(y, axis=0)
            y_pred = self.model_predict_and_correct(x, y, sep_pos)
            # 6.更新纠错表
            self.table.update({t: v for t, v in zip(timestamp, y_pred)})
        self.table = {k: self.table[k] for k in sorted(self.table.keys())}

    def save(self, save_path: str, file_name: str):
        """
        保存纠错表
        :param save_path: 纠错表存储路径
        :param file_name: 存储文件名
        :return: None
        """
        with open(os.path.join(save_path, file_name), 'w') as f:
            json.dump(self.table, f, indent=4)


# if __name__ == '__main__':
def leonard_correct_func():
    # root = '../data/model'
    # for p in os.listdir(root):
    #     file_name = p[:-3] + '.json'
    #     if os.path.exists(os.path.join(calibration_table_dir, file_name)):
    #         continue
    t_start = time.time()
    corrector = LeonardCorrector(model_path)
    corrector.get_calibration_table()
    corrector.save(calibration_table_dir, 'my_calibration_table.json')
    del corrector
    t_end = time.time()
    print(f'Correct cost: {t_end - t_start}')
