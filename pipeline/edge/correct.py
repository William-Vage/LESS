from config import device, model_name_appendex, ml_model_list
from tqdm import tqdm
import xgboost as xgb
import numpy as np
import config
import struct
import joblib
import torch
import os

# 数据参数
n_vocab = 13
# 模型参数
batch_size = 1024
seq_len = 10

class Corrector:
    def __init__(self, data_path, model_path, method_name):
        """
        纠错表构造类
        """
        # 1.读取编码数据
        self.data = np.load(data_path)
        self.wrong_pred = []
        # 2.加载模型
        self.method_name = method_name
        if self.method_name in ml_model_list:
            if self.method_name == 'xgboost':
                self.model = xgb.Booster()
                self.model.load_model(model_path)
            else:
                self.model = joblib.load(model_path)
        else:
            self.model = torch.load(model_path, device)
        # 3.校准表
        self.table = {}

    def model_predict_and_correct(self, x: np.ndarray, y: np.ndarray):
        """
        模型预测
        :param x: 输入数据
        :param y: 预测目标
        :return: 预测结果
        """
        # 1.模型预测
        x_ = torch.tensor(x, device=device)
        y_ = torch.tensor(y, dtype=torch.long, device=device).reshape(-1)
        if self.method_name in ml_model_list:
            x_  = x_.cpu().numpy()
            if self.method_name == 'xgboost':
                dtest = xgb.DMatrix(x_)
                y_pred = self.model.predict(dtest)
            else:
                y_pred = self.model.predict(x_)
            y_pred = torch.from_numpy(y_pred).to(device=device)
        else:
            y_pred = self.model(x_)
            y_pred = torch.argmax(y_pred, dim=-1)
        # 2.纠错
        res = torch.not_equal(y_, y_pred).cpu().numpy()
        return res

    def get_calibration_table(self):
        """
        生成校准表
        :return: None
        """
        # 1.计算需要几个batch
        n_batch = int(np.ceil(len(self.data) / batch_size))
        for i in tqdm(range(n_batch)):
            # 边界处理
            self.wrong_pred.append([1] * seq_len)
            # 2.按滑动窗口提取数据
            start = max(0, i * batch_size)
            end = min((i + 1) * batch_size, len(self.data))
            data = np.lib.stride_tricks.sliding_window_view(self.data[start: end],
                                                            window_shape=seq_len + 1, writeable=False)
            # 3.运行模型
            x = data[:, :-1]
            y = data[:, -1]
            res = self.model_predict_and_correct(x, y)
            # 4.更新纠错表
            self.wrong_pred.append(res)
        self.wrong_pred = np.concatenate(self.wrong_pred, axis=0)
        self.wrong_pred[-1] = 1
        if i == n_batch-1:
            print()

    def save(self, save_path: str, file_name: str):
        """
        保存并编码纠错表
        :param save_path: 纠错表存储路径
        :param file_name: 存储文件名
        :return: None
        """
        # 注意：纠错表的结尾不一定是全部的结尾，因为可能会一直预测正确
        with open(os.path.join(save_path, file_name), 'wb') as f:
            cnt = 0
            ws = sum(self.wrong_pred)
            print(f'Wrong prediction: {ws}, wrong ratio: {ws / len(self.data)}')
            for i in tqdm(range(len(self.data))):
                if self.wrong_pred[i]:
                    # 如果偏移量大于128，用2个字节写偏移量除以128的部分
                    n = cnt // 128
                    if n > 0:
                        v = struct.pack('<H', n | 0x8000)
                        f.write(v)
                    # 用2个字节写偏移量+真实值
                    res = cnt - n * 128
                    f.write(bytes([res, self.data[i]]))
                    # 重置偏移量
                    cnt = 1
                else:
                    cnt += 1


class Re_Corrector:
    def __init__(self, model_path, method_name):
        self.data = []
        self.wrong_pred = []
        # 加载模型
        self.method_name = method_name
        if self.method_name in ml_model_list:
            if self.method_name == 'xgboost':
                self.model = xgb.Booster()
                self.model.load_model(model_path)
            else:
                self.model = joblib.load(model_path)
        else:
            self.model = torch.load(model_path, device)

    def load(self, load_path: str, file_name: str):
        """
        从文件中读取并解码至self.data中
        :param load_path: 纠错表存储路径
        :param file_name: 存储文件名
        :return: None
        """
        with open(os.path.join(load_path, file_name), 'rb') as f:
            cnt = 0
            self.data = []  # 清空self.data
            self.wrong_pred = []  # 清空self.wrong_pred
            while True:
                n_byte = f.read(2)
                if not n_byte:
                    break
                n = struct.unpack('<H', n_byte)
                n = int(n[0])
                # 如果最高位为1，表示需要读取额外的偏移量
                if n & 0x8000:   
                    n &= ~0x8000
                    cnt += n * 128
                    #读取偏移量和数据值
                    res_data_byte =f.read(2)
                    if not res_data_byte:
                        break
                    res, data = struct.unpack('<BB', res_data_byte)
                else:
                    res, data = struct.unpack('<BB', n_byte)
                cnt += res
                #更新data和wrong_pred列表
                self.wrong_pred += [0] * (cnt - 1)
                self.data += [-1] * (cnt - 1)
                self.data.append(data)
                self.wrong_pred += [1]
                cnt = 0
        print("Data loaded successfully.")


    def reconstruct_correct_data(self, data_path_for_decode):
        """
        恢复正确的编码数据
        :return: None
        """
        n_batch = int(np.ceil(len(self.data) / batch_size))
        for i in tqdm(range(batch_size)):
            x = [self.data[i+j*batch_size:i+j*batch_size+seq_len] for j in range(0, n_batch-1)]
            # 运行模型
            if self.method_name in ml_model_list:
                if self.method_name == 'xgboost':
                    dtest = xgb.DMatrix(x)
                    y_pred = self.model.predict(dtest)
                else:
                    y_pred = self.model.predict(x)
                y_pred = torch.from_numpy(y_pred).to(device=device)
            else:
                x = torch.tensor(x, device=device)
                with torch.no_grad():
                    y_pred = self.model(x)
                y_pred = torch.argmax(y_pred, dim=-1)
            for j in range(0, n_batch-1):
                cur = i+j*batch_size+seq_len
                if not self.wrong_pred[cur]:
                    self.data[cur] = int(y_pred[j])
        cur = (n_batch-1)*batch_size
        while 1:
            if cur+seq_len+1 == len(self.data):
                break
            x = [self.data[cur:cur+seq_len]]
            # 运行模型
            if self.method_name in ml_model_list:
                if self.method_name == 'xgboost':
                    dtest = xgb.DMatrix(x)
                    y_pred = self.model.predict(dtest)
                else:
                    y_pred = self.model.predict(x)
                y_pred = torch.from_numpy(y_pred)
            else:
                x = torch.tensor(x, device=device)
                with torch.no_grad():
                    y_pred = self.model(x)
                y_pred = torch.argmax(y_pred, dim=-1)
            if not self.wrong_pred[cur+seq_len]:
                self.data[cur+seq_len] = int(y_pred)
            cur += 1
        # if predictions[-1] != 12: #不是全部记录末尾符号，则说明后续推断全部正确，继续推断，直至出现12
        #     while 1:
        #         x = torch.tensor([predictions[-seq_len:]], device=device)
        #         # 运行模型
        #         if self.method_name in ml_model_list:
        #             if self.method_name == 'xgboost':
        #                 dtest = xgb.DMatrix(x)
        #                 y_pred = self.model.predict(dtest)
        #             else:
        #                 y_pred = self.model.predict(x)
        #             y_pred = torch.from_numpy(y_pred)
        #         else:
        #             with torch.no_grad():
        #                 y_pred = self.model(x)
        #             y_pred = torch.argmax(y_pred, dim=-1)
        #         predictions.append(int(y_pred))
        #         if y_pred == 12:
        #             break
        
        # 4.将预测的正确list回传给self.data参数
        # for i, pred in enumerate(self.wrong_pred):
        #     if pred == 0:
        #         self.data[i] = predictions[i]
        # 保存为 .npy 文件
        data_array = np.array(self.data)
        np.save(data_path_for_decode, data_array)


def correct_edge(method_name:str):
    # 训练数据及模型路径
    data_path = os.path.join(config.project_root, 'data/encode/edge_encode.npy')
    model_path = os.path.join(config.project_root, 'data/model/'+ model_name_appendex[method_name])
    calibration_table_dir = os.path.join(config.project_root, 'data/correct')
    corrector = Corrector(data_path, model_path, method_name)
    corrector.get_calibration_table()
    corrector.save(calibration_table_dir, 'calibration_table.txt')


def re_construct_edge(method_name:str):
    model_path = os.path.join(config.project_root, 'data/model/' + model_name_appendex[method_name])
    data_path_for_decode = os.path.join(config.project_root, 'data/decode/edge_decode.npy')
    calibration_table_dir = os.path.join(config.project_root, 'data/correct')
    re_corrector = Re_Corrector(model_path, method_name)
    re_corrector.load(calibration_table_dir, 'calibration_table.txt')
    re_corrector.reconstruct_correct_data(data_path_for_decode)

    # if len(corrector.data) != len(re_corrector.data):
    #     print("列表长度不同")
    # else:
    #     for i in range(len(corrector.data)):
    #         if corrector.data[i] != re_corrector.data[i]:
    #             print("列表内容不同")
    #             break
    #     else:
    #         print("列表内容相同")
    # print()


if __name__ == '__main__':
    correct_edge()
