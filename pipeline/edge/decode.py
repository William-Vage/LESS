import numpy as np
from tqdm import tqdm
import config
import os
import csv
import time


def read_arrays_from_file(file_path):
    """
    读取edge连接关系文件，转成array
    :param file_path:
    :return:
    """
    res = [[], []]
    i = 0
    # 创建一个空字典，用于存储已读取的行
    hash_dict = {}
    with open(file_path, 'r') as f:
        # 跳过第一行
        next(f)
        # 读取文件的每一行
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines)):
            # 去除首尾空白符
            line = line.strip()
            if line:
                line_items = line.split(',')
                for item in line_items:
                    if item not in hash_dict:
                        hash_dict[item] = i
                        i += 1
                res[0].append(hash_dict[line_items[0]])
                res[1].append(hash_dict[line_items[1]])
    return res


def npy2dict(file_path):
    # 从.npy文件加载数据
    tensor_data_array = np.load(file_path)
    tensor_data = tensor_data_array.tolist()

    # 创建一个空字典用于存储数据
    my_dict = {}
    # cur_edge_id = 0
    # 将数据转换回原始字典形式
    value = []  # 更替存储差分个数、差分值的数组
    tmp_num = 0  # 把0-9的数字恢复成的完整的计数值
    # 从编码恢复更替存储差分个数、差分值的数组
    for _, num in enumerate(tensor_data):
        if num == 11 or num == 12:  # 遇到记录分隔符或全部记录停止符
            value.append(tmp_num)
            tmp_num = 0
            key = value[0]
            my_dict[key] = value[1:]
            value = []
        elif num == 10:  # 遇到字段分隔符
            value.append(tmp_num)
            tmp_num = 0
        else:  # 遇到0-9的数字
            tmp_num = tmp_num * 10 + num
    # 恢复所有差分值（去除差分个数这一项）、反差分
    for key, value in my_dict.items():
        tmp_value = []
        # 去除差分个数这一项
        for i in range(int(len(value) / 2)):
            tmp_value += [value[i * 2 + 1]] * value[i * 2]
        # 恢复原始值（反差分）
        for i in range(len(tmp_value) // 2 - 1):
            tmp_value[i + 1] += tmp_value[i]
        my_dict[key] = tmp_value  # 每一个字典项的第0个表示整个字典前面有多少条边了，从而方便获取此字典项内的边的id编号（每个字典项的第0条边的id即为此值）。
        # cur_edge_id += len(tmp_value)
    for key, value_list in my_dict.items():
        # 确定列表的长度（必为偶数）
        mid = len(value_list) // 2
        # 将前半部分和后半部分配对成二元组
        my_dict[key] = [(value_list[i], value_list[i + mid]) for i in range(mid)]
    return my_dict


def test_dict_to_csv(my_dict, out_file):
    # 提取字典中的键和值
    keys = list(my_dict.keys())
    values = list(my_dict.values())

    # 将字典的键和值分别写入 CSV 文件的两列
    with open(out_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])  # 写入标题行
        for key, value in zip(keys, values):
            writer.writerow([key, value])


def successors_to_predecessors(successors_dict):
    # 新字典
    predecessors_dict = {}
    # 遍历原始字典的每个键值对
    for orig_key, tuple_list in successors_dict.items():
        for first_elem, second_elem in tuple_list:
            # 如果新字典中不存在当前first_elem作为键的记录，创建一个空列表
            if first_elem not in predecessors_dict:
                predecessors_dict[first_elem] = []

            # 将原始键和二元组中的第二个元素组成新二元组，添加到新字典中
            predecessors_dict[first_elem].append((orig_key, second_elem))
    return predecessors_dict


def edge_decode():
    # 指定文件路径
    input_file_path = os.path.join(config.project_root, 'data/decode/edge_decode.npy')
    # out_file_path = os.path.join(config.project_root, 'data/preprocess/test_edge_dict_decode.csv')
    # out_file_path2 = os.path.join(config.project_root, 'data/preprocess/test_edge_dict_decode2.csv')
    # 读取npy文件为dict，并在此函数中反差分
    ret_dict = npy2dict(input_file_path)
    t_start = time.time()
    ret_dict2 = successors_to_predecessors(ret_dict)
    t_end = time.time()
    print(f'successors_to_predecessors cost: {t_end - t_start}')
    # test:将dict存储成文件
    # test_dict_to_csv(ret_dict, out_file_path)
    # test_dict_to_csv(ret_dict2, out_file_path2)
    return ret_dict, ret_dict2


if __name__ == '__main__':
    edge_decode()
