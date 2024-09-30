import numpy as np
from tqdm import tqdm
import config
import os
import csv

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


def insert_values_to_dict(keys, values):  # 将边的array转成dict
    result_dict = {}

    for key in keys:
        if key not in result_dict:
            result_dict[key] = []

    for key, value in zip(keys, values):
        result_dict[key].append(value)

    return result_dict


def digit2digit_array(number):
    # 将数字转换为字符串
    number_str = str(number)

    # 使用列表推导式将字符串中的每个字符转换为整数并存储在列表中
    number_array = [int(digit) for digit in number_str]
    return number_array


def dict2npy(my_dict, file_path):
    # 创建一个空列表用于存储数据
    tensor_data = []

    # 遍历字典的每个键值对
    sorted_keys = sorted(my_dict.keys()) #让键递增有序
    for i in range(len(sorted_keys)):
        key = sorted_keys[i]
        tensor_data.extend(digit2digit_array(key)) #存储key
        tensor_data.append(10)  # 每个字段的分隔符
        
        value = my_dict[key]
        # 对列表进行排序
        sorted_value = sorted(value)
        # 计算相邻元素之间的差分
        value = [sorted_value[0]] + [sorted_value[i + 1] - sorted_value[i] for i in range(len(sorted_value) - 1)]  # 差分
        # 将对应值的每个元素添加到列表中
        value_len = len(value)
        i = 0
        while i < value_len:
            before = i
            while i + 1 < value_len and value[i] == value[i + 1]:
                i += 1
            tensor_data.extend(digit2digit_array(i - before + 1))  # 相同差分的个数
            tensor_data.append(10)  # 每个字段的分隔符
            tensor_data.extend(digit2digit_array(value[i]))  # 差分值
            # 在值之间添加‘10’，除了最后一个值
            if i != len(value) - 1:
                tensor_data.append(10)  # 每个字段的分隔符
            else:  # 添加‘11’分割每条记录
                tensor_data.append(11)  # 每条记录的分隔符
            i += 1
    tensor_data[-1] = 12 #全部记录的结尾
    # 将列表转换为 NumPy 数组
    tensor_data_array = np.array(tensor_data)
    # with open('node.txt', 'w') as f:
    #     f.write(str(list(tensor_data_array)))
    # 将一维张量保存为.npy 文件
    np.save(file_path, tensor_data_array)


def edge_encode():
    # 指定文件路径
    input_file_path = os.path.join(config.project_root, 'data/preprocess/edge_id.csv')
    out_file_path = os.path.join(config.project_root, 'data/encode/edge_encode.npy')
    # 调用函数读取文件中的数组
    arrays = read_arrays_from_file(input_file_path)
    # 将数组转换成dict
    ret = insert_values_to_dict(arrays[0], arrays[1])
    # 将数组保存成npy文件，用于进一步的压缩
    dict2npy(ret, out_file_path)
    # # test: 将dict存成csv，以供解码对比
    # test_dict_to_csv(ret, os.path.join(config.project_root, 'data/preprocess/test_edge_dict_encode.csv'))


def test_dict_to_csv(my_dict, out_file):
    # 提取字典中的键和值
    sorted_keys = sorted(my_dict.keys())

    # 将字典的键和值分别写入 CSV 文件的两列
    with open(out_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])  # 写入标题行
        for key in sorted_keys:
            value = sorted(my_dict[key])
            writer.writerow([key, value])


if __name__ == '__main__':
    edge_encode()
