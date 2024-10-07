import os
import sys
import json
import numpy as np
import pandas as pd
import time
from leonard.preprocess.base import BaseEncoder
import leonard.config as config


def create_unique_mapping_dict(df: pd.DataFrame, exclude_col=None):
    """
    对表格的每一列进行去重，生成一个字典，其中键是每一列的列名，值是一个字典，
    内层是该列每一个不同的取值的集合。
    :param df: Pandas DataFrame，包含要处理的表格数据。
    :param exclude_col: 要排除的列
    :return: dict
    示例输入：
    data = {
        'A': ['apple', 'banana', 'apple', 'orange'],
        'B': [10, 20, 10, 30],
        'C': ['red', 'yellow', 'red', 'orange']
    }
    示例输出：
    output = {
        'A': {'apple', 'banana', 'orange'},
        'B': {10, 20, 30},
        'C': {'red', 'yellow', 'orange'}
    }
    """
    if exclude_col is None:
        exclude_col = []
    result_dict = {}
    for column in df.columns:
        if column in exclude_col:
            continue
        unique_values = df[column].dropna().drop_duplicates()
        result_dict[column] = set(unique_values)
    return result_dict


def calculate_min_values(df: pd.DataFrame, dataset):
    """
    计算指定列的最小值并存储在一维列表中。
    :param df: pandas DataFrame，包含待处理的数据。
    :return: list，包含timestampNanos、startTimestampNanos、sequence、size四列的最小值。
    示例输入：
    data = {
        'timestampNanos': [100, 200, 300, 400],
        'sequence': [1, 2, 3, 4],
    }
    示例输出：
    [100, 9223372036854775807, 1, 9223372036854775807]
    """
    min_values_list = [sys.maxsize] * 4  # 初始化为sys.maxsize
    # 指定列名
    columns_to_calculate = ['startTimestampNanos', 'event id', 'time', '-']
    if dataset == 'darpa_tc':
        columns_to_calculate = ['timestampNanos', 'sequence']#, 'size', '-']
    for column in columns_to_calculate:
        if column in df.columns:
            min_value = df[column].min()
            min_values_list[columns_to_calculate.index(column)] = min_value
    # 特判
    if 'time' in df.columns:
        min_values_list[2] = int(min([str(i).split('.')[0] for i in df['time']]))
        min_values_list[3] = int(min([str(i).split('.')[1] for i in df['time']]))
    return min_values_list


def extract_non_null_columns_list(df: pd.DataFrame):
    """
    提取DataFrame中每行的列名字符串列表。
    :param df: Pandas DataFrame
    :return: 每行列名字符串列表
    """

    def extract_non_null_columns(row):
        non_null_columns = [column for column, value in row.items() if pd.notna(value)]
        return ','.join(non_null_columns)

    # 对每一行应用函数
    s = df.apply(extract_non_null_columns, axis=1)
    # 转换为列表
    result_list = s.tolist()
    return result_list


class LeonardEncoder(BaseEncoder):
    def __init__(self):
        """
        Leonard原始数据编码类
        """
        super().__init__()
        self.vertex: pd.DataFrame = pd.DataFrame()
        self.edge: pd.DataFrame = pd.DataFrame()
        # 保存的参数
        self.parm_edges = [[] for _ in range(4)]
        self.id2char_dict = {}
        self.char2id_dict = {}
        self.min_values = []
        self.re_values = {}
        self.key_template_dict = {}
        self.processed_data: np.array = None
        # 列别名映射
        self.m = {}

    def read_vertex(self, df: pd.DataFrame, vertex_id: str, vertex_type: str, **kwargs):
        self.m.update({
            'vertex_id': vertex_id,
            'vertex_type': vertex_type
        })
        self.vertex = df

    def read_edge(self, df: pd.DataFrame, source_id: str, destination_id: str, edge_id: str, edge_type: str, **kwargs):
        self.m.update({
            'source_id': source_id,
            'destination_id': destination_id,
            'edge_id': edge_id,
            'edge_type': edge_type
        })
        self.edge = df

    def update_char_dict_and_translate(self, s: str, split=True):
        """
        更新字符到id的映射，并返回映射结果
        :param s: 输入串
        :param split: 是否逐字符遍历
        :return: str 映射结果
        """
        res = []
        if s is None:
            return res
        if split:
            for c in s:
                if c not in self.char2id_dict:
                    id_ = len(self.char2id_dict) + 2
                    self.char2id_dict[c] = id_
                    self.id2char_dict[id_] = c
                    res.append(id_)
                else:
                    res.append(self.char2id_dict[c])
        else:
            if s not in self.char2id_dict:
                id_ = len(self.char2id_dict) + 2
                self.char2id_dict[s] = id_
                self.id2char_dict[id_] = s
                res.append(id_)
            else:
                res.append(self.char2id_dict[s])
        # 属性分隔符
        res.append(0)
        return res

    def encode_other_parameters(self, df: pd.DataFrame, dataset):
        """
        编码其余参数
        :param df: 输入参数
        :return: 编码结果
        """
        res = []
        for col in df:
            if col in ["id", "predicate_id", "subject_id"]:
                continue
            # elif (dataset == 'leonard' and col == 'startTimestampNanos') or (dataset == 'darpa_tc' and col == 'timestampNanos'):
            #     value = str(df[col] - self.min_values[0])
            #     value = [str(x) for x in list(df[col] - self.min_values[0])]
            # elif (dataset == 'leonard' and col == 'event id') or (dataset == 'darpa_tc' and col == 'sequence'):
            #     value = [str(x) for x in list(df[col] - self.min_values[1])]
            elif col == 'time':
                value = [str(i).replace('.', ',') for i in df[col]]
            else:
                value = [str(self.re_values[col][i]) if pd.notna(i) else None for i in df[col]]
            value = [self.update_char_dict_and_translate(i) for i in value]
            res.append(value)
        return res


    def encode(self, dataset):
        # 1.统计所有不重复的串，并从0-N进行编码
        unique_values_vertex = create_unique_mapping_dict(self.vertex)
        unique_values_edge = create_unique_mapping_dict(self.edge, exclude_col=['id'])
        result_dict = unique_values_vertex.copy()
        # result_dict['id'] = result_dict['id'].union(unique_values_edge['subject_id'])
        # result_dict['id'] = result_dict['id'].union(unique_values_edge['predicate_id'])
        # 合并集合
        for key, value_set in unique_values_edge.items():
            if key in result_dict:
                result_dict[key] = result_dict[key].union(value_set)
            else:
                result_dict[key] = value_set
        # 对集合进行0-N-1编码
        # for key, value_set in result_dict.items():
        #     for i, value in enumerate(value_set):
        #         print({key:{value:i}})
        self.re_values = {key: {value: i for i, value in enumerate(value_set)} for key, value_set in
                          result_dict.items()}
        tmp_dict = {}
        cnt = len(self.re_values['id'])
        for key, _ in self.re_values['subject_id'].items():
            if(self.re_values['id'].get(key)):
                tmp_dict[key] = self.re_values['id'][key]
            else:
                tmp_dict[key] = cnt
                cnt += 1
        self.re_values['subject_id'] = tmp_dict.copy()

        tmp_dict = {}
        for key, _ in self.re_values['predicate_id'].items():
            if(self.re_values['id'].get(key)):
                tmp_dict[key] = self.re_values['id'][key]
            else:
                tmp_dict[key] = cnt
                cnt += 1
        self.re_values['predicate_id'] = tmp_dict.copy()
        # 2.求4列最小值的数组
        self.min_values = calculate_min_values(self.edge, dataset)
        # 3.更新hash及pid
        rows_vertex = self.vertex['pid'].notna()

        # self.parm_edges[0].extend([self.re_values['id'][i] for i in self.vertex.loc[rows_vertex, 'id']])
        # self.parm_edges[0].extend([self.re_values['id'][i] for i in self.vertex['id']])
        self.parm_edges[0].extend([self.re_values['id'][i] for i in self.re_values['id']])
        self.parm_edges[1].extend([self.re_values['pid'][i] for i in self.vertex.loc[rows_vertex, 'pid']])
        # 4.更新edge的source_id与destination_id，转换为id数值
        self.parm_edges[2] = [self.re_values[self.m['source_id']][i] for i in self.edge[self.m['source_id']]]
        self.parm_edges[3] = [self.re_values[self.m['destination_id']][i] for i in self.edge[self.m['destination_id']]]
        # 5.对每个hash值做编码
        # hash_vertex = [self.update_char_dict_and_translate(f"verteid:{self.re_values['id'][i]}")
        #                for i in self.vertex['id']]
        hash_vertex = [self.update_char_dict_and_translate(f"verteid:{self.re_values['id'][i]}")
                       for i in self.vertex['id']]
        hash_edge = [self.update_char_dict_and_translate(f"eventid:{i}")
                     for i in range(len(self.edge))]
        # 6.编码非空属性组合
        s_vertex = extract_non_null_columns_list(self.vertex)
        s_edge = extract_non_null_columns_list(self.edge)
        s = set(s_vertex + s_edge)
        self.key_template_dict = {key: id_ for id_, key in enumerate(s)}
        schema_vertex = [self.update_char_dict_and_translate(str(self.key_template_dict[i]), split=False)
                         for i in s_vertex]
        schema_edge = [self.update_char_dict_and_translate(str(self.key_template_dict[i]), split=False) for i in s_edge]
        # 7.编码其余属性
        other_prop_vertex = self.encode_other_parameters(self.vertex, dataset)
        other_prop_edge = self.encode_other_parameters(self.edge, dataset)
        # 8.整合编码结果
        stop_vertex = [[1] for _ in range(len(self.vertex))]
        stop_edge = [[1] for _ in range(len(self.edge))]
        encoded_vertex = [sum(i, []) for i in zip(hash_vertex, schema_vertex, *other_prop_vertex, stop_vertex)]
        encoded_vertex = np.concatenate(encoded_vertex)
        encoded_edge = [sum(i, []) for i in zip(hash_edge, schema_edge, *other_prop_edge, stop_edge)]
        encoded_edge = np.concatenate(encoded_edge)
        self.processed_data = np.concatenate((encoded_vertex, encoded_edge))
        # 9.将re_values去除value
        for k in self.re_values:
            self.re_values[k] = sorted(self.re_values[k], key=lambda k2: self.re_values[k][k2])
            self.re_values[k] = [str(int(i)) if isinstance(i, float) and round(i) == i
                                 else str(i) for i in self.re_values[k]]
        # 10.删除hash
        # 删除predicate_id，subject_id，以及全为空的列
        del_keys = [self.m['vertex_id'], self.m['source_id'], self.m['destination_id'], 'time']
        for k, v in self.re_values.items():
            if len(v) == 0:
                del_keys.append(k)
        for i in del_keys:
            if i in self.re_values:
                del self.re_values[i]

    def convert_to_native_type(self, obj):
        """递归转换字典中的 NumPy 类型为 Python 原生类型"""
        if isinstance(obj, dict):
            return {k: self.convert_to_native_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_native_type(i) for i in obj]
        elif isinstance(obj, (np.int64, np.float64)):
            return obj.item()  # 转换为 Python 的 int 或 float
        return obj  # 返回原始对象

    def save_model(self, path: str):
        """
        保存模型
        :param path: 存储路径名
        :return: None
        """
        # 1.编码参数json文件
        params = {
            "id2char_dict": self.id2char_dict,
            "char2id_dict": self.char2id_dict,
            "mins": self.min_values,
            "re_values_dict": self.re_values,
            "key_template_dict": self.key_template_dict,
        }
        params = self.convert_to_native_type(params)

        with open(os.path.join(path, 'params.json'), "w") as f:
            json.dump(params, f, indent=4)
        # 2.edges数组
        s = [str(i) + '\n' for i in self.parm_edges]
        with open(os.path.join(path, 'edges.txt'), 'w') as f:
            f.writelines(s)

    def save_data(self, path: str):
        """
        保存编码后的数据
        :param path: 存储路径
        :return: None
        """
        np.save(os.path.join(path, 'tensor.npy'), self.processed_data)


# if __name__ == '__main__':
def leonard_preprocess_func(leonard_edge_file='', leonard_vertex_file='', dataset='toy'):
    t_start = time.time()
    # 测试leonard样例数据
    encoder = LeonardEncoder()
    save_path = os.path.join(config.project_root, 'data/encode/')
    if dataset == 'darpa_tc':
        vertex = pd.read_csv(leonard_vertex_file, low_memory=False)
        edge = pd.read_csv(leonard_edge_file, low_memory=False)
        # 读取点和边
        encoder.read_vertex(vertex, vertex_id='id', vertex_type='type')
        encoder.read_edge(edge, source_id='subject_id', destination_id='predicate_id', edge_id='id',
                          edge_type='type')
    elif dataset == 'toy':
        vertex = pd.read_csv(leonard_vertex_file, low_memory=False)
        edge = pd.read_csv(leonard_edge_file, low_memory=False)
        vertex = vertex.rename(columns={
            'hash': 'id'
        })
        edge = edge.rename(columns={
            'hash': 'id',
            'parentVertexHash': 'predicate_id',
            'childVertexHash': 'subject_id'
        })
        # 读取点和边
        encoder.read_vertex(vertex, vertex_id='id', vertex_type='type')
        encoder.read_edge(edge, source_id='predicate_id', destination_id='subject_id', edge_id='id',
                          edge_type='type')

    # 编码
    encoder.encode(dataset)
    # 保存模型
    encoder.save_model(save_path)
    # 保存编码后的数据
    encoder.save_data(save_path)
    t_end = time.time()
    t_cost = t_end - t_start
    print(f"\033[33m Preprocessing Time: {t_cost:.1f}s \033[0m")
    return t_cost
