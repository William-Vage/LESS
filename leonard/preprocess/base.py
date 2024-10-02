import pandas as pd
from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    def __init__(self):
        """
        数据编码基类，应当能处理节点及边的csv文件。
        实现方法：
        1. read_vertex(df: pd.DataFrame)
        读取节点数据
        2. read_edge(df: pd.DataFrame)
        读取边数据
        3. encode()
        节点与边编码
        4. save_model(save_path: str)
        存储编码模型
        5. save_data(save_path: str)
        编码节点信息
        """

    @abstractmethod
    def read_vertex(self, df: pd.DataFrame, vertex_id: str, vertex_type: str, **kwargs):
        """
        读取节点数据
        :param df: 包含节点数据的DataFrame
        :param vertex_id: 节点id列名
        :param vertex_type: 节点属性列名
        输入：节点csv文件
        要求：至少包含节点id，节点类型两个属性。
        保存：节点向量numpy文件，编码映射json文件等。
        :return: None
        """

    @abstractmethod
    def read_edge(self, df: pd.DataFrame, source_id: str, destination_id: str, edge_id: str, edge_type: str, **kwargs):
        """
        读取边数据
        :param df: 包含节点数据的DataFrame
        :param source_id: 源节点id列名
        :param destination_id: 目的节点id列名
        :param edge_id: 边id列名
        :param edge_type: 边类型列名
        输入：节点csv文件
        要求：至少包含源节点id，目的节点id，边id，边类型四个属性。
        保存：节点向量numpy文件，编码映射json文件等。
        :return: None
        """

    @abstractmethod
    def encode(self):
        """
        编码节点以及边的数据
        :return: None
        """

    @abstractmethod
    def save_model(self, path: str):
        """
        存储编码模型
        :param path: 存储路径
        :return: None
        """

    @abstractmethod
    def save_data(self, path: str):
        """
        存储编码后的数据
        :param path: 存储路径
        :return: None
        """
