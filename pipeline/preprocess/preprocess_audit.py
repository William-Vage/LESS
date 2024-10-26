import os
import json
import pandas as pd
from typing import List, Dict
import config


def preprocess_audit(file: str):
    """
    预处理图数据JSON文件，并输出四个CSV文件：
    1. edge_id.csv
    2. edge_property.csv
    3. node_id.csv
    4. node_property.csv

    参数:
    file (str): 输入的JSON文件名，位于data/raw目录下。
    """
    # 构建输入文件路径
    input_path = os.path.join(config.project_root, 'data/raw', file)

    # 读取JSON文件
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误：无法读取JSON文件 - {e}")
        return

        # 初始化列表存储节点和边
    nodes: List[Dict] = []
    edges: List[Dict] = []

    # 分离节点和边
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"警告：第 {idx} 个元素不是字典，已跳过。")
            continue

        if 'id' in item and 'type' in item:
            nodes.append(item)
        elif 'from' in item and 'to' in item and 'type' in item:
            edges.append(item)
        else:
            print(f"警告：第 {idx} 行既不是节点也不是边，已跳过。")

    if edges:
        processed_edges = []
        edge_annotations_keys = set()

        for edge in edges:
            processed_edge = {
                'from': edge.get('from', ''),
                'to': edge.get('to', ''),
                'type': edge.get('type', ''),
            }

            annotations = edge.get('annotations', {})
            if isinstance(annotations, dict):
                for key, value in annotations.items():
                    processed_edge[key] = value
                    edge_annotations_keys.add(key)
            processed_edges.append(processed_edge)

        edges_df = pd.DataFrame(processed_edges)

        # 文件1: edge_id.csv
        if {'from', 'to'}.issubset(edges_df.columns):
            edge_id_df = edges_df[['from', 'to']].rename(columns={'from': 'parentVertexHash', 'to': 'childVertexHash'})
            edge_id_path = os.path.join(config.project_root, 'data/preprocess', 'edge_id.csv')
            edge_id_df.to_csv(edge_id_path, index=False)
        else:
            print("错误：边数据缺少 'from' 或 'to' 字段。")
            return

        # 文件2: edge_property.csv
        fixed_edge_columns = ['type', 'event id']
        dynamic_edge_columns = sorted(list(edge_annotations_keys - set(fixed_edge_columns)))
        all_edge_columns = fixed_edge_columns + dynamic_edge_columns

        for col in all_edge_columns:
            if col not in edges_df.columns:
                edges_df[col] = ''

        edge_property_df = edges_df[all_edge_columns].copy()

        edge_property_path = os.path.join(config.project_root, 'data/preprocess', 'edge_property.csv')
        edge_property_df.to_csv(edge_property_path, index=False)
    else:
        print("警告：数据中未找到任何边。")

    if nodes:
        processed_nodes = []
        node_annotations_keys = set()

        for node in nodes:
            processed_node = {
                'id': node.get('id', ''),
                'type': node.get('type', '')
            }

            annotations = node.get('annotations', {})
            if isinstance(annotations, dict):
                for key, value in annotations.items():
                    processed_node[key] = value
                    node_annotations_keys.add(key)
            processed_nodes.append(processed_node)

        nodes_df = pd.DataFrame(processed_nodes)

        # 文件3: node_id.csv
        if 'id' in nodes_df.columns:
            node_id_df = nodes_df[['id']].rename(columns={'id': 'hash'})
            node_id_path = os.path.join(config.project_root, 'data/preprocess', 'node_id.csv')
            node_id_df.to_csv(node_id_path, index=False)
        else:
            print("错误：节点数据缺少 'id' 字段。")
            return

        # 文件4: node_property.csv
        fixed_node_columns = ['type']
        dynamic_node_columns = sorted(list(node_annotations_keys - set(fixed_node_columns)))
        all_node_columns = fixed_node_columns + dynamic_node_columns

        for col in all_node_columns:
            if col not in nodes_df.columns:
                nodes_df[col] = ''

        node_property_df = nodes_df[all_node_columns].copy()


        node_property_path = os.path.join(config.project_root, 'data/preprocess', 'node_property.csv')
        node_property_df.to_csv(node_property_path, index=False)
    else:
        print("警告：数据中未找到任何节点。")

