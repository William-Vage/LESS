import pandas as pd
import config
import os


def preprocess_vertex(src_file: str, dest_dir: str):
    """
    预处理节点
    :return: None
    """
    df = pd.read_csv(src_file, low_memory=False)

    # 保留第一次出现的行
    df = df.drop_duplicates(subset='id', keep='first')

    node_hash = df['id']
    del df['id']
    node_hash.to_csv(os.path.join(dest_dir, 'node_id.csv'), index=False)
    df.to_csv(os.path.join(dest_dir, 'node_property.csv'), index=False)


def preprocess_edge(src_file: str, dest_dir: str):
    """
    预处理边
    :return: None
    """
    df = pd.read_csv(src_file, low_memory=False)
    edge_hash = df[['subject_id', 'predicate_id']]
    del_cols = ['id', 'subject_id', 'predicate_id']
    for i in del_cols:
        if i in df.columns:
            del df[i]
    edge_hash.to_csv(os.path.join(dest_dir, 'edge_id.csv'), index=False)
    df.to_csv(os.path.join(dest_dir, 'edge_property.csv'), index=False)


def preprocess(vertex_src: str, edge_src: str, dest_dir: str):
    """
    预处理leonard数据集
    :return: None
    """
    preprocess_vertex(vertex_src, dest_dir)
    preprocess_edge(edge_src, dest_dir)


def preprocess_darpa_tc(edge_file, vertex_file):
    vertex = os.path.join(config.project_root, 'data/raw', vertex_file)
    edge = os.path.join(config.project_root, 'data/raw', edge_file)
    dest = os.path.join(config.project_root, 'data/preprocess')
    preprocess(vertex, edge, dest)


if __name__ == '__main__':
    preprocess_darpa_tc()
