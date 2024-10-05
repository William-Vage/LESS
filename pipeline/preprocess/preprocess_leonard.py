import pandas as pd
import config
import os


def preprocess_vertex(src_file: str, dest_dir: str):
    """
    Preprocess vertices
    :return: None
    """
    df = pd.read_csv(src_file, low_memory=False)
    node_hash = df['hash']
    del df['hash']
    node_hash.to_csv(os.path.join(dest_dir, 'node_id.csv'), index=False)
    df.to_csv(os.path.join(dest_dir, 'node_property.csv'), index=False)


def preprocess_edge(src_file: str, dest_dir: str):
    """
    Preprocess edges
    :return: None
    """
    df = pd.read_csv(src_file, low_memory=False)
    edge_hash = df[['parentVertexHash', 'childVertexHash']]
    del_cols = ['hash', 'parentVertexHash', 'childVertexHash']
    for i in del_cols:
        del df[i]
    edge_hash.to_csv(os.path.join(dest_dir, 'edge_id.csv'), index=False)
    df.to_csv(os.path.join(dest_dir, 'edge_property.csv'), index=False)


def preprocess(vertex_src: str, edge_src: str, dest_dir: str):
    """
    Preprocess the Leonard dataset
    :return: None
    """
    preprocess_vertex(vertex_src, dest_dir)
    preprocess_edge(edge_src, dest_dir)


def preprocess_toy():
    vertex = os.path.join(config.project_root, 'data/raw/vertex200m.csv')
    edge = os.path.join(config.project_root, 'data/raw/edge200m.csv')
    dest = os.path.join(config.project_root, 'data/preprocess')
    preprocess(vertex, edge, dest)


def preprocess_exp():
    vertex = os.path.join(config.project_root, 'data/raw/exp_vertex.csv')
    edge = os.path.join(config.project_root, 'data/raw/exp_edge.csv')
    dest = os.path.join(config.project_root, 'data/preprocess')
    vertex = pd.read_csv(vertex, low_memory=False)
    edge = pd.read_csv(edge, low_memory=False)
    vertex.to_csv(os.path.join(dest, 'node_property.csv'), index=False)
    edge.to_csv(os.path.join(dest, 'edge_property.csv'), index=False)


if __name__ == '__main__':
    preprocess_toy()
