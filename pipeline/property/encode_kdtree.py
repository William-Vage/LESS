from pipeline.property.encode import merge_adjacent_operations, save_encode_result, UnionFindSet
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from Levenshtein import editops
from config import device
from tqdm import tqdm
import config
import struct
import os
import networkx as nx


# 模型设置
batch_size = 2048
leaf_size = 1
n_neighbors = 5
n_dim = 10


def minimum_spanning_tree(distances, indices):
    # 创建一个空的最小生成树边列表
    edge_list = []
    # 创建一个空的无向图
    G = nx.Graph()
    # 批量添加带权重的边到图中
    for i in tqdm(range(len(distances)), desc='build graph'):
        weighted_edges = [(i, j, dist) for j, dist in zip(indices[i], distances[i]) if i < j]
        G.add_weighted_edges_from(weighted_edges)
    # 找到图的所有连通分量
    connected_components = list(nx.connected_components(G))
    # 对每个连通分量应用最小生成树算法
    for component in tqdm(connected_components, desc='minimum spanning tree algo'):
        subgraph = G.subgraph(component)
        MST_component = nx.minimum_spanning_tree(subgraph)
        # 添加最小生成树的边到列表中
        edge_list.extend([(u, v) for u, v, weight in MST_component.edges(data='weight')])
    return edge_list


def compress_graph_new(file_path) -> list[list]:
    # 读取节点属性串
    with open(file_path, 'r') as file:
        node_attributes = [line.strip() for line in file]
    del node_attributes[0]
    # 使用transformer计算属性串的向量形式
    transformer = SentenceTransformer('all-MiniLM-L12-v2', device=device)
    # 获得每个属性的词向量
    embeddings = transformer.encode(node_attributes, batch_size=batch_size, show_progress_bar=True,
                                    convert_to_tensor=False, normalize_embeddings=True)
    # 数据降维
    model = PCA(n_dim)
    embeddings = model.fit_transform(embeddings)
    # 构建kd树
    tree = KDTree(embeddings, leaf_size, metric='euclidean')
    distance, indices = tree.query(embeddings, k=n_neighbors)
    # 构建最小生成树
    indices = minimum_spanning_tree(distance, indices)
    # 合并相似节点
    n = len(node_attributes)
    visit = set(range(n))
    u = UnionFindSet(n)
    ans = []
    for i, j in tqdm(indices, desc=f'merge nodes'):
        # 必须是两个集合的代表元
        if i in visit and j in visit and u.find(i) != u.find(j):
            # 获取编辑操作
            edits = editops(node_attributes[i], node_attributes[j])
            # 合并编辑操作
            edits = merge_adjacent_operations(edits, node_attributes[j])
            # 记录结果
            ans.append([i, j, edits])
            # 合并并查集
            u.union(i, j)
            # 代表元中删除j
            visit.discard(j)
    # 插入未替换的元素
    for i in visit:
        ans.append([i, i, [['insert', 0, node_attributes[i]]]])
    # 将结果排序
    ans.sort(key=lambda x: x[1])
    return ans


def encode_line2(a: list, file):
    """
    编码规则
    fa: max value
    fb: insert
    fc: replace
    fd: delete
    fe: op_sep
    ff: sep
    """
    # 差分存储父节点id
    # if prev_a is None or a[0] != prev_a[0]:
    # v = a[1] - a[0] + 32768
    # f_id = struct.pack('<H', v)
    # file.write(f_id)
    # 统计次数
    cnt = 0
    # 遍历操作
    for x in a[2]:
        op, pos, value = x
        # 记录操作
        if op == 'insert':
            file.write(bytes([0xfb]))
        elif op == 'replace':
            file.write(bytes([0xfc]))
        else:
            file.write(bytes([0xfd]))
        # 记录操作位置
        if pos <= 0xfa:
            file.write(bytes([pos]))
            cnt += 2
        else:
            file.write(bytes([0xfe]))
            pos_bin = struct.pack('<H', pos)
            file.write(pos_bin)
            cnt += 3
        # 如果是删除，且删除的是一个区间
        if op == 'delete':
            # if pos != value:
            file.write(bytes([value - pos + 1]))
            cnt += 1
        else:
            file.write(value.encode())
            cnt += len(value)
    # file.write(bytes([0xff]))
    return cnt


def save_encode_result2(compressed, out_file: str, out_dir: str):
    """
    记录编码结果
    :return:
    """
    fa = []
    cnt = [0]
    with open(os.path.join(out_dir, out_file + '_data'), 'wb') as f:
        for i in compressed:
            r = encode_line2(i, f)
            fa.append(i[0] - i[1])
            cnt.append(r + cnt[-1])
    cnt.pop(0)
    fa = [str(i) for i in fa]
    cnt = [str(i) for i in cnt]
    with open(os.path.join(out_dir, out_file + '_tree'), 'w') as f:
        f.write(' '.join(fa))
    with open(os.path.join(out_dir, out_file + '_cnt'), 'w') as f:
        f.write(' '.join(cnt))


def property_encode_leonard_opt():
    # 存储路径
    node_file = os.path.join(config.project_root, 'data/preprocess/node_property.csv')
    # edge_file = os.path.join(config.project_root, 'data/preprocess/edge_property.csv')
    out_dir = os.path.join(config.project_root, 'data/encode/')
    node_res = compress_graph_new(node_file)
    # edge_res = compress_graph_new(edge_file)
    # 保存结果
    save_encode_result2(node_res, f'vertex_property_kdtree', out_dir)
    # save_encode_result2(edge_res, f'edge_property', out_dir)


if __name__ == '__main__':
    property_encode_leonard_opt()
