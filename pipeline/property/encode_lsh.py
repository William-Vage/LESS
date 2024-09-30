from pipeline.property.encode import UnionFindSet, merge_adjacent_operations
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from Levenshtein import editops
from queue import PriorityQueue
from annoy import AnnoyIndex
from config import device
import numpy as np
import struct
import random
import config
import torch
import annoy
import tqdm
import os


# 模型设置
batch_size = 2048
n_index = 50
max_k = 5
n_dim = 10


def kruskal_lsh(x: np.array, index: annoy.AnnoyIndex):
    """
    使用kdtree辅助kruskal算法构造最小生成树
    :param x: 输入数据 [N, E]
    :param index: lsh模型
    :return: None
    """
    n = len(x)
    q = PriorityQueue()
    s = UnionFindSet(n)
    ans = []
    next_k = np.ones(n, dtype=int) * 3
    # 1.优先队列输入最近邻
    for i in tqdm.tqdm(range(n), desc='Get nearest neighbour'):
        j = index.get_nns_by_vector(x[i], 2, search_k=n_index, include_distances=False)[-1]
        d = 1 - np.dot(x[i], x[j]) + random.random() * 1e-9
        q.put((d, i, j))
    while q.qsize() > 0:
        d, i, j = q.get()
        if s.find(i) != s.find(j):
            s.union(i, j)
            ans.append((i, j))
            # 取得次近邻的距离，并加入到队列
            if next_k[i] <= max_k:
                next_i = index.get_nns_by_vector(x[i], next_k[i], search_k=n_index, include_distances=False)[-1]
                di = 1 - np.dot(x[i], x[next_i]) + random.random() * 1e-9
                q.put((di, i, next_i))
                next_k[i] += 1
            if next_k[j] <= max_k:
                next_j = index.get_nns_by_vector(x[j], next_k[j], include_distances=False)[-1]
                dj = 1 - np.dot(x[j], x[next_j]) + random.random() * 1e-9
                q.put((dj, j, next_j))
                next_k[j] += 1
    # 按首元素从小到大排序，保证拓扑有序
    ans.sort(key=lambda a: a[0])
    return ans


def compress_graph_lsh_vec(file_path) -> list[list[list]]:
    ans = []
    # 读取节点属性串
    with open(file_path, 'r') as file:
        node_attributes = [line.strip() for line in file]
    # 使用transformer计算属性串的向量形式
    transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
    # 获得每个属性的词向量
    embeddings = transformer.encode(node_attributes, batch_size=batch_size, show_progress_bar=True,
                                    convert_to_tensor=False, normalize_embeddings=True)
    # 数据降维
    model = PCA(n_dim)
    embeddings = model.fit_transform(embeddings)
    # 创建 AnnoyIndex 对象
    index = AnnoyIndex(embeddings.shape[1], 'angular')
    # 添加向量到索引中
    for i, vector in enumerate(embeddings):
        index.add_item(i, vector)
    # 构建索引
    index.build(n_index, -1)
    # 构建最小生成树
    tree = kruskal_lsh(embeddings, index)
    # 获取编辑操作，并直接存储0号属性串
    visit = set(range(len(embeddings)))
    for e in tree:
        i, j = e
        # 获取编辑操作
        edits = editops(node_attributes[i], node_attributes[j])
        # 整合编辑操作
        edits = merge_adjacent_operations(edits, node_attributes[j])
        # 记录结果
        ans.append([i, j, edits])
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
    node_res = compress_graph_lsh_vec(node_file)
    # edge_res = compress_graph_lsh_vec(edge_file)
    # 保存结果
    save_encode_result2(node_res, f'vertex_property', out_dir)
    # save_encode_result2(edge_res, f'edge_property', out_dir)


if __name__ == '__main__':
    property_encode_leonard_opt()
