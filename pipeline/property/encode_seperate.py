from sentence_transformers import SentenceTransformer
from Levenshtein import editops
import torch.nn.functional as F
from config import device
from tqdm import tqdm
import numpy as np
import config
import struct
import heapq
import torch
import os


# 模型设置
batch_size = 256
window_size = 15
max_distance = 0.5


class UnionFindSet:
    def __init__(self, n):
        # 初始化每个元素的父节点为自身，rank（秩）为0
        self.parent = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, x):
        # 查找元素x所在的集合的代表元
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # 将元素x所在的集合和元素y所在的集合合并
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


def merge_adjacent_operations(edits, s2):
    merged_edits = []
    current_edit = None
    pos = None

    for edit in edits:
        op, i, j = edit
        if current_edit is not None and op == current_edit[0] and (
            (i == pos + 1 and (op == 'replace' or op == 'delete')) or
            (i == pos and op == 'insert')
        ):
            if op == 'insert' or op == 'replace':
                # 插入元素
                current_edit[2] += s2[j]
            else:
                # 删除i-j的元素
                current_edit[2] = i
            if op == 'replace' or op == 'delete':
                pos += 1
        else:
            if current_edit is not None:
                merged_edits.append(current_edit)
            if op == 'insert' or op == 'replace':
                # 插入元素
                current_edit = [op, i, s2[j]]
            else:
                # 删除i-j的元素
                current_edit = [op, i, i]
            pos = i
    if current_edit is not None:
        merged_edits.append(current_edit)
    return merged_edits


def compute_cosine_similarity_matrix(matrix: np.array, window_size):
    """
    计算词向量矩阵中每一条记录与其上下相邻 window_size 个元素的余弦相似度

    参数：
    - matrix: 词向量矩阵，每一行代表一个词向量
    - window_size: 窗口大小，表示每个记录与其上下相邻的记录的数量

    返回：
    一个包含余弦相似度的矩阵，大小为 (len(matrix), window_size*2+1)
    """
    # 取出每个记录与其上下相邻 window_size 个元素的余弦相似度
    result_matrix = np.zeros((len(matrix), window_size * 2 + 1))
    for i in tqdm(range(len(matrix)), desc='Computing cosine similarity'):
        # 计算余弦相似度
        start = max(0, i - window_size)
        end = min(len(matrix), i + window_size+1)
        distance = 1 - np.sum(matrix[start: end] * matrix[i], axis=1)
        if i < window_size:
            pad_width = (window_size - i, 0)
            distance = np.pad(distance, pad_width, mode='constant', constant_values=0.)
        elif i + window_size + 1 > len(matrix):
            pad_width = (0, i + window_size + 1 - len(matrix))
            distance = np.pad(distance, pad_width, mode='constant', constant_values=0.)
        result_matrix[i] = distance
    return result_matrix


def compress_graph(file_path) -> list[list]:
    # 读取节点属性串
    with open(file_path, 'r') as file:
        node_attributes = [line.strip() for line in file]
    del node_attributes[0]
    # 使用transformer计算属性串的向量形式
    transformer = SentenceTransformer('all-MiniLM-L12-v2', device=device)
    # 获得每个属性的词向量
    embeddings = transformer.encode(node_attributes, batch_size=batch_size, show_progress_bar=True,
                                    convert_to_tensor=False, normalize_embeddings=True)
    # 计算滑动窗口距离
    distance = compute_cosine_similarity_matrix(embeddings, window_size)
    # 初始化并查集和代表元编号集合
    n = len(node_attributes)
    visit = set(range(n))
    q = []
    u = UnionFindSet(n)
    ans = []
    # 处理每个节点属性串
    for i in tqdm(range(n), desc=f'window scan'):
        # 计算新的代表元间的距离
        for j in range(2 * window_size + 1):
            p = i + j - window_size
            if p == i:
                continue
            # 代表元之间的距离不能超过window_size
            dij = distance[i, j]
            # 判断距离是否小于阈值
            if dij <= max_distance:
                # 加入优先队列
                heapq.heappush(q, (dij, i, p))
    # 合并相似节点
    bar = tqdm(desc=f'merge nodes', total=len(q))
    while len(q) > 0:
        bar.update()
        dij, i, j = heapq.heappop(q)
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
    v = a[1] - a[0] + 32768
    # f_id = struct.pack('<H', v)
    # file.write(f_id)
    # 统计次数
    cnt1, cnt2 = 0, 0
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
            cnt1 += 2
        else:
            file.write(bytes([0xfe]))
            pos_bin = struct.pack('<H', pos)
            file.write(pos_bin)
            cnt1 += 3
        # 如果是删除，且删除的是一个区间
        if op == 'delete':
            # if pos != value:
            file.write(bytes([value - pos + 1]))
            cnt1 += 1
        else:
            file.write(value.encode())
            cnt2 += len(value)
    # file.write(bytes([0xff]))
    return cnt1, cnt2


def save_encode_result2(compressed, out_file: str, out_dir: str):
    """
    记录编码结果
    :return:
    """
    fa = []
    cnt1 = [0]
    cnt2 = [0]
    with open(os.path.join(out_dir, out_file + '_data'), 'wb') as f:
        for i in compressed:
            r1, r2 = encode_line2(i, f)
            fa.append(i[0] - i[1])
            cnt1.append(r1 + cnt1[-1])
            cnt2.append(r2 + cnt2[-1])
    cnt1.pop(0)
    cnt2.pop(0)
    print(sum(cnt1))
    print(sum(cnt2))
    # fa = [str(i) for i in fa]
    # cnt = [str(i) for i in cnt]
    # with open(os.path.join(out_dir, out_file + '_tree'), 'w') as f:
    #     f.write(' '.join(fa))
    # with open(os.path.join(out_dir, out_file + '_cnt'), 'w') as f:
    #     f.write(' '.join(cnt))


def property_encode():
    # 存储路径
    node_file = os.path.join(config.project_root, 'data/preprocess/node_property.csv')
    edge_file = os.path.join(config.project_root, 'data/preprocess/edge_property.csv')
    out_dir = os.path.join(config.project_root, 'data/encode/')
    node_res = compress_graph(node_file)
    edge_res = compress_graph(edge_file)
    # 保存结果
    save_encode_result2(node_res, 'vertex_property', out_dir)
    save_encode_result2(edge_res, 'edge_property', out_dir)


if __name__ == '__main__':
    property_encode()
