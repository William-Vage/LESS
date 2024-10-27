from numpy.lib.stride_tricks import sliding_window_view
from sentence_transformers import SentenceTransformer
from Levenshtein import editops, distance
from collections import Counter
from config import device
from tqdm import tqdm
import numpy as np
import config
import struct
import heapq
import time
import os


# 模型设置
batch_size = 1024
max_loop_time = 5
max_dis = 100
n_limit = 512000


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


def compute_edit_distance(node_attributes, window_size):
    """
    使用编辑距离计算
    :param node_attributes:
    :param window_size:
    :return: None
    """
    t1 = time.time()
    n = len(node_attributes)
    ans = np.zeros((n, 2 * window_size + 1), dtype=np.short)
    for i in range(n):
        for j in range(2 * window_size + 1):
            p = i + j - window_size
            if 0 <= p < n:
                ans[i][j] = distance(node_attributes[i], node_attributes[p])
    t2 = time.time()
    print('compute distance', t2 - t1)
    return ans


def sliding_window_matrix(matrix, k):
    padded_matrix = np.pad(matrix, np.array([(k, k), (0, 0)]), mode='constant', constant_values=0)
    res = sliding_window_view(padded_matrix, (2 * k + 1, matrix.shape[1]))
    return res.squeeze().transpose(0, 2, 1)


def compute_wordbag_distance(node_attributes, window_size):
    """
    使用字符串哈希计算距离
    :param node_attributes:
    :param window_size:
    :return: None
    """
    n = len(node_attributes)
    enc = np.zeros((n, 256), dtype=np.short)
    t1 = time.time()
    # 遍历node_attributes，同时统计字符计数并更新enc数组
    words = Counter()
    for i, attr in enumerate(node_attributes):
        words.clear()
        words.update(attr)
        for char, count in words.items():
            enc[i, ord(char) - 32] = count
    # print(enc)
    slide = sliding_window_matrix(enc, window_size)
    # 适当调整分块大小
    chunk_size = 1024
    res = np.zeros((enc.shape[0], slide.shape[2]), dtype=np.short)
    num_chunks = (res.shape[0] - 1) // chunk_size + 1
    for i in tqdm(range(num_chunks), desc='compute distance'):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, res.shape[0])
        chunk = enc[start:end, :]
        res[start:end] = np.sum(np.abs(chunk[:, :, np.newaxis] - slide[start:end, :, :]), axis=1)
    # res = np.sum(np.abs(enc[:, :, np.newaxis] - slide, dtype=np.short), axis=1, dtype=np.short)
    t4 = time.time()
    # print(res)
    print('compute distance', t4 - t1)
    return res


def merge_nodes(s1: str, s2: str):
    # 获取编辑操作
    edits = editops(s1, s2)
    # 合并编辑操作
    edits = merge_adjacent_operations(edits, s2)
    return edits


def compress_graph(file, n_limit, method: str, window_size) -> [list[list], bool]:
    # 读取节点属性串
    # with open(file_path, 'r') as file:
    #     node_attributes = [line.strip() for line in file]
    node_attributes = []
    flag = True
    for i in range(n_limit):
        s = file.readline().strip()
        if not s:
            flag = False
            break
        node_attributes.append(s)
    # 初始化并查集和代表元编号集合
    n = len(node_attributes)
    visit = set(range(n))
    u = UnionFindSet(n)
    ans = []
    # 处理每个节点属性串
    q = []
    t = 0
    t1 = time.time()
    while len(visit) > 1 and t < max_loop_time:
        loc = list(visit)
        attr = [node_attributes[i] for i in loc]
        if method == 'transformer':
            # 使用transformer计算属性串的向量形式
            transformer = SentenceTransformer('all-MiniLM-L12-v2', device=device)
            # 获得每个属性的词向量
            embeddings = transformer.encode(attr, batch_size=batch_size, show_progress_bar=True,
                                            convert_to_tensor=False, normalize_embeddings=True)
            # 计算滑动窗口距离
            d = compute_cosine_similarity_matrix(embeddings, window_size)
        elif method == 'edit_distance':
            # 计算任意两个串的编辑距离
            d = compute_edit_distance(attr, window_size)
        elif method == 'word_bag':
            d = compute_wordbag_distance(attr, window_size)
        else:
            raise Exception(f'方法{method}不存在')
        for i in tqdm(range(len(visit)), desc=f'window scan'):
            # 计算新的代表元间的距离
            for j in range(2 * window_size + 1):
                p = i + j - window_size
                if p < 0 or p == i or p > len(visit) - 1:
                    continue
                dij = d[i, j]
                # if dij <= max_distance:
                # 加入优先队列
                heapq.heappush(q, (dij, loc[i], loc[p]))
        # 合并相似节点
        bar = tqdm(desc=f'merge nodes: 1. find neighbours', total=len(q))
        # pos = []
        while len(q) > 0:
            bar.update()
            dij, i, j = heapq.heappop(q)
            # 必须是两个集合的代表元
            if i in visit and j in visit and u.find(i) != u.find(j) and abs(i - j) <= max_dis:
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
        t += 1

    # 插入未替换的元素
    for i in visit:
        ans.append([i, i, [['insert', 0, node_attributes[i]]]])
    # pool.close()
    # pool.join()
    # 将结果排序
    ans.sort(key=lambda x: x[1])
    # print(ans)
    t2 = time.time()
    print(f'merge nodes time cost: {t2 - t1}')
    return ans, flag


def encode_line(a: list, file):
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
    v = 127 + a[0] - a[1]
    f_id = bytes([v])
    file.write(f_id)
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
        else:
            file.write(bytes([0xfe]))
            pos_bin = struct.pack('<H', pos)
            file.write(pos_bin)
        # 如果是删除，且删除的是一个区间
        if op == 'delete':
            diff = value - pos + 1
            if diff <= 0xfa:
                file.write(bytes([diff]))
            else:
                file.write(bytes([0xfe]))
                diff_bin = struct.pack('<H', diff)
                file.write(diff_bin)
        else:
            file.write(value.encode())
    file.write(bytes([0xff]))


m = {'insert': 1, 'replace': 2, 'delete': 3}


def save_encode_result(compressed, out_file: str, out_dir: str):
    """
    记录编码结果
    :return:
    """
    t_start = time.time()
    with open(os.path.join(out_dir, out_file), 'ab') as f:
        for i in tqdm(compressed, desc=f'encode line'):
            encode_line(i, f)
    t_end = time.time()
    print(f'encode line cost: {t_end - t_start}')


def property_encode(encode_vertex: bool, method: str, window_size: int, suffix: str = ''):
    # 存储路径
    out_dir = os.path.join(config.project_root, 'data/encode/')
    edge_file = os.path.join(config.project_root, 'data/preprocess/edge_property.csv')
    with open(edge_file, 'r') as file:
        flag = True
        out_file = f'edge_property{suffix}.txt'
        out_path = os.path.join(out_dir, out_file)
        if os.path.exists(out_path):
            os.remove(out_path)
        # file.readline()
        while flag:
            edge_res, flag = compress_graph(file, n_limit, method, window_size)
            save_encode_result(edge_res, f'edge_property{suffix}.txt', out_dir)
    if encode_vertex:
        node_file = os.path.join(config.project_root, 'data/preprocess/node_property.csv')
        flag = True
        out_file = f'vertex_property{suffix}.txt'
        out_path = os.path.join(out_dir, out_file)
        if os.path.exists(out_path):
            os.remove(out_path)
        with open(node_file, 'r') as file:
            # file.readline()
            while flag:
                node_res, flag = compress_graph(file, n_limit, method, window_size)
                save_encode_result(node_res, out_file, out_dir)


if __name__ == '__main__':
    # 1.Attribute Vectorization（S6）
    print('exp1 edit_distance:')
    property_encode(False, 'edit_distance', 2)
    print('exp1 word_bag:')
    property_encode(False, 'word_bag', 2)
    # 2.Similarity Matrix Computing （S6）
    window_sizes = [5, 10, 20, 30, 40]
    for i in window_sizes:
        print('exp2 wind_size:', i)
        window_size = i
        property_encode(False, 'word_bag', window_size)
    window_size = 15
    # 3.Parameters of Tree Generation
    max_distances = [5, 10, 20, 30, 40]
    for i in max_distances:
        print('exp3 max_distance:', i)
        max_distance = i
        property_encode(False, 'word_bag', window_size)