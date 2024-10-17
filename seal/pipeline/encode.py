import json
from tqdm import tqdm
import pandas as pd


# Golomb编码参数，选择合适的M值
M = 1557239724513000000


def golomb_encode(N, M):
    q = (N - 1) // M  
    r = N - q * M - 1
    return q, r


def compress_edges(df, M):
    # 添加一个组合键
    df['key'] = list(zip(df['predicate_id'], df['type']))

    # 使用 groupby 对相同 predicate_id 和 type 的组合进行分组
    grouped = df.groupby('key')

    # 创建压缩边的结果列表
    result = []

    l = 0
    cnt = 0

    # 基于 groupby 结果对数据压缩
    for key, group in tqdm(grouped):
        src_list = group['subject_id'].tolist()
        l += len(src_list)
        cnt += 1
        dst = key[0]
        op = key[1]

        timestamps = group['timestampNanos'].tolist()
        edge_id_list = group['id'].tolist()

        if timestamps:
            t0 = timestamps[0]
            deltas = [0] + [t - t_prev for t_prev, t in zip(timestamps[:-1], timestamps[1:])]
            q0, r0 = golomb_encode(t0, M)
            # r0 = bin(r0)
        else:
            q0 = None
            r0 = None
            deltas = []

        # 追加压缩的边信息到结果中
        result.append([src_list, dst, op, q0, r0, deltas, edge_id_list])

    l /= cnt
    print(l)

    return result


if __name__ == "__main__":
    edges = pd.read_csv('../data/preprocess/edges_seal.csv')

    compressed = compress_edges(edges, M)

    # 输出压缩后的边
    with open('../data/encode/encode_seal.json', 'w') as f:
        json.dump(compressed, f)
