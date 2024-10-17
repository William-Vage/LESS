def golomb_decode(q, r, M):
    return q * M + r + 1

def seal_graph_decompression(compressed_edges, M):
    """
    解压缩已压缩的边列表，恢复原始边列表。

    参数:
    - compressed_edges (list of dict): 压缩后的边列表。
    - M (int): Golomb 编码参数。

    返回:
    - edges (list of dict): 解压缩后的边列表。
    """
    edges = []
    for comp_edge in compressed_edges:
        src_list = comp_edge['src_list']
        dst = comp_edge['dst']
        op = comp_edge['op']
        q0 = comp_edge['q0']
        r0 = int(comp_edge['r0'], 2)
        deltas = comp_edge['deltas']
        edge_id_list = comp_edge['edge_id_list']

        # 解码时间差
        t0 = golomb_decode(q0, r0, M)
        # 重新构建时间戳列表
        timestamps = [t0]
        for delta in deltas[1:]:
            timestamps.append(timestamps[-1] + delta)

        # 构建原始边
        for i in range(len(timestamps)):
            edge = {
                'edge_id': edge_id_list[i],
                'src': src_list[i],
                'dst': dst,
                'op': op,
                'timestamp': timestamps[i]
            }
            edges.append(edge)
    return edges


decompressed = seal_graph_decompression(compressed, M)
# 输出解压后的边
for edge in decompressed:
    print(edge)
