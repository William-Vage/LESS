import pandas as pd

# 读取第一个 CSV 文件，代表节点信息
nodes_df = pd.read_csv('../../data/vertex200m_hash.csv')
num_nodes = len(nodes_df)

# 创建节点映射，将原始节点 id 映射到新的节点 id（0 到 len(nodes_df)-1）
node_mapping = {original_id: new_id for new_id, original_id in enumerate(nodes_df['hash'])}

# 替换节点 id 列
nodes_df['hash'] = nodes_df['hash'].map(node_mapping)

# 读取第二个 CSV 文件，代表边信息
edges_df = pd.read_csv('../../data/edge200m_hash.csv')

# 替换源节点 id 和目的节点 id 列
edges_df['parentVertexHash'] = edges_df['parentVertexHash'].map(node_mapping)
edges_df['childVertexHash'] = edges_df['childVertexHash'].map(node_mapping)

# 保存结果到文本文件
with open('output_edges.txt', 'w') as file:
    for _, row in edges_df[['parentVertexHash', 'childVertexHash']].iterrows():
        file.write(f"{row['parentVertexHash']},{row['childVertexHash']}\n")

# 打印结果
print("Node Mapping:")
print(node_mapping)
print("\nEdges after Replacement:")
print(edges_df[['parentVertexHash', 'childVertexHash']])
