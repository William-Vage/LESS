import pandas as pd

# 加载数据
vertices_df = pd.read_csv('../data/raw/ta1-trace-3-e5-official-1.bin.1_concatenated_vertices.csv', low_memory=False)
edges_df = pd.read_csv('../data/raw/ta1-trace-3-e5-official-1.bin.1_concatenated_edges.csv', low_memory=False)

# 收集所有独特的 id
all_ids = pd.concat([vertices_df['id'], edges_df['subject_id'], edges_df['predicate_id']]).unique()

# 构建全局唯一值字典
global_id_mapping = {val: idx + 1 for idx, val in enumerate(all_ids)}

# 替换 vertices 的 id
vertices_df['id'] = vertices_df['id'].map(global_id_mapping)

# 替换 edges 的 subject_id 和 predicate_id
edges_df['subject_id'] = edges_df['subject_id'].map(global_id_mapping)
edges_df['predicate_id'] = edges_df['predicate_id'].map(global_id_mapping)

# 替换 edges 的 id
edges_df['id'] = range(1, len(edges_df) + 1)

# 保存 vertices.csv
vertices_df.to_csv('../data/preprocess/vertices.csv', index=False)

# 创建 edges_seal.csv
edges_seal_columns = ['id', 'timestampNanos', 'subject_id', 'predicate_id', 'type']
edges_seal_df = edges_df[edges_seal_columns]
edges_seal_df.to_csv('../data/preprocess/edges_seal.csv', index=False)

# 创建 edges_other.csv
edges_other_columns = ['id', 'sequence', 'threadId', 'size', 'flags', 'opm', 'protection', 'signal', 'mode', 'operation']
edges_other_df = edges_df[edges_other_columns]
edges_other_df.to_csv('../data/preprocess/edges_other.csv', index=False)