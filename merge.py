import numpy as np
import os

# 文件路径列表
data_paths = [
    ('/home/jwt/1/mig_npz/graphs.npz', '/home/jwt/1/mig_npz/labels.npz', '/home/jwt/DeepMap_Dataset/graph_label_npz/merged_mig_graphs.npz'),
    ('/home/jwt/1/xmg_npz/graphs.npz', '/home/jwt/1/xmg_npz/labels.npz', '/home/jwt/DeepMap_Dataset/graph_label_npz/merged_xmg_graphs.npz'),
    ('/home/jwt/1/xag_npz/graphs.npz', '/home/jwt/1/xag_npz/labels.npz', '/home/jwt/DeepMap_Dataset/graph_label_npz/merged_xag_graphs.npz'),
]

# 合并函数
def merge_graphs_and_labels(graphs_path, labels_path, output_path):
    # 加载 graphs.npz
    graphs_data = np.load(graphs_path, allow_pickle=True)['circuits'].item()
    print(f"Loaded {len(graphs_data)} circuits from {graphs_path}")

    # 加载 labels.npz
    labels_data = np.load(labels_path, allow_pickle=True)['labels'].item()
    print(f"Loaded {len(labels_data)} labels from {labels_path}")

    # 合并数据
    merged_data = {}
    for circuit_name, graph in graphs_data.items():
        # 如果 labels.npz 中有对应的电路名称
        if circuit_name in labels_data:
            graph['prob'] = labels_data[circuit_name]['prob']  # 添加 prob 数据
        else:
            print(f"Warning: No prob data for circuit {circuit_name}")
        merged_data[circuit_name] = graph

    # 保存合并后的数据
    np.savez_compressed(output_path, circuits=merged_data)
    print(f"Merged data saved to {output_path}")

    # 验证合并后的结构
    print("\nValidating merged data structure...")
    merged_data = np.load(output_path, allow_pickle=True)['circuits'].item()
    for circuit_name, graph in merged_data.items():
        keys = list(graph.keys())
        print(f"\nCircuit: {circuit_name}")
        print(f"Keys: {keys}")
        if not all(key in keys for key in ['x', 'edge_index', 'prob']):
            print(f"Warning: Circuit {circuit_name} is missing one or more expected keys (x, edge_index, prob).")
        # 可选: 检查 shape 信息
        if 'x' in graph:
            print(f"  x shape: {graph['x'].shape}")
        if 'edge_index' in graph:
            print(f"  edge_index shape: {graph['edge_index'].shape}")
        if 'prob' in graph:
            print(f"  prob length: {len(graph['prob'])}")

# 处理每组数据
for graphs_path, labels_path, output_path in data_paths:
    if not os.path.exists(graphs_path):
        print(f"Graphs file not found: {graphs_path}")
        continue
    if not os.path.exists(labels_path):
        print(f"Labels file not found: {labels_path}")
        continue

    print(f"\nProcessing:\n  Graphs: {graphs_path}\n  Labels: {labels_path}\n  Output: {output_path}")
    merge_graphs_and_labels(graphs_path, labels_path, output_path)
