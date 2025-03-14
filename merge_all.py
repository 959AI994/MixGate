import numpy as np
import os

# 文件路径列表
data_paths = [
    ('/home/jwt/DeepGate2_mig/mig_data150/train/graphs150.npz', '/home/jwt/DeepGate2_mig/mig_data150/train/labels150.npz', '/home/jwt/DeepGate2_mig/mig_data150/train/graphs_150.npz'),
    ('/home/jwt/DeepGate2_mig/xmg_data150/train/graphs150.npz', '/home/jwt/DeepGate2_mig/xmg_data150/train/labels150.npz', '/home/jwt/DeepGate2_mig/xmg_data150/train/graphs_150.npz'),
    ('/home/jwt/DeepGate2_mig/xag_data150/train/graphs150.npz', '/home/jwt/DeepGate2_mig/xag_data150/train/labels150.npz', '/home/jwt/DeepGate2_mig/xag_data150/train/graphs_150.npz'),
]

output_paths = ['/home/jwt/DeepGate2_mig/mig_data150/train/graphs_150.npz', '/home/jwt/DeepGate2_mig/xmg_data150/train/graphs_150.npz', '/home/jwt/DeepGate2_mig/xag_data150/train/graphs_150.npz', '/home/jwt/1/aig_npz/small_graphs150.npz']
final_output_path = '/home/jwt/1/150/merged_all150.npz'
# aiggraph_path = ['/home/jwt/1/aig_npz/aig_graphs.npz']

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
            graph['tt_pair_index'] = labels_data[circuit_name]['tt_pair_index']
            graph['tt_dis'] = labels_data[circuit_name]['tt_dis']
            graph['min_tt_dis'] = labels_data[circuit_name]['min_tt_dis']
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

def filter_dict_by_keys_inplace(input_dict, keys_to_keep):
    """
    原地删除字典中不在指定列表中的键。

    参数:
        input_dict (dict): 输入的字典。
        keys_to_keep (list): 需要保留的键列表。
    """
    # 获取字典的所有键
    keys = list(input_dict.keys())
    for key in keys:
        if key not in keys_to_keep:
            del input_dict[key]

def merge_all(output_paths, final_output_path):
    # 初始化合并后的字典
    merged = {}
    circuit = []
    circuit_names = []
    count_aig = 0
    # 加载 graphs.npz
    for i, path in enumerate(output_paths):
        graphs_data = np.load(path, allow_pickle=True)['circuits'].item()
        print(f"Loaded {len(graphs_data)} circuits from {path}")

        # 合并数据
        for circuit_name, graph in graphs_data.items():
            # 如果电路名称不在 merged 中，初始化一个空字典
            if circuit_name not in merged:
                merged[circuit_name] = {}
            if i != 3:
                circuit_names.append(circuit_name)
            # 根据文件类型合并数据
            if i == 0:  # mig
                merged[circuit_name]['mig_x'] = graph['x']
                merged[circuit_name]['mig_edge_index'] = graph['edge_index']
                merged[circuit_name]['mig_prob'] = graph['prob']
                merged[circuit_name]['mig_tt_pair_index'] = graph['tt_pair_index']
                merged[circuit_name]['mig_tt_dis'] = graph['tt_dis']
                merged[circuit_name]['mig_min_tt_dis'] = graph['min_tt_dis']
            elif i == 1:  # xmg
                merged[circuit_name]['xmg_x'] = graph['x']
                merged[circuit_name]['xmg_edge_index'] = graph['edge_index']
                merged[circuit_name]['xmg_prob'] = graph['prob']
                merged[circuit_name]['xmg_tt_pair_index'] = graph['tt_pair_index']
                merged[circuit_name]['xmg_tt_dis'] = graph['tt_dis']
                merged[circuit_name]['xmg_min_tt_dis'] = graph['min_tt_dis']
            elif i == 2:  # xag
                merged[circuit_name]['xag_x'] = graph['x']
                merged[circuit_name]['xag_edge_index'] = graph['edge_index']
                merged[circuit_name]['xag_prob'] = graph['prob']
                merged[circuit_name]['xag_tt_pair_index'] = graph['tt_pair_index']
                merged[circuit_name]['xag_tt_dis'] = graph['tt_dis']
                merged[circuit_name]['xag_min_tt_dis'] = graph['min_tt_dis']
            elif i == 3:  # aig
                circuit_names = tuple(circuit_names)
                if circuit_name in circuit_names:
                    count_aig += 1
                    merged[circuit_name]['aig_x'] = graph['x']
                    merged[circuit_name]['aig_edge_index'] = graph['edge_index']
                    merged[circuit_name]['aig_prob'] = graph['prob']
                    merged[circuit_name]['aig_forward_level'] = graph['forward_level']
                    merged[circuit_name]['aig_backward_level'] = graph['backward_level']
                    merged[circuit_name]['aig_forward_index'] = graph['forward_index']
                    merged[circuit_name]['aig_backward_index'] = graph['backward_index']
                    merged[circuit_name]['aig_gate'] = graph['gate']
                    merged[circuit_name]['aig_tt_sim'] = graph['tt_sim']
                    merged[circuit_name]['aig_tt_pair_index'] = graph['tt_pair_index']

    # 保存合并后的数据
    np.savez_compressed(final_output_path, circuits=merged, allowZip64=True)
    print("aig graphs count = ", count_aig)
    print(f"Final Merged data saved to {final_output_path}")
    merged_data = np.load(final_output_path, allow_pickle=True)['circuits'].item() 
    for circuit_name, graph in merged_data.items():
        if len(merged_data[circuit_name]) != 17:
                # print("circuit_name =", len(circuit_name))
                circuit.append(circuit_name)
    #print("circuit = ", circuit)

    # filter_dict_by_keys_inplace(merged, circuit)

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

merge_all(output_paths, final_output_path)


