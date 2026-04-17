import numpy as np
import torch
from collections import defaultdict

def prepare_equivalent_pairs_data(circuits_data, sat_sweep_results):
    """
    准备等价节点对数据（只处理AIG到其他视图的等价对）
    
    Args:
        circuits_data: 原始电路数据
        sat_sweep_results: SAT Sweeping结果，包含等价节点对信息
    
    Returns:
        更新后的circuits_data，包含等价对信息
    """
    
    for circuit_name in circuits_data:
        if circuit_name in sat_sweep_results:
            # 获取SAT Sweeping的等价对结果
            sat_pairs = sat_sweep_results[circuit_name]
            
            # 初始化等价对字典（只处理AIG到其他视图）
            equivalent_pairs = {
                'aig_to_xmg': [],
                'aig_to_xag': [],
                'aig_to_mig': []
            }
            
            # 处理SAT Sweeping结果，映射到AIG到其他视图的等价对
            for pair in sat_pairs:
                aig_node_id = pair['aig_node']
                xmg_node_id = pair.get('xmg_node', -1)
                xag_node_id = pair.get('xag_node', -1)
                mig_node_id = pair.get('mig_node', -1)
                
                # 只添加AIG到其他视图的等价对
                if xmg_node_id != -1:
                    equivalent_pairs['aig_to_xmg'].append([aig_node_id, xmg_node_id])
                if xag_node_id != -1:
                    equivalent_pairs['aig_to_xag'].append([aig_node_id, xag_node_id])
                if mig_node_id != -1:
                    equivalent_pairs['aig_to_mig'].append([aig_node_id, mig_node_id])
            
            # 将等价对信息添加到电路数据中
            circuits_data[circuit_name]['equivalent_pairs'] = equivalent_pairs
    
    return circuits_data

def create_synthetic_equivalent_pairs(circuits_data, equivalent_ratio=0.1):
    """
    创建合成的等价对数据（用于测试）
    只创建AIG到其他视图的等价对
    
    Args:
        circuits_data: 原始电路数据
        equivalent_ratio: 等价对占节点总数的比例
    
    Returns:
        更新后的circuits_data，包含合成的等价对信息
    """
    
    for circuit_name in circuits_data:
        circuit = circuits_data[circuit_name]
        
        # 获取各视图的节点数量
        aig_nodes = len(circuit['aig_x'])
        xmg_nodes = len(circuit['xmg_x'])
        xag_nodes = len(circuit['xag_x'])
        mig_nodes = len(circuit['mig_x'])
        
        # 初始化等价对字典（只处理AIG到其他视图）
        equivalent_pairs = {
            'aig_to_xmg': [],
            'aig_to_xag': [],
            'aig_to_mig': []
        }
        
        # 为AIG到其他视图创建随机等价对
        view_pairs = [
            ('aig_to_xmg', aig_nodes, xmg_nodes),
            ('aig_to_xag', aig_nodes, xag_nodes),
            ('aig_to_mig', aig_nodes, mig_nodes)
        ]
        
        for pair_name, aig_nodes_count, other_nodes_count in view_pairs:
            num_pairs = int(min(aig_nodes_count, other_nodes_count) * equivalent_ratio)
            if num_pairs > 0:
                # 随机选择等价对
                aig_indices = np.random.choice(aig_nodes_count, num_pairs, replace=False)
                other_indices = np.random.choice(other_nodes_count, num_pairs, replace=False)
                
                # 创建等价对
                pairs = [[int(i), int(j)] for i, j in zip(aig_indices, other_indices)]
                equivalent_pairs[pair_name] = pairs
        
        # 将等价对信息添加到电路数据中
        circuits_data[circuit_name]['equivalent_pairs'] = equivalent_pairs
    
    return circuits_data

def save_equivalent_pairs_data(circuits_data, output_path):
    """
    保存包含等价对信息的数据
    
    Args:
        circuits_data: 包含等价对信息的电路数据
        output_path: 输出文件路径
    """
    np.savez_compressed(output_path, circuits=circuits_data)
    print(f"Saved equivalent pairs data to {output_path}")

if __name__ == "__main__":
    print("Creating synthetic equivalent pairs for testing...")
    
    # 加载原始数据
    # circuits_data = np.load('your_original_data.npz', allow_pickle=True)['circuits'].item()
    
    # 创建合成等价对（用于测试）
    # circuits_data = create_synthetic_equivalent_pairs(circuits_data, equivalent_ratio=0.1)
    
    # 保存更新后的数据
    # save_equivalent_pairs_data(circuits_data, 'data_with_equivalent_pairs.npz')
    
    print("Example script completed. Please modify according to your actual data format.") 