import torch
import torch.nn as nn
import numpy as np

class MultiViewAlignmentLoss(nn.Module):
    """
    多视图对齐损失函数
    使用L1损失来对齐AIG到其他视图的等价节点对的功能隐藏状态（hf）
    支持使用parse_pair.py中的等价标签数据
    """
    
    def __init__(self, loss_weight=1.0):
        super(MultiViewAlignmentLoss, self).__init__()
        self.loss_weight = loss_weight
        self.l1_loss = nn.L1Loss(reduction='mean')
        
    def forward(self, graph, hf_dict):
        """
        计算多视图对齐损失
        
        Args:
            graph: 包含等价对信息的图对象，包含以下等价标签：
                - graph.aig_mig_equ: AIG到MIG的等价对（AIG节点索引）
                - graph.mig_aig_equ: MIG到AIG的等价对（MIG节点索引）
                - graph.aig_xmg_equ: AIG到XMG的等价对（AIG节点索引）
                - graph.xmg_aig_equ: XMG到AIG的等价对（XMG节点索引）
                - graph.aig_xag_equ: AIG到XAG的等价对（AIG节点索引）
                - graph.xag_aig_equ: XAG到AIG的等价对（XAG节点索引）
            hf_dict: 包含各视图功能隐藏状态的字典
                {
                    'aig': aig_hf,  # shape: [num_aig_nodes, hidden_dim]
                    'xmg': xmg_hf,   # shape: [num_xmg_nodes, hidden_dim] 
                    'xag': xag_hf,   # shape: [num_xag_nodes, hidden_dim]
                    'mig': mig_hf    # shape: [num_mig_nodes, hidden_dim]
                }
        
        Returns:
            total_loss: 总的对齐损失
            loss_dict: 各视图对的对齐损失详情
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 定义视图对映射（AIG节点索引，其他视图节点索引）
        view_pairs = [
            ('aig', 'mig', 'aig_mig_equ', 'mig_aig_equ'),
            ('aig', 'xmg', 'aig_xmg_equ', 'xmg_aig_equ'), 
            ('aig', 'xag', 'aig_xag_equ', 'xag_aig_equ')
        ]
        
        # 添加调试信息
        debug_info = {}
        
        for view1, view2, equ_key1, equ_key2 in view_pairs:
            # 获取等价对数据
            equivalent_indices1 = getattr(graph, equ_key1, None)  # AIG节点索引
            equivalent_indices2 = getattr(graph, equ_key2, None)  # 其他视图节点索引
            
            # 调试信息
            debug_info[f'{equ_key1}_exists'] = equivalent_indices1 is not None
            debug_info[f'{equ_key2}_exists'] = equivalent_indices2 is not None
            if equivalent_indices1 is not None:
                debug_info[f'{equ_key1}_len'] = len(equivalent_indices1)
                debug_info[f'{equ_key1}_type'] = type(equivalent_indices1).__name__
                debug_info[f'{equ_key1}_sample'] = equivalent_indices1[:3] if len(equivalent_indices1) > 0 else []
            if equivalent_indices2 is not None:
                debug_info[f'{equ_key2}_len'] = len(equivalent_indices2)
                debug_info[f'{equ_key2}_type'] = type(equivalent_indices2).__name__
                debug_info[f'{equ_key2}_sample'] = equivalent_indices2[:3] if len(equivalent_indices2) > 0 else []
            
            if equivalent_indices1 is not None and equivalent_indices2 is not None and \
               len(equivalent_indices1) > 0 and len(equivalent_indices2) > 0:
                
                # 获取对应的功能隐藏状态
                hf1 = hf_dict.get(view1)  # AIG的hf
                hf2 = hf_dict.get(view2)  # 其他视图的hf
                
                debug_info[f'{view1}_hf_exists'] = hf1 is not None
                debug_info[f'{view2}_hf_exists'] = hf2 is not None
                if hf1 is not None:
                    debug_info[f'{view1}_hf_shape'] = hf1.shape
                if hf2 is not None:
                    debug_info[f'{view2}_hf_shape'] = hf2.shape
                
                if hf1 is not None and hf2 is not None:
                    try:
                        # 确保等价对索引是tensor格式，并处理数据不一致的问题
                        if not isinstance(equivalent_indices1, torch.Tensor):
                            # 检查数据一致性
                            if isinstance(equivalent_indices1, (list, tuple)):
                                # 处理numpy数组列表的情况
                                if len(equivalent_indices1) > 0 and hasattr(equivalent_indices1[0], '__len__'):
                                    # 如果是numpy数组列表，取第一个数组
                                    equivalent_indices1 = equivalent_indices1[0]
                                # 确保所有元素都是标量
                                equivalent_indices1 = [int(idx) if isinstance(idx, (int, float, np.integer)) else idx for idx in equivalent_indices1]
                                # 过滤掉无效值
                                equivalent_indices1 = [idx for idx in equivalent_indices1 if isinstance(idx, int) and idx >= 0]
                            equivalent_indices1 = torch.tensor(equivalent_indices1, dtype=torch.long, device=hf1.device)
                        
                        if not isinstance(equivalent_indices2, torch.Tensor):
                            # 检查数据一致性
                            if isinstance(equivalent_indices2, (list, tuple)):
                                # 处理numpy数组列表的情况
                                if len(equivalent_indices2) > 0 and hasattr(equivalent_indices2[0], '__len__'):
                                    # 如果是numpy数组列表，取第一个数组
                                    equivalent_indices2 = equivalent_indices2[0]
                                # 确保所有元素都是标量
                                equivalent_indices2 = [int(idx) if isinstance(idx, (int, float, np.integer)) else idx for idx in equivalent_indices2]
                                # 过滤掉无效值
                                equivalent_indices2 = [idx for idx in equivalent_indices2 if isinstance(idx, int) and idx >= 0]
                            equivalent_indices2 = torch.tensor(equivalent_indices2, dtype=torch.long, device=hf2.device)
                        
                        debug_info[f'{equ_key1}_tensor_shape'] = equivalent_indices1.shape
                        debug_info[f'{equ_key2}_tensor_shape'] = equivalent_indices2.shape
                        debug_info[f'{equ_key1}_tensor_sample'] = equivalent_indices1[:3].tolist()
                        debug_info[f'{equ_key2}_tensor_sample'] = equivalent_indices2[:3].tolist()
                        
                        # 确保两个索引列表长度一致
                        min_len = min(len(equivalent_indices1), len(equivalent_indices2))
                        if min_len == 0:
                            loss_dict[f'{view1}_to_{view2}_hf_alignment_loss'] = 0.0
                            loss_dict[f'{view1}_to_{view2}_num_pairs'] = 0
                            continue
                        
                        equivalent_indices1 = equivalent_indices1[:min_len]
                        equivalent_indices2 = equivalent_indices2[:min_len]
                        
                        # 确保索引在有效范围内
                        valid_indices1 = (equivalent_indices1 < len(hf1)).all()
                        valid_indices2 = (equivalent_indices2 < len(hf2)).all()
                        
                        debug_info[f'{equ_key1}_valid'] = valid_indices1.item()
                        debug_info[f'{equ_key2}_valid'] = valid_indices2.item()
                        debug_info[f'{equ_key1}_max_idx'] = equivalent_indices1.max().item()
                        debug_info[f'{equ_key2}_max_idx'] = equivalent_indices2.max().item()
                        debug_info[f'{view1}_hf_len'] = len(hf1)
                        debug_info[f'{view2}_hf_len'] = len(hf2)
                        
                        if valid_indices1 and valid_indices2:
                            # 获取等价节点的功能隐藏状态
                            hf1_equivalent = hf1[equivalent_indices1]  # shape: [num_pairs, hidden_dim]
                            hf2_equivalent = hf2[equivalent_indices2]  # shape: [num_pairs, hidden_dim]
                            
                            debug_info[f'{view1}_hf_equiv_shape'] = hf1_equivalent.shape
                            debug_info[f'{view2}_hf_equiv_shape'] = hf2_equivalent.shape
                            
                            # 计算功能隐藏状态对齐损失（L1损失）
                            hf_alignment_loss = self.l1_loss(hf1_equivalent, hf2_equivalent)
                            
                            debug_info[f'{view1}_to_{view2}_loss'] = hf_alignment_loss.item()
                            
                            total_loss += hf_alignment_loss
                            loss_dict[f'{view1}_to_{view2}_hf_alignment_loss'] = hf_alignment_loss.item()
                            loss_dict[f'{view1}_to_{view2}_num_pairs'] = len(equivalent_indices1)
                        else:
                            print(f"Warning: Invalid indices in {equ_key1} or {equ_key2}, skipping this pair")
                            loss_dict[f'{view1}_to_{view2}_hf_alignment_loss'] = 0.0
                            loss_dict[f'{view1}_to_{view2}_num_pairs'] = 0
                    except Exception as e:
                        print(f"Error processing {equ_key1}/{equ_key2}: {e}")
                        debug_info[f'{view1}_to_{view2}_error'] = str(e)
                        loss_dict[f'{view1}_to_{view2}_hf_alignment_loss'] = 0.0
                        loss_dict[f'{view1}_to_{view2}_num_pairs'] = 0
                else:
                    loss_dict[f'{view1}_to_{view2}_hf_alignment_loss'] = 0.0
                    loss_dict[f'{view1}_to_{view2}_num_pairs'] = 0
            else:
                loss_dict[f'{view1}_to_{view2}_hf_alignment_loss'] = 0.0
                loss_dict[f'{view1}_to_{view2}_num_pairs'] = 0
        
        # 打印调试信息（只在第一次调用时）
        if not hasattr(self, '_debug_printed'):
            print("Alignment Loss Debug Info:")
            for key, value in debug_info.items():
                print(f"  {key}: {value}")
            self._debug_printed = True
        
        return total_loss * self.loss_weight, loss_dict 