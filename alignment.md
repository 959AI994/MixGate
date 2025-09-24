# 多视图对齐机制 (Multi-View Alignment)

## 概述

实现了AIG到其他三个视图（XMG、XAG、MIG）的等价节点对对齐机制，使用L1损失来对齐等价节点对的功能隐藏状态（hf）。

## 数据结构

### 1. 等价对数据结构

在 `parse_pair.py` 中，每个电路图包含以下等价对信息：

```python
# 等价对索引 (shape: [num_pairs, 2])
graph[f'{view_pair}_equivalent_indices'] = pair_indices
# 等价对数量
graph[f'{view_pair}_num_equivalent_pairs'] = torch.tensor(len(pairs), dtype=torch.long)
```

其中 `view_pair` 可以是：
- `aig_to_xmg`: AIG到XMG的等价对
- `aig_to_xag`: AIG到XAG的等价对  
- `aig_to_mig`: AIG到MIG的等价对

### 2. 功能隐藏状态 (hf)

每个视图的编码器返回：
```python
# 从各编码器获取hf
aig_hs, aig_hf = self.deepgate_aig(G)  # AIG的结构和功能状态
mig_hs, mig_hf = self.deepgate_mig(G)  # MIG的结构和功能状态
xmg_hs, xmg_hf = self.deepgate_xmg(G)  # XMG的结构和功能状态
xag_hs, xag_hf = self.deepgate_xag(G)  # XAG的结构和功能状态
```

其中 `hf` 是功能隐藏状态，维度为 `[num_nodes, hidden_dim]`。

## 对齐策略

### 1. 等价对识别

使用SAT Sweeping技术识别等价节点对：
- 输入：AIG电路和其他视图电路
- 输出：等价节点对列表 `[(aig_node_idx, other_node_idx), ...]`
- 判断标准：真值表等价性

### 2. 对齐目标

只对齐AIG到其他三个视图的等价节点对：
- AIG ↔ XMG 等价节点对
- AIG ↔ XAG 等价节点对  
- AIG ↔ MIG 等价节点对

### 3. 对齐内容

**只对齐功能隐藏状态（hf）**：
- 不包含真值表（tt）对齐
- 不包含概率（prob）对齐
- 专注于功能隐藏状态的对齐

## 损失函数

### 1. 对齐损失定义

```python
class MultiViewAlignmentLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.l1_loss = nn.L1Loss(reduction='mean')
    
    def forward(self, graph, hf_dict):
        total_loss = 0.0
        loss_dict = {}
        
        # 只处理AIG到其他视图的等价对
        view_pairs = ['aig_to_xmg', 'aig_to_xag', 'aig_to_mig']
        
        for view_pair in view_pairs:
            equivalent_indices = getattr(graph, f'{view_pair}_equivalent_indices', None)
            num_pairs = getattr(graph, f'{view_pair}_num_equivalent_pairs', torch.tensor(0))
            
            if equivalent_indices is not None and num_pairs > 0:
                view1, view2 = view_pair.split('_to_')
                hf1 = hf_dict.get(view1)  # AIG的hf
                hf2 = hf_dict.get(view2)  # 其他视图的hf
                
                if hf1 is not None and hf2 is not None:
                    # 提取等价节点的功能隐藏状态
                    node1_indices = equivalent_indices[:, 0]  # AIG节点索引
                    node2_indices = equivalent_indices[:, 1]  # 其他视图节点索引
                    
                    hf1_equivalent = hf1[node1_indices]  # shape: [num_pairs, hidden_dim]
                    hf2_equivalent = hf2[node2_indices]  # shape: [num_pairs, hidden_dim]
                    
                    # 计算功能隐藏状态对齐损失（L1损失）
                    hf_alignment_loss = self.l1_loss(hf1_equivalent, hf2_equivalent)
                    
                    total_loss += hf_alignment_loss
                    loss_dict[f'{view_pair}_hf_alignment_loss'] = hf_alignment_loss.item()
        
        return total_loss * self.loss_weight, loss_dict
```

### 2. 损失计算

对于每个等价对 `(aig_node_idx, other_node_idx)`：

```python
# 获取等价节点的功能隐藏状态
aig_hf_equivalent = aig_hf[aig_node_idx]      # shape: [hidden_dim]
other_hf_equivalent = other_hf[other_node_idx] # shape: [hidden_dim]

# 计算L1损失
hf_alignment_loss = L1_loss(aig_hf_equivalent, other_hf_equivalent)
```

### 3. 总损失

```python
# 在训练循环中
total_loss = prob_loss * weight[0] + func_loss * weight[2] + alignment_loss
```

## 数据准备

### 1. 等价对数据格式

在NPZ文件中添加等价对信息：（TODO:@wentao）

```python
circuits[circuit_name]["equivalent_pairs"] = {
    "aig_to_xmg": [(aig_idx, xmg_idx), ...],
    "aig_to_xag": [(aig_idx, xag_idx), ...], 
    "aig_to_mig": [(aig_idx, mig_idx), ...]
}
```

### 2. 数据准备脚本

使用 `prepare_equivalent_pairs.py` 准备等价对数据：

```python
# 创建合成等价对（用于测试）（TODO:@wentao）
create_synthetic_equivalent_pairs(circuits_data, equivalent_ratio=0.1)

# 处理真实SAT Sweeping结果（TODO:@wentao）
prepare_equivalent_pairs_data(circuits_data, sat_sweep_results)
```

## 模型集成

### 1. 模型修改

在 `top_model.py` 中返回hf：

```python
def forward(self, G):
    # ... 现有代码 ...
    return mcm_predicted_tokens, mask_indices, selected_tokens, masked_prob, \
           aig_prob, mig_prob, xmg_prob, xag_prob, \
           aig_hf, mig_hf, xmg_hf, xag_hf  # 新增hf返回
```

### 2. 训练器集成

在 `top_trainer.py` 中计算对齐损失：

```python
def run_batch(self, batch):
    # 获取模型输出，包括hf
    mcm_pm_tokens, mask_indices, pm_tokens, pm_prob, \
    aig_prob, mig_prob, xmg_prob, xag_prob, \
    aig_hf, mig_hf, xmg_hf, xag_hf = self.model(batch)
    
    # 构建hf字典
    hf_dict = {
        'aig': aig_hf,
        'mig': mig_hf, 
        'xmg': xmg_hf,
        'xag': xag_hf
    }
    
    # 计算对齐损失
    alignment_loss, alignment_loss_dict = self.alignment_loss(batch, hf_dict)
    
    # 添加到总损失
    total_loss = ... + alignment_loss
```

## 训练监控

### 1. 损失显示

训练过程中显示对齐损失：
```
HF_Align: 0.0000  # 当没有等价对时为0
```

### 2. 日志记录

每个epoch记录详细的对齐损失：
```
Epoch: 0/60 |Prob: 0.1240 |Func: 0.8097 |HF_Align: 0.0000
```

## 使用说明

### 1. 准备等价对数据（暂定）

```python
# 使用SAT Sweeping找到等价对
sat_results = run_sat_sweeping(aig_circuit, other_circuits)

# 准备数据
prepare_equivalent_pairs_data(circuits_data, sat_results)

# 保存数据
save_equivalent_pairs_data(circuits_data, output_path)
```

### 2. 训练模型

```bash
torchrun --nproc_per_node=1 --master_port=29921 train_mask.py \
    --exp_id 01_deepgate2_0.00 \
    --batch_size 8 \
    --num_epochs 60 \
    --mask_ratio 0.00 \
    --gpus 0 \
    --hier_tf \
    --aig_encoder dg2
```

### 3. 监控对齐效果

- 观察 `HF_Align` 损失值的变化
- 当有等价对数据时，损失值应该逐渐下降
- 对齐损失为0表示没有等价对数据

## 优势

1. **专注功能对齐**：只对齐功能隐藏状态，避免复杂的多目标优化
2. **L1损失**：使用L1损失确保等价节点对的hf尽可能相似
3. **高效实现**：直接使用模型返回的hf，避免重复计算
4. **灵活扩展**：可以轻松添加更多视图的对齐

## 注意事项

1. **等价对质量**：SAT Sweeping的准确性直接影响对齐效果
2. **数据平衡**：确保各视图的等价对数量相对平衡
3. **损失权重**：需要调整对齐损失的权重以平衡其他损失
4. **梯度传播**：确保hf保持梯度以支持反向传播
