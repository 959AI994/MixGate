from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import forward

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from torch_geometric.nn import GATConv
from torch.nn import Linear, LayerNorm

class GATTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, concat=False, dropout=0.1, ff_hidden_dim=128):
        super(GATTransformerEncoderLayer, self).__init__()
        
        # GAT multi-head attention
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=concat)
        
        # Feed-forward network (FFN)
        self.ffn = torch.nn.Sequential(
            Linear(out_channels*heads if concat else out_channels, ff_hidden_dim),
            torch.nn.ReLU(),
            Linear(ff_hidden_dim, out_channels*heads if concat else out_channels)
        )
        
        # Layer normalization
        self.norm1 = LayerNorm(out_channels*heads if concat else out_channels)
        self.norm2 = LayerNorm(out_channels*heads if concat else out_channels)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # GAT layer with residual connection
        x_residual = x.clone()
        x = self.gat(x, edge_index)
        x = self.dropout(x)
        x = x + x_residual  # Residual connection
        x = self.norm1(x)   # Layer normalization
        
        # Feed-forward network with residual connection
        x_residual = x.clone()
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_residual  # Residual connection
        x = self.norm2(x)   # Layer normalization
        
        return x


# 改为自定义支持注意力输出的版本
class TransformerEncoderWithAttention(nn.TransformerEncoder):
    def forward(self, src, **kwargs):
        attentions = []
        output = src
        final_attn = None
        for mod in self.layers:
            output, attn = mod(output, **kwargs)  # 需要自定义Layer
            final_attn = attn  # 只保留最后一层注意力
            # attentions.append(attn)
            del attn  # 显式释放中间层注意力
        print(f"Final attention shape: {final_attn.shape}")  # 打印最后一层的注意力矩阵形状
        return output, final_attn # attentions[-1]  # 只返回最后一层注意力

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, **kwargs):
        # 重写forward以返回注意力权重
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            average_attn_weights=False
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


class HierarchicalTransformer(nn.Module):
    def __init__(self, args, modalities=['aig', 'xag', 'xmg', 'mig']):
        super(HierarchicalTransformer, self).__init__()
        self.dim = args.dim_hidden
        self.heads = args.tf_head
        self.depth = args.tf_layer
        self.hier_tf_head = args.hier_tf_head
        self.hier_tf_layer = args.hier_tf_layer
        self.max_hop_once = args.max_hop_once
        self.modalities = modalities

        # 新增注意力记录系统
        self.attention_buffer = {}  # 存储注意力权重 {batch_id: tensor}
        self.token_name_buffer = {} # 存储token名称 {batch_id: [name1, name2...]}

        
        self.hop_tfs = nn.ModuleList([
            GATTransformerEncoderLayer(self.dim*2, self.dim*2, heads=self.hier_tf_head, ff_hidden_dim=self.dim*8)
            for i in range(args.hier_tf_layer) for modal in modalities
        ])
        self.lev_tfs = nn.ModuleList([
            GATTransformerEncoderLayer(self.dim*2, self.dim*2, heads=self.hier_tf_head, ff_hidden_dim=self.dim*8)
            for i in range(args.hier_tf_layer) for modal in modalities
        ])
        self.graph_tfs = nn.ModuleList([
            GATTransformerEncoderLayer(self.dim*2, self.dim*2, heads=self.hier_tf_head, ff_hidden_dim=self.dim*8)
            for i in range(args.hier_tf_layer) for modal in modalities
        ])
        # self.mcm_tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim * 2, nhead=self.heads, batch_first=True), num_layers=self.depth)
        # 初始化时使用自定义层

        self.mcm_tf = TransformerEncoderWithAttention(
            CustomTransformerEncoderLayer(
                d_model=self.dim * 2,
                nhead=self.heads,
                batch_first=True
            ),
            num_layers=self.depth
        )

        self.hop_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2)) 
        self.subg_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2))
        self.graph_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2))

        self.reset_buffers()  # 初始化时清空

    def reset_buffers(self):
        self.attention_buffer = {}
        self.token_name_buffer = {}
        self.last_attention = []  # 新增清空逻辑

    def forward(self, g, tokens, masked_tokens,mask_indices,masked_modal='mig' ):
        # 每次前向传播前清空旧缓存
        self.attention_buffer.clear()
        self.token_name_buffer.clear()
        device = next(self.parameters()).device
        mcm_predicted_tokens = torch.zeros(0, self.dim * 2).to(device)
        # 确保 mask_indices 和其他张量与模型在同一设备上
        mask_indices = mask_indices.to(device)

        for batch_id in range(g['aig_batch'].max().item()+1):
            other_modal_tokens = torch.zeros((0, self.dim * 2)).to(device) # 未被mask的modal tokens
            
            # 在拼接batch_all_tokens时记录token含义（关键修改）
            current_names = []
        
            # ========== 处理各模态的层次结构 ==========
            for modal_k, modal in enumerate(self.modalities):
                # 如果被mask的modal，直接跳过
                if modal == masked_modal:
                    # print(f"mask_indices: {mask_indices}")
                    # print(f"g['{}_batch'.format(modal)]: {g['{}_batch'.format(modal)]}")
                    batch_masked_tokens = masked_tokens[g['{}_batch'.format(modal)] == batch_id]
                    
                    # masked_token =  masked_tokens[mask_indices]
                    # masked_token2 =  batch_masked_tokens[mask_indices]
                    # print(f"Masked token: {masked_token}")
                    # print(f"Masked token2: {masked_token2}")

                    continue
                # 选择当前批次的hop，# 获取当前模态的层级信息
                select_hop = (g['{}_batch'.format(modal)][g['{}_hop_node'.format(modal)]] == batch_id)
                hop_list = g['{}_hop'.format(modal)][select_hop]
                hop_lev_list = g['{}_hop_lev'.format(modal)][select_hop]
                hop_length_list = g['{}_hop_length'.format(modal)][select_hop]
                max_hop_lev = hop_lev_list.max().item()
                max_hop_length = hop_length_list.max().item()
                node_tokens = tokens[modal_k]
                all_hop_tokens = torch.zeros((0, self.dim * 2)).to(device)
                all_subg_tokens = torch.zeros((0, self.dim * 2)).to(device)
                
                # ===== 处理每个层级 =====
                for lev in range(max_hop_lev + 1):
                    # 获得 hop tokens
                    hop_flag = hop_lev_list == lev
                    no_hops_in_level = hop_list[hop_flag].size(0)
                    if no_hops_in_level == 0:
                        continue


                    # 处理hop时记录名称
                    hop_names = [f"{modal}_h{lev}_{i}" for i in range(no_hops_in_level)]
                    current_names.extend(hop_names)  # 新增记录


                    level_hop_tokens = torch.zeros((0, self.dim * 2)).to(device)
                    # === 处理当前层级的hops === 为了节约显存，每次最多处理 MAX_HOP_ONCE 个hop
                    for i in range(0, no_hops_in_level, self.max_hop_once):
                        nodes_in_hop = node_tokens[hop_list[hop_flag][i:i+self.max_hop_once]]
                        nodes_in_hop_flatten = torch.zeros((0, self.dim * 2)).to(device)
                        no_hops_once = min(self.max_hop_once, no_hops_in_level - i)
                        # Add the hop-level tokens as the beginning of nodes_in_hop_flatten
                        nodes_in_hop_flatten = torch.cat([self.hop_nodes.repeat(no_hops_once, 1), nodes_in_hop_flatten], dim=0)
                        no_nodes_once = 0
                        hop_attn = []
                        for j, length in enumerate(hop_length_list[hop_flag][i:i+self.max_hop_once]):
                            nodes_in_hop_flatten = torch.cat([nodes_in_hop_flatten, nodes_in_hop[j, :length, :]], dim=0)
                            hop_attn.append([j, j])
                            for k in range(length):
                                hop_attn.append([no_hops_once + no_nodes_once + k, j])
                            no_nodes_once += length  
                        hop_attn = torch.tensor(hop_attn, dtype=torch.long).t().contiguous().to(device)   
                        output_nodes_in_hop = self.hop_tfs[modal_k](nodes_in_hop_flatten, hop_attn)
                        hop_tokens = output_nodes_in_hop[:no_hops_once, :]
                        all_hop_tokens = torch.cat([all_hop_tokens, hop_tokens], dim=0)
                        level_hop_tokens = torch.cat([level_hop_tokens, hop_tokens], dim=0)

                    # 获得 subg tokens
                    hops_in_subg = torch.cat([self.subg_nodes, level_hop_tokens], dim=0)
                    subg_attn = torch.tensor([[i, 0] for i in range(hops_in_subg.size(0))], dtype=torch.long).t().contiguous().to(device)
                    output_subg_tokens = self.lev_tfs[modal_k](hops_in_subg, subg_attn)
                    subg_tokens = output_subg_tokens[0:1, :]
                    
                    # 处理subg时记录名称
                    subg_name = f"{modal}_s{lev}"
                    current_names.append(subg_name)  # 新增记录

                    all_subg_tokens = torch.cat([all_subg_tokens, subg_tokens], dim=0)

                # 处理graph时记录名称
                if all_subg_tokens.size(0) > 0:
                    graph_name = f"{modal}_g"
                    current_names.append(graph_name)  # 新增记录

                # 获得 graph tokens
                subg_in_graph = torch.cat([self.graph_nodes, all_subg_tokens], dim=0)
                graph_attn = torch.tensor([[i, 0] for i in range(subg_in_graph.size(0))], dtype=torch.long).t().contiguous().to(device)
                output_graph_tokens = self.graph_tfs[modal_k](subg_in_graph, graph_attn)
                graph_tokens = output_graph_tokens[0:1, :]
                
                # 一个模态的tokens由多层次组成：hop、level、graph
                modal_tokens = torch.cat([all_hop_tokens, all_subg_tokens, graph_tokens], dim=0)
                other_modal_tokens = torch.cat([other_modal_tokens, modal_tokens], dim=0)
                
            
            # 获取batch_masked_tokens中对应mask_indices的token
            masked_tokens_from_indices = batch_masked_tokens[mask_indices]
            # 获取这些掩码token的数量
            masked_count = masked_tokens_from_indices.shape[0]
            # print("masked_count",masked_count)
            # 首先将被掩码的token单独命名并加入current_names
            current_names += [f"{masked_modal}_masked_{i}" for i in range(masked_count)]
            # print(current_names)
            # 继续处理其他节点并正常编号
            for i in range(len(batch_masked_tokens)):
                if i not in mask_indices:  # 排除已经命名的掩码token
                    current_names.append(f"{masked_modal}_{i}")
            # print(current_names)
            
            batch_all_tokens = torch.cat([batch_masked_tokens, other_modal_tokens], dim=0)
            # print(f"Total batch tokens: {batch_all_tokens.shape[0]}")
            # print(f"Total batch  masked tokens: {batch_masked_tokens.shape[0]}")
            assert len(current_names) == batch_all_tokens.size(0), "名称数量不匹配"
            # batch_predicted_tokens = self.mcm_tf(batch_all_tokens)
            batch_predicted_tokens, batch_attentions = self.mcm_tf(batch_all_tokens)

            # 保存到缓冲区
            self.token_name_buffer[batch_id] = current_names
            self.attention_buffer[batch_id] = batch_attentions
            

            batch_pred_masked_tokens = batch_predicted_tokens[:batch_masked_tokens.shape[0], :]
            mcm_predicted_tokens = torch.cat([mcm_predicted_tokens, batch_pred_masked_tokens], dim=0)
            
        return mcm_predicted_tokens
    

# --------------
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from turtle import forward

# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch import Tensor
# from torchvision.transforms import Compose, Resize, ToTensor
# from torch_geometric.nn import GATConv
# from torch.nn import Linear, LayerNorm
# from .custom_transformer import CustomTransformerEncoder, CustomTransformerEncoderLayer

# class GATTransformerEncoderLayer(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, heads=8, concat=False, dropout=0.1, ff_hidden_dim=128):
#         super(GATTransformerEncoderLayer, self).__init__()
        
#         # GAT multi-head attention
#         self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=concat)
        
#         # Feed-forward network (FFN)
#         self.ffn = torch.nn.Sequential(
#             Linear(out_channels*heads if concat else out_channels, ff_hidden_dim),
#             torch.nn.ReLU(),
#             Linear(ff_hidden_dim, out_channels*heads if concat else out_channels)
#         )
        
#         # Layer normalization
#         self.norm1 = LayerNorm(out_channels*heads if concat else out_channels)
#         self.norm2 = LayerNorm(out_channels*heads if concat else out_channels)
        
#         # Dropout
#         self.dropout = torch.nn.Dropout(dropout)

#     def forward(self, x, edge_index):
#         # GAT layer with residual connection
#         x_residual = x.clone()
#         x = self.gat(x, edge_index)
#         x = self.dropout(x)
#         x = x + x_residual  # Residual connection
#         x = self.norm1(x)   # Layer normalization
        
#         # Feed-forward network with residual connection
#         x_residual = x.clone()
#         x = self.ffn(x)
#         x = self.dropout(x)
#         x = x + x_residual  # Residual connection
#         x = self.norm2(x)   # Layer normalization
        
#         return x

# class HierarchicalTransformer(nn.Module):
#     def __init__(self, args, modalities=['aig', 'xag', 'xmg', 'mig']):
#         super(HierarchicalTransformer, self).__init__()
#         self.dim = args.dim_hidden
#         self.heads = args.tf_head
#         self.depth = args.tf_layer
#         self.hier_tf_head = args.hier_tf_head
#         self.hier_tf_layer = args.hier_tf_layer
#         self.max_hop_once = args.max_hop_once
#         self.modalities = modalities
        
#         self.hop_tfs = nn.ModuleList([
#             GATTransformerEncoderLayer(self.dim*2, self.dim*2, heads=self.hier_tf_head, ff_hidden_dim=self.dim*8)
#             for i in range(args.hier_tf_layer) for modal in modalities
#         ])
#         self.lev_tfs = nn.ModuleList([
#             GATTransformerEncoderLayer(self.dim*2, self.dim*2, heads=self.hier_tf_head, ff_hidden_dim=self.dim*8)
#             for i in range(args.hier_tf_layer) for modal in modalities
#         ])
#         self.graph_tfs = nn.ModuleList([
#             GATTransformerEncoderLayer(self.dim*2, self.dim*2, heads=self.hier_tf_head, ff_hidden_dim=self.dim*8)
#             for i in range(args.hier_tf_layer) for modal in modalities
#         ])
#         # self.mcm_tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim * 2, nhead=self.heads, batch_first=True), num_layers=self.depth)
#         # 修改mcm_tf结构-----------新增⬇️
#         self.mcm_tf = CustomTransformerEncoder(
#             CustomTransformerEncoderLayer(
#                 d_model=self.dim * 2,
#                 nhead=self.heads,
#                 batch_first=True
#             ),
#             num_layers=self.depth
#         )
#         # --------------------------⬆️
#         self.hop_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2)) 
#         self.subg_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2))
#         self.graph_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2))
        
#     def forward(self, g, tokens, masked_tokens, masked_modal='mig'):
#         device = next(self.parameters()).device
#         mcm_predicted_tokens = torch.zeros(0, self.dim * 2).to(device)
        
#         attention_info = []  # 存储注意力信息

#         for batch_id in range(g['mig_batch'].max().item()+1):
#             other_modal_tokens = torch.zeros((0, self.dim * 2)).to(device) # 未被mask的modal tokens
            
#             # ------新增--⬇️
#             # 生成类型信息
#             all_token_types = []
#             # 添加被mask节点的类型
#             # num_masked = batch_masked_tokens.size(0)
#             # all_token_types.extend([(masked_modal, 'node')] * num_masked)
#             # -----------⬆️
#             # 生成类型信息
            

#             for modal_k, modal in enumerate(self.modalities):
#                 # 如果被mask的modal，直接跳过
#                 if modal == masked_modal:
#                     batch_masked_tokens = masked_tokens[g['{}_batch'.format(modal)] == batch_id]
#                     # 添加被mask节点的类型⬇️
#                     num_masked = batch_masked_tokens.size(0)
#                     all_token_types.extend([(masked_modal, 'node')] * num_masked)
#                     # -----------⬆️
#                     continue
#                 # all_token_types = []
#                 all_hop_tokens = torch.zeros((0, self.dim * 2)).to(device)
#                 all_subg_tokens = torch.zeros((0, self.dim * 2)).to(device)
#                 graph_tokens = torch.zeros((0, self.dim * 2)).to(device)

#                 # 记录当前模态的token类型------------------------------⬇️
#                 num_hop = all_hop_tokens.size(0)
#                 num_subg = all_subg_tokens.size(0)
#                 num_graph = graph_tokens.size(0)
#                 all_token_types.extend([(modal, 'hop')] * num_hop)
#                 all_token_types.extend([(modal, 'subg')] * num_subg)
#                 all_token_types.extend([(modal, 'graph')] * num_graph)
#                 # ------------------------------------------------------⬆️

#                 # 选择当前批次的hop
#                 select_hop = (g['{}_batch'.format(modal)][g['{}_hop_node'.format(modal)]] == batch_id)
#                 hop_list = g['{}_hop'.format(modal)][select_hop]
#                 hop_lev_list = g['{}_hop_lev'.format(modal)][select_hop]
#                 hop_length_list = g['{}_hop_length'.format(modal)][select_hop]
#                 max_hop_lev = hop_lev_list.max().item()
#                 max_hop_length = hop_length_list.max().item()
#                 node_tokens = tokens[modal_k]
#                 all_hop_tokens = torch.zeros((0, self.dim * 2)).to(device)
#                 all_subg_tokens = torch.zeros((0, self.dim * 2)).to(device)
#                 for lev in range(max_hop_lev + 1):
#                     # 获得 hop tokens
#                     hop_flag = hop_lev_list == lev
#                     no_hops_in_level = hop_list[hop_flag].size(0)
#                     if no_hops_in_level == 0:
#                         continue
#                     level_hop_tokens = torch.zeros((0, self.dim * 2)).to(device)
#                     # 为了节约显存，每次最多处理 MAX_HOP_ONCE 个hop
#                     for i in range(0, no_hops_in_level, self.max_hop_once):
#                         nodes_in_hop = node_tokens[hop_list[hop_flag][i:i+self.max_hop_once]]
#                         nodes_in_hop_flatten = torch.zeros((0, self.dim * 2)).to(device)
#                         no_hops_once = min(self.max_hop_once, no_hops_in_level - i)
#                         # Add the hop-level tokens as the beginning of nodes_in_hop_flatten
#                         nodes_in_hop_flatten = torch.cat([self.hop_nodes.repeat(no_hops_once, 1), nodes_in_hop_flatten], dim=0)
#                         no_nodes_once = 0
#                         hop_attn = []
#                         for j, length in enumerate(hop_length_list[hop_flag][i:i+self.max_hop_once]):
#                             nodes_in_hop_flatten = torch.cat([nodes_in_hop_flatten, nodes_in_hop[j, :length, :]], dim=0)
#                             hop_attn.append([j, j])
#                             for k in range(length):
#                                 hop_attn.append([no_hops_once + no_nodes_once + k, j])
#                             no_nodes_once += length  
#                         hop_attn = torch.tensor(hop_attn, dtype=torch.long).t().contiguous().to(device)   
#                         output_nodes_in_hop = self.hop_tfs[modal_k](nodes_in_hop_flatten, hop_attn)
#                         hop_tokens = output_nodes_in_hop[:no_hops_once, :]
#                         all_hop_tokens = torch.cat([all_hop_tokens, hop_tokens], dim=0)
#                         level_hop_tokens = torch.cat([level_hop_tokens, hop_tokens], dim=0)
#                     # 获得 subg tokens
#                     hops_in_subg = torch.cat([self.subg_nodes, level_hop_tokens], dim=0)
#                     subg_attn = torch.tensor([[i, 0] for i in range(hops_in_subg.size(0))], dtype=torch.long).t().contiguous().to(device)
#                     output_subg_tokens = self.lev_tfs[modal_k](hops_in_subg, subg_attn)
#                     subg_tokens = output_subg_tokens[0:1, :]
#                     all_subg_tokens = torch.cat([all_subg_tokens, subg_tokens], dim=0)
                    
#                 # 获得 graph tokens
#                 subg_in_graph = torch.cat([self.graph_nodes, all_subg_tokens], dim=0)
#                 graph_attn = torch.tensor([[i, 0] for i in range(subg_in_graph.size(0))], dtype=torch.long).t().contiguous().to(device)
#                 output_graph_tokens = self.graph_tfs[modal_k](subg_in_graph, graph_attn)
#                 graph_tokens = output_graph_tokens[0:1, :]
                
#                 # 一个模态的tokens由多层次组成：hop、level、graph
#                 modal_tokens = torch.cat([all_hop_tokens, all_subg_tokens, graph_tokens], dim=0)
#                 other_modal_tokens = torch.cat([other_modal_tokens, modal_tokens], dim=0)
                
#             batch_all_tokens = torch.cat([batch_masked_tokens, other_modal_tokens], dim=0)
            
#             # 处理注意力权重-----------------------------⬇️
#             batch_all_tokens = batch_all_tokens.unsqueeze(0)  # 添加batch维度
#             batch_predicted_tokens, attn_weights = self.mcm_tf(batch_all_tokens)
#             batch_predicted_tokens = batch_predicted_tokens.squeeze(0)  # 去除batch维度 -> [seq_len, dim]
#             last_layer_attn = attn_weights[-1].mean(dim=1)[0]  # [seq_len, seq_len]
#             attention_info.append({
#                 'attention': last_layer_attn,
#                 'token_types': all_token_types,
#                 'mask_indices': list(range(num_masked))  # 被mask节点的位置
#             })
#             # ------------------------------------------------------⬆️

#             # batch_predicted_tokens = self.mcm_tf(batch_all_tokens)
#             batch_pred_masked_tokens = batch_predicted_tokens[:batch_masked_tokens.shape[0], :]
#             mcm_predicted_tokens = torch.cat([mcm_predicted_tokens, batch_pred_masked_tokens], dim=0)
            
#         return mcm_predicted_tokens, attention_info