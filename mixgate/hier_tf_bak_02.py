from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import forward

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor

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
        
        self.hop_tfs = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim * 2, nhead=self.hier_tf_head, batch_first=True), num_layers=self.hier_tf_layer) for modal in modalities
        ])
        self.lev_tfs = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim * 2, nhead=self.hier_tf_head, batch_first=True), num_layers=self.hier_tf_layer) for modal in modalities
        ])
        self.graph_tfs = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim * 2, nhead=self.hier_tf_head, batch_first=True), num_layers=self.hier_tf_layer) for modal in modalities
        ])
        self.mcm_tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim * 2, nhead=self.heads, batch_first=True), num_layers=self.depth)
        self.hop_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2)) 
        self.subg_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2))
        self.graph_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2))
        
    def forward(self, g, tokens, masked_tokens, masked_modal='aig'):
        device = next(self.parameters()).device
        mcm_predicted_tokens = torch.zeros(0, self.dim * 2).to(device)
        for batch_id in range(g['aig_batch'].max().item()+1):
            other_modal_tokens = torch.zeros((0, self.dim * 2)).to(device) # 未被mask的modal tokens
            for modal_k, modal in enumerate(self.modalities):
                # 如果被mask的modal，直接跳过
                if modal == masked_modal:
                    batch_masked_tokens = masked_tokens[g['{}_batch'.format(modal)] == batch_id]
                    continue
                # 选择当前批次的hop
                select_hop = (g['{}_batch'.format(modal)][g['{}_hop_node'.format(modal)]] == batch_id)
                hop_list = g['{}_hop'.format(modal)][select_hop]
                hop_lev_list = g['{}_hop_lev'.format(modal)][select_hop]
                hop_length_list = g['{}_hop_length'.format(modal)][select_hop]
                max_hop_lev = hop_lev_list.max().item()
                max_hop_length = hop_length_list.max().item()
                node_tokens = tokens[modal_k]
                all_hop_tokens = torch.zeros((0, self.dim * 2)).to(device)
                all_subg_tokens = torch.zeros((0, self.dim * 2)).to(device)
                for lev in range(max_hop_lev + 1):
                    # 获得 hop tokens
                    hop_flag = hop_lev_list == lev
                    no_hops_in_level = hop_list[hop_flag].size(0)
                    if no_hops_in_level == 0:
                        continue
                    level_hop_tokens = torch.zeros((0, self.dim * 2)).to(device)
                    # 为了节约显存，每次最多处理 MAX_HOP_ONCE 个hop
                    for i in range(0, no_hops_in_level, self.max_hop_once):
                        hop_tokens = self.hop_nodes.repeat(min(self.max_hop_once, no_hops_in_level - i), 1)
                        hop_tokens = hop_tokens.unsqueeze(1)
                        nodes_in_hop = node_tokens[hop_list[hop_flag][i:i+self.max_hop_once]]
                        max_hop_length = hop_length_list[hop_flag][i:i+self.max_hop_once].max().item()
                        nodes_in_hop = nodes_in_hop[:, :max_hop_length, :]
                        nodes_in_hop = torch.cat([hop_tokens, nodes_in_hop], dim=1)
                        key_padding_mask = torch.zeros((min(self.max_hop_once, no_hops_in_level - i), max_hop_length+1), dtype=torch.bool).to(device)
                        for j, length in enumerate(hop_length_list[hop_flag][i:i+self.max_hop_once]):
                            key_padding_mask[j, length + 1:] = True
                        output_all_hop_tokens = self.hop_tfs[modal_k](nodes_in_hop, src_key_padding_mask=key_padding_mask)
                        # print(nodes_in_hop.shape)
                        hop_tokens = output_all_hop_tokens[:, 0, :]
                        all_hop_tokens = torch.cat([all_hop_tokens, hop_tokens], dim=0)
                        level_hop_tokens = torch.cat([level_hop_tokens, hop_tokens], dim=0)
                        # del nodes_in_hop, key_padding_mask, output_all_hop_tokens   # TODO: 释放显存是否有效？会影响梯度回传？
                    # 获得 subg tokens
                    subg_tokens = self.subg_nodes.repeat(1, 1)
                    hops_in_subg = torch.cat([subg_tokens, level_hop_tokens], dim=0).unsqueeze(0)
                    output_all_subg_tokens = self.lev_tfs[modal_k](hops_in_subg)
                    subg_tokens = output_all_subg_tokens[:, 0, :]
                    all_subg_tokens = torch.cat([all_subg_tokens, subg_tokens], dim=0)
                    
                # 获得 graph tokens
                graph_tokens = self.graph_nodes.repeat(1, 1)
                subg_in_graph = torch.cat([graph_tokens, all_subg_tokens], dim=0).unsqueeze(0)
                output_all_graph_tokens = self.graph_tfs[modal_k](subg_in_graph)
                graph_tokens = output_all_graph_tokens[:, 0, :]
                
                # 一个模态的tokens由多层次组成：hop、level、graph
                modal_tokens = torch.cat([all_hop_tokens, all_subg_tokens, graph_tokens], dim=0)
                other_modal_tokens = torch.cat([other_modal_tokens, modal_tokens], dim=0)
                
            batch_all_tokens = torch.cat([batch_masked_tokens, other_modal_tokens], dim=0)
            batch_predicted_tokens = self.mcm_tf(batch_all_tokens)
            batch_pred_masked_tokens = batch_predicted_tokens[:batch_masked_tokens.shape[0], :]
            mcm_predicted_tokens = torch.cat([mcm_predicted_tokens, batch_pred_masked_tokens], dim=0)
            
        return mcm_predicted_tokens