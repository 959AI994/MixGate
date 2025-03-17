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
        self.modalities = modalities
        
        self.hop_tfs = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim * 2, nhead=self.heads, batch_first=True), num_layers=self.depth) for modal in modalities
        ])
        self.lev_tfs = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim * 2, nhead=self.heads, batch_first=True), num_layers=self.depth) for modal in modalities
        ])
        self.graph_tfs = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim * 2, nhead=self.heads, batch_first=True), num_layers=self.depth) for modal in modalities
        ])
        self.mcm_tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim * 2, nhead=self.heads, batch_first=True), num_layers=self.depth)
        self.hop_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2)) 
        self.level_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2))
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
                no_hops = hop_list.size(0)
                
                # 获得hop tokens
                node_tokens = tokens[modal_k]
                hop_tokens = self.hop_nodes.repeat(hop_list.size(0), 1)
                hop_tokens = hop_tokens.unsqueeze(1)
                all_nodes_in_hop = node_tokens[hop_list]
                all_nodes_in_hop = all_nodes_in_hop[:, :max_hop_length, :]
                all_nodes_in_hop = torch.cat([hop_tokens, all_nodes_in_hop], dim=1)
                key_padding_mask = torch.zeros((no_hops, max_hop_length+1), dtype=torch.bool).to(device)
                for i, length in enumerate(hop_length_list):
                    key_padding_mask[i, length + 1:] = True
                output_all_hop_tokens = self.hop_tfs[modal_k](all_nodes_in_hop, src_key_padding_mask=key_padding_mask)
                hop_tokens = output_all_hop_tokens[:, 0, :]
                
                # 获得level tokens
                max_level_length = 0
                hops_in_level_list = []
                for lev in range(max_hop_lev+1): 
                    hops_in_level = hop_tokens[hop_lev_list == lev]
                    if hops_in_level.size(0) == 0:
                        continue
                    if hops_in_level.size(0) > max_level_length:
                        max_level_length = hops_in_level.size(0)
                    hops_in_level_list.append(hops_in_level)
                all_hops_in_level = torch.zeros((len(hops_in_level_list), max_level_length, self.dim * 2)).to(device)
                for i, hops_in_level in enumerate(hops_in_level_list):
                    all_hops_in_level[i, :hops_in_level.size(0), :] = hops_in_level
                level_tokens = self.level_nodes.repeat(len(hops_in_level_list), 1)
                level_tokens = level_tokens.unsqueeze(1)
                all_hops_in_level = torch.cat([level_tokens, all_hops_in_level], dim=1)
                key_padding_mask = torch.zeros((len(hops_in_level_list), max_level_length+1), dtype=torch.bool).to(device)
                for i, hops_in_level in enumerate(hops_in_level_list):
                    key_padding_mask[i, hops_in_level.size(0) + 1:] = True
                output_all_level_tokens = self.lev_tfs[modal_k](all_hops_in_level, src_key_padding_mask=key_padding_mask)
                level_tokens = output_all_level_tokens[:, 0, :]
                
                # 获得graph tokens
                graph_tokens = self.graph_nodes.repeat(1, 1)
                all_levels_in_graph = torch.cat([graph_tokens, level_tokens], dim=0).unsqueeze(0)
                output_all_graph_tokens = self.graph_tfs[modal_k](all_levels_in_graph)
                graph_tokens = output_all_graph_tokens[:, 0, :]
                
                # 一个模态的tokens由多层次组成：hop、level、graph
                modal_tokens = torch.cat([hop_tokens, level_tokens, graph_tokens], dim=0)
                other_modal_tokens = torch.cat([other_modal_tokens, modal_tokens], dim=0)

            batch_all_tokens = torch.cat([batch_masked_tokens, other_modal_tokens], dim=0)
            batch_predicted_tokens = self.mcm_tf(batch_all_tokens)
            batch_pred_masked_tokens = batch_predicted_tokens[:batch_masked_tokens.shape[0], :]
            mcm_predicted_tokens = torch.cat([mcm_predicted_tokens, batch_pred_masked_tokens], dim=0)
            
        return mcm_predicted_tokens