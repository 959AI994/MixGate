from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from torch import nn
from torch.nn import GRU
from .utils.dag_utils import subgraph, custom_backward_subgraph
from .arch.mlp import MLP
from .arch.mlp_aggr import MlpAggr
from .arch.tfmlp import TFMlpAggr
from .arch.gcn_conv import AggConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from .digae_layer import DirectedInnerProductDecoder  

EPS        = 1e-15
MAX_LOGSTD = 10

class Model(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits. (XAG)
    This is the integrated version where structural and functional states are combined.
    '''
    def __init__(self, 
                 xag_struct_encoder,  # Structural encoder (from DirectedGAE)
                 num_rounds=1, 
                 dim_hidden=128, 
                 enable_encode=True,
                 enable_reverse=True):
        super(Model, self).__init__()
        
        # 结构编码器 (来自 DirectedGAE)
        self.xag_struct_encoder = xag_struct_encoder
        self.decoder = DirectedInnerProductDecoder()
        self.hs_linear = nn.Linear(dim_hidden * 2, dim_hidden)
        self.hs_decompose = nn.Linear(dim_hidden, dim_hidden * 2)

        # Configuration
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse

        # Dimension settings
        self.dim_hidden = dim_hidden
        self.dim_mlp = 32

        # 功能部分：聚合时输入为 [structure, function]
        self.aggr_and_func = TFMlpAggr(self.dim_hidden * 2, self.dim_hidden)
        self.aggr_not_func = TFMlpAggr(self.dim_hidden * 2, self.dim_hidden)
        self.aggr_xor_func = TFMlpAggr(self.dim_hidden * 2, self.dim_hidden)        
            
        # 更新模块 (只用于更新功能隐藏状态 hf)
        self.update_and_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_xor_func = GRU(self.dim_hidden, self.dim_hidden)

        # Readout
        self.readout_prob = MLP(self.dim_hidden, self.dim_mlp, 1, num_layer=3, 
                                p_drop=0.2, norm_layer='batchnorm', act_layer='relu')


    def forward(self, G):
        device = next(self.parameters()).device
        num_nodes = len(G.xag_gate)
        num_layers_f = max(G.xag_forward_level).item() + 1

        # 使用结构编码器获得结构编码 s 和 t
        x, edge_index = G.xag_x, G.xag_edge_index
        one_hot = torch.nn.functional.one_hot(G.xag_x[:, 1].to(int), num_classes=6).to(device)
        s, t = self.xag_struct_encoder(one_hot, one_hot, edge_index)  # s 为结构信息，t 可用于后续重构

        # 初始化功能隐藏状态 hf (结构信息 s 不再更新)
        hf = torch.zeros(num_nodes, self.dim_hidden, device=device)
        # 初始节点状态为结构信息和功能状态的拼接
        hs = self.hs_linear(torch.cat([s, t], dim=-1))
        node_state = torch.cat([hs, hf], dim=-1)

        # 掩码定义（门类型）
        not_mask = G.xag_gate.squeeze(1) == 2  # NOT 门
        and_mask = G.xag_gate.squeeze(1) == 3  # AND 门
        xor_mask = G.xag_gate.squeeze(1) == 5  # XOR 门

        for _ in range(self.num_rounds):
            for level in range(1, num_layers_f):
                layer_mask = G.xag_forward_level == level

                # AND Gate update
                l_and_node = G.xag_forward_index[layer_mask & and_mask]
                if l_and_node.size(0) > 0:
                    and_edge_index, and_edge_attr = subgraph(l_and_node, edge_index, dim=1)
                    # 此处不更新结构状态，直接用结构编码 hs 参与消息聚合
                    msg = self.aggr_and_func(node_state, and_edge_index, and_edge_attr)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hf_and = torch.index_select(hf, dim=0, index=l_and_node)
                    _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
                    hf[l_and_node, :] = hf_and.squeeze(0)

                # NOT Gate update
                l_not_node = G.xag_forward_index[layer_mask & not_mask]
                if l_not_node.size(0) > 0:
                    not_edge_index, not_edge_attr = subgraph(l_not_node, edge_index, dim=1)
                    msg = self.aggr_not_func(node_state, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hf_not = torch.index_select(hf, dim=0, index=l_not_node)
                    _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                    hf[l_not_node, :] = hf_not.squeeze(0)
                
                # XOR Gate update
                l_xor_node = G.xag_forward_index[layer_mask & xor_mask]
                if l_xor_node.size(0) > 0:
                    xor_edge_index, xor_edge_attr = subgraph(l_xor_node, edge_index, dim=1)
                    msg = self.aggr_xor_func(node_state, xor_edge_index, xor_edge_attr)
                    xor_msg = torch.index_select(msg, dim=0, index=l_xor_node)
                    hf_xor = torch.index_select(hf, dim=0, index=l_xor_node)
                    _, hf_xor = self.update_xor_func(xor_msg.unsqueeze(0), hf_xor.unsqueeze(0))
                    hf[l_xor_node, :] = hf_xor.squeeze(0)
                
                # 更新节点状态：结构信息 hs 不变，hf 更新后重新拼接
                node_state = torch.cat([hs, hf], dim=-1)

        # 返回结构编码 s、t 以及最终的功能隐藏状态 hf
        return hs, hf

    def pred_prob(self, hf):
        prob = self.readout_prob(hf)
        return torch.clamp(prob, min=0.0, max=1.0)
    
    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        
    def load_pretrained(self, pretrained_model_path=''):
        if pretrained_model_path == '':
            pretrained_model_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'model.pth')
        self.load(pretrained_model_path)

    def recon_loss(self, hs, pos_edge_index, neg_edge_index=None):
        s, t = self.hs_decompose(hs).chunk(2, dim=-1)
        pos_pred = self.decoder(s, t, pos_edge_index, sigmoid=True)
        pos_pred_bin = (pos_pred > 0.5).float()
        pos_gt = torch.ones_like(pos_pred)
        pos_loss = -torch.log(pos_pred + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, s.size(0))
        neg_pred = self.decoder(s, t, neg_edge_index, sigmoid=True)
        neg_pred_bin = (neg_pred > 0.5).float()
        neg_gt = torch.zeros_like(neg_pred)
        neg_loss = -torch.log(1 - neg_pred + EPS).mean()
        
        pred_bin = torch.cat([pos_pred_bin, neg_pred_bin], dim=0)
        gt_bin = torch.cat([pos_gt, neg_gt], dim=0)
        pred_bin = pred_bin.int()
        gt_bin = gt_bin.int()

        return pos_loss + neg_loss, pred_bin, gt_bin

