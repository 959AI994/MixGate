import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from ..arch.gcn_conv import AggConv
from ..arch.mlp import MLP

class MultiGCNEncoder(nn.Module): 
    def __init__(self, num_rounds, dim_hidden, dim_feature, enable_reverse, layernorm):
        super(MultiGCNEncoder, self).__init__()
        
        # configuration
        self.num_rounds = num_rounds
        self.enable_reverse = enable_reverse 
        self.layernorm = layernorm

        # dimensions
        self.dim_hidden = dim_hidden

        # Network 
        self.aggr = AggConv(self.dim_hidden*1, self.dim_hidden)
        self.update = nn.GRU(self.dim_hidden + dim_feature, self.dim_hidden)
        if self.enable_reverse:
            self.aggr_r = AggConv(self.dim_hidden*1, self.dim_hidden)
            self.update_r = nn.GRU(self.dim_hidden + dim_feature, self.dim_hidden)
        if self.layernorm:
            self.ln = nn.LayerNorm(self.dim_hidden)
        
            
    def forward(self, x, edge_index):
        device = next(self.parameters()).device
        num_nodes = len(x)
        node_state = torch.ones(1, num_nodes, self.dim_hidden).to(device)
        r_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        
        for _ in range(self.num_rounds):
            msg = self.aggr(node_state, edge_index)
            _, node_state = self.update(torch.cat([msg, x.unsqueeze(0)], dim=-1), node_state)
            if self.layernorm:
                node_state = self.ln(node_state)
            if self.enable_reverse:
                msg = self.aggr_r(node_state, r_edge_index)
                _, node_state = self.update_r(torch.cat([msg, x.unsqueeze(0)], dim=-1), node_state)
                if self.layernorm:
                    node_state = self.ln(node_state)
            
        return node_state.squeeze(0)

class DirectMultiGCNEncoder(nn.Module):
    def __init__(self, 
                 dim_feature = 3, 
                 dim_hidden = 128, 
                 s_rounds = 4, 
                 t_rounds = 1, 
                 enable_reverse = True,
                 layernorm = False 
                ):
        super(DirectMultiGCNEncoder, self).__init__()
        self.dim_hidden = dim_hidden
        self.source_conv = MultiGCNEncoder(s_rounds, dim_hidden*2, dim_feature, enable_reverse, layernorm)
        self.target_conv = MultiGCNEncoder(t_rounds, dim_hidden*2, dim_feature, enable_reverse, layernorm)
        
        self.readout_prob = MLP(128, 32, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.linear = nn.Linear(dim_hidden * 2, dim_hidden * 2)
        
    def forward(self, g):
        s = g.x
        edge_index = g.edge_index
        s = self.source_conv(s, edge_index)
        # t = self.target_conv(t, edge_index)
        state = self.linear(s)
        hs = state[:, :self.dim_hidden]
        hf = state[:, self.dim_hidden:]
        
        return hs, hf
    
    def pred_prob(self, hf):
        prob = self.readout_prob(hf)
        prob = torch.clamp(prob, min=0.0, max=1.0)
        return prob
        
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