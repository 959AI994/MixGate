from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from .digae_layer import InnerProductDecoder, DirectedInnerProductDecoder
from .utils.dag_utils import subgraph
from .utils.utils import generate_hs_init
from .arch.mlp import MLP
from .arch.tfmlp import TFMlpAggr

EPS = 1e-15
MAX_LOGSTD = 10

class Model(nn.Module):
    '''
    GCN-VAE Enhanced Graph Neural Network for Circuits
    '''
    def __init__(self, 
                 num_rounds=1, 
                 dim_hidden=128, 
                 enable_encode=True,
                 enable_reverse=False):
        super(Model, self).__init__()
        
        # 原始配置保留
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse
        self.dim_hidden = dim_hidden
        self.dim_mlp = 32

        # GCN编码器
        self.gcn_encoder = GCNConv(dim_hidden, dim_hidden)
        
        # VAE组件
        self.vae_mu = nn.Linear(dim_hidden, dim_hidden)
        self.vae_logstd = nn.Linear(dim_hidden, dim_hidden)
        self.decoder = DirectedInnerProductDecoder()

        # 门类型聚合器（保留原始结构）
        self.aggr_and_strc = TFMlpAggr(dim_hidden, dim_hidden)
        self.aggr_not_strc = TFMlpAggr(dim_hidden, dim_hidden)
        self.aggr_and_func = TFMlpAggr(dim_hidden*2, dim_hidden)
        self.aggr_not_func = TFMlpAggr(dim_hidden, dim_hidden)

        # 门类型更新网络
        self.update_and = nn.Sequential(
            nn.Linear(dim_hidden*2, dim_hidden),
            nn.ReLU()
        )
        self.update_not = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh()
        )

        # 保持原始Readout
        self.readout_prob = MLP(dim_hidden, self.dim_mlp, 1, 
                               num_layer=3, p_drop=0.2, 
                               norm_layer='batchnorm', 
                               act_layer='relu')

    def reparameterize(self, mu, logstd):
        if self.training:
            return mu + torch.exp(logstd.clamp(max=MAX_LOGSTD)) * torch.randn_like(mu)
        else:
            return mu

    def forward(self, G):
        device = next(self.parameters()).device
        num_nodes = len(G.aig_gate)
        
        # 保持原始初始化
        if self.enable_encode:
            hs = generate_hs_init(G, torch.zeros(num_nodes, self.dim_hidden), 
                                 self.dim_hidden, aig=True).to(device)
        else:
            hs = torch.zeros(num_nodes, self.dim_hidden).to(device)
            
        hf = torch.zeros(num_nodes, self.dim_hidden).to(device)
        edge_index = G.aig_edge_index.to(device)
        
        # 门类型掩码
        not_mask = G.aig_gate.squeeze(1) == 2
        and_mask = G.aig_gate.squeeze(1) == 1

        for _ in range(self.num_rounds):
            # GCN编码
            hs_gcn = self.gcn_encoder(hs, edge_index)
            
            # VAE潜在空间
            mu = self.vae_mu(hs_gcn)
            logstd = self.vae_logstd(hs_gcn)
            hs = self.reparameterize(mu, logstd)
            
            # 分层处理保持原始逻辑
            for level in range(1, max(G.aig_forward_level).item()+1):
                layer_mask = G.aig_forward_level == level
                
                # AND门处理
                if torch.any(and_mask & layer_mask):
                    l_and_node = G.aig_forward_index[layer_mask & and_mask]
                    and_edges, _ = subgraph(l_and_node, edge_index, dim=1)
                    
                    # 结构更新
                    msg = self.aggr_and_strc(hs, and_edges)
                    hs_and = self.update_and(torch.cat([hs[l_and_node], msg[l_and_node]], dim=1))
                    hs[l_and_node] = hs_and
                    
                    # 功能更新
                    func_msg = self.aggr_and_func(torch.cat([hs, hf], dim=1), and_edges)
                    hf[l_and_node] = func_msg[l_and_node]

                # NOT门处理
                if torch.any(not_mask & layer_mask):
                    l_not_node = G.aig_forward_index[layer_mask & not_mask]
                    not_edges, _ = subgraph(l_not_node, edge_index, dim=1)
                    
                    # 结构更新
                    msg = self.aggr_not_strc(hs, not_edges)
                    hs_not = self.update_not(msg[l_not_node])
                    hs[l_not_node] = hs_not
                    
                    # 功能更新
                    func_msg = self.aggr_not_func(hf, not_edges)
                    hf[l_not_node] = func_msg[l_not_node]

            # 解码重建
            hf = self.decoder(hs, edge_index)

        return hs, hf

    # 保持原始方法
    def pred_prob(self, hf):
        prob = self.readout_prob(hf)
        return torch.clamp(prob, 0.0, 1.0)

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
                    print(f'Skipping {k}, required shape {model_state_dict[k].shape}, loaded {state_dict[k].shape}')
                    state_dict[k] = model_state_dict[k]
            else:
                print(f'Dropping {k}')
        for k in model_state_dict:
            if k not in state_dict:
                print(f'Initializing {k}')
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        
    def load_pretrained(self, pretrained_model_path=''):
        if not pretrained_model_path:
            pretrained_model_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'model.pth')
        self.load(pretrained_model_path)
