# coding=utf-8
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..arch.pg_layer import create_spectral_features, MLP, PolarGateConv, restPolarGateConv
import os

class PolarGate(nn.Module):
    def __init__(
            self,
            args,
            node_num: int = 0,
            in_dim: int = 64,
            out_dim: int = 64,
            layer_num: int = 2,
            lamb: float = 5,
            norm_emb: bool = False,
            **kwargs
    ):

        super().__init__(**kwargs)

        self.args = args
        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lamb = lamb

        self.pos_edge_index = None
        self.neg_edge_index = None

        self.x = None

        self.conv1 = PolarGateConv(in_dim, out_dim // 2, first_aggr=True)

        self.convs = torch.nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(
                restPolarGateConv(out_dim // 2, out_dim // 2, first_aggr=False,
                             norm_emb=norm_emb))
        self.weight = torch.nn.Linear(self.out_dim, self.out_dim)
        self.readout_prob = MLP(self.out_dim, self.out_dim, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm',
                                act_layer='relu')

        self.proj = torch.nn.Linear(self.out_dim, self.out_dim * 2)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()

    def get_x_edge_index(self, init_emb, edge_index_s):
        device = next(self.parameters()).device
        self.pos_edge_index = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        self.neg_edge_index = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()
        if init_emb is None:
            init_emb = create_spectral_features(
                pos_edge_index=self.pos_edge_index,
                neg_edge_index=self.neg_edge_index,
                node_num=self.node_num,
                dim=self.in_dim
            ).to(device)
        else:
            init_emb = init_emb
        self.x = init_emb

    def get_feature(self, init_emb, edge_index_s) -> Tuple[Tensor, Tensor]:
        self.get_x_edge_index(init_emb, edge_index_s)
        z = torch.tanh(self.conv1(
            self.x, self.pos_edge_index, self.neg_edge_index))
        for conv in self.convs:
            z = torch.tanh(conv(z, self.pos_edge_index, self.neg_edge_index))
        z = torch.tanh(self.weight(z))

        # prob = self.readout_prob(z)
        # prob = F.sigmoid(prob)

        return z
    
    def forward(self, G):
        device = next(self.parameters()).device
        pos_edge_index = []
        neg_edge_index = []
        not_fanin_list = [-1] * len(G.x)
        for edge in G.edge_index.T:
            src = edge[0].item()
            dst = edge[1].item()
            if G.gate[dst] == 2:
                not_fanin_list[dst] = src
        for edge in G.edge_index.T:
            src = edge[0].item()
            dst = edge[1].item()
            if G.gate[dst] == 1:
                if G.gate[src] != 2:
                    pos_edge_index.append([src, dst])
                else:
                    neg_edge_index.append([not_fanin_list[src], dst])
        pos_edge_index = torch.tensor(pos_edge_index).to(device).T
        neg_edge_index = torch.tensor(neg_edge_index).to(device).T
        
        z = torch.tanh(self.conv1(
            G.x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = torch.tanh(conv(z, pos_edge_index, neg_edge_index))
        z = torch.tanh(self.weight(z))
        state = self.proj(z)
        hs = state[:, :self.out_dim]
        hf = state[:, self.out_dim:]
        
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
        
    def load_pretrained(self, pretrained_model_path = ''):
        if pretrained_model_path == '':
            pretrained_model_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'model.pth')
        self.load(pretrained_model_path)
