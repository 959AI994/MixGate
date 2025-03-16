import torch
import os
from torch import nn
from torch_geometric.nn import GCNConv
from .digae_layer import DirectedInnerProductDecoder
from .utils.dag_utils import subgraph
from .utils.utils import generate_hs_init
from .arch.mlp import MLP
from .arch.tfmlp import TFMlpAggr

class Model(nn.Module):
    '''
    GCN-VAE Enhanced MIG Network
    '''
    def __init__(self, 
                 num_rounds=1, 
                 dim_hidden=128, 
                 enable_encode=True,
                 enable_reverse=False):
        super(Model, self).__init__()
        
        # 保持原始配置
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

        # 门类型聚合器（保持原始结构）
        self.aggr_and_strc = TFMlpAggr(dim_hidden, dim_hidden)
        self.aggr_not_strc = TFMlpAggr(dim_hidden, dim_hidden)
        self.aggr_or_strc = TFMlpAggr(dim_hidden, dim_hidden)
        self.aggr_maj_strc = TFMlpAggr(dim_hidden, dim_hidden)
        
        self.aggr_and_func = TFMlpAggr(dim_hidden*2, dim_hidden)
        self.aggr_not_func = TFMlpAggr(dim_hidden, dim_hidden)
        self.aggr_or_func = TFMlpAggr(dim_hidden*2, dim_hidden)
        self.aggr_maj_func = TFMlpAggr(dim_hidden*2, dim_hidden)

        # 门类型更新网络
        self.gate_update = nn.ModuleDict({
            'and': nn.Sequential(nn.Linear(dim_hidden*2, dim_hidden), nn.ReLU()),
            'not': nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.Tanh()),
            'or': nn.Sequential(nn.Linear(dim_hidden*2, dim_hidden), nn.ReLU()),
            'maj': nn.Sequential(nn.Linear(dim_hidden*2, dim_hidden), nn.ReLU())
        })

        # 保持原始Readout
        self.readout_prob = MLP(dim_hidden, self.dim_mlp, 1, 
                               num_layer=3, p_drop=0.2, 
                               norm_layer='batchnorm', 
                               act_layer='relu')

    def reparameterize(self, mu, logstd):
        if self.training:
            return mu + torch.exp(logstd.clamp(max=10)) * torch.randn_like(mu)
        else:
            return mu

    def forward(self, G):
        device = next(self.parameters()).device
        num_nodes = len(G.mig_gate)
        
        # 保持原始初始化
        if self.enable_encode:
            hs = generate_hs_init(G, torch.zeros(num_nodes, self.dim_hidden), 
                                 self.dim_hidden, mig=True).to(device)
        else:
            hs = torch.zeros(num_nodes, self.dim_hidden).to(device)
            
        hf = torch.zeros(num_nodes, self.dim_hidden).to(device)
        edge_index = G.mig_edge_index.to(device)
        
        # 门类型掩码
        gate_masks = {
            'and': G.mig_gate.squeeze(1) == 3,
            'not': G.mig_gate.squeeze(1) == 2,
            'or': G.mig_gate.squeeze(1) == 4,
            'maj': G.mig_gate.squeeze(1) == 1
        }

        for _ in range(self.num_rounds):
            # GCN编码
            hs_gcn = self.gcn_encoder(hs, edge_index)
            
            # VAE潜在空间
            mu = self.vae_mu(hs_gcn)
            logstd = self.vae_logstd(hs_gcn)
            hs = self.reparameterize(mu, logstd)
            
            # 分层处理
            for level in range(1, max(G.mig_forward_level).item()+1):
                layer_mask = G.mig_forward_level == level
                
                # 统一处理各门类型
                for gate_type in ['and', 'not', 'or', 'maj']:
                    mask = gate_masks[gate_type] & layer_mask
                    if not torch.any(mask):
                        continue
                        
                    l_nodes = G.mig_forward_index[mask]
                    sub_edges, _ = subgraph(l_nodes, edge_index, dim=1)
                    
                    # 结构更新
                    aggr = getattr(self, f'aggr_{gate_type}_strc')
                    msg = aggr(hs, sub_edges)
                    hs_update = self.gate_update[gate_type](torch.cat([hs[l_nodes], msg[l_nodes]], dim=1))
                    hs[l_nodes] = hs_update
                    
                    # 功能更新
                    func_aggr = getattr(self, f'aggr_{gate_type}_func')
                    func_msg = func_aggr(torch.cat([hs, hf], dim=1), sub_edges)
                    hf[l_nodes] = func_msg[l_nodes]

            # 解码重建
            hf = self.decoder(hs, edge_index)

        return hs, hf

    # 保持原始方法
    def pred_prob(self, hf):
        prob = self.readout_prob(hf)
        return torch.clamp(prob, 0.0, 1.0)

    def load(self, model_path):
        # 保持原始加载逻辑
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
                    print(f'Skipping {k}, shape mismatch')
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
