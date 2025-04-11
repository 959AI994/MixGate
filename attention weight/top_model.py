# ----------------------------------------------- Hierarchical Transformer-----------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from torch import nn
from torch.nn import LSTM, GRU
from .utils.dag_utils import subgraph, custom_backward_subgraph
from .utils.utils import generate_hs_init

from .arch.mlp import MLP
from .arch.mlp_aggr import MlpAggr
from .arch.tfmlp import TFMlpAggr
from .arch.gcn_conv import AggConv

from .dg_model_mig import Model as DeepGate_Mig
from .dg_model_xmg import Model as DeepGate_Xmg
from .dg_model_xag import Model as DeepGate_Xag
from .dg_model import Model as DeepGate_Aig
from .hier_tf import HierarchicalTransformer
import numpy as np

from linformer import Linformer

class TopModel(nn.Module):
    def __init__(self, 
                 args, 
                 # add aig,mig,xmg,xag pth
                 dg_ckpt_aig, 
                 dg_ckpt_mig, 
                 dg_ckpt_xmg, 
                 dg_ckpt_xag
                ):
        super(TopModel, self).__init__()
        self.args = args
        self.mask_ratio = args.mask_ratio
        
        # DeepGate for AIG, MIG, xmg, XAG
        self.deepgate_aig = DeepGate_Aig(dim_hidden=args.dim_hidden)
        self.deepgate_aig.load(dg_ckpt_aig)
        
        self.deepgate_mig = DeepGate_Mig(dim_hidden=args.dim_hidden)
        self.deepgate_mig.load(dg_ckpt_mig)
        
        self.deepgate_xmg = DeepGate_Xmg(dim_hidden=args.dim_hidden)
        self.deepgate_xmg.load(dg_ckpt_xmg)
        
        self.deepgate_xag = DeepGate_Xag(dim_hidden=args.dim_hidden)
        self.deepgate_xag.load(dg_ckpt_xag)

        # # 关键：冻结参数
        # for model in [self.deepgate_aig, self.deepgate_mig, self.deepgate_xmg, self.deepgate_xag]:
        #     for param in model.parameters():
        #         param.requires_grad = False  
                
        # Transformer
        if self.args.linear_tf:
            self.mask_tf = Linformer(
                dim = args.dim_hidden * 2, k=args.dim_hidden*2, 
                heads = args.tf_head, depth = args.tf_layer, seq_len=8192, 
                one_kv_head=True, share_kv=True, 
            )
        elif self.args.hier_tf:
            self.mask_tf = HierarchicalTransformer(args)
        else:
            one_tf_layer = nn.TransformerEncoderLayer(d_model=args.dim_hidden * 2, nhead=args.tf_head, batch_first=True)
            self.mask_tf = nn.TransformerEncoder(one_tf_layer, num_layers=args.tf_layer)
        
        # Token masking
        self.mask_token = nn.Parameter(torch.randn(1, args.dim_hidden))  # learnable mask token
    
    def mask_tokens(self, G, tokens, mask_ratio=0.005, k_hop=4): 
        """
        Randomly mask a ratio of tokens and extract its k_hop
        Args:
            G: Input graph
            tokens: Input tokens (batch_size, seq_len, dim_hidden)
            mask_ratio: Percentage of tokens to mask
            k_hop: Number of hops to extract
        Returns:
            masked_tokens: Tokens with some positions replaced by mask token
            mask_indices: Indices of masked tokens
        """
        if mask_ratio == 0:
            masked_tokens = tokens.clone()
            return masked_tokens, None
        
        seq_len = len(tokens)
        mask_indices = torch.randperm(seq_len)[:int(mask_ratio * seq_len)]  # randomly select tokens to mask
        mask_flag = torch.zeros(seq_len, dtype=torch.bool)
        mask_flag[mask_indices] = True
        
        # Extract k-hop subgraph
        current_nodes = mask_indices
        for hop in range(k_hop):
            fanin_nodes, _ = subgraph(current_nodes, G.edge_index, dim=1)
            mask_indices = mask_indices.to('cpu')
            fanin_nodes = fanin_nodes.to('cpu')
            fanin_nodes = torch.unique(fanin_nodes)
            current_nodes = fanin_nodes
            mask_flag[fanin_nodes] = True
            mask_indices = torch.cat([mask_indices, fanin_nodes])
        
        mask_indices = torch.unique(mask_indices)
        masked_tokens = tokens.clone()
        masked_tokens[mask_indices, self.args.dim_hidden:] = self.mask_token
        # print(f"Masked indices: {mask_indices}")  # 打印掩码的 token 索引
        # print(f"Masked tokens: {masked_tokens.shape}")  # 打印掩码后 tokens 的形状

        return masked_tokens, mask_indices

    # def mask_tokens(self, G, tokens, mask_ratio=0.005, k_hop=4): 
    #     """
    #     Randomly mask one token based on the mask_ratio.
    #     Args:
    #         G: Input graph (not used in this case)
    #         tokens: Input tokens (batch_size, seq_len, dim_hidden)
    #         mask_ratio: Percentage of tokens to mask
    #     Returns:
    #         masked_tokens: Tokens with one token replaced by mask token
    #         mask_indices: Indices of masked token
    #         masked_token: The actual masked token
    #     """
    #     # If mask_ratio is 0, return the tokens unchanged
    #     if mask_ratio == 0:
    #         return tokens.clone(), None, None
        
    #     # Get the number of tokens to mask (just 1 token here)
    #     seq_len = tokens.size(0)  # Get the sequence length
    #     num_masked = 1  # Only mask one token
        
    #     # Randomly select one token to mask
    #     mask_index = torch.randint(0, seq_len, (1,))  # Randomly select one index
    #     mask_indices = mask_index  # Only one token is being masked
        
    #     # Clone the tokens and replace the selected token with the mask token
    #     masked_tokens = tokens.clone()
    #     masked_tokens[mask_indices, self.args.dim_hidden:] = self.mask_token  # Replace with mask token
    #     # print(f"Masked indices: {mask_indices}")  # 打印掩码的 token 索引
    #     # print(f"Masked tokens: {masked_tokens.shape}")  # 打印掩码后 tokens 的形状
        
    #     # Extract the actual token that was masked
    #     masked_token = tokens[mask_indices]
    #     # print(f"Masked token: {masked_token}")

    #     return masked_tokens, mask_indices



    def forward(self, G):
        G.mig_batch = G.batch
        self.device = next(self.parameters()).device
        mcm_predicted_tokens = torch.zeros(0, self.args.dim_hidden * 2).to(self.device)

        # Get tokens from AIG, MIG, xmg, XAG
        aig_hs, aig_hf = self.deepgate_aig(G)
        aig_hs = aig_hs.detach()
        aig_hf = aig_hf.detach()
        aig_tokens = torch.cat([aig_hs, aig_hf], dim=1)

        mig_hs, mig_hf = self.deepgate_mig(G)
        mig_hs = mig_hs.detach()
        mig_hf = mig_hf.detach()
        mig_tokens = torch.cat([mig_hs, mig_hf], dim=1)
        
        xmg_hs, xmg_hf = self.deepgate_xmg(G)
        xmg_hs = xmg_hs.detach()
        xmg_hf = xmg_hf.detach()
        xmg_tokens = torch.cat([xmg_hs, xmg_hf], dim=1)
        
        xag_hs, xag_hf = self.deepgate_xag(G)
        xag_hs = xag_hs.detach()
        xag_hf = xag_hf.detach()
        xag_tokens = torch.cat([xag_hs, xag_hf], dim=1)

        
        # 模态列表
        modalities = ['aig', 'mig', 'xmg', 'xag']
        tokens_dict = {
            'aig': (aig_tokens, aig_hf, self.deepgate_aig),
            'mig': (mig_tokens, mig_hf, self.deepgate_mig),
            'xmg': (xmg_tokens, xmg_hf, self.deepgate_xmg),
            'xag': (xag_tokens, xag_hf, self.deepgate_xag),
        }
        # 随机选择一个模态
        # selected_modality = modalities[torch.randint(0, len(modalities), (1,)).item()]
        selected_modality = 'mig'
        selected_tokens, masked_hf, encoder = tokens_dict[selected_modality]
        # 对选定模态进行掩码
        masked_tokens, mask_indices = self.mask_tokens(G, selected_tokens, self.mask_ratio, k_hop=4)

        # Reconstruction: Mask Circuit Modeling 
        # Hierarchical Transformer / Stone 03.13
        if self.args.hier_tf:
            tokens = [aig_tokens, xag_tokens, xmg_tokens, mig_tokens]
            mcm_predicted_tokens = self.mask_tf(G, tokens, masked_tokens,mask_indices, masked_modal=selected_modality)
            transformer_output = mcm_predicted_tokens
            
        else:
            for batch_id in range(G.batch.max().item() + 1): 
                # batch_pm_tokens_masked = pm_tokens_masked[G.batch == batch_id]
                batch_aig_tokens = aig_tokens[G.aig_batch == batch_id]
                batch_mig_tokens = mig_tokens[G.batch == batch_id]
                batch_xmg_tokens = xmg_tokens[G.xmg_batch == batch_id]
                batch_xag_tokens = xag_tokens[G.xag_batch == batch_id]
                
                # 根据被掩码的模态，排除该模态的原 token，拼接其余模态
                if selected_modality == 'aig':
                    other_tokens = torch.cat([batch_mig_tokens, batch_xmg_tokens, batch_xag_tokens], dim=0)
                    batch_masked_tokens = masked_tokens[G.aig_batch == batch_id]
                elif selected_modality == 'mig':
                    other_tokens = torch.cat([batch_aig_tokens, batch_xmg_tokens, batch_xag_tokens], dim=0)
                    batch_masked_tokens = masked_tokens[G.batch == batch_id]
                elif selected_modality == 'xmg':
                    other_tokens = torch.cat([batch_aig_tokens, batch_mig_tokens, batch_xag_tokens], dim=0)
                    batch_masked_tokens = masked_tokens[G.xmg_batch == batch_id]
                elif selected_modality == 'xag':
                    other_tokens = torch.cat([batch_aig_tokens, batch_mig_tokens, batch_xmg_tokens], dim=0)
                    batch_masked_tokens = masked_tokens[G.xag_batch == batch_id]

                # 将掩码后的 token 与其他模态的 token 拼接
                batch_all_tokens = torch.cat([batch_masked_tokens, other_tokens], dim=0)
                
                # Transformer forward
                if self.args.linear_tf:
                    batch_predicted_tokens = self.mask_tf(batch_all_tokens.unsqueeze(0))
                    batch_predicted_tokens = batch_predicted_tokens.squeeze(0)
                else:
                    batch_predicted_tokens = self.mask_tf(batch_all_tokens)
                batch_pred_masked_tokens = batch_predicted_tokens[:batch_masked_tokens.shape[0], :]
                # 收集预测的被掩码的 token
                mcm_predicted_tokens = torch.cat([mcm_predicted_tokens, batch_pred_masked_tokens], dim=0)

            # attention_info = None  # 非hier模式返回空值
            # 获取其他模态的 tokens
            other_tokens = torch.cat([tokens[0] for modality, tokens in tokens_dict.items() if modality != selected_modality], dim=0)
            
        hf_tokens = mcm_predicted_tokens[:, self.args.dim_hidden:]
        masked_prob = encoder.pred_prob(hf_tokens)

        # 获取每个模态的预测概率
        aig_prob = self.deepgate_aig.pred_prob(aig_hf)
        mig_prob = self.deepgate_mig.pred_prob(mig_hf)
        xmg_prob = self.deepgate_xmg.pred_prob(xmg_hf)
        xag_prob = self.deepgate_xag.pred_prob(xag_hf)

        return mcm_predicted_tokens, mask_indices, selected_tokens, masked_prob, aig_prob, mig_prob, xmg_prob, xag_prob,{
                'attentions': self.mask_tf.attention_buffer,
                'token_names': self.mask_tf.token_name_buffer
            }
   

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


# ---------------------------------------------------Dense Transformer---------------------------------------------------------------------
# [新版本代码：dense_tf调用了一次，计算hf'的prob loss获取是从四个模态一起得到的]
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import torch
# import os
# from torch import nn
# from torch.nn import LSTM, GRU
# from .utils.dag_utils import subgraph, custom_backward_subgraph
# from .utils.utils import generate_hs_init

# from .arch.mlp import MLP
# from .arch.mlp_aggr import MlpAggr
# from .arch.tfmlp import TFMlpAggr
# from .arch.gcn_conv import AggConv

# from .dc_model import Model as DeepCell
# from .dg_model_mig import Model as DeepGate_Mig
# from .dg_model_xmg import Model as DeepGate_Xmg
# from .dg_model_xag import Model as DeepGate_Xag
# from .dg_model import Model as DeepGate_Aig
# from .hier_tf import HierarchicalTransformer
# import numpy as np

## from linformer import Linformer

# class TopModel(nn.Module):
#     def __init__(self, 
#                  args, 
#                  # add aig,mig,xmg,xam pt
#                  dg_ckpt_aig, 
#                  dg_ckpt_mig, 
#                  dg_ckpt_xmg, 
#                  dg_ckpt_xag
#                 ):
#         super(TopModel, self).__init__()
#         self.args = args
#         self.mask_ratio = args.mask_ratio
        
#         # DeepGate for AIG, MIG, xmg, XAG
#         self.deepgate_aig = DeepGate_Aig(dim_hidden=args.dim_hidden)
#         self.deepgate_aig.load(dg_ckpt_aig)
        
#         self.deepgate_mig = DeepGate_Mig(dim_hidden=args.dim_hidden)
#         self.deepgate_mig.load(dg_ckpt_mig)
        
#         self.deepgate_xmg = DeepGate_Xmg(dim_hidden=args.dim_hidden)
#         self.deepgate_xmg.load(dg_ckpt_xmg)
        
#         self.deepgate_xag = DeepGate_Xag(dim_hidden=args.dim_hidden)
#         self.deepgate_xag.load(dg_ckpt_xag)

#         # 关键：冻结参数
#         # for model in [self.deepgate_aig, self.deepgate_mig, self.deepgate_xmg, self.deepgate_xag]:
#         #     for param in model.parameters():
#         #         param.requires_grad = False  
                
#         # Transformer
#         if self.args.linear_tf:
#             self.mask_tf = Linformer(
#                 dim = args.dim_hidden * 2, k=args.dim_hidden*2, 
#                 heads = args.tf_head, depth = args.tf_layer, seq_len=8192, 
#                 one_kv_head=True, share_kv=True, 
#             )
#         elif self.args.hier_tf:
#             self.mask_tf = HierarchicalTransformer(args)
#         else:
#             one_tf_layer = nn.TransformerEncoderLayer(d_model=args.dim_hidden * 2, nhead=args.tf_head, batch_first=True)
#             self.mask_tf = nn.TransformerEncoder(one_tf_layer, num_layers=args.tf_layer)
        
#         # Token masking
#         self.mask_token = nn.Parameter(torch.randn(1, args.dim_hidden))  # learnable mask token
    
#     def mask_tokens(self, G, tokens, mask_ratio=0.05, k_hop=4): 
#         """
#         Randomly mask a ratio of tokens and extract its k_hop
#         Args:
#             G: Input graph
#             tokens: Input tokens (batch_size, seq_len, dim_hidden)
#             mask_ratio: Percentage of tokens to mask
#             k_hop: Number of hops to extract
#         Returns:
#             masked_tokens: Tokens with some positions replaced by mask token
#             mask_indices: Indices of masked tokens
#         """
#         if mask_ratio == 0:
#             masked_tokens = tokens.clone()
#             return masked_tokens, None
        
#         if mask_ratio == 0:
#             masked_tokens = tokens.clone()
#             return masked_tokens, None
        
#         seq_len = len(tokens)
#         mask_indices = torch.randperm(seq_len)[:int(mask_ratio * seq_len)]  # randomly select tokens to mask
#         mask_flag = torch.zeros(seq_len, dtype=torch.bool)
#         mask_flag[mask_indices] = True
        
#         # Extract k-hop subgraph
#         current_nodes = mask_indices
#         for hop in range(k_hop):
#             fanin_nodes, _ = subgraph(current_nodes, G.edge_index, dim=1)
#             mask_indices = mask_indices.to('cpu')
#             fanin_nodes = fanin_nodes.to('cpu')
#             fanin_nodes = torch.unique(fanin_nodes)
#             current_nodes = fanin_nodes
#             mask_flag[fanin_nodes] = True
#             mask_indices = torch.cat([mask_indices, fanin_nodes])
        
#         mask_indices = torch.unique(mask_indices)
#         masked_tokens = tokens.clone()
#         masked_tokens[mask_indices, self.args.dim_hidden:] = self.mask_token
#         return masked_tokens, mask_indices

#     def forward(self, G):
#         G.mig_batch = G.batch
#         self.device = next(self.parameters()).device

#         mcm_predicted_tokens = torch.zeros(0, self.args.dim_hidden * 2).to(self.device)
        
#         # Get tokens from AIG, MIG, xmg, XAG
#         aig_hs, aig_hf = self.deepgate_aig(G)
#         aig_hs = aig_hs.detach()
#         aig_hf = aig_hf.detach()
#         aig_tokens = torch.cat([aig_hs, aig_hf], dim=1)

#         mig_hs, mig_hf = self.deepgate_mig(G)
#         mig_hs = mig_hs.detach()
#         mig_hf = mig_hf.detach()
#         mig_tokens = torch.cat([mig_hs, mig_hf], dim=1)
        
#         xmg_hs, xmg_hf = self.deepgate_xmg(G)
#         xmg_hs = xmg_hs.detach()
#         xmg_hf = xmg_hf.detach()
#         xmg_tokens = torch.cat([xmg_hs, xmg_hf], dim=1)
        
#         xag_hs, xag_hf = self.deepgate_xag(G)
#         xag_hs = xag_hs.detach()
#         xag_hf = xag_hf.detach()
#         xag_tokens = torch.cat([xag_hs, xag_hf], dim=1)
        
#         # 模态列表
#         modalities = ['aig', 'mig', 'xmg', 'xag']
#         tokens_dict = {
#             'aig': (aig_tokens, aig_hf, self.deepgate_aig),
#             'mig': (mig_tokens, mig_hf, self.deepgate_mig),
#             'xmg': (xmg_tokens, xmg_hf, self.deepgate_xmg),
#             'xag': (xag_tokens, xag_hf, self.deepgate_xag),
#         }
#         # 随机选择一个模态
#         # selected_modality = modalities[torch.randint(0, len(modalities), (1,)).item()]
#         selected_modality = 'mig'
#         selected_tokens, masked_hf, encoder = tokens_dict[selected_modality]
#         # 对选定模态进行掩码
#         masked_tokens, mask_indices = self.mask_tokens(G, selected_tokens, self.mask_ratio, k_hop=4)
    
#         # Reconstruction: Mask Circuit Modeling 
#         for batch_id in range(G.batch.max().item() + 1): 
#             # batch_pm_tokens_masked = pm_tokens_masked[G.batch == batch_id]
#             batch_aig_tokens = aig_tokens[G.aig_batch == batch_id]
#             batch_mig_tokens = mig_tokens[G.batch == batch_id]
#             batch_xmg_tokens = xmg_tokens[G.xmg_batch == batch_id]
#             batch_xag_tokens = xag_tokens[G.xag_batch == batch_id]
            
#             # 根据被掩码的模态，排除该模态的原 token，拼接其余模态
#             if selected_modality == 'aig':
#                 other_tokens = torch.cat([batch_mig_tokens, batch_xmg_tokens, batch_xag_tokens], dim=0)
#                 batch_masked_tokens = masked_tokens[G.aig_batch == batch_id]
#             elif selected_modality == 'mig':
#                 other_tokens = torch.cat([batch_aig_tokens, batch_xmg_tokens, batch_xag_tokens], dim=0)
#                 batch_masked_tokens = masked_tokens[G.batch == batch_id]
#             elif selected_modality == 'xmg':
#                 other_tokens = torch.cat([batch_aig_tokens, batch_mig_tokens, batch_xag_tokens], dim=0)
#                 batch_masked_tokens = masked_tokens[G.xmg_batch == batch_id]
#             elif selected_modality == 'xag':
#                 other_tokens = torch.cat([batch_aig_tokens, batch_mig_tokens, batch_xmg_tokens], dim=0)
#                 batch_masked_tokens = masked_tokens[G.xag_batch == batch_id]

#                 # 将掩码后的 token 与其他模态的 token 拼接
#                 batch_all_tokens = torch.cat([batch_masked_tokens, other_tokens], dim=0)
                
#                 # Transformer forward
#                 if self.args.linear_tf:
#                     batch_predicted_tokens = self.mask_tf(batch_all_tokens.unsqueeze(0))
#                     batch_predicted_tokens = batch_predicted_tokens.squeeze(0)
#                 else:
#                     batch_predicted_tokens = self.mask_tf(batch_all_tokens)
#                 batch_pred_masked_tokens = batch_predicted_tokens[:batch_masked_tokens.shape[0], :]
#                 # 收集预测的被掩码的 token
#                 mcm_predicted_tokens = torch.cat([mcm_predicted_tokens, batch_pred_masked_tokens], dim=0)
        
#         # # 获取其他模态的 tokens
#         # other_tokens = torch.cat([tokens[0] for modality, tokens in tokens_dict.items() if modality != selected_modality], dim=0)
#         # input_tokens = masked_tokens
#         # # Transformer 层处理
#         # transformer_output = self.mask_tf(input_tokens)
#         # # 从 Transformer 输出中获取处理后的 hf
#         # transformer_hf =transformer_output[:, self.args.dim_hidden:] 
#         # # 用 transformer_hf 进行预测
#         # masked_prob = encoder.pred_prob(transformer_hf)
#         hf_tokens = mcm_predicted_tokens[:, self.args.dim_hidden:]
#         masked_prob = encoder.pred_prob(hf_tokens)


#         # 获取每个模态的预测概率
#         aig_prob = self.deepgate_aig.pred_prob(aig_hf)
#         mig_prob = self.deepgate_mig.pred_prob(mig_hf)
#         xmg_prob = self.deepgate_xmg.pred_prob(xmg_hf)
#         xag_prob = self.deepgate_xag.pred_prob(xag_hf)

#         return mcm_predicted_tokens, mask_indices, selected_tokens, masked_prob, aig_prob, mig_prob, xmg_prob, xag_prob
   

#     def load(self, model_path):
#         checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
#         state_dict_ = checkpoint['state_dict']
#         state_dict = {}
#         for k in state_dict_:
#             if k.startswith('module') and not k.startswith('module_list'):
#                 state_dict[k[7:]] = state_dict_[k]
#             else:
#                 state_dict[k] = state_dict_[k]
#         model_state_dict = self.state_dict()
        
#         for k in state_dict:
#             if k in model_state_dict:
#                 if state_dict[k].shape != model_state_dict[k].shape:
#                     print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
#                         k, model_state_dict[k].shape, state_dict[k].shape))
#                     state_dict[k] = model_state_dict[k]
#             else:
#                 print('Drop parameter {}.'.format(k))
#         for k in model_state_dict:
#             if not (k in state_dict):
#                 print('No param {}.'.format(k))
#                 state_dict[k] = model_state_dict[k]
#         self.load_state_dict(state_dict, strict=False)


# ----------------------------------------------------2 Dense Transformer--------------------------------------------------------------------
# [旧版本代码：dense_tf调用了两次，hf'的获取仅仅是从mig单分支获得]
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import torch
# import os
# from torch import nn
# from torch.nn import LSTM, GRU
# from .utils.dag_utils import subgraph, custom_backward_subgraph
# from .utils.utils import generate_hs_init

# from .arch.mlp import MLP
# from .arch.mlp_aggr import MlpAggr
# from .arch.tfmlp import TFMlpAggr
# from .arch.gcn_conv import AggConv

# from .dc_model import Model as DeepCell
# from .dg_model_mig import Model as DeepGate_Mig
# from .dg_model_xmg import Model as DeepGate_Xmg
# from .dg_model_xag import Model as DeepGate_Xag
# from .dg_model import Model as DeepGate_Aig
# import numpy as np


# class TopModel(nn.Module):
#     def __init__(self, 
#                  args, 
#                  # add aig,mig,xmg,xam pt
#                  dg_ckpt_aig, 
#                  dg_ckpt_mig, 
#                  dg_ckpt_xmg, 
#                  dg_ckpt_xag
#                 ):
#         super(TopModel, self).__init__()
#         self.args = args
#         self.mask_ratio = args.mask_ratio
        
#         # DeepGate for AIG, MIG, xmg, XAG
#         self.deepgate_aig = DeepGate_Aig(dim_hidden=args.dim_hidden)
#         self.deepgate_aig.load(dg_ckpt_aig)
        
#         self.deepgate_mig = DeepGate_Mig(dim_hidden=args.dim_hidden)
#         self.deepgate_mig.load(dg_ckpt_mig)
        
#         self.deepgate_xmg = DeepGate_Xmg(dim_hidden=args.dim_hidden)
#         self.deepgate_xmg.load(dg_ckpt_xmg)
        
#         self.deepgate_xag = DeepGate_Xag(dim_hidden=args.dim_hidden)
#         self.deepgate_xag.load(dg_ckpt_xag)

#         # # 关键：冻结参数
#         # for model in [self.deepgate_aig, self.deepgate_mig, self.deepgate_xmg, self.deepgate_xag]:
#         #     for param in model.parameters():
#         #         param.requires_grad = False  
                
#         # Transformer
#         tf_layer = nn.TransformerEncoderLayer(d_model=args.dim_hidden * 2, nhead=args.tf_head, batch_first=True)
#         self.mask_tf = nn.TransformerEncoder(tf_layer, num_layers=args.tf_layer)
        
#         # Token masking
#         self.mask_token = nn.Parameter(torch.randn(1, args.dim_hidden))  # learnable mask token
    
#     def mask_tokens(self, G, tokens, mask_ratio=0.05, k_hop=4): 
#         """
#         Randomly mask a ratio of tokens and extract its k_hop
#         Args:
#             G: Input graph
#             tokens: Input tokens (batch_size, seq_len, dim_hidden)
#             mask_ratio: Percentage of tokens to mask
#             k_hop: Number of hops to extract
#         Returns:
#             masked_tokens: Tokens with some positions replaced by mask token
#             mask_indices: Indices of masked tokens
#         """
#         seq_len = len(tokens)
#         mask_indices = torch.randperm(seq_len)[:int(mask_ratio * seq_len)]  # randomly select tokens to mask
#         mask_flag = torch.zeros(seq_len, dtype=torch.bool)
#         mask_flag[mask_indices] = True
        
#         # Extract k-hop subgraph
#         current_nodes = mask_indices
#         for hop in range(k_hop):
#             fanin_nodes, _ = subgraph(current_nodes, G.edge_index, dim=1)
#             mask_indices = mask_indices.to('cpu')
#             fanin_nodes = fanin_nodes.to('cpu')
#             fanin_nodes = torch.unique(fanin_nodes)
#             current_nodes = fanin_nodes
#             mask_flag[fanin_nodes] = True
#             mask_indices = torch.cat([mask_indices, fanin_nodes])
        
#         mask_indices = torch.unique(mask_indices)
#         masked_tokens = tokens.clone()
#         masked_tokens[mask_indices, self.args.dim_hidden:] = self.mask_token
#         return masked_tokens, mask_indices

#     def forward(self, G):
#         self.device = next(self.parameters()).device
#         mcm_predicted_tokens = torch.zeros(0, self.args.dim_hidden * 2).to(self.device)
        
#         # Get tokens from AIG, MIG, xmg, XAG
#         aig_hs, aig_hf = self.deepgate_aig(G)
#         aig_hs = aig_hs.detach()
#         aig_hf = aig_hf.detach()
#         aig_tokens = torch.cat([aig_hs, aig_hf], dim=1)

#         mig_hs, mig_hf = self.deepgate_mig(G)
#         mig_hs = mig_hs.detach()
#         mig_hf = mig_hf.detach()
#         mig_tokens = torch.cat([mig_hs, mig_hf], dim=1)
        
#         xmg_hs, xmg_hf = self.deepgate_xmg(G)
#         xmg_hs = xmg_hs.detach()
#         xmg_hf = xmg_hf.detach()
#         xmg_tokens = torch.cat([xmg_hs, xmg_hf], dim=1)
        
#         xag_hs, xag_hf = self.deepgate_xag(G)
#         xag_hs = xag_hs.detach()
#         xag_hf = xag_hf.detach()
#         xag_tokens = torch.cat([xag_hs, xag_hf], dim=1)
        
#         # 模态列表
#         modalities = ['aig', 'mig', 'xmg', 'xag']
#         tokens_dict = {
#             'aig': (aig_tokens, aig_hf, self.deepgate_aig),
#             'mig': (mig_tokens, mig_hf, self.deepgate_mig),
#             'xmg': (xmg_tokens, xmg_hf, self.deepgate_xmg),
#             'xag': (xag_tokens, xag_hf, self.deepgate_xag),
#         }
#         # 随机选择一个模态
#         # selected_modality = modalities[torch.randint(0, len(modalities), (1,)).item()]
#         selected_modality = 'mig'
#         selected_tokens, masked_hf, encoder = tokens_dict[selected_modality]
#         # 对选定模态进行掩码
#         masked_tokens, mask_indices = self.mask_tokens(G, selected_tokens, self.mask_ratio, k_hop=4)

#         # Reconstruction: Mask Circuit Modeling 
#         for batch_id in range(G.batch.max().item() + 1): 
#             # batch_pm_tokens_masked = pm_tokens_masked[G.batch == batch_id]
#             batch_aig_tokens = aig_tokens[G.aig_batch == batch_id]
#             batch_mig_tokens = mig_tokens[G.batch == batch_id]
#             batch_xmg_tokens = xmg_tokens[G.xmg_batch == batch_id]
#             batch_xag_tokens = xag_tokens[G.xag_batch == batch_id]
            
#             # 根据被掩码的模态，排除该模态的原 token，拼接其余模态
#             if selected_modality == 'aig':
#                 other_tokens = torch.cat([batch_mig_tokens, batch_xmg_tokens, batch_xag_tokens], dim=0)
#                 batch_masked_tokens = masked_tokens[G.aig_batch == batch_id]
#             elif selected_modality == 'mig':
#                 other_tokens = torch.cat([batch_aig_tokens, batch_xmg_tokens, batch_xag_tokens], dim=0)
#                 batch_masked_tokens = masked_tokens[G.batch == batch_id]
#             elif selected_modality == 'xmg':
#                 other_tokens = torch.cat([batch_aig_tokens, batch_mig_tokens, batch_xag_tokens], dim=0)
#                 batch_masked_tokens = masked_tokens[G.xmg_batch == batch_id]
#             elif selected_modality == 'xag':
#                 other_tokens = torch.cat([batch_aig_tokens, batch_mig_tokens, batch_xmg_tokens], dim=0)
#                 batch_masked_tokens = masked_tokens[G.xag_batch == batch_id]

#             # 将掩码后的 token 与其他模态的 token 拼接
#             batch_all_tokens = torch.cat([batch_masked_tokens, other_tokens], dim=0)
            
#             # Transformer forward
#             batch_predicted_tokens = self.mask_tf(batch_all_tokens)
#             batch_pred_masked_tokens = batch_predicted_tokens[:batch_masked_tokens.shape[0], :]
#             # 收集预测的被掩码的 token
#             mcm_predicted_tokens = torch.cat([mcm_predicted_tokens, batch_pred_masked_tokens], dim=0)
        
#         # Transformer 层处理
#         # 获取其他模态的 tokens
#         # other_tokens = torch.cat([tokens[0] for modality, tokens in tokens_dict.items() if modality != selected_modality], dim=0)
#         # input_tokens = masked_tokens  
#         # transformer_output = self.mask_tf(input_tokens)
#         # transformer_hf =transformer_output[:, self.args.dim_hidden:] 
#         # masked_prob = encoder.pred_prob(transformer_hf)
#         # 改为直接使用循环中生成的预测结果
#         hf_tokens = mcm_predicted_tokens[:, self.args.dim_hidden:]
#         masked_prob = encoder.pred_prob(hf_tokens)

#         # 获取每个模态的预测概率
#         aig_prob = self.deepgate_aig.pred_prob(aig_hf)
#         mig_prob = self.deepgate_mig.pred_prob(mig_hf)
#         xmg_prob = self.deepgate_xmg.pred_prob(xmg_hf)
#         xag_prob = self.deepgate_xag.pred_prob(xag_hf)

#         return mcm_predicted_tokens, mask_indices, selected_tokens, masked_prob,aig_prob, mig_prob, xmg_prob, xag_prob
   

#     def load(self, model_path):
#         checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
#         state_dict_ = checkpoint['state_dict']
#         state_dict = {}
#         for k in state_dict_:
#             if k.startswith('module') and not k.startswith('module_list'):
#                 state_dict[k[7:]] = state_dict_[k]
#             else:
#                 state_dict[k] = state_dict_[k]
#         model_state_dict = self.state_dict()
        
#         for k in state_dict:
#             if k in model_state_dict:
#                 if state_dict[k].shape != model_state_dict[k].shape:
#                     print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
#                         k, model_state_dict[k].shape, state_dict[k].shape))
#                     state_dict[k] = model_state_dict[k]
#             else:
#                 print('Drop parameter {}.'.format(k))
#         for k in model_state_dict:
#             if not (k in state_dict):
#                 print('No param {}.'.format(k))
#                 state_dict[k] = model_state_dict[k]
#         self.load_state_dict(state_dict, strict=False)
        
        