from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from torch import nn
from .utils.dag_utils import subgraph
from .utils.utils import generate_hs_init
from .arch.mlp import MLP
from .arch.tfmlp import TFMlpAggr

from .dc_model import Model as DeepCell
from .dg_model_mig import Model as DeepGate_Mig
from .dg_model_xmg import Model as DeepGate_Xmg
from .dg_model_xag import Model as DeepGate_Xag
from .dg_model import Model as DeepGate_Aig

class TopModel(nn.Module):
    def __init__(self, 
                 args, 
                 dg_ckpt_aig, 
                 dg_ckpt_mig, 
                 dg_ckpt_xmg, 
                 dg_ckpt_xag):
        super(TopModel, self).__init__()
        self.args = args
        self.mask_ratio = args.mask_ratio
        self.latent_dim = 64  # VAE潜在空间维度
        
        # 初始化各结构模型
        self.deepgate_aig = self._init_model(DeepGate_Aig, dg_ckpt_aig, args)
        self.deepgate_mig = self._init_model(DeepGate_Mig, dg_ckpt_mig, args)
        self.deepgate_xmg = self._init_model(DeepGate_Xmg, dg_ckpt_xmg, args)
        self.deepgate_xag = self._init_model(DeepGate_Xag, dg_ckpt_xag, args)

        # Transformer配置
        tf_layer = nn.TransformerEncoderLayer(
            d_model=args.dim_hidden * 2 + self.latent_dim,
            nhead=args.tf_head,
            dim_feedforward=512,
            batch_first=True
        )
        self.mask_tf = nn.TransformerEncoder(tf_layer, num_layers=args.tf_layer)
        
        # Mask token和投影层
        self.mask_token = nn.Parameter(torch.randn(1, args.dim_hidden + self.latent_dim))
        self.vae_proj = nn.Sequential(
            nn.Linear(args.dim_hidden * 2 + self.latent_dim, args.dim_hidden * 2),
            nn.ReLU(),
            nn.LayerNorm(args.dim_hidden * 2)
        )

        # 重建预测头
        self.recon_head = nn.Sequential(
            nn.Linear(args.dim_hidden * 2, args.dim_hidden),
            nn.ReLU(),
            nn.Linear(args.dim_hidden, 1),
            nn.Sigmoid()
        )

    def _init_model(self, model_class, ckpt_path, args):
        """初始化并加载预训练模型"""
        model = model_class(
            num_rounds=1,
            dim_hidden=args.dim_hidden,
            enable_encode=True
        )
        
        # 加载检查点并冻结非VAE参数
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if 'decoder' in k or 'vae' in k:  # 不加载解码器参数
                    continue
                state_dict[k] = v
            model.load_state_dict(state_dict, strict=False)
            
            # 冻结非VAE参数
            for name, param in model.named_parameters():
                if 'vae' not in name and 'decoder' not in name:
                    param.requires_grad_(False)
        return model

    def _get_latent_features(self, model, G):
        """获取VAE潜在特征"""
        with torch.no_grad():
            # 前向传播获取mu和logstd
            _ = model(G)  # 触发VAE的前向计算
            mu = model.vae_mu.buffer  # 假设底层模型存储了mu
            logstd = model.vae_logstd.buffer  # 假设存储了logstd
            latent = mu + torch.exp(logstd) * torch.randn_like(logstd)
        return latent

    def mask_tokens(self, G, tokens, mask_ratio=0.05, k_hop=4):
        """改进的掩码方法，考虑电路结构"""
        if mask_ratio == 0:
            return tokens.clone(), None
            
        device = tokens.device
        seq_len = len(tokens)
        mask_indices = torch.randperm(seq_len, device=device)[:int(mask_ratio * seq_len)]
        
        # 基于电路结构的k-hop掩码
        edge_index = G.edge_index.to(device)
        for _ in range(k_hop):
            _, neighbors = subgraph(mask_indices, edge_index, dim=1)
            mask_indices = torch.unique(torch.cat([mask_indices, neighbors[0]]))
        
        # 应用掩码
        masked_tokens = tokens.clone()
        mask_token = self.mask_token.expand(len(mask_indices), -1)
        masked_tokens[mask_indices] = mask_token
        
        return masked_tokens, mask_indices

    def forward(self, G):
        device = next(self.parameters()).device
        G = G.to(device)
        
        # 获取各模型的隐藏状态和潜在特征
        def get_features(model, G):
            with torch.no_grad():
                hs, hf = model(G)
                latent = self._get_latent_features(model, G)
            return hs, hf, latent

        aig_hs, aig_hf, aig_latent = get_features(self.deepgate_aig, G)
        mig_hs, mig_hf, mig_latent = get_features(self.deepgate_mig, G)
        xmg_hs, xmg_hf, xmg_latent = get_features(self.deepgate_xmg, G)
        xag_hs, xag_hf, xag_latent = get_features(self.deepgate_xag, G)

        # 构建多模态tokens
        def build_tokens(hs, hf, latent):
            return torch.cat([
                hs, 
                hf, 
                latent[:, :self.latent_dim]  # 取前64维潜在特征
            ], dim=1)

        aig_tokens = build_tokens(aig_hs, aig_hf, aig_latent)
        mig_tokens = build_tokens(mig_hs, mig_hf, mig_latent)
        xmg_tokens = build_tokens(xmg_hs, xmg_hf, xmg_latent)
        xag_tokens = build_tokens(xag_hs, xag_hf, xag_latent)

        # 随机选择要掩码的模态
        modalities = {
            'aig': (aig_tokens, self.deepgate_aig, G.aig_batch),
            'mig': (mig_tokens, self.deepgate_mig, G.mig_batch),
            'xmg': (xmg_tokens, self.deepgate_xmg, G.xmg_batch),
            'xag': (xag_tokens, self.deepgate_xag, G.xag_batch)
        }
        selected_modality = list(modalities.keys())[torch.randint(0, 4, (1,)).item()]
        masked_tokens, model, batch_indices = modalities[selected_modality]
        
        # 应用结构感知掩码
        masked_tokens, mask_indices = self.mask_tokens(G, masked_tokens, self.mask_ratio)

        # 多模态融合
        batch_preds = []
        for batch_id in range(G.batch.max().item() + 1):
            # 收集当前batch的所有tokens
            batch_tokens = []
            for mod in modalities.values():
                tokens, _, indices = mod
                batch_mask = (indices == batch_id)
                batch_tokens.append(tokens[batch_mask])
            
            # 拼接并处理
            combined_tokens = torch.cat(batch_tokens, dim=0)
            tf_output = self.mask_tf(combined_tokens)
            
            # 投影到目标空间
            projected = self.vae_proj(tf_output)
            
            # 仅保留被掩码部分的预测
            if mask_indices is not None:
                batch_mask = (batch_indices[mask_indices] == batch_id)
                batch_preds.append(projected[batch_mask])
        
        # 重建预测
        reconstructed = torch.cat(batch_preds, dim=0)
        pred_prob = self.recon_head(reconstructed)

        # 各模态原始预测
        with torch.no_grad():
            aig_prob = self.deepgate_aig.pred_prob(aig_hf)
            mig_prob = self.deepgate_mig.pred_prob(mig_hf)
            xmg_prob = self.deepgate_xmg.pred_prob(xmg_hf)
            xag_prob = self.deepgate_xag.pred_prob(xag_hf)

        return {
            'reconstructed': reconstructed,
            'mask_indices': mask_indices,
            'pred_prob': pred_prob,
            'aig_prob': aig_prob,
            'mig_prob': mig_prob,
            'xmg_prob': xmg_prob,
            'xag_prob': xag_prob
        }

    def load_pretrained(self, pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module'):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
        self.load_state_dict(state_dict, strict=False)

    def configure_optimizers(self, lr=1e-4):
        return torch.optim.AdamW([
            {'params': self.mask_tf.parameters()},
            {'params': self.vae_proj.parameters()},
            {'params': self.recon_head.parameters()},
            {'params': self.mask_token, 'lr': lr*0.1}
        ], lr=lr)
