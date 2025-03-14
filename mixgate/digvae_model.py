from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import roc_auc_score, average_precision_score

import torch
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from .digae_layer import InnerProductDecoder, DirectedInnerProductDecoder

EPS        = 1e-15
MAX_LOGSTD = 10

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.
    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

class DirectedGVAE(torch.nn.Module):
    def __init__(self, encoder, dim_hidden, decoder=None):
        super(DirectedGVAE, self).__init__()
        self.encoder = encoder
        self.decoder = DirectedInnerProductDecoder() if decoder is None else decoder
        DirectedGVAE.reset_parameters(self)
        self.dim_hidden = dim_hidden
        self.fc_s_mu = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_s_logstd = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_t_mu = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_t_logstd = torch.nn.Linear(dim_hidden, dim_hidden)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # Encode
        s, t = self.encoder(x, x, edge_index)
        
        # Sample
        sample_s, sample_t = self.sample(s, t)
        
        # Decode
        adj_pred = self.decoder(sample_s, sample_t)
        return adj_pred
    
    def sample(self, s, t):
        self.s_mu, self.s_logstd = self.fc_s_mu(s), self.fc_s_logstd(s)
        self.t_mu, self.t_logstd = self.fc_t_mu(t), self.fc_t_logstd(t)
        guassian_noise_s = torch.randn_like(self.s_mu).to(self.s_mu.device)
        sample_s = self.s_mu + torch.exp(self.s_logstd) * guassian_noise_s
        guassian_noise_t = torch.randn_like(self.t_mu).to(self.t_mu.device)
        sample_t = self.t_mu + torch.exp(self.t_logstd) * guassian_noise_t
        
        return sample_s, sample_t

    
    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    
    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    
    def recon_loss(self, s, t, pos_edge_index, neg_edge_index=None):
        s, t = self.sample(s, t)
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

    def test(self, s, t, pos_edge_index, neg_edge_index):
        # XXX
        pos_y = s.new_ones(pos_edge_index.size(1))
        neg_y = s.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(s, t, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(s, t, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

