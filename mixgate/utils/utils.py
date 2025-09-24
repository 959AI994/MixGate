from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import math
import copy
import torch
import random
import numpy as np

from .circuit_utils import random_pattern_generator, logic

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def zero_normalization(x):
    mean_x = torch.mean(x)
    std_x = torch.std(x)
    # 避免除零问题
    if std_x == 0:
        return x - mean_x
    z_x = (x - mean_x) / std_x
    return z_x

class custom_DataParallel(nn.parallel.DataParallel):
# define a custom DataParallel class to accomodate igraph inputs
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(custom_DataParallel, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        # to overwride nn.parallel.scatter() to adapt igraph batch inputs
        G = inputs[0]
        scattered_G = []
        n = math.ceil(len(G) / len(device_ids))
        mini_batch = []
        for i, g in enumerate(G):
            mini_batch.append(g)
            if len(mini_batch) == n or i == len(G)-1:
                scattered_G.append((mini_batch, ))
                mini_batch = []
        return tuple(scattered_G), tuple([{}]*len(scattered_G))

def collate_fn(G):
    return [copy.deepcopy(g) for g in G]

def pyg_simulation(g, pattern=[]):
    # PI, Level list
    max_level = 0
    PI_indexes = []
    fanin_list = []
    for idx, ele in enumerate(g.forward_level):
        level = int(ele)
        fanin_list.append([])
        if level > max_level:
            max_level = level
        if level == 0:
            PI_indexes.append(idx)
    level_list = []
    for level in range(max_level + 1):
        level_list.append([])
    for idx, ele in enumerate(g.forward_level):
        level_list[int(ele)].append(idx)
    # Fanin list 
    for k in range(len(g.edge_index[0])):
        src = g.edge_index[0][k]
        dst = g.edge_index[1][k]
        fanin_list[dst].append(src)
    
    ######################
    # Simulation
    ######################
    y = [0] * len(g.x)
    if len(pattern) == 0:
        pattern = random_pattern_generator(len(PI_indexes))
    j = 0
    for i in PI_indexes:
        y[i] = pattern[j]
        j = j + 1
    for level in range(1, len(level_list), 1):
        for node_idx in level_list[level]:
            source_signals = []
            for pre_idx in fanin_list[node_idx]:
                source_signals.append(y[pre_idx])
            if len(source_signals) > 0:
                if int(g.x[node_idx][1]) == 1:
                    gate_type = 1
                elif int(g.x[node_idx][2]) == 1:
                    gate_type = 5
                else:
                    raise("This is PI")
                y[node_idx] = logic(gate_type, source_signals)

    # Output
    if len(level_list[-1]) > 1:
        raise('Too many POs')
    return y[level_list[-1][0]], pattern

def get_function_acc(g, node_emb):
    MIN_GAP = 0.05
    # Sample
    retry = 10000
    tri_sample_idx = 0
    correct = 0
    total = 0
    while tri_sample_idx < 100 and retry > 0:
        retry -= 1
        sample_pair_idx = torch.LongTensor(random.sample(range(len(g.tt_pair_index[0])), 2))
        pair_0 = sample_pair_idx[0]
        pair_1 = sample_pair_idx[1]
        pair_0_gt = g.tt_dis[pair_0]
        pair_1_gt = g.tt_dis[pair_1]
        if pair_0_gt == pair_1_gt:
            continue
        if abs(pair_0_gt - pair_1_gt) < MIN_GAP:
            continue

        total += 1
        tri_sample_idx += 1
        pair_0_sim = torch.cosine_similarity(node_emb[g.tt_pair_index[0][pair_0]].unsqueeze(0), node_emb[g.tt_pair_index[1][pair_0]].unsqueeze(0), eps=1e-8)
        pair_1_sim = torch.cosine_similarity(node_emb[g.tt_pair_index[0][pair_1]].unsqueeze(0), node_emb[g.tt_pair_index[1][pair_1]].unsqueeze(0), eps=1e-8)
        pair_0_predDis = 1 - pair_0_sim
        pair_1_predDis = 1 - pair_1_sim
        succ = False
        if pair_0_gt > pair_1_gt and pair_0_predDis > pair_1_predDis:
            succ = True
        elif pair_0_gt < pair_1_gt and pair_0_predDis < pair_1_predDis:
            succ = True
        if succ:
            correct += 1

    if total > 0:
        acc = correct * 1.0 / total
        return acc
    return -1
            
def generate_orthogonal_vectors(n, dim):
    if n < dim * 8:
        # Choice 1: Generate n random orthogonal vectors in R^dim
        # Generate an initial random vector
        v0 = np.random.randn(dim)
        v0 /= np.linalg.norm(v0)
        # Generate n-1 additional vectors
        vectors = [v0]
        for i in range(n-1):
            while True:
                # Generate a random vector
                v = np.random.randn(dim)

                # Project the vector onto the subspace spanned by the previous vectors
                for j in range(i+1):
                    v -= np.dot(v, vectors[j]) * vectors[j]

                if np.linalg.norm(v) > 0:
                    # Normalize the vector
                    v /= np.linalg.norm(v)
                    break

            # Append the vector to the list
            vectors.append(v)
    else: 
        # Choice 2: Generate n random vectors:
        vectors = np.random.rand(n, dim) - 0.5
        for i in range(n):
            vectors[i] = vectors[i] / np.linalg.norm(vectors[i])

    # # calculate the max cosine similarity between any two vectors
    # max_cos_sim = 0
    # for i in range(n):
    #     for j in range(i+1, n):
    #         vi = vectors[i]
    #         vj = vectors[j]
    #         cos_sim = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
    #         if cos_sim > max_cos_sim:
    #             max_cos_sim = cos_sim

    return vectors

def generate_hs_init(G, hs, no_dim, aig=False, mig=False, xmg=False, xag=False):
    
    if aig: 
        if G.batch == None:
            batch_size = 1
        else:
            batch_size = G.batch.max().item() + 1
        for batch_idx in range(batch_size):
            if G.batch == None:
                pi_mask = (G.forward_level == 0)
            else:
                pi_mask = (G.batch == batch_idx) & (G.forward_level == 0)
            pi_node = G.forward_index[pi_mask]
            pi_vec = generate_orthogonal_vectors(len(pi_node), no_dim)
            pi_vec = np.array(pi_vec, dtype=np.float32)
            hs[pi_node] = torch.tensor(pi_vec, dtype=torch.float)

    if xmg:
        if G.batch == None:
            batch_size = 1
        else:
            batch_size = G.xmg_batch.max().item() + 1
        for batch_idx in range(batch_size):
            if G.batch == None:
                pi_mask = (G.xmg_forward_level == 0)
            else:
                pi_mask = (G.xmg_batch == batch_idx) & (G.xmg_forward_level == 0)
            pi_node = G.xmg_forward_index[pi_mask]
            pi_vec = generate_orthogonal_vectors(len(pi_node), no_dim)
            pi_vec = np.array(pi_vec, dtype=np.float32)
            hs[pi_node] = torch.tensor(pi_vec, dtype=torch.float)

    if mig:
        if G.batch == None:
                batch_size = 1
        else:
            batch_size = G.mig_batch.max().item() + 1
        for batch_idx in range(batch_size):
            if G.batch == None:
                pi_mask = (G.mig_forward_level == 0)
            else:
                pi_mask = (G.mig_batch == batch_idx) & (G.mig_forward_level == 0)
            pi_node = G.mig_forward_index[pi_mask]
            pi_vec = generate_orthogonal_vectors(len(pi_node), no_dim)
            pi_vec = np.array(pi_vec, dtype=np.float32)
            hs[pi_node] = torch.tensor(pi_vec, dtype=torch.float)

    if xag:
        if G.batch == None:
                batch_size = 1
        else:
            batch_size = G.xag_batch.max().item() + 1
        for batch_idx in range(batch_size):
            if G.batch == None:
                pi_mask = (G.xag_forward_level == 0)
            else:
                pi_mask = (G.xag_batch == batch_idx) & (G.xag_forward_level == 0)
            pi_node = G.xag_forward_index[pi_mask]
            pi_vec = generate_orthogonal_vectors(len(pi_node), no_dim)
            pi_vec = np.array(pi_vec, dtype=np.float32)
            hs[pi_node] = torch.tensor(pi_vec, dtype=torch.float)    
    return hs

def generate_k_hop_tensor(g, k):
    n = g.x.size(0)  # Number of nodes
    max_hops = k*2 + 1
    hop_x = torch.zeros((n, max_hops, g.x.size(1)))  # Initialize output tensor
    x = g.x.to('cpu')
    edge_index = g.edge_index.to('cpu')
    
    for i in range(n):
        current_nodes = [i]
        tp = 0
        hop_x[i, tp, :] = x[current_nodes[0]]
        tp += 1
        for hop_lev in range(0, k-1):
            neighbors = []
            for current_node in current_nodes:
                neighbor = edge_index[0][edge_index[1] == current_node]
                neighbors += neighbor.tolist()
            neighbors = list(set(neighbors))
            neighbors = torch.tensor(neighbors)
            if len(neighbors) == 0:
                continue
            if tp+len(neighbors) >= max_hops:
                hop_x[i, tp:tp+len(neighbors), :] = x[neighbors[:max_hops-tp]]
                break
            else:
                hop_x[i, tp:tp+len(neighbors), :] = x[neighbors]
            tp += len(neighbors)
            current_nodes = neighbors
            
    g.aig_hop_x = hop_x.to(g.x.device)
    return g
        