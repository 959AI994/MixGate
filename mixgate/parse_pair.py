from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Callable, List
import os.path as osp

import numpy as np 
import torch
import shutil
import os
import copy
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from mixgate.utils.data_utils import read_npz_file
from mixgate.utils.aiger_utils import aig_to_xdata
from mixgate.utils.circuit_utils import get_fanin_fanout, read_file, add_node_index, feature_gen_connect
from mixgate.utils.dataset_utils import *
from mixgate.utils.data_utils import construct_node_feature
from mixgate.utils.dag_utils import return_order_info
import mixgate.utils.circuit_utils as circuit_utils

K_HOP = 7
MAX_HOP_LENGTH = 511

class NpzParser_Pair():
    '''
        Parse the npz file into an inmemory torch_geometric.data.Data object
    '''
    def __init__(self, data_dir, circuit_path, \
                 random_shuffle=True, trainval_split=0.9): 
        self.data_dir = data_dir
        dataset = self.inmemory_dataset(data_dir, circuit_path)
        if random_shuffle:
            perm = torch.randperm(len(dataset))
            dataset = dataset[perm]
        data_len = len(dataset)
        training_cutoff = int(data_len * trainval_split)
        self.train_dataset = dataset[:training_cutoff]
        self.val_dataset = dataset[training_cutoff:]
        # self.train_dataset = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        # self.val_dataset = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    def get_dataset(self):
        return self.train_dataset, self.val_dataset
    
    class inmemory_dataset(InMemoryDataset):
        def __init__(self, root, circuit_path, transform=None, pre_transform=None, pre_filter=None):
            self.name = 'npz_inmm_dataset'
            self.root = root
            self.circuit_path = circuit_path
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
        @property
        def raw_dir(self):
            return self.root

        @property
        def processed_dir(self):
            name = 'merge_consistency1500_1%_inmemory'
            # name = 'merge_consistency1508_inmemory'
            return osp.join(self.root, name)

        @property
        def raw_file_names(self) -> List[str]:
            return [self.circuit_path]

        @property
        def processed_file_names(self) -> str:
            return ['data.pt']

        def download(self):
            pass

        ### for aig
        def process(self):
            data_list = []
            tot_pairs = 0
            circuits = read_npz_file(self.circuit_path)['circuits'].item()
            j = 0
            # # epfl dataset
            # for cir_idx, cir_name in enumerate(circuits):
            #     print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100))
            #     #--------------------if model type = aig exchange "circuits" with "labels"--------------------
            #     # x = circuits[cir_name]["x"]
            #     # edge_index = circuits[cir_name]["edge_index"]
            #     # tt_dis = circuits[cir_name]['tt_sim']#in aig_graphs there is no tt_dis but tt_sim
            #     # tt_pair_index = circuits[cir_name]['tt_pair_index']
            #     # prob = circuits[cir_name]['prob']
                
            #     #---------------------------------------------------------------------------------------------
            #     x = circuits[cir_name]["aig_x"]
            #     # gate = circuits[cir_name]["gate"]
            #     #print("gate =", gate.shape)
            #     forward_level = circuits[cir_name]["forward_level"]
            #     #print("forward_level =",forward_level.shape)

            #     print("x =", x)
            #     edge_index = circuits[cir_name]["edge_index"]
            #     print("edge_index =", edge_index.shape)
            #     tt_dis = circuits[cir_name]['tt_sim']
            #     tt_pair_index = circuits[cir_name]['tt_pair_index']
            #     prob = circuits[cir_name]['prob']

            #     if len(tt_pair_index) == 0 :
            #         print('No tt or rc pairs: ', cir_name)
            #         continue

            #     tot_pairs += len(tt_dis)

            #     # check the gate types

            # 1500
            for cir_idx, cir_name in enumerate(circuits):
                # if len(circuits[cir_name]) != 16:
                #     print(f"Skipping circuit {cir_name} with length {len(circuits[cir_name])}")
        
                #if len(circuits[cir_name]) == 17:
                print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx+1, len(circuits), (cir_idx+1) / len(circuits) *100))
                
                #aig:
                x = circuits[cir_name]["aig_x"]
                edge_index = circuits[cir_name]["aig_edge_index"]
                # is_pi = circuits[cir_name]["is_pi"]
                # no_edges = circuits[cir_name]["no_edges"]
                prob = circuits[cir_name]["aig_prob"]
                # backward_level = circuits[cir_name]["backward_level"]
                # forward_index = circuits[cir_name]["forward_index"]
                # forward_level = circuits[cir_name]["forward_level"]
                # no_nodes = circuits[cir_name]["no_nodes"]
                # backward_index = circuits[cir_name]["backward_index"]
                tt_dis = circuits[cir_name]["aig_tt_sim"]
                tt_pair_index = circuits[cir_name]["aig_tt_pair_index"]

                if len(tt_pair_index) == 0:
                    print('No tt : ', cir_name)
                    continue

                connect_label = None
                connect_pair_index = None


                graph = parse_pyg_mlpgate(
                    x, edge_index, tt_dis, tt_pair_index, prob
                )
                graph.num_nodes = len(x)
                graph.batch = torch.zeros(len(graph.x), dtype=torch.long)
                graph.gate = torch.tensor(circuits[cir_name]["aig_gate"], dtype=torch.float)
                graph.aig_mig_equ = circuits[cir_name]["aig_mig_equ"]
                graph.mig_aig_equ = circuits[cir_name]["mig_aig_equ"]
                graph.aig_xmg_equ = circuits[cir_name]["aig_xmg_equ"]
                graph.xmg_aig_equ = circuits[cir_name]["xmg_aig_equ"]
                graph.aig_xag_equ = circuits[cir_name]["aig_xag_equ"]
                graph.xag_aig_equ = circuits[cir_name]["xag_aig_equ"]
                # Hop / Stone 03.13
                for modal in ['mig', 'aig', 'xmg', 'xag']:
                    x = circuits[cir_name]["{}_x".format(modal)]
                    edge_index = circuits[cir_name]["{}_edge_index".format(modal)]
                    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
                        edge_index = edge_index.T
                    elif edge_index.shape[0] == 2 and edge_index.shape[1] != 2:
                        edge_index = edge_index
                    else:
                        continue
                    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, len(x))
                    po_list = forward_index[backward_level == 0]
                    max_level = forward_level.max().item()
                    level_list = [[] for _ in range(max_level+1)]
                    for idx, lev in enumerate(forward_level):
                        level_list[lev.item()].append(idx)
                    cur_node_level = K_HOP
                    cur_hop_level = 0
                    hop_list = np.zeros((0, MAX_HOP_LENGTH), dtype=int)
                    hop_lev_list = []
                    hop_node_list = []
                    hop_length = []
                    max_hop = 0
                    while cur_node_level < max_level:
                        for idx in level_list[cur_node_level]:
                            hop_nodes = circuit_utils.get_hops(idx, edge_index, k_hop=K_HOP)
                            hop_nodes_pad = [-1] * MAX_HOP_LENGTH
                            hop_nodes_pad = np.array(hop_nodes_pad).reshape(1, MAX_HOP_LENGTH)
                            hop_nodes_pad[0, :len(hop_nodes)] = hop_nodes
                            hop_list = np.concatenate((hop_list, hop_nodes_pad), axis=0)
                            hop_lev_list.append(cur_hop_level)
                            hop_node_list.append(idx)
                            hop_length.append(len(hop_nodes))
                            if len(hop_node_list) > max_hop:
                                max_hop = len(hop_node_list)
                            
                        cur_node_level += (K_HOP - 2)
                        cur_hop_level += 1
                    for po_idx in po_list:
                        if po_idx not in hop_node_list:
                            hop_nodes = circuit_utils.get_hops(po_idx, edge_index, k_hop=K_HOP)
                            hop_nodes_pad = np.zeros((1, MAX_HOP_LENGTH), dtype=int)
                            hop_nodes_pad[0, :len(hop_nodes)] = hop_nodes
                            hop_list = np.concatenate((hop_list, hop_nodes_pad), axis=0)
                            hop_lev_list.append(cur_hop_level)
                            hop_node_list.append(po_idx)
                            hop_length.append(len(hop_nodes))
                            if len(hop_node_list) > max_hop:
                                max_hop = len(hop_node_list)
                                
                    print('{} hop list length: {}/ max hop: {}'.format(modal, len(hop_list), max_hop))
                    graph['{}_hop'.format(modal)] = torch.tensor(hop_list)
                    graph['{}_hop_lev'.format(modal)] = torch.tensor(hop_lev_list)
                    graph['{}_hop_node'.format(modal)] = torch.tensor(hop_node_list)
                    graph['{}_hop_length'.format(modal)] = torch.tensor(hop_length)

                #xmg:
                xmg_edge_index =  torch.tensor(circuits[cir_name]["xmg_edge_index"], dtype=torch.long).contiguous()
                xmg_x = circuits[cir_name]["xmg_x"]
                xmg_forward_level, xmg_forward_index, xmg_backward_level, xmg_backward_index = return_order_info(xmg_edge_index, torch.LongTensor(xmg_x).size(0))
                graph.xmg_tt_dis = circuits[cir_name]["xmg_tt_sim"]
                graph.xmg_tt_pair_index = torch.tensor(circuits[cir_name]["xmg_tt_pair_index"], dtype=torch.long).contiguous()
                if len(graph.xmg_tt_pair_index) == 0:
                    print('No tt : ', cir_name)
                    continue
                graph.xmg_x = torch.tensor(circuits[cir_name]["xmg_x"])
                graph.xmg_edge_index = torch.tensor(circuits[cir_name]["xmg_edge_index"], dtype=torch.long).contiguous()
                graph.xmg_prob = torch.tensor(circuits[cir_name]["xmg_prob"])
                graph.xmg_forward_level = xmg_forward_level
                graph.xmg_forward_index = xmg_forward_index
                graph.xmg_backward_level = xmg_backward_level
                graph.xmg_backward_index = xmg_backward_index
                graph.xmg_batch = torch.zeros(len(graph.xmg_x), dtype=torch.long)
                graph.xmg_gate = torch.tensor(xmg_x[:, 1:2], dtype=torch.float)

                #xag
                xag_edge_index = torch.tensor(circuits[cir_name]["xag_edge_index"], dtype=torch.long).contiguous()
                xag_x = circuits[cir_name]["xag_x"]
                xag_forward_level, xag_forward_index, xag_backward_level, xag_backward_index = return_order_info(xag_edge_index, torch.LongTensor(xag_x).size(0))
                graph.xag_tt_dis = circuits[cir_name]["xag_tt_sim"]
                graph.xag_tt_pair_index =  torch.tensor(circuits[cir_name]["xag_tt_pair_index"], dtype=torch.long).contiguous()
                if len(graph.xag_tt_pair_index) == 0:
                    print('No tt : ', cir_name)
                    continue
                graph.xag_x = torch.tensor(circuits[cir_name]["xag_x"])
                graph.xag_edge_index = torch.tensor(circuits[cir_name]["xag_edge_index"], dtype=torch.long).contiguous()
                graph.xag_prob = torch.tensor(circuits[cir_name]["xag_prob"])
                graph.xag_forward_level = xag_forward_level
                graph.xag_forward_index = xag_forward_index
                graph.xag_backward_level = xag_backward_level
                graph.xag_backward_index = xag_backward_index
                graph.xag_batch = torch.zeros(len(graph.xag_x), dtype=torch.long)
                graph.xag_gate = torch.tensor(circuits[cir_name]["xag_x"][:, 1:2], dtype=torch.float)

                #mig
                graph.mig_tt_dis = circuits[cir_name]["mig_tt_sim"]
                mig_x = circuits[cir_name]["mig_x"]
                mig_edge_index = torch.tensor(circuits[cir_name]["mig_edge_index"], dtype=torch.long).contiguous()
                graph.mig_tt_pair_index =  torch.tensor(circuits[cir_name]["mig_tt_pair_index"], dtype=torch.long).contiguous()
                graph.mig_x = torch.tensor(circuits[cir_name]["mig_x"])
                graph.mig_edge_index = torch.tensor(circuits[cir_name]["mig_edge_index"], dtype=torch.long).contiguous()
                if len(graph.mig_tt_pair_index) == 0:
                    print('No tt : ', cir_name)
                    continue
                mig_forward_level, mig_forward_index, mig_backward_level, mig_backward_index = return_order_info(mig_edge_index, torch.LongTensor(mig_x).size(0))
                graph.mig_prob = torch.tensor(circuits[cir_name]["mig_prob"])
                graph.mig_forward_level = mig_forward_level
                graph.mig_forward_index = mig_forward_index
                graph.mig_backward_level = mig_backward_level
                graph.mig_backward_index = mig_backward_index
                graph.mig_batch = torch.zeros(len(graph.mig_x), dtype=torch.long)
                graph.mig_gate = torch.tensor(circuits[cir_name]["mig_x"][:, 1:2], dtype=torch.float)
                graph.name = cir_name

                # Equivalent-node-pair labels (AIG to other views only)
                # Expected at circuits[cir_name]["equivalent_pairs"] if enabled
                # if "equivalent_pairs" in circuits[cir_name]:
                #     equivalent_pairs = circuits[cir_name]["equivalent_pairs"]
                    
                #     # AIG-to-other-view pairs only
                #     aig_to_other_pairs = ['aig_to_xmg', 'aig_to_xag', 'aig_to_mig']
                    
                #     for view_pair in aig_to_other_pairs:
                #         if view_pair in equivalent_pairs and len(equivalent_pairs[view_pair]) > 0:
                #             pairs = equivalent_pairs[view_pair]
                            
                #             # Build pair index tensors
                #             pair_indices = torch.tensor(pairs, dtype=torch.long)  # shape: [num_pairs, 2]
                            
                #             # Store on graph
                #             graph[f'{view_pair}_equivalent_indices'] = pair_indices
                #             graph[f'{view_pair}_num_equivalent_pairs'] = torch.tensor(len(pairs), dtype=torch.long)
                #         else:
                #             # Empty tensor if no pairs
                #             graph[f'{view_pair}_equivalent_indices'] = torch.empty((0, 2), dtype=torch.long)
                #             graph[f'{view_pair}_num_equivalent_pairs'] = torch.tensor(0, dtype=torch.long)
                # else:
                #     # No equivalent-pair data: use empty tensors for AIG-other pairs
                #     aig_to_other_pairs = ['aig_to_xmg', 'aig_to_xag', 'aig_to_mig']
                #     for view_pair in aig_to_other_pairs:
                #         graph[f'{view_pair}_equivalent_indices'] = torch.empty((0, 2), dtype=torch.long)
                #         graph[f'{view_pair}_num_equivalent_pairs'] = torch.tensor(0, dtype=torch.long)

                data_list.append(graph)
                
                # # Stone: Add for debug
                # if len(data_list) > 20:
                #     break
                
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))

        def __repr__(self) -> str:
            return f'{self.name}({len(self)})'

class AigParser():
    def __init__(self):
        pass
    
    def read_aiger(self, aig_path):
        circuit_name = os.path.basename(aig_path).split('.')[0]
        # tmp_aag_path = os.path.join(self.tmp_dir, '{}.aag'.format(circuit_name))
        x_data, edge_index = aig_to_xdata(aig_path)
        # os.remove(tmp_aag_path)
        # Construct graph object 
        x_data = np.array(x_data)
        edge_index = np.array(edge_index)
        tt_dis = []
        tt_pair_index = []
        prob = [0] * len(x_data)
        rc_pair_index = []
        is_rc = []
        graph = parse_pyg_mlpgate(
            x_data, edge_index, tt_dis, tt_pair_index, prob, rc_pair_index, is_rc
        )
        graph.name = circuit_name
        graph.PIs = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] != 0)]
        graph.POs = graph.backward_index[(graph['backward_level'] == 0) & (graph['forward_level'] != 0)]
        graph.no_connect = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] == 0)]
        
        return graph        
        
class BenchParser():
    def __init__(self, gate_to_index={'INPUT': 0, 'MAJ': 1, 'NOT': 2, 'AND': 3, 'OR': 4, 'XOR': 5}):
        self.gate_to_index = gate_to_index
        pass
    
    def read_bench(self, bench_path):
        circuit_name = os.path.basename(bench_path).split('.')[0]
        x_data = read_file(bench_path)
        x_data, num_nodes, _ = add_node_index(x_data)
        x_data, edge_index = feature_gen_connect(x_data, self.gate_to_index)
        for idx in range(len(x_data)):
            x_data[idx] = [idx, int(x_data[idx][1])]
        # os.remove(tmp_aag_path)
        # Construct graph object 
        x_data = np.array(x_data)
        edge_index = np.array(edge_index)
        tt_dis = []
        tt_pair_index = []
        prob = [0] * len(x_data)
        rc_pair_index = []
        is_rc = []
        graph = parse_pyg_mlpgate(
            x_data, edge_index, tt_dis, tt_pair_index, prob, rc_pair_index, is_rc
        )
        graph.name = circuit_name
        graph.PIs = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] != 0)]
        graph.POs = graph.backward_index[(graph['backward_level'] == 0) & (graph['forward_level'] != 0)]
        graph.no_connect = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] == 0)]
        
        return graph       
