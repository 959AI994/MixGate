import torch
import torch.nn as nn
import numpy as np

class MultiViewAlignmentLoss(nn.Module):
    """
    Multi-view alignment loss.
    Uses L1 to align functional hidden states (hf) at equivalent nodes (AIG vs other views).
    Expects equivalence labels as produced by parse_pair.py.
    """

    def __init__(self, loss_weight=1.0):
        super(MultiViewAlignmentLoss, self).__init__()
        self.loss_weight = loss_weight
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, graph, hf_dict):
        """
        Compute multi-view alignment loss.

        Args:
            graph: PyG-style batch with equivalence attributes, e.g.:
                - graph.aig_mig_equ: AIG-side indices for AIG–MIG pairs
                - graph.mig_aig_equ: MIG-side indices
                - graph.aig_xmg_equ / graph.xmg_aig_equ
                - graph.aig_xag_equ / graph.xag_aig_equ
            hf_dict: Per-view hf tensors, e.g.:
                {
                    'aig': aig_hf,
                    'xmg': xmg_hf,
                    'xag': xag_hf,
                    'mig': mig_hf
                }

        Returns:
            total_loss: Weighted sum of pairwise alignment losses.
            loss_dict: Per-pair diagnostics (loss value, num pairs, etc.).
        """
        total_loss = 0.0
        loss_dict = {}

        # (view_a, view_b, attr on graph for AIG indices, attr for other view)
        view_pairs = [
            ('aig', 'mig', 'aig_mig_equ', 'mig_aig_equ'),
            ('aig', 'xmg', 'aig_xmg_equ', 'xmg_aig_equ'),
            ('aig', 'xag', 'aig_xag_equ', 'xag_aig_equ')
        ]

        debug_info = {}

        for view1, view2, equ_key1, equ_key2 in view_pairs:
            equivalent_indices1 = getattr(graph, equ_key1, None)  # AIG indices
            equivalent_indices2 = getattr(graph, equ_key2, None)  # other view indices

            debug_info[f'{equ_key1}_exists'] = equivalent_indices1 is not None
            debug_info[f'{equ_key2}_exists'] = equivalent_indices2 is not None
            if equivalent_indices1 is not None:
                debug_info[f'{equ_key1}_len'] = len(equivalent_indices1)
                debug_info[f'{equ_key1}_type'] = type(equivalent_indices1).__name__
                debug_info[f'{equ_key1}_sample'] = equivalent_indices1[:3] if len(equivalent_indices1) > 0 else []
            if equivalent_indices2 is not None:
                debug_info[f'{equ_key2}_len'] = len(equivalent_indices2)
                debug_info[f'{equ_key2}_type'] = type(equivalent_indices2).__name__
                debug_info[f'{equ_key2}_sample'] = equivalent_indices2[:3] if len(equivalent_indices2) > 0 else []

            if equivalent_indices1 is not None and equivalent_indices2 is not None and \
               len(equivalent_indices1) > 0 and len(equivalent_indices2) > 0:

                hf1 = hf_dict.get(view1)
                hf2 = hf_dict.get(view2)

                debug_info[f'{view1}_hf_exists'] = hf1 is not None
                debug_info[f'{view2}_hf_exists'] = hf2 is not None
                if hf1 is not None:
                    debug_info[f'{view1}_hf_shape'] = hf1.shape
                if hf2 is not None:
                    debug_info[f'{view2}_hf_shape'] = hf2.shape

                if hf1 is not None and hf2 is not None:
                    try:
                        if not isinstance(equivalent_indices1, torch.Tensor):
                            if isinstance(equivalent_indices1, (list, tuple)):
                                if len(equivalent_indices1) > 0 and hasattr(equivalent_indices1[0], '__len__'):
                                    equivalent_indices1 = equivalent_indices1[0]
                                equivalent_indices1 = [int(idx) if isinstance(idx, (int, float, np.integer)) else idx for idx in equivalent_indices1]
                                equivalent_indices1 = [idx for idx in equivalent_indices1 if isinstance(idx, int) and idx >= 0]
                            equivalent_indices1 = torch.tensor(equivalent_indices1, dtype=torch.long, device=hf1.device)

                        if not isinstance(equivalent_indices2, torch.Tensor):
                            if isinstance(equivalent_indices2, (list, tuple)):
                                if len(equivalent_indices2) > 0 and hasattr(equivalent_indices2[0], '__len__'):
                                    equivalent_indices2 = equivalent_indices2[0]
                                equivalent_indices2 = [int(idx) if isinstance(idx, (int, float, np.integer)) else idx for idx in equivalent_indices2]
                                equivalent_indices2 = [idx for idx in equivalent_indices2 if isinstance(idx, int) and idx >= 0]
                            equivalent_indices2 = torch.tensor(equivalent_indices2, dtype=torch.long, device=hf2.device)

                        debug_info[f'{equ_key1}_tensor_shape'] = equivalent_indices1.shape
                        debug_info[f'{equ_key2}_tensor_shape'] = equivalent_indices2.shape
                        debug_info[f'{equ_key1}_tensor_sample'] = equivalent_indices1[:3].tolist()
                        debug_info[f'{equ_key2}_tensor_sample'] = equivalent_indices2[:3].tolist()

                        min_len = min(len(equivalent_indices1), len(equivalent_indices2))
                        if min_len == 0:
                            loss_dict[f'{view1}_to_{view2}_hf_alignment_loss'] = 0.0
                            loss_dict[f'{view1}_to_{view2}_num_pairs'] = 0
                            continue

                        equivalent_indices1 = equivalent_indices1[:min_len]
                        equivalent_indices2 = equivalent_indices2[:min_len]

                        valid_indices1 = (equivalent_indices1 < len(hf1)).all()
                        valid_indices2 = (equivalent_indices2 < len(hf2)).all()

                        debug_info[f'{equ_key1}_valid'] = valid_indices1.item()
                        debug_info[f'{equ_key2}_valid'] = valid_indices2.item()
                        debug_info[f'{equ_key1}_max_idx'] = equivalent_indices1.max().item()
                        debug_info[f'{equ_key2}_max_idx'] = equivalent_indices2.max().item()
                        debug_info[f'{view1}_hf_len'] = len(hf1)
                        debug_info[f'{view2}_hf_len'] = len(hf2)

                        if valid_indices1 and valid_indices2:
                            hf1_equivalent = hf1[equivalent_indices1]
                            hf2_equivalent = hf2[equivalent_indices2]

                            debug_info[f'{view1}_hf_equiv_shape'] = hf1_equivalent.shape
                            debug_info[f'{view2}_hf_equiv_shape'] = hf2_equivalent.shape

                            hf_alignment_loss = self.l1_loss(hf1_equivalent, hf2_equivalent)

                            debug_info[f'{view1}_to_{view2}_loss'] = hf_alignment_loss.item()

                            total_loss += hf_alignment_loss
                            loss_dict[f'{view1}_to_{view2}_hf_alignment_loss'] = hf_alignment_loss.item()
                            loss_dict[f'{view1}_to_{view2}_num_pairs'] = len(equivalent_indices1)
                        else:
                            print(f"Warning: Invalid indices in {equ_key1} or {equ_key2}, skipping this pair")
                            loss_dict[f'{view1}_to_{view2}_hf_alignment_loss'] = 0.0
                            loss_dict[f'{view1}_to_{view2}_num_pairs'] = 0
                    except Exception as e:
                        print(f"Error processing {equ_key1}/{equ_key2}: {e}")
                        debug_info[f'{view1}_to_{view2}_error'] = str(e)
                        loss_dict[f'{view1}_to_{view2}_hf_alignment_loss'] = 0.0
                        loss_dict[f'{view1}_to_{view2}_num_pairs'] = 0
                else:
                    loss_dict[f'{view1}_to_{view2}_hf_alignment_loss'] = 0.0
                    loss_dict[f'{view1}_to_{view2}_num_pairs'] = 0
            else:
                loss_dict[f'{view1}_to_{view2}_hf_alignment_loss'] = 0.0
                loss_dict[f'{view1}_to_{view2}_num_pairs'] = 0

        if not hasattr(self, '_debug_printed'):
            print("Alignment Loss Debug Info:")
            for key, value in debug_info.items():
                print(f"  {key}: {value}")
            self._debug_printed = True

        return total_loss * self.loss_weight, loss_dict
