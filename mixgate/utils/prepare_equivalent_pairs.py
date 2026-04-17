import numpy as np
import torch
from collections import defaultdict

def prepare_equivalent_pairs_data(circuits_data, sat_sweep_results):
    """
    Prepare equivalent-node-pair data (AIG-to-other-view pairs only).

    Args:
        circuits_data: Raw circuit data dict.
        sat_sweep_results: SAT sweeping output containing equivalent pairs.

    Returns:
        Updated circuits_data with equivalent-pair fields.
    """

    for circuit_name in circuits_data:
        if circuit_name in sat_sweep_results:
            # SAT sweeping equivalent pairs for this circuit
            sat_pairs = sat_sweep_results[circuit_name]

            # Initialize pair lists (AIG to other views only)
            equivalent_pairs = {
                'aig_to_xmg': [],
                'aig_to_xag': [],
                'aig_to_mig': []
            }

            # Map SAT sweeping results to AIG-to-other-view pairs
            for pair in sat_pairs:
                aig_node_id = pair['aig_node']
                xmg_node_id = pair.get('xmg_node', -1)
                xag_node_id = pair.get('xag_node', -1)
                mig_node_id = pair.get('mig_node', -1)

                # Only add AIG-to-other-view pairs
                if xmg_node_id != -1:
                    equivalent_pairs['aig_to_xmg'].append([aig_node_id, xmg_node_id])
                if xag_node_id != -1:
                    equivalent_pairs['aig_to_xag'].append([aig_node_id, xag_node_id])
                if mig_node_id != -1:
                    equivalent_pairs['aig_to_mig'].append([aig_node_id, mig_node_id])

            # Attach equivalent pairs to circuit record
            circuits_data[circuit_name]['equivalent_pairs'] = equivalent_pairs

    return circuits_data

def create_synthetic_equivalent_pairs(circuits_data, equivalent_ratio=0.1):
    """
    Create synthetic equivalent pairs (for testing).
    Only AIG-to-other-view pairs.

    Args:
        circuits_data: Raw circuit data dict.
        equivalent_ratio: Fraction of node count used as pair count (capped by view sizes).

    Returns:
        Updated circuits_data with synthetic equivalent-pair fields.
    """

    for circuit_name in circuits_data:
        circuit = circuits_data[circuit_name]

        # Node counts per view
        aig_nodes = len(circuit['aig_x'])
        xmg_nodes = len(circuit['xmg_x'])
        xag_nodes = len(circuit['xag_x'])
        mig_nodes = len(circuit['mig_x'])

        equivalent_pairs = {
            'aig_to_xmg': [],
            'aig_to_xag': [],
            'aig_to_mig': []
        }

        # Random AIG-to-other-view pairs
        view_pairs = [
            ('aig_to_xmg', aig_nodes, xmg_nodes),
            ('aig_to_xag', aig_nodes, xag_nodes),
            ('aig_to_mig', aig_nodes, mig_nodes)
        ]

        for pair_name, aig_nodes_count, other_nodes_count in view_pairs:
            num_pairs = int(min(aig_nodes_count, other_nodes_count) * equivalent_ratio)
            if num_pairs > 0:
                aig_indices = np.random.choice(aig_nodes_count, num_pairs, replace=False)
                other_indices = np.random.choice(other_nodes_count, num_pairs, replace=False)

                pairs = [[int(i), int(j)] for i, j in zip(aig_indices, other_indices)]
                equivalent_pairs[pair_name] = pairs

        circuits_data[circuit_name]['equivalent_pairs'] = equivalent_pairs

    return circuits_data

def save_equivalent_pairs_data(circuits_data, output_path):
    """
    Save circuits data including equivalent pairs.

    Args:
        circuits_data: Circuit dict with equivalent pairs.
        output_path: Output .npz path.
    """
    np.savez_compressed(output_path, circuits=circuits_data)
    print(f"Saved equivalent pairs data to {output_path}")

if __name__ == "__main__":
    print("Creating synthetic equivalent pairs for testing...")

    # Load raw data
    # circuits_data = np.load('your_original_data.npz', allow_pickle=True)['circuits'].item()

    # Create synthetic pairs (for testing)
    # circuits_data = create_synthetic_equivalent_pairs(circuits_data, equivalent_ratio=0.1)

    # Save updated data
    # save_equivalent_pairs_data(circuits_data, 'data_with_equivalent_pairs.npz')

    print("Example script completed. Please modify according to your actual data format.")
