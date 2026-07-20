import torch

from etflow.ecir.bac_constraints import sparse_clash_edges


def test_sparse_clash_excludes_bonded_and_one_three_pairs():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.4, 0.0]])
    bonds = torch.tensor([[0, 1], [1, 2]])
    result = sparse_clash_edges(
        coordinates, bonds, cutoff=1.0, allowed_contact=0.8, exclude_topology_distance=2
    )
    pairs = {tuple(pair) for pair in result["edge_index"].t().tolist()}
    assert (0, 1) not in pairs and (0, 2) not in pairs
    assert (0, 3) in pairs
