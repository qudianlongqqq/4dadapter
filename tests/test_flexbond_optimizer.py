import torch

from etflow.commons.flexbond_jacobian import (
    apply_bond_jacobian,
    jacobian_sanity_check,
    solve_q_star_least_squares,
)
from etflow.commons.kabsch_utils import (
    kabsch_align,
    kabsch_rmsd,
    select_best_reference_conformer,
)
from etflow.models.components.light_egnn_refiner import LightEGNNRefinerBackbone


def _rotation():
    return torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )


def test_kabsch_selects_best_reference_and_aligns_to_init():
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.5]]
    )
    transformed = x @ _rotation().T + torch.tensor([3.0, -2.0, 1.0])
    poor = transformed.clone()
    poor[2, 2] += 2.0
    ref, aligned, index, rmsds = select_best_reference_conformer(
        x, torch.stack([poor, transformed])
    )
    assert index == 1
    assert rmsds[1] < 1.0e-5
    torch.testing.assert_close(aligned, x, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(kabsch_align(ref, x), x, atol=1.0e-5, rtol=1.0e-5)
    assert kabsch_rmsd(ref, x) < 1.0e-5


def test_flexbond_jacobian_solve_and_rotation_covariance():
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 2.0]]
    )
    target = {
        "anchor_index": torch.tensor([0]),
        "moving_index": torch.tensor([1]),
        "affected_atom_index": torch.tensor([1, 2, 3]),
        "affected_bond_index": torch.tensor([0, 0, 0]),
    }
    q = torch.tensor([[0.1, 0.2, -0.3, 0.4]])
    velocity, _ = apply_bond_jacobian(x, q, target)
    q_star, valid, stats = solve_q_star_least_squares(
        x, velocity, target, ridge_eps=1.0e-8, max_condition=1.0e10
    )
    assert valid.tolist() == [True]
    assert stats["q_star_nan_count"] == 0
    torch.testing.assert_close(q_star, q, atol=1.0e-5, rtol=1.0e-5)
    rotated_velocity, _ = apply_bond_jacobian(x @ _rotation().T, q, target)
    torch.testing.assert_close(
        rotated_velocity, velocity @ _rotation().T, atol=1.0e-5, rtol=1.0e-5
    )
    assert jacobian_sanity_check(x, q, target)["finite"]


def test_light_egnn_cartesian_and_q_outputs_are_rotation_consistent():
    torch.manual_seed(3)
    model = LightEGNNRefinerBackbone(num_layers=2, hidden_dim=32, edge_hidden_dim=32)
    node_attr = torch.randn(4, 10)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.5, 1.0, 0.4], [0.1, 1.0, 0.3]]
    )
    edge = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
    edge_attr = torch.zeros(edge.size(1), 1)
    time = torch.full((4,), 0.4)
    anchor, moving = torch.tensor([0]), torch.tensor([1])
    velocity, q = model(node_attr, pos, edge, edge_attr, time, anchor, moving)
    moved_velocity, moved_q = model(
        node_attr,
        pos @ _rotation().T + torch.tensor([2.0, -1.0, 0.5]),
        edge,
        edge_attr,
        time,
        anchor,
        moving,
    )
    torch.testing.assert_close(
        moved_velocity, velocity @ _rotation().T, atol=1.0e-5, rtol=1.0e-5
    )
    torch.testing.assert_close(moved_q, q, atol=1.0e-5, rtol=1.0e-5)
