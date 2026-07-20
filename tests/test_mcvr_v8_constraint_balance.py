import torch

from etflow.ecir.v8_constraint_layer import DifferentiableMolecularConstraintLayer


def _payload(duplicate: bool):
    pairs = [[0, 0], [1, 1]] if duplicate else [[0], [1]]
    ranges = [[0.9, 1.1, 0.1], [0.9, 1.1, 0.1]] if duplicate else [[0.9, 1.1, 0.1]]
    return {
        "ptr": torch.tensor([0, 3]),
        "active_bond_constraint_index": torch.tensor(pairs),
        "bond_allowed_range": torch.tensor(ranges),
        "active_angle_constraint_index": torch.tensor([[0], [1], [2]]),
        "angle_allowed_range": torch.tensor([[1.4, 1.7, 0.1]]),
    }


def test_duplicate_bonds_do_not_double_normalized_contribution():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.0, 0.0]])
    prior = torch.zeros_like(coordinates)
    layer = DifferentiableMolecularConstraintLayer({}, scales={"bond": 0.1, "angle": 0.1})
    once = layer(coordinates, prior, torch.ones(3), _payload(False))
    twice = layer(coordinates, prior, torch.ones(3), _payload(True))
    assert torch.allclose(once["delta_final"], twice["delta_final"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        once["solver_angle_contribution"], twice["solver_angle_contribution"], atol=1e-7
    )


def test_disabling_normalization_exposes_duplicate_imbalance():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.0, 0.0]])
    layer = DifferentiableMolecularConstraintLayer(
        {"normalize_by_active_count": False}, scales={"bond": 0.1, "angle": 0.1}
    )
    once = layer(coordinates, torch.zeros_like(coordinates), torch.ones(3), _payload(False))
    twice = layer(coordinates, torch.zeros_like(coordinates), torch.ones(3), _payload(True))
    assert float(twice["bond_normal_trace"]) > float(once["bond_normal_trace"]) * 1.9
