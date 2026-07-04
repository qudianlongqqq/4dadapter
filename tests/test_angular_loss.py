import torch

from etflow.commons.angular_loss import compute_target_dot_tau


def _masked_mse(prediction, target, valid_mask):
    if not valid_mask.any():
        return prediction.new_zeros(())
    return (prediction[valid_mask] - target[valid_mask]).square().mean()


def test_no_rotatable_bonds_returns_empty_target_and_zero_loss():
    pos = torch.zeros((3, 3), dtype=torch.float64)
    velocity = torch.randn_like(pos)
    rotatable_bond_index = torch.empty((2, 0), dtype=torch.long)
    influence_index = torch.empty((2, 0), dtype=torch.long)

    target, valid_mask = compute_target_dot_tau(
        pos,
        velocity,
        rotatable_bond_index,
        influence_index,
    )

    assert target.shape == (0,)
    assert valid_mask.shape == (0,)
    assert valid_mask.dtype == torch.bool
    assert _masked_mse(target, target, valid_mask).item() == 0.0


def test_no_influenced_atoms_marks_all_bonds_invalid_and_zero_loss():
    pos = torch.tensor(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    velocity = torch.zeros_like(pos)
    rotatable_bond_index = torch.tensor([[0], [1]], dtype=torch.long)
    influence_index = torch.empty((2, 0), dtype=torch.long)

    target, valid_mask = compute_target_dot_tau(
        pos,
        velocity,
        rotatable_bond_index,
        influence_index,
    )

    torch.testing.assert_close(target, torch.zeros(1, dtype=pos.dtype))
    assert not valid_mask.any()
    assert _masked_mse(target, target, valid_mask).item() == 0.0


def test_synthetic_rotation_recovers_scalar_dot_tau():
    pos = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    rotatable_bond_index = torch.tensor([[0], [1]], dtype=torch.long)
    influence_index = torch.tensor(
        [[2, 3, 4], [0, 0, 0]],
        dtype=torch.long,
    )
    dot_tau_true = pos.new_tensor(2.75)

    axis = pos.new_tensor([1.0, 0.0, 0.0])
    center = 0.5 * (pos[0] + pos[1])
    velocity = torch.zeros_like(pos)
    influenced_atoms = influence_index[0]
    basis = torch.cross(
        axis.expand(influenced_atoms.numel(), -1),
        pos[influenced_atoms] - center,
        dim=-1,
    )
    velocity[influenced_atoms] = dot_tau_true * basis

    target, valid_mask = compute_target_dot_tau(
        pos,
        velocity,
        rotatable_bond_index,
        influence_index,
        batch=torch.zeros(pos.size(0), dtype=torch.long),
    )

    assert valid_mask.tolist() == [True]
    assert target.dtype == pos.dtype
    assert target.device == pos.device
    torch.testing.assert_close(target[0], dot_tau_true, rtol=1.0e-8, atol=1.0e-8)
