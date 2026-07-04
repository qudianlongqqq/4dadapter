import torch

from etflow.commons.bond_local_velocity import (
    bond_local_velocity_loss,
    build_bond_frame,
    compute_bond_local_velocity,
)


def test_bond_frame_is_orthonormal_for_axis_and_fallback_cases():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    bond_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

    frame, valid_mask = build_bond_frame(pos, bond_index)

    identity = torch.eye(3).expand(2, -1, -1)
    torch.testing.assert_close(frame.transpose(1, 2) @ frame, identity)
    assert valid_mask.tolist() == [True, True]


def test_compute_bond_local_velocity_projects_relative_velocity():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocity = torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    bond_index = torch.tensor([[0], [1]], dtype=torch.long)

    q, valid_mask = compute_bond_local_velocity(pos, velocity, bond_index)

    torch.testing.assert_close(q, torch.tensor([[2.0, -3.0, -4.0]]))
    assert valid_mask.tolist() == [True]


def test_loss_excludes_degenerate_and_masked_bonds():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    pred = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )
    target = torch.zeros_like(pred)
    bond_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

    loss, stats = bond_local_velocity_loss(
        pos,
        pred,
        target,
        bond_index,
        bond_mask=torch.tensor([True, True]),
    )

    torch.testing.assert_close(loss, torch.tensor(14.0 / 3.0))
    torch.testing.assert_close(stats["parallel_loss"], torch.tensor(1.0))
    torch.testing.assert_close(stats["perp_loss"], torch.tensor(6.5))
    assert stats["valid_bonds"].item() == 1.0
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_empty_or_fully_masked_bonds_return_differentiable_zero():
    pos = torch.zeros((2, 3))
    pred = torch.randn((2, 3), requires_grad=True)
    target = torch.zeros_like(pred)
    empty_bonds = torch.empty((2, 0), dtype=torch.long)

    empty_loss, empty_stats = bond_local_velocity_loss(
        pos, pred, target, empty_bonds
    )
    masked_loss, masked_stats = bond_local_velocity_loss(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        pred,
        target,
        torch.tensor([[0], [1]], dtype=torch.long),
        bond_mask=torch.tensor([False]),
    )

    torch.testing.assert_close(empty_loss, torch.tensor(0.0))
    torch.testing.assert_close(masked_loss, torch.tensor(0.0))
    assert empty_stats["valid_bonds"].item() == 0.0
    assert masked_stats["valid_bonds"].item() == 0.0
    (empty_loss + masked_loss).backward()
    assert pred.grad is not None
