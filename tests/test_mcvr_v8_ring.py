import torch

from etflow.ecir.v8_losses import ring_loss


def test_ring_loss_is_applicable_only_with_ring_bonds():
    source = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    moved = source.clone()
    moved[1, 0] = 1.2
    value, diag = ring_loss(
        moved,
        source,
        {"protected_ring_bond_index": torch.tensor([[0], [1]])},
        residual_scale=0.01,
    )
    assert value > 0 and diag["applicable_ring_bond_count"] == 1
    empty, _ = ring_loss(moved, source, {})
    assert empty == 0
