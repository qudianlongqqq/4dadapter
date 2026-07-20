import torch

from etflow.ecir.v8_losses import chirality_barrier


def test_chirality_flip_has_larger_barrier():
    source = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    batch = {"protected_chirality_constraint_index": torch.tensor([[0], [1], [2], [3]])}
    preserved, _ = chirality_barrier(source, source, batch)
    flipped_coordinates = source.clone()
    flipped_coordinates[3, 2] = -1.0
    flipped, diag = chirality_barrier(flipped_coordinates, source, batch)
    assert flipped > preserved
    assert diag["chirality_sign_flip_count"] == 1
