from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Batch, Data


ROOT = Path(__file__).resolve().parents[1]
D1_CHECKPOINT = ROOT / "artifacts/ecir_mvr/formal_large/d1_b_seed43/best_noninferior_validity.ckpt"
D1_SHA256 = "c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca"


def rotation(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    angle = torch.tensor(0.7, dtype=dtype)
    return torch.stack(
        (
            torch.stack((angle.cos(), -angle.sin(), angle.new_zeros(()))),
            torch.stack((angle.sin(), angle.cos(), angle.new_zeros(()))),
            torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
        )
    )


def graph(offset: float = 0.0, *, distorted: bool = True) -> Data:
    length = 1.5 if distorted else 1.0
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [length, 0.0, 0.0], [length, 1.0, 0.0], [3.0, 0.2, 0.0]],
        dtype=torch.float32,
    )
    pos[:, 0] += offset
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    return Data(
        num_nodes=4,
        node_attr=torch.randn(4, 10, generator=torch.Generator().manual_seed(5)),
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.size(1), 1),
        x_input=pos,
        x_target=pos
        + torch.tensor([[0.0, 0.0, 0.0], [-0.1, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        deterministic_error_features=torch.zeros(1, 10),
        upstream_metadata=torch.zeros(1, 4),
        active_mode_mask=torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        active_bond_constraint_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        bond_allowed_range=torch.tensor([[0.9, 1.1, 0.1], [0.9, 1.1, 0.1]]),
        active_angle_constraint_index=torch.tensor([[0], [1], [2]], dtype=torch.long),
        angle_allowed_range=torch.tensor([[1.45, 1.70, 0.1]]),
        protected_ring_bond_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        protected_chirality_constraint_index=torch.empty((4, 0), dtype=torch.long),
        canonical_angle_index=torch.tensor([[0], [1], [2]], dtype=torch.long),
        canonical_bond_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        canonical_ring_bond_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        canonical_torsion_index=torch.empty((4, 0), dtype=torch.long),
        rotatable_bond_index=torch.empty((2, 0), dtype=torch.long),
        bond_is_in_ring=torch.tensor([True, True, True, True]),
        num_rotatable_bonds=torch.tensor([0]),
    )


def batch(two: bool = False) -> Batch:
    values = [graph()]
    if two:
        values.append(graph(10.0))
    return Batch.from_data_list(values)
