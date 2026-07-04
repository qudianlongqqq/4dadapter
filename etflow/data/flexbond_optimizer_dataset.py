"""Generator-agnostic cached dataset for FlexBond-4D refinement.

The cache is deliberately upstream-neutral: every ``.pt`` file contains one
generated conformer, its molecular graph, and its matched reference set.  The
network never needs to know whether ``x_init`` came from ETFlow, RDKit, or a
future generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import Tensor
from torch_geometric.data import Dataset

from etflow.commons.featurization import MoleculeData
from etflow.commons.kabsch_utils import (
    kabsch_sanity_check,
    select_best_reference_conformer,
)


REQUIRED_CACHE_FIELDS = (
    "mol_id",
    "atomic_numbers",
    "node_attr",
    "edge_index",
    "rotatable_bond_mask",
    "rotatable_bond_index",
    "atom_bond_influence_index",
    "x_init",
    "x_ref_candidates",
)


class FlexBondData(MoleculeData):
    """PyG object with flattened reference candidates for safe batching."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "reference_conformer_ptr":
            return int(self.x_ref_candidates.size(0))
        if key == "selected_reference_index":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def _tensor(record: Mapping[str, Any], key: str, dtype=None) -> Tensor:
    value = torch.as_tensor(record[key])
    return value.to(dtype=dtype) if dtype is not None else value


def validate_cache_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate atom order, graph sizes, coordinates, and Kabsch matching."""

    missing = [key for key in REQUIRED_CACHE_FIELDS if key not in record]
    if missing:
        raise ValueError(f"FlexBond cache record is missing fields: {missing}.")
    atomic_numbers = _tensor(record, "atomic_numbers", torch.long).view(-1)
    x_init = _tensor(record, "x_init", torch.float32)
    refs = _tensor(record, "x_ref_candidates", torch.float32)
    if refs.ndim == 2:
        refs = refs.unsqueeze(0)
    num_atoms = int(atomic_numbers.numel())
    if tuple(x_init.shape) != (num_atoms, 3):
        raise ValueError(
            f"x_init shape {tuple(x_init.shape)} does not match {num_atoms} atoms."
        )
    if refs.ndim != 3 or tuple(refs.shape[1:]) != (num_atoms, 3):
        raise ValueError(
            "x_ref_candidates must have shape [C, N, 3], got "
            f"{tuple(refs.shape)} for {num_atoms} atoms."
        )
    if not torch.isfinite(x_init).all() or not torch.isfinite(refs).all():
        raise ValueError("Coordinate cache contains NaN or Inf.")
    init_numbers = record.get("x_init_atomic_numbers")
    ref_numbers = record.get("x_ref_atomic_numbers")
    for label, numbers in (
        ("x_init", init_numbers),
        ("x_ref", ref_numbers),
    ):
        if numbers is not None and not torch.equal(
            torch.as_tensor(numbers).long().view(-1), atomic_numbers
        ):
            raise ValueError(f"{label} atomic-number order does not match topology.")
    edge_index = _tensor(record, "edge_index", torch.long)
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}.")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_atoms):
        raise ValueError("edge_index contains an out-of-range atom index.")
    if edge_index.size(1) < max(0, 2 * (num_atoms - 1)):
        raise ValueError(
            f"edge count {edge_index.size(1)} is too small for a connected graph."
        )
    rotatable_mask = torch.as_tensor(record["rotatable_bond_mask"]).bool().view(-1)
    if rotatable_mask.numel() != edge_index.size(1):
        raise ValueError(
            "rotatable_bond_mask length must equal the directed edge count."
        )
    x_ref, x_ref_aligned, selected, rmsds = select_best_reference_conformer(
        x_init, refs
    )
    check = kabsch_sanity_check(x_init, x_ref)
    if not check["rmsd_non_increasing"] or not check["center_aligned"]:
        raise ValueError(f"Kabsch sanity check failed: {check}.")
    return {
        "atomic_numbers": atomic_numbers,
        "x_init": x_init,
        "x_ref_candidates": refs,
        "x_ref": x_ref,
        "x_ref_aligned": x_ref_aligned,
        "selected_reference_index": selected,
        "selected_rmsd": float(rmsds[selected]),
        **check,
    }


class FlexBondOptimizerDataset(Dataset):
    """Read one validated FlexBond cache record per generated conformer."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: Optional[str] = None,
        max_molecules: Optional[int] = None,
        validate: bool = True,
    ) -> None:
        super().__init__()
        root = Path(cache_dir).expanduser()
        if split is not None and (root / split).is_dir():
            root = root / split
        self.cache_dir = root
        self.data_files = sorted(root.glob("*.pt"))
        if max_molecules is not None:
            self.data_files = self.data_files[: int(max_molecules)]
        if not self.data_files:
            raise ValueError(f"No .pt FlexBond cache files found in {root}.")
        self.validate = bool(validate)

    def len(self) -> int:
        return len(self.data_files)

    def get(self, idx: int) -> FlexBondData:
        path = self.data_files[idx]
        record = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(record, Mapping):
            raise TypeError(f"Cache {path} must contain a mapping.")
        if self.validate:
            matched = validate_cache_record(record)
        else:
            refs = _tensor(record, "x_ref_candidates", torch.float32)
            if refs.ndim == 2:
                refs = refs.unsqueeze(0)
            x_init = _tensor(record, "x_init", torch.float32)
            if all(
                key in record
                for key in ("x_ref", "x_ref_aligned", "selected_reference_index")
            ):
                selected = int(record["selected_reference_index"])
                matched = {
                    "atomic_numbers": _tensor(record, "atomic_numbers", torch.long),
                    "x_init": x_init,
                    "x_ref_candidates": refs,
                    "x_ref": _tensor(record, "x_ref", torch.float32),
                    "x_ref_aligned": _tensor(record, "x_ref_aligned", torch.float32),
                    "selected_reference_index": selected,
                    "selected_rmsd": float(
                        record.get("selected_reference_rmsd", 0.0)
                    ),
                }
            else:
                x_ref, aligned, selected, rmsds = select_best_reference_conformer(
                    x_init, refs
                )
                matched = {
                    "atomic_numbers": _tensor(record, "atomic_numbers", torch.long),
                    "x_init": x_init,
                    "x_ref_candidates": refs,
                    "x_ref": x_ref,
                    "x_ref_aligned": aligned,
                    "selected_reference_index": selected,
                    "selected_rmsd": float(rmsds[selected]),
                }
        refs = matched["x_ref_candidates"]
        num_refs, num_atoms = refs.shape[:2]
        edge_attr = record.get("edge_attr")
        if edge_attr is None:
            edge_attr = torch.zeros(
                (torch.as_tensor(record["edge_index"]).size(1), 1),
                dtype=torch.float32,
            )
        rotatable = _tensor(record, "rotatable_bond_index", torch.long)
        metadata = dict(record.get("metadata", {}))
        return FlexBondData(
            num_nodes=num_atoms,
            mol_id=str(record["mol_id"]),
            source_mol_id=str(record.get("source_mol_id", record["mol_id"])),
            smiles=str(record.get("smiles", "")),
            atomic_numbers=matched["atomic_numbers"],
            node_attr=_tensor(record, "node_attr", torch.float32),
            edge_index=_tensor(record, "edge_index", torch.long),
            bond_index=_tensor(record, "edge_index", torch.long),
            edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32),
            rotatable_bond_mask=torch.as_tensor(
                record["rotatable_bond_mask"], dtype=torch.bool
            ),
            rotatable_bond_index=rotatable,
            atom_bond_influence_index=_tensor(
                record, "atom_bond_influence_index", torch.long
            ),
            x_init=matched["x_init"],
            x_ref=matched["x_ref"],
            x_ref_aligned=matched["x_ref_aligned"],
            # Flattening preserves all candidates while allowing molecules with
            # different atom counts to coexist in a PyG batch.
            x_ref_candidates=refs.reshape(num_refs * num_atoms, 3),
            reference_conformer_ptr=torch.arange(
                0, (num_refs + 1) * num_atoms, num_atoms, dtype=torch.long
            ),
            num_reference_conformers=torch.tensor([num_refs], dtype=torch.long),
            selected_reference_index=torch.tensor(
                [matched["selected_reference_index"]], dtype=torch.long
            ),
            selected_reference_rmsd=torch.tensor(
                [matched["selected_rmsd"]], dtype=torch.float32
            ),
            num_rotatable_bonds=torch.tensor(
                [rotatable.size(1)], dtype=torch.long
            ),
            metadata=metadata,
        )
