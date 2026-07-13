"""Strict label-free dataset for post-generation FlexBond inference."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import torch
from torch_geometric.data import Dataset

from etflow.commons.featurization import MoleculeData
from etflow.commons.record_identity import source_record_identity
from etflow.data.flexbond_cache_schema import (
    validate_inference_record,
)


class FlexBondInferenceDataset(Dataset):
    """Read inference records that structurally cannot contain training labels."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        root = Path(cache_dir).expanduser()
        if split is not None and (root / split).is_dir():
            root = root / split
        self.cache_dir = root
        self.data_files = sorted(root.glob("*.pt"))
        if max_samples is not None:
            self.data_files = self.data_files[: int(max_samples)]
        if not self.data_files:
            raise ValueError(f"No .pt FlexBond inference files found in {root}.")

    def len(self) -> int:
        return len(self.data_files)

    def get(self, idx: int) -> MoleculeData:
        path = self.data_files[idx]
        record = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(record, Mapping):
            raise TypeError(f"Inference cache {path} must contain a mapping.")
        checked = validate_inference_record(record)
        rotatable = checked["rotatable_bond_index"]
        return MoleculeData(
            num_nodes=int(checked["atomic_numbers"].numel()),
            mol_id=str(record["mol_id"]),
            sample_id=str(record.get("sample_id", record["mol_id"])),
            source_mol_id=str(record.get("source_mol_id", record["mol_id"])),
            source_record_id=source_record_identity(record),
            dataset_index=record.get("dataset_index"),
            generated_conformer_index=record.get(
                "generated_conformer_index",
                record.get("metadata", {}).get("generated_conformer_index"),
            ),
            smiles=str(record.get("smiles", "")),
            atomic_numbers=checked["atomic_numbers"],
            node_attr=checked["node_attr"].to(dtype=torch.float32),
            edge_index=checked["edge_index"],
            bond_index=checked["edge_index"],
            edge_attr=checked["edge_attr"].to(dtype=torch.float32),
            rotatable_bond_index=rotatable,
            atom_bond_influence_index=checked["atom_bond_influence_index"],
            x_init=checked["x_init"],
            x_init_hash=str(checked["x_init_hash"]),
            num_rotatable_bonds=torch.tensor([rotatable.size(1)], dtype=torch.long),
            metadata=dict(record.get("metadata", {})),
        )
