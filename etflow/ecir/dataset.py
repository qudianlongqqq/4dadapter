"""Unified real/synthetic/identity dataset for ECIR-Flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch_geometric.data import Dataset

from etflow.data.flexbond_optimizer_dataset import FlexBondData

from .geometry import geometry_error_vector
from .structured_corruption import CORRUPTION_MODES, corrupt_conformer
from .target_building import build_real_error_target


SOURCE_TYPES = ("real_error", "synthetic_error", "clean_identity")


def _jsonable(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _as_record(path: Path) -> Mapping[str, Any]:
    record = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(record, Mapping):
        raise TypeError(f"ECIR source cache must be a mapping: {path}")
    return record


def _tensor(record: Mapping[str, Any], name: str, dtype=None, default=None):
    value = record.get(name, default)
    if value is None:
        raise KeyError(name)
    result = torch.as_tensor(value)
    return result.to(dtype=dtype) if dtype is not None else result


class ECIRMixedDataset(Dataset):
    """Molecule cache with configurable real/synthetic/identity sampling ratios."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        *,
        atlas_path: str | Path | None = None,
        target_cache_dir: str | Path | None = None,
        real_error_ratio: float = 0.50,
        synthetic_error_ratio: float = 0.35,
        clean_identity_ratio: float = 0.15,
        seed: int = 42,
        max_records: int | None = None,
        max_molecules: int | None = None,
        allow_online_target_building: bool = False,
    ) -> None:
        super().__init__()
        root = Path(cache_dir).expanduser()
        if (root / split).is_dir():
            root = root / split
        if atlas_path is not None:
            frame = pd.read_parquet(Path(atlas_path).expanduser())
            if "split" in frame:
                frame = frame[frame["split"] == split]
            self.entries = [
                {
                    "path": Path(row.source_path),
                    "target_path": Path(row.target_cache_path),
                    "coordinate_key": str(getattr(row, "coordinate_key", "x_init")),
                    "source_type": str(row.source_type),
                    "NFE": int(getattr(row, "NFE", 0)),
                    "seed": int(getattr(row, "seed", 0)),
                    "checkpoint": str(getattr(row, "checkpoint", "")),
                }
                for row in frame.itertuples(index=False)
            ]
        else:
            self.entries = [
                {"path": path, "coordinate_key": "x_init"}
                for path in sorted(root.glob("*.pt"))
            ]
        target_candidate_root = None
        if target_cache_dir is not None:
            target_candidate_root = Path(target_cache_dir).expanduser()
            if (target_candidate_root / split).is_dir():
                target_candidate_root = target_candidate_root / split
            self.entries = [
                entry
                for entry in self.entries
                if entry.get("target_path", target_candidate_root / entry["path"].name).is_file()
            ]
        if max_molecules is not None:
            selected: list[dict[str, Any]] = []
            molecule_ids: set[str] = set()
            for entry in self.entries:
                record = _as_record(entry["path"])
                molecule = str(record.get("source_mol_id", record.get("mol_id")))
                if molecule in molecule_ids:
                    selected.append(entry)
                elif len(molecule_ids) < int(max_molecules):
                    molecule_ids.add(molecule)
                    selected.append(entry)
            self.entries = selected
        if max_records is not None:
            self.entries = self.entries[: int(max_records)]
        self.files = [entry["path"] for entry in self.entries]
        if not self.entries:
            raise ValueError(f"No ECIR source records in {root}")
        ratios = torch.tensor(
            [real_error_ratio, synthetic_error_ratio, clean_identity_ratio],
            dtype=torch.float64,
        )
        if bool((ratios < 0).any()) or not torch.isclose(ratios.sum(), ratios.new_tensor(1.0)):
            raise ValueError("ECIR source ratios must be nonnegative and sum to one")
        self.ratios = ratios
        self.seed = int(seed)
        self.epoch = 0
        self.split = str(split)
        self.allow_online_target_building = bool(allow_online_target_building)
        self.target_root = None
        if target_cache_dir is not None:
            self.target_root = target_candidate_root

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def len(self) -> int:
        return len(self.files)

    def _generator(self, index: int) -> torch.Generator:
        return torch.Generator().manual_seed(
            self.seed + 1_000_003 * self.epoch + 97 * int(index)
        )

    def _source_type(self, generator: torch.Generator) -> str:
        selected = int(torch.multinomial(self.ratios.float(), 1, generator=generator))
        return SOURCE_TYPES[selected]

    def _real_target(
        self, index: int, source_path: Path, record: Mapping[str, Any], generator: torch.Generator
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        cached = self.entries[index].get("target_path")
        if cached is None and self.target_root is not None:
            cached = self.target_root / source_path.name
        if cached is not None and cached.is_file():
            payload = torch.load(cached, map_location="cpu", weights_only=False)
            if str(payload.get("sample_id")) != str(record.get("sample_id")):
                raise ValueError("ECIR target cache sample_id mismatch")
            return torch.as_tensor(payload["x_target"], dtype=torch.float32), dict(
                payload.get("target_metadata") or {}
            )
        if not self.allow_online_target_building:
            raise FileNotFoundError(
                f"Missing offline ECIR target {cached}; online target building is disabled"
            )
        result = build_real_error_target(record, generator=generator)
        return result.pop("x_target"), result

    def get(self, index: int) -> FlexBondData:
        source_path = self.files[index]
        entry = self.entries[index]
        record = _as_record(source_path)
        generator = self._generator(index)
        source_type = self._source_type(generator)
        reference = _tensor(
            record,
            "x_ref_aligned",
            torch.float32,
            default=record.get("x_init"),
        )
        target_metadata: dict[str, Any] = {}
        corruption_metadata: dict[str, Any] = {}
        if source_type == "real_error":
            x_input = _tensor(record, entry.get("coordinate_key", "x_init"), torch.float32)
            x_target, target_metadata = self._real_target(
                index, source_path, record, generator
            )
            corruption_mode = "real"
        elif source_type == "synthetic_error":
            candidates = [
                mode
                for mode in CORRUPTION_MODES
                if mode not in {"zero"}
                and (mode != "ring" or bool(_tensor(record, "bond_is_in_ring", torch.bool).any()))
                and (
                    mode not in {"torsion", "multi_torsion", "bond_angle"}
                    or _tensor(record, "rotatable_bond_index", torch.long).size(1) > 0
                )
            ]
            corruption_mode = candidates[
                int(torch.randint(len(candidates), (1,), generator=generator))
            ]
            x_input, corruption_metadata = corrupt_conformer(
                record,
                mode=corruption_mode,
                coordinates=reference,
                generator=generator,
            )
            x_target = reference.clone()
        else:
            x_input = reference.clone()
            x_target = reference.clone()
            corruption_mode = "zero"
            _, corruption_metadata = corrupt_conformer(
                record, mode="zero", coordinates=reference, generator=generator
            )
        x_target = torch.as_tensor(x_target, dtype=torch.float32)
        error_label = geometry_error_vector(x_input, x_target, record)
        edge_index = _tensor(record, "edge_index", torch.long)
        edge_attr = torch.as_tensor(
            record.get("edge_attr", torch.zeros(edge_index.size(1), 1)),
            dtype=torch.float32,
        )
        rotatable = _tensor(record, "rotatable_bond_index", torch.long)
        metadata = dict(record.get("metadata") or {})
        nfe = float(entry.get("NFE", metadata.get("NFE", metadata.get("nfe", 0.0))) or 0.0)
        seed = float(entry.get("seed", record.get("sample_seed", metadata.get("seed", 0.0))) or 0.0)
        checkpoint_step = float(metadata.get("checkpoint_step", 0.0) or 0.0)
        upstream_metadata = torch.tensor(
            [min(nfe / 100.0, 10.0), (seed % 10_000) / 10_000.0, min(checkpoint_step / 200_000.0, 10.0), 1.0],
            dtype=torch.float32,
        ).view(1, -1)
        return FlexBondData(
            num_nodes=x_input.size(0),
            mol_id=str(record.get("source_mol_id", record.get("mol_id"))),
            sample_id=str(record.get("sample_id", record.get("mol_id"))),
            source_mol_id=str(record.get("source_mol_id", record.get("mol_id"))),
            smiles=str(record.get("smiles", "")),
            atomic_numbers=_tensor(record, "atomic_numbers", torch.long),
            node_attr=_tensor(record, "node_attr", torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            bond_type=torch.as_tensor(record.get("bond_type", torch.zeros(edge_index.size(1))), dtype=torch.long),
            bond_is_aromatic=torch.as_tensor(record.get("bond_is_aromatic", torch.zeros(edge_index.size(1))), dtype=torch.bool),
            bond_is_in_ring=torch.as_tensor(record.get("bond_is_in_ring", torch.zeros(edge_index.size(1))), dtype=torch.bool),
            rotatable_bond_index=rotatable,
            atom_bond_influence_index=_tensor(record, "atom_bond_influence_index", torch.long),
            x_init=x_input,
            x_input=x_input,
            x_target=x_target,
            error_label=error_label.view(1, -1),
            upstream_metadata=upstream_metadata,
            source_type=source_type,
            upstream_source_type=str(entry.get("source_type", record.get("generator_name", "unknown"))),
            source_type_code=torch.tensor([SOURCE_TYPES.index(source_type)], dtype=torch.long),
            corruption_mode=corruption_mode,
            is_clean=torch.tensor([source_type == "clean_identity"], dtype=torch.bool),
            target_metadata_json=json.dumps(target_metadata, sort_keys=True, default=str),
            corruption_metadata_json=json.dumps(
                _jsonable(corruption_metadata), sort_keys=True, default=str
            ),
            num_rotatable_bonds=torch.tensor([rotatable.size(1)], dtype=torch.long),
        )
