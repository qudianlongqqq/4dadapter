"""Generator-agnostic cached dataset for FlexBond-4D refinement.

The cache is deliberately upstream-neutral: every ``.pt`` file contains one
generated conformer, its molecular graph, and its matched reference set.  The
network never needs to know whether ``x_init`` came from ETFlow, RDKit, or a
future generator.
"""

from __future__ import annotations

import math
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
from etflow.commons.record_identity import source_record_identity
from etflow.data.flexbond_cache_schema import (
    CACHE_SCHEMA_VERSION,
    atom_map_ids_from_record,
    validate_graph_record,
    x_init_sha256,
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
    "x_init_hash",
    "x_ref_candidates",
    "x_init_atomic_numbers",
    "x_ref_atomic_numbers",
    "x_init_topology_signature",
    "x_ref_topology_signature",
    "cache_schema_version",
    "generator_name",
    "generator_checkpoint",
    "sample_seed",
    "DATA_DIR",
    "created_at",
)

RMSD_REL_TOL = 1.0e-6
RMSD_ABS_TOL = 1.0e-4


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


def validate_rmsd_diagnostic(
    persisted: float,
    recomputed: float,
    *,
    field: str = "selected_reference_rmsd",
    rel_tol: float = RMSD_REL_TOL,
    abs_tol: float = RMSD_ABS_TOL,
) -> dict[str, float | str]:
    """Validate a persisted float diagnostic without weakening hard identities."""

    persisted = float(persisted)
    recomputed = float(recomputed)
    if rel_tol != RMSD_REL_TOL or abs_tol != RMSD_ABS_TOL:
        raise ValueError(
            "Stage 2 RMSD validation tolerances are fixed at "
            f"rel_tol={RMSD_REL_TOL} and abs_tol={RMSD_ABS_TOL}."
        )
    if not math.isfinite(persisted) or not math.isfinite(recomputed):
        raise ValueError(
            f"Persisted {field} is not numerically valid: "
            f"persisted={persisted}, recomputed={recomputed}."
        )
    absolute_delta = abs(persisted - recomputed)
    scale = max(abs(persisted), abs(recomputed))
    relative_delta = absolute_delta / scale if scale else 0.0
    effective_tolerance = max(abs_tol, rel_tol * scale)
    if not math.isclose(
        persisted,
        recomputed,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    ):
        raise ValueError(
            f"Persisted {field} is stale or incorrect: persisted={persisted}, "
            f"recomputed={recomputed}, absolute_delta={absolute_delta}, "
            f"relative_delta={relative_delta}, "
            f"effective_tolerance={effective_tolerance}."
        )
    return {
        "persisted_rmsd": persisted,
        "recomputed_rmsd": recomputed,
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
        "effective_tolerance": effective_tolerance,
        "validation_status": (
            "PASS" if persisted == recomputed else "PASS_NUMERICALLY_CLOSE"
        ),
    }


def validate_cache_record(
    record: Mapping[str, Any], *, require_persisted_pair: bool = False
) -> dict[str, Any]:
    """Validate atom order, graph sizes, coordinates, and Kabsch matching."""

    missing = [key for key in REQUIRED_CACHE_FIELDS if key not in record]
    if missing:
        raise ValueError(f"FlexBond cache record is missing fields: {missing}.")
    if str(record["cache_schema_version"]) != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Expected cache schema {CACHE_SCHEMA_VERSION}, got "
            f"{record['cache_schema_version']!r}."
        )
    graph = validate_graph_record(record)
    atomic_numbers = graph["atomic_numbers"]
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
    if record.get("x_init_hash") != x_init_sha256(x_init, atomic_numbers):
        raise ValueError("x_init_hash is missing or does not match x_init.")
    init_numbers = record["x_init_atomic_numbers"]
    ref_numbers = record["x_ref_atomic_numbers"]
    for label, numbers in (
        ("x_init", init_numbers),
        ("x_ref", ref_numbers),
    ):
        if not torch.equal(torch.as_tensor(numbers).long().view(-1), atomic_numbers):
            raise ValueError(f"{label} atomic-number order does not match topology.")
    init_maps = record.get("x_init_atom_map_ids")
    ref_maps = record.get("x_ref_atom_map_ids")
    if init_maps is not None or ref_maps is not None:
        if init_maps is None or ref_maps is None:
            raise ValueError("Atom-map ids must be present for both x_init and x_ref.")
        init_maps = torch.as_tensor(init_maps, dtype=torch.long).view(-1)
        ref_maps = torch.as_tensor(ref_maps, dtype=torch.long).view(-1)
        if init_maps.numel() != num_atoms or not torch.equal(init_maps, ref_maps):
            raise ValueError("x_init and x_ref atom_map_ids are not aligned.")
        graph_maps = atom_map_ids_from_record(record)
        if graph_maps is not None and not torch.equal(init_maps, graph_maps):
            raise ValueError("Coordinate atom_map_ids do not match graph atom_map_ids.")
    if record["x_init_topology_signature"] != record["x_ref_topology_signature"]:
        raise ValueError("x_init and x_ref ordered topology signatures differ.")
    if record["x_init_topology_signature"] != record.get("topology_signature"):
        raise ValueError("Coordinate topology signature does not match cached graph.")

    edge_index = graph["edge_index"]
    rotatable_mask = torch.as_tensor(record["rotatable_bond_mask"]).bool().view(-1)
    if rotatable_mask.numel() != edge_index.size(1):
        raise ValueError(
            "rotatable_bond_mask length must equal the directed edge count."
        )
    marked_pairs = {
        tuple(sorted((int(edge_index[0, index]), int(edge_index[1, index]))))
        for index in torch.nonzero(rotatable_mask, as_tuple=False).view(-1).tolist()
    }
    rotatable_pairs = {
        tuple(sorted((int(atom_a), int(atom_b))))
        for atom_a, atom_b in graph["rotatable_bond_index"].t().tolist()
    }
    if marked_pairs != rotatable_pairs:
        raise ValueError("rotatable_bond_mask does not match rotatable_bond_index.")
    x_ref, x_ref_aligned, selected, rmsds = select_best_reference_conformer(
        x_init, refs
    )
    check = kabsch_sanity_check(x_init, x_ref)
    if not check["rmsd_non_increasing"] or not check["center_aligned"]:
        raise ValueError(f"Kabsch sanity check failed: {check}.")
    if require_persisted_pair:
        pair_fields = (
            "selected_reference_index",
            "selected_reference_rmsd",
            "selected_ref_id",
            "rmsd_before",
            "rmsd_after",
        )
        missing_pair = [key for key in pair_fields if key not in record]
        if missing_pair:
            raise ValueError(
                f"Cache is missing persisted pair diagnostics: {missing_pair}."
            )
        if int(record["selected_reference_index"]) != selected:
            raise ValueError(
                "Persisted selected_reference_index is stale or incorrect."
            )
        if not str(record["selected_ref_id"]).strip():
            raise ValueError("selected_ref_id must be a non-empty stable identifier.")
        persisted_x_ref = _tensor(record, "x_ref", torch.float32)
        if persisted_x_ref.shape != x_ref.shape or not torch.equal(
            persisted_x_ref, x_ref
        ):
            raise ValueError(
                "Persisted selected reference identity does not match "
                "selected_reference_index."
            )
        persisted = (
            ("selected_reference_rmsd", float(rmsds[selected])),
            ("rmsd_before", check["rmsd_before"]),
            ("rmsd_after", check["rmsd_after"]),
        )
        numeric_diagnostics = {}
        for key, expected in persisted:
            numeric_diagnostics[key] = validate_rmsd_diagnostic(
                record[key], expected, field=key
            )
    else:
        numeric_diagnostics = {}
    return {
        "atomic_numbers": atomic_numbers,
        "x_init": x_init,
        "x_ref_candidates": refs,
        "x_ref": x_ref,
        "x_ref_aligned": x_ref_aligned,
        "selected_reference_index": selected,
        "selected_rmsd": float(rmsds[selected]),
        "selected_ref_id": str(
            record.get("selected_ref_id", f"reference_{selected:04d}")
        ),
        "numeric_diagnostics": numeric_diagnostics,
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
            limit = int(max_molecules)
            if limit < 1:
                raise ValueError("max_molecules must be positive.")
            selected_ids: set[str] = set()
            selected_files = []
            for path in self.data_files:
                header = torch.load(path, map_location="cpu", weights_only=False)
                source_id = source_record_identity(header)
                if source_id in selected_ids:
                    selected_files.append(path)
                elif len(selected_ids) < limit:
                    selected_ids.add(source_id)
                    selected_files.append(path)
            self.data_files = selected_files
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
            matched = validate_cache_record(record, require_persisted_pair=True)
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
                    "selected_rmsd": float(record.get("selected_reference_rmsd", 0.0)),
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
        num_directed_edges = torch.as_tensor(record["edge_index"]).size(1)
        bond_type = record.get("bond_type")
        bond_is_aromatic = record.get("bond_is_aromatic")
        bond_is_in_ring = record.get("bond_is_in_ring")
        metadata = dict(record.get("metadata", {}))
        source_record_id = source_record_identity(record)
        return FlexBondData(
            num_nodes=num_atoms,
            mol_id=str(record["mol_id"]),
            sample_id=str(record.get("sample_id", record["mol_id"])),
            source_mol_id=str(record.get("source_mol_id", record["mol_id"])),
            source_record_id=source_record_id,
            dataset_index=record.get("dataset_index"),
            generated_conformer_index=record.get(
                "generated_conformer_index",
                metadata.get("generated_conformer_index"),
            ),
            smiles=str(record.get("smiles", "")),
            atomic_numbers=matched["atomic_numbers"],
            node_attr=_tensor(record, "node_attr", torch.float32),
            edge_index=_tensor(record, "edge_index", torch.long),
            bond_index=_tensor(record, "edge_index", torch.long),
            edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32),
            bond_type=(
                torch.as_tensor(bond_type)
                if bond_type is not None
                else torch.zeros(num_directed_edges, dtype=torch.long)
            ),
            bond_is_aromatic=(
                torch.as_tensor(bond_is_aromatic, dtype=torch.bool)
                if bond_is_aromatic is not None
                else torch.zeros(num_directed_edges, dtype=torch.bool)
            ),
            bond_is_in_ring=(
                torch.as_tensor(bond_is_in_ring, dtype=torch.bool)
                if bond_is_in_ring is not None
                else torch.zeros(num_directed_edges, dtype=torch.bool)
            ),
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
            selected_ref_id=matched.get("selected_ref_id", ""),
            num_rotatable_bonds=torch.tensor([rotatable.size(1)], dtype=torch.long),
            metadata=metadata,
        )
