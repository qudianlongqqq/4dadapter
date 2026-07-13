"""Shared schema, hashing, and graph-contract helpers for FlexBond caches."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional

import torch
from torch import Tensor


CACHE_SCHEMA_VERSION = "2.0"
INFERENCE_FORBIDDEN_FIELDS = frozenset(
    {
        "x_ref",
        "x_ref_aligned",
        "x_ref_candidates",
        "u_t",
        "target_velocity",
        "residual_velocity",
        "r_vel",
        "q_b_star",
        "q_star",
    }
)


def _record_field(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    getter = getattr(record, "get", None)
    if callable(getter):
        return getter(key, None)
    return getattr(record, key, None)


def stable_record_identities(record: Any) -> list[tuple[str, str]]:
    """Return only explicit identities suitable for cross-file matching."""

    identities: list[tuple[str, str]] = []
    for key in (
        "source_record_id",
        "source_mol_id",
        "molecule_id",
        "id",
        "mol_id",
    ):
        value = _record_field(record, key)
        if value is not None and str(value).strip():
            identities.append(("record_id", str(value).strip()))
    dataset_index = _record_field(record, "dataset_index")
    if dataset_index is not None and str(dataset_index).strip():
        identities.append(("dataset_index", str(dataset_index).strip()))
    for key in ("smiles", "canonical_smiles", "smi"):
        value = _record_field(record, key)
        if value is not None and str(value).strip():
            identities.append(("smiles", str(value).strip()))
    atom_map_id = _record_field(record, "atom_map_id")
    if atom_map_id is not None:
        try:
            value = torch.as_tensor(atom_map_id).reshape(-1)
            if value.numel() == 1:
                identities.append(("atom_map_id", str(value.item())))
        except (TypeError, ValueError):
            if str(atom_map_id).strip():
                identities.append(("atom_map_id", str(atom_map_id).strip()))
    return list(dict.fromkeys(identities))


def strict_reference_lookup(records: list[tuple[str, Any]]) -> dict[tuple[str, str], Any]:
    """Build an external-reference lookup without ever using fallback indices."""

    lookup: dict[tuple[str, str], Any] = {}
    ambiguous_structure_ids: set[tuple[str, str]] = set()
    for fallback, record in records:
        identities = stable_record_identities(record)
        if not identities:
            raise ValueError(
                "External reference record has no stable mol_id, smiles, "
                f"canonical_smiles, or atom_map_id (record {fallback!r})."
            )
        for identity in identities:
            if identity in ambiguous_structure_ids:
                continue
            if identity in lookup and lookup[identity] is not record:
                if identity[0] == "smiles":
                    # SMILES is a structural lookup aid, not a unique record key.
                    # Removing an ambiguous entry keeps explicit record IDs usable
                    # while making SMILES-only matching fail closed.
                    del lookup[identity]
                    ambiguous_structure_ids.add(identity)
                    continue
                raise ValueError(f"Duplicate external reference identity: {identity!r}.")
            lookup[identity] = record
    return lookup


def tensor_sha256(value: Any) -> str:
    """Hash a tensor using a stable CPU, contiguous representation."""

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode("utf-8")
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def x_init_sha256(x_init: Any, atomic_numbers: Optional[Any] = None) -> str:
    """Hash coordinates, optionally binding the hash to ordered atom identities."""

    coordinate_hash = tensor_sha256(torch.as_tensor(x_init, dtype=torch.float32))
    if atomic_numbers is None:
        return coordinate_hash
    atom_hash = tensor_sha256(torch.as_tensor(atomic_numbers, dtype=torch.long))
    return hashlib.sha256(f"{coordinate_hash}:{atom_hash}".encode("ascii")).hexdigest()


def atom_map_ids_from_record(record: Any) -> Optional[Tensor]:
    """Read ordered per-atom map ids without confusing them with molecule ids."""

    keys = ("atom_map_ids", "atom_map_numbers", "atom_maps")
    for key in keys:
        value = record.get(key) if isinstance(record, Mapping) else getattr(record, key, None)
        if value is not None:
            ids = torch.as_tensor(value, dtype=torch.long).view(-1)
            return ids if ids.numel() and bool((ids != 0).any()) else None
    return None


def validate_graph_record(record: Mapping[str, Any]) -> dict[str, Tensor]:
    """Validate ordered atoms and the directed molecular graph contract."""

    required = (
        "atomic_numbers",
        "node_attr",
        "edge_index",
        "bond_type",
        "bond_is_aromatic",
        "bond_is_in_ring",
        "rotatable_bond_index",
        "atom_bond_influence_index",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Graph record is missing fields: {missing}.")

    atomic_numbers = torch.as_tensor(record["atomic_numbers"], dtype=torch.long).view(-1)
    num_atoms = int(atomic_numbers.numel())
    if num_atoms < 1:
        raise ValueError("atomic_numbers must contain at least one atom.")
    if "num_atoms" in record and int(record["num_atoms"]) != num_atoms:
        raise ValueError("num_atoms does not match atomic_numbers length.")

    node_attr = torch.as_tensor(record["node_attr"])
    if node_attr.ndim < 2 or node_attr.size(0) != num_atoms:
        raise ValueError(
            f"node_attr must start with num_atoms={num_atoms}, got {tuple(node_attr.shape)}."
        )

    edge_index = torch.as_tensor(record["edge_index"], dtype=torch.long)
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}.")
    num_edges = int(edge_index.size(1))
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_atoms
    ):
        raise ValueError("edge_index contains an out-of-range atom index.")
    if edge_index.numel() and bool((edge_index[0] == edge_index[1]).any()):
        raise ValueError("edge_index contains a self edge.")

    edge_attr = record.get("edge_attr")
    if edge_attr is not None and torch.as_tensor(edge_attr).size(0) != num_edges:
        raise ValueError("edge_attr length must equal the directed edge count.")
    bond_type = record.get("bond_type")
    bond_type = torch.as_tensor(bond_type, dtype=torch.long).view(-1)
    if bond_type.numel() != num_edges:
        raise ValueError("bond_type length must equal the directed edge count.")
    if edge_attr is not None:
        attr = torch.as_tensor(edge_attr)
        attr_type = attr[:, 0] if attr.ndim > 1 else attr
        if not torch.equal(attr_type.to(dtype=torch.long), bond_type):
            raise ValueError("bond_type does not match the first edge_attr column.")

    aromatic = torch.as_tensor(
        record["bond_is_aromatic"], dtype=torch.bool
    ).view(-1)
    in_ring = torch.as_tensor(
        record["bond_is_in_ring"], dtype=torch.bool
    ).view(-1)
    if aromatic.numel() != num_edges or in_ring.numel() != num_edges:
        raise ValueError("bond aromatic/ring flag lengths must equal edge count.")

    directed: dict[tuple[int, int], tuple[int, bool, bool]] = {}
    for index in range(num_edges):
        pair = (int(edge_index[0, index]), int(edge_index[1, index]))
        signature = (int(bond_type[index]), bool(aromatic[index]), bool(in_ring[index]))
        if pair in directed:
            raise ValueError(f"edge_index contains duplicate directed edge {pair}.")
        directed[pair] = signature
    for (source, target), signature in directed.items():
        reverse = directed.get((target, source))
        if reverse is None:
            raise ValueError(f"edge ({source}, {target}) has no reciprocal edge.")
        if reverse != signature:
            raise ValueError(f"reciprocal edge signature mismatch for ({source}, {target}).")

    rotatable = torch.as_tensor(record["rotatable_bond_index"], dtype=torch.long)
    if rotatable.ndim != 2 or rotatable.size(0) != 2:
        raise ValueError("rotatable_bond_index must have shape [2, B].")
    if rotatable.numel() and (
        int(rotatable.min()) < 0 or int(rotatable.max()) >= num_atoms
    ):
        raise ValueError("rotatable_bond_index contains an invalid atom index.")
    for atom_a, atom_b in rotatable.t().tolist():
        if (int(atom_a), int(atom_b)) not in directed:
            raise ValueError(f"rotatable bond ({atom_a}, {atom_b}) is not a graph edge.")

    influence = torch.as_tensor(record["atom_bond_influence_index"], dtype=torch.long)
    if influence.ndim != 2 or influence.size(0) != 2:
        raise ValueError("atom_bond_influence_index must have shape [2, K].")
    if influence.size(1):
        if int(influence[0].min()) < 0 or int(influence[0].max()) >= num_atoms:
            raise ValueError("atom_bond_influence_index contains an invalid atom index.")
        if int(influence[1].min()) < 0 or int(influence[1].max()) >= rotatable.size(1):
            raise ValueError("atom_bond_influence_index contains an invalid bond index.")

    maps = atom_map_ids_from_record(record)
    if maps is not None:
        if maps.numel() != num_atoms:
            raise ValueError("atom_map_ids length must equal num_atoms.")
        if maps.unique().numel() != num_atoms:
            raise ValueError("non-zero atom_map_ids must be unique within a molecule.")

    return {
        "atomic_numbers": atomic_numbers,
        "node_attr": node_attr,
        "edge_index": edge_index,
        "edge_attr": torch.as_tensor(edge_attr) if edge_attr is not None else bond_type[:, None],
        "bond_type": bond_type,
        "bond_is_aromatic": aromatic,
        "bond_is_in_ring": in_ring,
        "rotatable_bond_index": rotatable,
        "atom_bond_influence_index": influence,
    }


def validate_inference_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a schema-v2 cache while structurally rejecting all labels."""

    leaked = sorted(
        field
        for field in record
        if field in INFERENCE_FORBIDDEN_FIELDS
        or field.startswith("x_ref")
        or field.startswith("selected_ref")
        or field in {"rmsd_before", "rmsd_after"}
    )
    if leaked:
        raise ValueError(f"Inference cache contains forbidden label fields: {leaked}.")
    required = (
        "mol_id",
        "x_init",
        "cache_schema_version",
        "generator_name",
        "generator_checkpoint",
        "sample_seed",
        "DATA_DIR",
        "created_at",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Inference cache is missing fields: {missing}.")
    if str(record["cache_schema_version"]) != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Expected cache schema {CACHE_SCHEMA_VERSION}, got "
            f"{record['cache_schema_version']!r}."
        )
    graph = validate_graph_record(record)
    x_init = torch.as_tensor(record["x_init"], dtype=torch.float32)
    expected = (graph["atomic_numbers"].numel(), 3)
    if tuple(x_init.shape) != expected:
        raise ValueError(f"x_init must have shape {expected}, got {tuple(x_init.shape)}.")
    if not torch.isfinite(x_init).all():
        raise ValueError("x_init contains NaN or Inf.")
    expected_hash = x_init_sha256(x_init, graph["atomic_numbers"])
    cached_hash = record.get("x_init_hash")
    if cached_hash is not None and str(cached_hash) != expected_hash:
        raise ValueError("x_init_hash does not match cached coordinates.")
    return {**graph, "x_init": x_init, "x_init_hash": expected_hash}
