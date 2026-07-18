"""Model-independent sparse constraints for unified BAC refinement."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from typing import Any, Mapping

import torch
from torch import Tensor
from torch_geometric.nn import radius_graph

from .audit import field
from .geometry import bond_angles, bond_lengths, training_bond_index


CONSTRAINT_SCHEMA_VERSION = "mcvr-bac-canonical-constraints-v1"
CONSTRAINT_FEATURE_VERSION = "mcvr-bac-static-features-v1"
MAX_ABS_STANDARDIZED_RESIDUAL = 10.0


def canonical_identity_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_angle_triplets(edge_index: Tensor, num_atoms: int) -> Tensor:
    """Return deterministic `(min(i,k), j, max(i,k))` angle identities."""

    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
    neighbors: list[set[int]] = [set() for _ in range(int(num_atoms))]
    for left, right in edge_index.t().tolist():
        if left == right:
            continue
        neighbors[int(left)].add(int(right))
        neighbors[int(right)].add(int(left))
    triplets = []
    for center, adjacent in enumerate(neighbors):
        ordered = sorted(adjacent)
        for offset, left in enumerate(ordered):
            for right in ordered[offset + 1 :]:
                triplets.append((left, center, right))
    return torch.tensor(triplets, dtype=torch.long).reshape(-1, 3)


def stable_angle_cosine(coordinates: Tensor, triplets: Tensor) -> Tensor:
    coordinates = torch.as_tensor(coordinates)
    triplets = torch.as_tensor(
        triplets, device=coordinates.device, dtype=torch.long
    ).reshape(-1, 3)
    if not triplets.numel():
        return coordinates.new_empty(0)
    left = coordinates[triplets[:, 0]] - coordinates[triplets[:, 1]]
    right = coordinates[triplets[:, 2]] - coordinates[triplets[:, 1]]
    left_norm = torch.linalg.vector_norm(left, dim=-1).clamp_min(1.0e-8)
    right_norm = torch.linalg.vector_norm(right, dim=-1).clamp_min(1.0e-8)
    return ((left * right).sum(-1) / (left_norm * right_norm)).clamp(
        -1.0 + 1.0e-7, 1.0 - 1.0e-7
    )


def standardized_interval_residual(
    values: Tensor, intervals: Tensor
) -> tuple[Tensor, Tensor]:
    """Return signed standardized residual and absolute severity."""

    values = torch.as_tensor(values)
    intervals = torch.as_tensor(
        intervals, device=values.device, dtype=values.dtype
    ).reshape(-1, 3)
    if not values.numel():
        empty = values.new_empty(0)
        return empty, empty
    lower, upper, scale = intervals.unbind(-1)
    signed = torch.where(
        values < lower,
        values - lower,
        torch.where(values > upper, values - upper, torch.zeros_like(values)),
    )
    residual = (signed / scale.clamp_min(1.0e-8)).clamp(
        -MAX_ABS_STANDARDIZED_RESIDUAL, MAX_ABS_STANDARDIZED_RESIDUAL
    )
    return residual, residual.abs()


def canonical_constraint_fields(
    validity: Any,
    record: Any,
    *,
    source_identity_sha256: str,
    feature_version: str = CONSTRAINT_FEATURE_VERSION,
) -> dict[str, Any]:
    """Build model-independent static constraint fields for one record."""

    prepared = validity._prepare(record)
    edge_index = torch.as_tensor(prepared["edge_index"], dtype=torch.long)
    bonds = torch.as_tensor(prepared["bonds"], dtype=torch.long).reshape(2, -1)
    num_atoms = len(torch.as_tensor(field(record, "atomic_numbers")))
    angles = canonical_angle_triplets(edge_index, num_atoms)
    # ChemicalValidity already uses the same deterministic neighbor traversal.
    prepared_angles = torch.as_tensor(prepared["angles"], dtype=torch.long)
    if not torch.equal(angles, prepared_angles):
        angles = prepared_angles
        swap = angles[:, 0] > angles[:, 2]
        angles = angles.clone()
        angles[swap, 0], angles[swap, 2] = (
            angles[swap, 2].clone(),
            angles[swap, 0].clone(),
        )
        order = sorted(range(len(angles)), key=lambda i: tuple(angles[i].tolist()))
        angles = angles[order]
        angle_stats = torch.as_tensor(prepared["angle_stats"])[order]
    else:
        angle_stats = torch.as_tensor(prepared["angle_stats"])
    ring_bonds = bonds[:, torch.as_tensor(prepared["ring_mask"], dtype=torch.bool)]
    chirality = torch.tensor(
        list(prepared["centers"]), dtype=torch.long
    ).reshape(-1, 4)
    sample_id = str(field(record, "sample_id", field(record, "mol_id", "")))
    identity = canonical_identity_sha256(
        {
            "schema_version": CONSTRAINT_SCHEMA_VERSION,
            "feature_version": feature_version,
            "source_identity_sha256": source_identity_sha256,
            "sample_id": sample_id,
            "bonds": bonds.t().tolist(),
            "angles": angles.tolist(),
            "ring_bonds": ring_bonds.t().tolist(),
            "chirality": chirality.tolist(),
        }
    )
    return {
        "constraint_schema_version": CONSTRAINT_SCHEMA_VERSION,
        "constraint_feature_version": str(feature_version),
        "constraint_identity_sha256": identity,
        "constraint_source_identity_sha256": str(source_identity_sha256),
        "active_bond_constraint_index": bonds,
        "bond_allowed_range": torch.as_tensor(
            prepared["bond_stats"], dtype=torch.float32
        ).reshape(-1, 3),
        "active_angle_constraint_index": angles.t().contiguous(),
        "angle_allowed_range": angle_stats.to(torch.float32).reshape(-1, 3),
        "protected_ring_bond_index": ring_bonds,
        "protected_chirality_constraint_index": chirality.t().contiguous(),
    }


def _topology_distances(
    num_atoms: int, bonds: Tensor, max_distance: int
) -> dict[tuple[int, int], int]:
    neighbors: list[list[int]] = [[] for _ in range(int(num_atoms))]
    for left, right in torch.as_tensor(bonds, dtype=torch.long).t().tolist():
        neighbors[left].append(right)
        neighbors[right].append(left)
    result: dict[tuple[int, int], int] = {}
    for start in range(int(num_atoms)):
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            current, distance = queue.popleft()
            if distance >= int(max_distance):
                continue
            for neighbor in sorted(neighbors[current]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                next_distance = distance + 1
                pair = tuple(sorted((start, neighbor)))
                result[pair] = min(result.get(pair, next_distance), next_distance)
                queue.append((neighbor, next_distance))
    return result


def _cell_key(point: Tensor, cell_size: float) -> tuple[int, int, int]:
    values = torch.floor(point.detach().cpu() / float(cell_size)).to(torch.long)
    return tuple(int(value) for value in values.tolist())


def _topology_key_tensors(
    atom_count: int,
    bonds: Tensor,
    atom_batch: Tensor,
    *,
    max_distance: int = 3,
) -> tuple[Tensor, Tensor]:
    device = bonds.device
    cpu_bonds = bonds.detach().cpu()
    cpu_batch = atom_batch.detach().cpu()
    keys: list[int] = []
    distances: list[int] = []
    graphs = int(cpu_batch.max()) + 1 if cpu_batch.numel() else 1
    for graph in range(graphs):
        atom_ids = torch.nonzero(cpu_batch == graph, as_tuple=False).reshape(-1)
        local = {int(global_id): offset for offset, global_id in enumerate(atom_ids)}
        keep = (
            cpu_batch[cpu_bonds[0]] == graph
            if cpu_bonds.numel()
            else torch.empty(0, dtype=torch.bool)
        )
        local_bonds = torch.tensor(
            [
                (local[left], local[right])
                for left, right in cpu_bonds[:, keep].t().tolist()
            ],
            dtype=torch.long,
        ).t().reshape(2, -1)
        topology = _topology_distances(len(atom_ids), local_bonds, max_distance)
        for (local_left, local_right), distance in topology.items():
            left = int(atom_ids[local_left])
            right = int(atom_ids[local_right])
            keys.append(min(left, right) * int(atom_count) + max(left, right))
            distances.append(int(distance))
    if not keys:
        return (
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long),
        )
    key_tensor = torch.tensor(keys, device=device, dtype=torch.long)
    distance_tensor = torch.tensor(distances, device=device, dtype=torch.long)
    order = torch.argsort(key_tensor, stable=True)
    return key_tensor[order], distance_tensor[order]


def _stable_candidate_order(
    graph_index: Tensor, penetration: Tensor, left: Tensor, right: Tensor
) -> Tensor:
    order = torch.arange(left.numel(), device=left.device)
    for values, descending in (
        (right, False),
        (left, False),
        (penetration, True),
        (graph_index, False),
    ):
        local = torch.argsort(values[order], descending=descending, stable=True)
        order = order[local]
    return order


def _sparse_clash_edges_cuda(
    coordinates: Tensor,
    bonds: Tensor,
    atom_batch: Tensor,
    *,
    cutoff: float,
    allowed_contact: float,
    exclude_topology_distance: int,
    max_edges_per_graph: int,
) -> dict[str, Tensor]:
    edge_index = radius_graph(
        coordinates,
        r=float(cutoff),
        batch=atom_batch,
        loop=False,
        max_num_neighbors=max(int(coordinates.size(0)), 256),
    )
    keep = edge_index[0] < edge_index[1]
    edge_index = edge_index[:, keep]
    if not edge_index.numel():
        return _empty_clash_payload(coordinates)
    left, right = edge_index
    atom_count = int(coordinates.size(0))
    pair_keys = left * atom_count + right
    topology_keys, topology_values = _topology_key_tensors(
        atom_count, bonds, atom_batch, max_distance=max(3, exclude_topology_distance)
    )
    topology_distance = torch.zeros_like(left)
    if topology_keys.numel():
        locations = torch.searchsorted(topology_keys, pair_keys)
        valid = locations < topology_keys.numel()
        safe_locations = locations.clamp(max=topology_keys.numel() - 1)
        matched = valid & (topology_keys[safe_locations] == pair_keys)
        topology_distance[matched] = topology_values[safe_locations[matched]]
    keep = ~(
        (topology_distance > 0)
        & (topology_distance <= int(exclude_topology_distance))
    )
    edge_index = edge_index[:, keep]
    topology_distance = topology_distance[keep]
    if not edge_index.numel():
        return _empty_clash_payload(coordinates)
    left, right = edge_index
    relative = coordinates[left] - coordinates[right]
    distance = torch.linalg.vector_norm(relative, dim=-1)
    contacts = distance.new_full(distance.shape, float(allowed_contact))
    penetration = (contacts - distance).clamp_min(0.0)
    graph_index = atom_batch[left]
    order = _stable_candidate_order(graph_index, penetration, left, right)
    selected = []
    graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
    for graph in range(graphs):
        graph_order = order[graph_index[order] == graph]
        selected.append(graph_order[: int(max_edges_per_graph)])
    selected_index = torch.cat(selected) if selected else left.new_empty(0)
    edge_index = edge_index[:, selected_index]
    topology_distance = topology_distance[selected_index]
    relative = relative[selected_index]
    distance = distance[selected_index]
    contacts = contacts[selected_index]
    penetration = penetration[selected_index]
    graph_index = graph_index[selected_index]
    fallback = coordinates.new_tensor([1.0, 0.0, 0.0]).expand_as(relative)
    direction = torch.where(
        (distance > 1.0e-8)[:, None],
        relative / distance.clamp_min(1.0e-8)[:, None],
        fallback,
    )
    return {
        "edge_index": edge_index,
        "distance": distance,
        "allowed_contact": contacts,
        "penetration": penetration,
        "direction": direction,
        "topology_distance": topology_distance,
        "active_mask": penetration > 0.0,
        "graph_index": graph_index,
    }


def _empty_clash_payload(coordinates: Tensor) -> dict[str, Tensor]:
    device = coordinates.device
    return {
        "edge_index": torch.empty((2, 0), device=device, dtype=torch.long),
        "distance": coordinates.new_empty(0),
        "allowed_contact": coordinates.new_empty(0),
        "penetration": coordinates.new_empty(0),
        "direction": coordinates.new_empty((0, 3)),
        "topology_distance": torch.empty(0, device=device, dtype=torch.long),
        "active_mask": torch.empty(0, device=device, dtype=torch.bool),
        "graph_index": torch.empty(0, device=device, dtype=torch.long),
    }


def sparse_clash_edges(
    coordinates: Tensor,
    bonds: Tensor,
    *,
    atom_batch: Tensor | None = None,
    cutoff: float = 2.0,
    allowed_contact: float = 1.0,
    exclude_topology_distance: int = 2,
    max_edges_per_graph: int = 128,
) -> dict[str, Tensor]:
    """Build deterministic radius edges without an O(N^2) tensor allocation."""

    coordinates = torch.as_tensor(coordinates)
    device = coordinates.device
    bonds = torch.as_tensor(bonds, device=device, dtype=torch.long).reshape(2, -1)
    if atom_batch is None:
        atom_batch = torch.zeros(
            coordinates.size(0), device=device, dtype=torch.long
        )
    else:
        atom_batch = torch.as_tensor(atom_batch, device=device, dtype=torch.long)
    if coordinates.is_cuda:
        return _sparse_clash_edges_cuda(
            coordinates,
            bonds,
            atom_batch,
            cutoff=cutoff,
            allowed_contact=allowed_contact,
            exclude_topology_distance=exclude_topology_distance,
            max_edges_per_graph=max_edges_per_graph,
        )
    graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
    rows: list[tuple[float, int, int, int, int]] = []
    for graph in range(graphs):
        atom_ids = torch.nonzero(atom_batch == graph, as_tuple=False).reshape(-1)
        if not atom_ids.numel():
            continue
        local = {int(global_id): offset for offset, global_id in enumerate(atom_ids)}
        keep = atom_batch[bonds[0]] == graph if bonds.numel() else bonds.new_empty(0)
        local_bonds = [
            (local[left], local[right])
            for left, right in bonds[:, keep].t().tolist()
        ]
        local_bond_tensor = torch.tensor(local_bonds, dtype=torch.long).t().reshape(2, -1)
        topology = _topology_distances(
            len(atom_ids), local_bond_tensor, int(exclude_topology_distance)
        )
        cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for global_id in atom_ids.tolist():
            cells[_cell_key(coordinates[global_id], cutoff)].append(int(global_id))
        candidates: list[tuple[float, int, int, int, int]] = []
        for left in atom_ids.tolist():
            base = _cell_key(coordinates[left], cutoff)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for right in cells.get(
                            (base[0] + dx, base[1] + dy, base[2] + dz), []
                        ):
                            if right <= left:
                                continue
                            local_pair = tuple(sorted((local[left], local[right])))
                            topology_distance = topology.get(local_pair, 0)
                            if 0 < topology_distance <= int(exclude_topology_distance):
                                continue
                            distance = float(
                                torch.linalg.vector_norm(
                                    coordinates[left] - coordinates[right]
                                ).detach()
                            )
                            if distance > float(cutoff):
                                continue
                            penetration = max(float(allowed_contact) - distance, 0.0)
                            candidates.append(
                                (-penetration, int(left), int(right), topology_distance, graph)
                            )
        candidates.sort(key=lambda value: value[:4])
        rows.extend(candidates[: int(max_edges_per_graph)])
    rows.sort(key=lambda value: (value[4], value[0], value[1], value[2]))
    if not rows:
        return _empty_clash_payload(coordinates)
    edge_index = torch.tensor(
        [[row[1], row[2]] for row in rows], device=device, dtype=torch.long
    ).t()
    relative = coordinates[edge_index[0]] - coordinates[edge_index[1]]
    distance = torch.linalg.vector_norm(relative, dim=-1)
    fallback = coordinates.new_tensor([1.0, 0.0, 0.0]).expand_as(relative)
    direction = torch.where(
        (distance > 1.0e-8)[:, None],
        relative / distance.clamp_min(1.0e-8)[:, None],
        fallback,
    )
    contacts = distance.new_full(distance.shape, float(allowed_contact))
    penetration = (contacts - distance).clamp_min(0.0)
    return {
        "edge_index": edge_index,
        "distance": distance,
        "allowed_contact": contacts,
        "penetration": penetration,
        "direction": direction,
        "topology_distance": torch.tensor(
            [row[3] for row in rows], device=device, dtype=torch.long
        ),
        "active_mask": penetration > 0.0,
        "graph_index": torch.tensor(
            [row[4] for row in rows], device=device, dtype=torch.long
        ),
    }


def angle_equivariant_directions(
    coordinates: Tensor, triplets: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return stable negative-angle-gradient directions for each triplet."""

    coordinates = torch.as_tensor(coordinates)
    triplets = torch.as_tensor(
        triplets, device=coordinates.device, dtype=torch.long
    ).reshape(-1, 3)
    if not triplets.numel():
        empty = coordinates.new_empty((0, 3))
        return empty, empty, empty
    i, j, k = triplets.unbind(-1)
    left = coordinates[i] - coordinates[j]
    right = coordinates[k] - coordinates[j]
    left_norm = torch.linalg.vector_norm(left, dim=-1, keepdim=True)
    right_norm = torch.linalg.vector_norm(right, dim=-1, keepdim=True)
    left_unit = left / left_norm.clamp_min(1.0e-8)
    right_unit = right / right_norm.clamp_min(1.0e-8)
    cosine = (left_unit * right_unit).sum(-1, keepdim=True).clamp(
        -1.0 + 1.0e-7, 1.0 - 1.0e-7
    )
    sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1.0e-7))
    grad_left = -(
        right_unit - cosine * left_unit
    ) / (left_norm.clamp_min(1.0e-8) * sine)
    grad_right = -(
        left_unit - cosine * right_unit
    ) / (right_norm.clamp_min(1.0e-8) * sine)
    valid = (left_norm > 1.0e-7) & (right_norm > 1.0e-7)
    grad_left = torch.where(valid, grad_left, torch.zeros_like(grad_left))
    grad_right = torch.where(valid, grad_right, torch.zeros_like(grad_right))
    grad_center = -(grad_left + grad_right)
    norm = torch.sqrt(
        grad_left.square().sum(-1)
        + grad_center.square().sum(-1)
        + grad_right.square().sum(-1)
    ).clamp_min(1.0e-8)
    return (
        grad_left / norm[:, None],
        grad_center / norm[:, None],
        grad_right / norm[:, None],
    )


def constraint_graph_index(batch: Any, indices: Tensor, atom_batch: Tensor) -> Tensor:
    indices = torch.as_tensor(indices, device=atom_batch.device, dtype=torch.long)
    if not indices.numel():
        return atom_batch.new_empty(0)
    anchor = indices[1] if indices.ndim == 2 and indices.size(1) >= 3 else indices[0]
    return atom_batch[anchor]


def constraint_counts(graph_index: Tensor, graphs: int) -> Tensor:
    if not graph_index.numel():
        return torch.zeros(graphs, device=graph_index.device, dtype=torch.long)
    return torch.bincount(graph_index, minlength=int(graphs))


def source_constraint_values(batch: Any, coordinates: Tensor) -> dict[str, Tensor]:
    bonds = field(batch, "active_bond_constraint_index")
    if bonds is None:
        bonds = training_bond_index(batch, coordinates.device)
    angles = field(batch, "active_angle_constraint_index")
    if angles is None:
        angles = canonical_angle_triplets(
            field(batch, "edge_index"), coordinates.size(0)
        ).to(coordinates.device)
    return {
        "bond": bond_lengths(coordinates, bonds),
        "angle": bond_angles(coordinates, angles),
        "angle_cosine": stable_angle_cosine(coordinates, angles),
    }


def finite_constraint_payload(payload: Mapping[str, Tensor]) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for value in payload.values()
        if isinstance(value, Tensor) and value.dtype.is_floating_point
    )


def estimate_sparse_work(payload: Mapping[str, Tensor]) -> int:
    return sum(
        int(value.numel())
        for value in payload.values()
        if isinstance(value, Tensor)
    )


def radians_from_cosine(cosine: Tensor) -> Tensor:
    return torch.acos(
        torch.as_tensor(cosine).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    )


def topology_relation_name(distance: int) -> str:
    return {1: "1-2", 2: "1-3", 3: "1-4"}.get(int(distance), "nonbonded")


def default_clash_cutoff(allowed_contact: float) -> float:
    return max(2.0, math.ceil(float(allowed_contact) * 2.0 * 10.0) / 10.0)
