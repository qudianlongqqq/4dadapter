"""Learned Structured Geometry Objective (LSGO) research primitives.

The module learns graph-conditioned Bond/Angle distributions from Reference
geometry only.  It deliberately contains no Cartesian target or force-field
interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import Tensor, nn

from .audit import field
from .geometry import angle_triplets, bond_lengths, unique_bonds
from .rdkit_utils import chiral_center_quads
from .target_building import _record_to_rdkit_mapping


SCHEMA_VERSION = "mcvr-lsgo-v1"
SIGMA_FLOORS = {"bond": 0.005, "angle": 0.010}
SIGMA_MAX = {"bond": 0.200, "angle": 0.400}
HYBRIDIZATION = {
    "UNSPECIFIED": 0, "S": 1, "SP": 2, "SP2": 3, "SP3": 4,
    "SP2D": 5, "SP3D": 6, "SP3D2": 7, "OTHER": 8,
}
BOND_TYPE = {"UNSPECIFIED": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}


@dataclass(frozen=True)
class GraphGeometry:
    atom_categorical: Tensor
    edge_index: Tensor
    edge_categorical: Tensor
    bonds: Tensor
    bond_categorical: Tensor
    angles: Tensor
    angle_edge_categorical: Tensor
    rings: tuple[tuple[int, ...], ...]
    chirality: tuple[tuple[int, int, int, int], ...]
    bond_fixed: Tensor | None = None
    angle_fixed: Tensor | None = None
    ring_fixed: Tensor | None = None
    backoff: Mapping[str, Sequence[str]] | None = None

    def to(self, device: torch.device | str) -> "GraphGeometry":
        values = {}
        for name in (
            "atom_categorical", "edge_index", "edge_categorical", "bonds",
            "bond_categorical", "angles", "angle_edge_categorical",
            "bond_fixed", "angle_fixed", "ring_fixed",
        ):
            value = getattr(self, name)
            values[name] = value.to(device) if value is not None else None
        return GraphGeometry(
            **values,
            rings=self.rings,
            chirality=self.chirality,
            backoff=self.backoff,
        )


def _metadata(record: Any):
    mol, mapping = _record_to_rdkit_mapping(record)
    inverse = {rdkit: cache for cache, rdkit in mapping.items()}
    atoms = {}
    for cache, rdkit in mapping.items():
        atom = mol.GetAtomWithIdx(rdkit)
        atoms[int(cache)] = {
            "z": int(atom.GetAtomicNum()),
            "charge": int(atom.GetFormalCharge()),
            "hybridization": str(atom.GetHybridization()),
            "aromatic": bool(atom.GetIsAromatic()),
            "ring": bool(atom.IsInRing()),
            "degree": int(atom.GetDegree()),
        }
    bonds = {}
    for bond in mol.GetBonds():
        pair = tuple(sorted((inverse[bond.GetBeginAtomIdx()], inverse[bond.GetEndAtomIdx()])))
        bonds[pair] = {
            "type": str(bond.GetBondType()),
            "aromatic": bool(bond.GetIsAromatic()),
            "ring": bool(bond.IsInRing()),
        }
    rings = tuple(tuple(inverse[index] for index in ring) for ring in mol.GetRingInfo().AtomRings())
    fused = tuple(
        any(i != j and len(set(ring) & set(other)) >= 2 for j, other in enumerate(rings))
        for i, ring in enumerate(rings)
    )
    return atoms, bonds, rings, fused


def _bond_contexts(left: int, right: int, atoms, bonds) -> tuple[str, ...]:
    a, b = atoms[left], atoms[right]
    pair = tuple(sorted((a["z"], b["z"])))
    charge = tuple(sorted((a["charge"], b["charge"])))
    value = bonds[tuple(sorted((left, right)))]
    contexts = (
        (pair, value["type"], value["aromatic"], value["ring"], charge),
        (pair, value["type"], value["aromatic"], value["ring"]),
        (pair, value["type"], value["aromatic"]),
        (pair, value["type"]), (value["type"],), ("all",),
    )
    return tuple(f"B{i}|{context!r}" for i, context in enumerate(contexts))


def _angle_contexts(left: int, center: int, right: int, atoms, bonds, rings) -> tuple[str, ...]:
    atom = atoms[center]
    orders = tuple(sorted((
        bonds[tuple(sorted((left, center)))]["type"],
        bonds[tuple(sorted((right, center)))]["type"],
    )))
    containing = [len(ring) for ring in rings if {left, center, right}.issubset(ring)]
    ring_size = min(containing) if containing else 0
    aromatic = atom["aromatic"] or all(
        bonds[tuple(sorted(pair))]["aromatic"] for pair in ((left, center), (right, center))
    )
    in_ring = ring_size > 0
    contexts = (
        (atom["z"], atom["hybridization"], orders, aromatic, in_ring, ring_size),
        (atom["z"], atom["hybridization"], orders, aromatic, in_ring),
        (atom["z"], atom["hybridization"], orders),
        (atom["z"], atom["hybridization"]), (atom["z"],), ("all",),
    )
    return tuple(f"A{i}|{context!r}" for i, context in enumerate(contexts))


def _ring_contexts(index: int, atoms, rings, fused) -> tuple[str, ...]:
    ring = rings[index]
    aromatic = all(atoms[atom]["aromatic"] for atom in ring)
    composition = tuple(sorted(atoms[atom]["z"] for atom in ring))
    contexts = (
        (aromatic, len(ring), fused[index], composition),
        (aromatic, len(ring), fused[index]), (aromatic, len(ring)),
        (aromatic,), ("all",),
    )
    return tuple(f"R{i}|{context!r}" for i, context in enumerate(contexts))


def _select_stat(calibration: Mapping[str, Any], mode: str, contexts: Sequence[str]):
    statistics = calibration["primitive_stats"][mode]
    fallback = None
    for key in contexts:
        if key in statistics:
            fallback = (statistics[key], key)
            if int(statistics[key]["count"]) >= int(calibration["minimum_context_count"]):
                return fallback
    if fallback is None:
        raise KeyError(f"missing frozen DRCSR {mode} fallback")
    return fallback


def _bond_code(value: Mapping[str, Any]) -> list[int]:
    return [
        BOND_TYPE.get(str(value["type"]), 0),
        int(bool(value["aromatic"])),
        int(bool(value["ring"])),
    ]


def prepare_graph(record: Any, calibration: Mapping[str, Any] | None = None) -> GraphGeometry:
    atoms, metadata_bonds, rings, fused = _metadata(record)
    atom_values = []
    for index in range(len(atoms)):
        atom = atoms[index]
        atom_values.append([
            min(max(atom["z"], 0), 118),
            min(max(atom["charge"] + 5, 0), 10),
            HYBRIDIZATION.get(atom["hybridization"], HYBRIDIZATION["OTHER"]),
            int(atom["aromatic"]), int(atom["ring"]), min(atom["degree"], 8),
        ])
    edge_index = torch.as_tensor(field(record, "edge_index"), dtype=torch.long).reshape(2, -1)
    edge_codes = [_bond_code(metadata_bonds[tuple(sorted(pair))]) for pair in edge_index.t().tolist()]
    bonds = unique_bonds(edge_index)
    bond_codes = [_bond_code(metadata_bonds[tuple(sorted(pair))]) for pair in bonds.t().tolist()]
    angles = angle_triplets(edge_index, len(atoms))
    angle_codes = []
    for left, center, right in angles.tolist():
        angle_codes.append(
            _bond_code(metadata_bonds[tuple(sorted((left, center)))])
            + _bond_code(metadata_bonds[tuple(sorted((right, center)))])
        )
    bond_fixed = angle_fixed = ring_fixed = None
    backoff = None
    if calibration is not None:
        selected_bonds, selected_angles, selected_rings = [], [], []
        backoff = {"bond": [], "angle": [], "ring": []}
        for left, right in bonds.t().tolist():
            stat, key = _select_stat(calibration, "bond", _bond_contexts(left, right, atoms, metadata_bonds))
            selected_bonds.append((stat["median"], stat["scale"]))
            backoff["bond"].append(key.split("|", 1)[0])
        for left, center, right in angles.tolist():
            stat, key = _select_stat(
                calibration, "angle",
                _angle_contexts(left, center, right, atoms, metadata_bonds, rings),
            )
            selected_angles.append((stat["median"], stat["scale"]))
            backoff["angle"].append(key.split("|", 1)[0])
        for index, ring in enumerate(rings):
            if len(ring) < 4:
                continue
            stat, key = _select_stat(calibration, "ring", _ring_contexts(index, atoms, rings, fused))
            selected_rings.append((index, stat["median"], stat["scale"]))
            backoff["ring"].append(key.split("|", 1)[0])
        bond_fixed = torch.tensor(selected_bonds, dtype=torch.float64).reshape(-1, 2)
        angle_fixed = torch.tensor(selected_angles, dtype=torch.float64).reshape(-1, 2)
        ring_fixed = torch.tensor(selected_rings, dtype=torch.float64).reshape(-1, 3)
    return GraphGeometry(
        atom_categorical=torch.tensor(atom_values, dtype=torch.long),
        edge_index=edge_index,
        edge_categorical=torch.tensor(edge_codes, dtype=torch.long).reshape(-1, 3),
        bonds=bonds,
        bond_categorical=torch.tensor(bond_codes, dtype=torch.long).reshape(-1, 3),
        angles=angles,
        angle_edge_categorical=torch.tensor(angle_codes, dtype=torch.long).reshape(-1, 6),
        rings=rings,
        chirality=chiral_center_quads(record),
        bond_fixed=bond_fixed,
        angle_fixed=angle_fixed,
        ring_fixed=ring_fixed,
        backoff=backoff,
    )


def stable_angle_cosine(coordinates: Tensor, triplets: Tensor, eps: float = 1.0e-8) -> Tensor:
    coordinates = torch.as_tensor(coordinates)
    triplets = torch.as_tensor(triplets, device=coordinates.device, dtype=torch.long).reshape(-1, 3)
    if not triplets.numel():
        return coordinates.new_empty(0)
    left = coordinates[triplets[:, 0]] - coordinates[triplets[:, 1]]
    right = coordinates[triplets[:, 2]] - coordinates[triplets[:, 1]]
    denominator = (
        torch.linalg.vector_norm(left, dim=-1).clamp_min(eps)
        * torch.linalg.vector_norm(right, dim=-1).clamp_min(eps)
    )
    return ((left * right).sum(-1) / denominator).clamp(-1 + 1.0e-7, 1 - 1.0e-7)


def geometry_values(coordinates: Tensor, graph: GraphGeometry) -> tuple[Tensor, Tensor]:
    return bond_lengths(coordinates, graph.bonds), stable_angle_cosine(coordinates, graph.angles)


def gaussian_nll(value: Tensor, mean: Tensor, sigma: Tensor) -> Tensor:
    sigma = torch.as_tensor(sigma, device=value.device, dtype=value.dtype).clamp_min(1.0e-8)
    return 0.5 * ((value - mean) / sigma).square() + torch.log(sigma) + 0.5 * math.log(2 * math.pi)


class MessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node: Tensor, edge_index: Tensor, edge: Tensor) -> Tensor:
        left, right = edge_index
        messages = self.message(torch.cat((node[left], node[right], edge), dim=-1))
        aggregated = node.new_zeros(node.shape)
        aggregated.index_add_(0, right, messages)
        counts = torch.bincount(right, minlength=node.size(0)).clamp_min(1).to(node.dtype)
        return self.norm(node + self.update(torch.cat((node, aggregated / counts[:, None]), dim=-1)))


class LearnedGeometryObjective(nn.Module):
    """Small invariant GNN predicting local conditional means and uncertainty."""

    def __init__(self, *, hidden_dim: int = 128, layers: int = 3, learned_sigma: bool = True) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers_count = int(layers)
        self.learned_sigma = bool(learned_sigma)
        self.element = nn.Embedding(119, 32)
        self.charge = nn.Embedding(11, 8)
        self.hybridization = nn.Embedding(9, 12)
        self.aromatic = nn.Embedding(2, 4)
        self.ring = nn.Embedding(2, 4)
        self.degree = nn.Embedding(9, 8)
        self.atom_projection = nn.Sequential(nn.Linear(68, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.order = nn.Embedding(5, 12)
        self.edge_aromatic = nn.Embedding(2, 4)
        self.edge_ring = nn.Embedding(2, 4)
        edge_dim = 20
        self.layers = nn.ModuleList(MessageLayer(hidden_dim, edge_dim) for _ in range(layers))
        output_dim = 2 if learned_sigma else 1
        self.bond_head = nn.Sequential(
            nn.Linear(3 * hidden_dim + edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, output_dim),
        )
        self.angle_head = nn.Sequential(
            nn.Linear(4 * hidden_dim + 2 * edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, output_dim),
        )

    def _edge(self, value: Tensor) -> Tensor:
        return torch.cat((
            self.order(value[:, 0]), self.edge_aromatic(value[:, 1]), self.edge_ring(value[:, 2])
        ), dim=-1)

    def forward(self, graph: GraphGeometry) -> dict[str, Tensor]:
        atoms = graph.atom_categorical
        node = self.atom_projection(torch.cat((
            self.element(atoms[:, 0]), self.charge(atoms[:, 1]),
            self.hybridization(atoms[:, 2]), self.aromatic(atoms[:, 3]),
            self.ring(atoms[:, 4]), self.degree(atoms[:, 5]),
        ), dim=-1))
        directed_edge = self._edge(graph.edge_categorical)
        for layer in self.layers:
            node = layer(node, graph.edge_index, directed_edge)
        left, right = graph.bonds
        bond_edge = self._edge(graph.bond_categorical)
        bond_raw = self.bond_head(torch.cat((
            node[left] + node[right], (node[left] - node[right]).abs(),
            node[left] * node[right], bond_edge,
        ), dim=-1))
        bond_mu = F.softplus(bond_raw[:, 0])
        angle = graph.angles
        if angle.numel():
            aleft, center, aright = angle.t()
            angle_edges = graph.angle_edge_categorical
            first_edge = self._edge(angle_edges[:, :3])
            second_edge = self._edge(angle_edges[:, 3:])
            angle_raw = self.angle_head(torch.cat((
                node[center], node[aleft] + node[aright],
                (node[aleft] - node[aright]).abs(), node[aleft] * node[aright],
                first_edge + second_edge, (first_edge - second_edge).abs(),
            ), dim=-1))
            angle_mu = torch.tanh(angle_raw[:, 0])
        else:
            angle_raw = node.new_empty((0, 2 if self.learned_sigma else 1))
            angle_mu = node.new_empty(0)
        result = {"bond_mu": bond_mu, "angle_mu": angle_mu, "node_embedding": node}
        if self.learned_sigma:
            bond_unclamped = SIGMA_FLOORS["bond"] + F.softplus(bond_raw[:, 1])
            angle_unclamped = SIGMA_FLOORS["angle"] + F.softplus(angle_raw[:, 1])
            result.update({
                "bond_sigma": bond_unclamped.clamp_max(SIGMA_MAX["bond"]),
                "angle_sigma": angle_unclamped.clamp_max(SIGMA_MAX["angle"]),
                "bond_sigma_unclamped": bond_unclamped,
                "angle_sigma_unclamped": angle_unclamped,
            })
        return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def distribution_parameters(
    graph: GraphGeometry,
    *,
    model: LearnedGeometryObjective | None,
    variant: str,
) -> dict[str, Tensor]:
    variant = variant.upper()
    if variant == "A":
        if graph.bond_fixed is None or graph.angle_fixed is None:
            raise ValueError("Variant A requires frozen DRCSR statistics")
        return {
            "bond_mu": graph.bond_fixed[:, 0], "bond_sigma": graph.bond_fixed[:, 1],
            "angle_mu": graph.angle_fixed[:, 0], "angle_sigma": graph.angle_fixed[:, 1],
        }
    if model is None:
        raise ValueError("neural variants require a model")
    prediction = model(graph)
    if variant == "B":
        if graph.bond_fixed is None or graph.angle_fixed is None:
            raise ValueError("Variant B requires frozen DRCSR uncertainty")
        prediction["bond_sigma"] = graph.bond_fixed[:, 1]
        prediction["angle_sigma"] = graph.angle_fixed[:, 1]
    elif variant != "C":
        raise ValueError(f"unknown LSGO variant: {variant}")
    return prediction


def structured_objective(
    coordinates: Tensor,
    graph: GraphGeometry,
    parameters: Mapping[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    bonds, angles = geometry_values(coordinates, graph)
    bond_nll = gaussian_nll(bonds, parameters["bond_mu"], parameters["bond_sigma"])
    angle_nll = gaussian_nll(angles, parameters["angle_mu"], parameters["angle_sigma"])
    groups = {
        "bond": bond_nll.mean() if bond_nll.numel() else coordinates.new_zeros(()),
        "angle": angle_nll.mean() if angle_nll.numel() else coordinates.new_zeros(()),
    }
    # Equal group averaging avoids primitive-count dominance; it is a disclosed
    # aggregation inductive bias, not a manually tuned BAC coefficient.
    return torch.stack(tuple(groups.values())).mean(), groups


def remove_rigid_component(vector: Tensor, coordinates: Tensor) -> Tensor:
    vector = torch.as_tensor(vector)
    centered = coordinates - coordinates.mean(0, keepdim=True)
    modes = []
    for axis in torch.eye(3, device=vector.device, dtype=vector.dtype):
        modes.append(axis.expand_as(vector).reshape(-1))
        modes.append(torch.linalg.cross(axis.expand_as(centered), centered, dim=-1).reshape(-1))
    matrix = torch.stack(modes, dim=1)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    valid = torch.diagonal(r).abs() > 1.0e-10
    flat = vector.reshape(-1)
    if bool(valid.any()):
        basis = q[:, valid]
        flat = flat - basis @ (basis.t() @ flat)
    return flat.reshape_as(vector)


def trust_project(source: Tensor, delta: Tensor, *, rms_budget: float, atom_cap: float) -> tuple[Tensor, dict[str, float]]:
    source = torch.as_tensor(source)
    delta = remove_rigid_component(torch.as_tensor(delta, device=source.device, dtype=source.dtype), source)
    norms = torch.linalg.vector_norm(delta, dim=-1)
    atom_scale = torch.clamp(delta.new_tensor(atom_cap) / norms.clamp_min(1.0e-15), max=1.0)
    delta = delta * atom_scale[:, None]
    raw_rms = torch.sqrt(delta.square().sum(-1).mean())
    graph_scale = min(1.0, float(rms_budget) / max(float(raw_rms), 1.0e-15))
    delta = delta * graph_scale
    return source + delta, {
        "raw_rms": float(raw_rms),
        "graph_scale": float(graph_scale),
        "atom_scale_min": float(atom_scale.min()) if atom_scale.numel() else 1.0,
        "final_rms": float(torch.sqrt(delta.square().sum(-1).mean())),
        "final_atom_max": float(torch.linalg.vector_norm(delta, dim=-1).max()),
    }


def direct_gradient_update(
    source: Tensor,
    graph: GraphGeometry,
    parameters: Mapping[str, Tensor],
    *,
    rms_budget: float,
    atom_cap: float,
    steps: int = 1,
) -> dict[str, Any]:
    source = torch.as_tensor(source, dtype=torch.float64)
    current = source.clone()
    trace = []
    for index in range(int(steps)):
        coordinates = current.detach().clone().requires_grad_(True)
        objective, groups = structured_objective(coordinates, graph, parameters)
        gradient, = torch.autograd.grad(objective, coordinates)
        gradient = remove_rigid_component(gradient, coordinates)
        gradient_rms = torch.sqrt(gradient.square().sum(-1).mean())
        if not bool(torch.isfinite(gradient).all()) or float(gradient_rms) <= 1.0e-14:
            break
        direction = -gradient / gradient_rms
        current, trust = trust_project(
            source,
            current - source + direction * (float(rms_budget) / int(steps)),
            rms_budget=float(rms_budget),
            atom_cap=float(atom_cap),
        )
        trace.append({
            "step": index + 1, "objective": float(objective),
            "bond_objective": float(groups["bond"]), "angle_objective": float(groups["angle"]),
            "gradient_rms": float(gradient_rms), **trust,
        })
    return {"coordinates": current, "trace": trace, "finite": bool(torch.isfinite(current).all())}


def precision_projection_update(
    source: Tensor,
    graph: GraphGeometry,
    parameters: Mapping[str, Tensor],
    *,
    ridge: float,
    rank_tolerance: float,
    maximum_condition: float,
    rms_budget: float,
    atom_cap: float,
    steps: int = 1,
) -> dict[str, Any]:
    source = torch.as_tensor(source, dtype=torch.float64)
    current = source.clone()
    trace = []
    for index in range(int(steps)):
        def values(flat: Tensor) -> Tensor:
            coordinates = flat.reshape_as(current)
            bond, angle = geometry_values(coordinates, graph)
            return torch.cat((bond, angle))

        flat = current.detach().clone().reshape(-1).requires_grad_(True)
        q = values(flat)
        mean = torch.cat((parameters["bond_mu"], parameters["angle_mu"]))
        sigma = torch.cat((parameters["bond_sigma"], parameters["angle_sigma"])).clamp_min(1.0e-8)
        jacobian = torch.autograd.functional.jacobian(values, flat, vectorize=True)
        residual = q - mean
        sqrt_precision = sigma.reciprocal()
        weighted = jacobian * sqrt_precision[:, None]
        singular = torch.linalg.svdvals(weighted)
        positive = singular[singular > float(rank_tolerance)]
        rank = int(positive.numel())
        condition = float(positive.max() / positive.min()) if positive.numel() else float("inf")
        normal = weighted.t() @ weighted + float(ridge) * torch.eye(
            flat.numel(), device=flat.device, dtype=flat.dtype
        )
        rhs = weighted.t() @ (sqrt_precision * residual)
        backend = "solve"
        try:
            if not math.isfinite(condition) or condition > float(maximum_condition):
                raise RuntimeError("ill-conditioned weighted Jacobian")
            delta = -torch.linalg.solve(normal, rhs).reshape_as(current)
        except RuntimeError:
            backend = "svd_pinv"
            delta = -(torch.linalg.pinv(normal, rcond=float(rank_tolerance)) @ rhs).reshape_as(current)
        candidate, trust = trust_project(
            source,
            current - source + delta,
            rms_budget=float(rms_budget), atom_cap=float(atom_cap),
        )
        finite = bool(torch.isfinite(candidate).all())
        if not finite:
            candidate = current
        current = candidate.detach()
        trace.append({
            "step": index + 1, "primitive_count": int(q.numel()),
            "cartesian_dimension": int(flat.numel()), "effective_rank": rank,
            "condition_number": condition, "singular_min": float(positive.min()) if positive.numel() else 0.0,
            "singular_max": float(positive.max()) if positive.numel() else 0.0,
            "solver_backend": backend, "finite": finite, **trust,
        })
    return {"coordinates": current, "trace": trace, "finite": bool(torch.isfinite(current).all())}


def ring_planarity(coordinates: Tensor, ring: Sequence[int]) -> Tensor:
    if len(ring) < 4:
        return coordinates.new_zeros(())
    points = coordinates[list(ring)]
    centered = points - points.mean(0, keepdim=True)
    return torch.linalg.svdvals(centered)[-1] / math.sqrt(len(ring))


def frozen_ring_score(coordinates: Tensor, graph: GraphGeometry) -> Tensor:
    if graph.ring_fixed is None or not graph.ring_fixed.numel():
        return coordinates.new_zeros(())
    values, means, sigmas = [], [], []
    for ring_index, mean, sigma in graph.ring_fixed.tolist():
        values.append(ring_planarity(coordinates, graph.rings[int(ring_index)]))
        means.append(mean)
        sigmas.append(sigma)
    value = torch.stack(values)
    return gaussian_nll(value, value.new_tensor(means), value.new_tensor(sigmas)).mean()


def chirality_preserved(source: Tensor, candidate: Tensor, quads) -> bool:
    source = torch.as_tensor(source, dtype=torch.float64)
    candidate = torch.as_tensor(candidate, dtype=torch.float64)
    for center, first, second, third in quads:
        before = torch.linalg.det(torch.stack((
            source[first] - source[center], source[second] - source[center], source[third] - source[center]
        )))
        after = torch.linalg.det(torch.stack((
            candidate[first] - candidate[center], candidate[second] - candidate[center], candidate[third] - candidate[center]
        )))
        if abs(float(before)) >= 1.0e-5 and float(before * after) <= 0:
            return False
    return True


def catastrophic_penetration(coordinates: Tensor, graph: GraphGeometry) -> Tensor:
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64)
    neighbors = [set() for _ in range(coordinates.size(0))]
    for left, right in graph.bonds.t().tolist():
        neighbors[left].add(right)
        neighbors[right].add(left)
    excluded = set()
    for start in range(coordinates.size(0)):
        for one in neighbors[start]:
            excluded.add(tuple(sorted((start, one))))
            for two in neighbors[one]:
                excluded.add(tuple(sorted((start, two))))
    table = Chem.GetPeriodicTable()
    values = []
    for left in range(coordinates.size(0)):
        for right in range(left + 1, coordinates.size(0)):
            if (left, right) in excluded:
                continue
            radius = float(
                table.GetRvdw(int(graph.atom_categorical[left, 0]))
                + table.GetRvdw(int(graph.atom_categorical[right, 0]))
            )
            ratio = torch.linalg.vector_norm(coordinates[left] - coordinates[right]) / radius
            values.append(torch.relu(coordinates.new_tensor(0.50) - ratio) / 0.05)
    return torch.stack(values).max() if values else coordinates.new_zeros(())


def safety_accept(source: Tensor, candidate: Tensor, graph: GraphGeometry) -> tuple[Tensor, dict[str, Any]]:
    source = torch.as_tensor(source, dtype=torch.float64)
    candidate = torch.as_tensor(candidate, dtype=torch.float64)
    finite = bool(torch.isfinite(candidate).all())
    chirality = chirality_preserved(source, candidate, graph.chirality) if finite else False
    ring_before = frozen_ring_score(source, graph)
    ring_after = frozen_ring_score(candidate, graph) if finite else source.new_tensor(float("inf"))
    ring_ok = bool(ring_after <= ring_before + 0.05)
    penetration_before = catastrophic_penetration(source, graph)
    penetration_after = catastrophic_penetration(candidate, graph) if finite else source.new_tensor(float("inf"))
    clash_ok = bool(penetration_after <= penetration_before + 0.02)
    accepted = finite and chirality and ring_ok and clash_ok
    return (candidate if accepted else source), {
        "accepted": accepted, "fallback": not accepted, "finite": finite,
        "chirality_preserved": chirality, "ring_nonregression": ring_ok,
        "catastrophic_clash_nonregression": clash_ok,
        "ring_score_source": float(ring_before), "ring_score_candidate": float(ring_after),
        "catastrophic_source": float(penetration_before), "catastrophic_candidate": float(penetration_after),
    }
