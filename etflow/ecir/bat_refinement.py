"""BAT refinement primitives: frozen BA, periodic torsion modes and steric safety.

This module contains no Cartesian teacher, force-field/PB interface, learned
uncertainty rescue or amortized coordinate predictor.
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
from .learned_geometry import (
    GraphGeometry,
    direct_gradient_update,
    distribution_parameters,
    remove_rigid_component,
    safety_accept,
    structured_objective,
    trust_project,
)
from .target_building import _record_to_rdkit_mapping


SCHEMA_VERSION = "mcvr-bat-refinement-v1"


@dataclass(frozen=True)
class BATGraph:
    base: GraphGeometry
    torsions: Tensor
    torsion_categorical: Tensor
    nonbonded_pairs: Tensor
    safe_distances: Tensor
    catastrophic_distances: Tensor
    topology_distances: Tensor
    canonical_metadata: tuple[dict[str, Any], ...]

    def to(self, device: torch.device | str) -> "BATGraph":
        return BATGraph(
            base=self.base.to(device),
            torsions=self.torsions.to(device),
            torsion_categorical=self.torsion_categorical.to(device),
            nonbonded_pairs=self.nonbonded_pairs.to(device),
            safe_distances=self.safe_distances.to(device),
            catastrophic_distances=self.catastrophic_distances.to(device),
            topology_distances=self.topology_distances.to(device),
            canonical_metadata=self.canonical_metadata,
        )


def wrap_periodic(value: Tensor) -> Tensor:
    value = torch.as_tensor(value)
    return torch.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def circular_difference(value: Tensor, target: Tensor) -> Tensor:
    return wrap_periodic(torch.as_tensor(value) - torch.as_tensor(target))


def _atom_key(molecule: Chem.Mol, index: int, rank: Sequence[int], cache: int) -> tuple[int, ...]:
    atom = molecule.GetAtomWithIdx(index)
    return (
        int(rank[index]), int(atom.GetAtomicNum()), int(atom.GetFormalCharge()),
        int(atom.GetIsAromatic()), int(cache),
    )


def _restricted_carbonyl_like_bond(molecule: Chem.Mol, left: int, right: int) -> bool:
    """Exclude amide/sulfonamide-like central bonds from free torsion labels."""

    first, second = molecule.GetAtomWithIdx(left), molecule.GetAtomWithIdx(right)
    for hetero, center in ((first, second), (second, first)):
        if hetero.GetAtomicNum() != 7 or center.GetAtomicNum() not in (6, 15, 16):
            continue
        for adjacent in center.GetBonds():
            other = adjacent.GetOtherAtom(center)
            if other.GetIdx() == hetero.GetIdx():
                continue
            if adjacent.GetBondType() == Chem.BondType.DOUBLE and other.GetAtomicNum() in (8, 16):
                return True
    return False


def canonical_rotatable_torsions(record: Any) -> tuple[Tensor, Tensor, tuple[dict[str, Any], ...]]:
    """Return one deterministic non-ring torsion quadruple per project rotor."""

    molecule, cache_to_rdkit = _record_to_rdkit_mapping(record)
    rdkit_to_cache = {rdkit: cache for cache, rdkit in cache_to_rdkit.items()}
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True, includeChirality=True))
    symmetry_ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=False, includeChirality=True))
    raw = torch.as_tensor(
        field(record, "rotatable_bond_index", torch.empty((2, 0))), dtype=torch.long
    ).reshape(2, -1)
    central_pairs = {tuple(sorted((int(left), int(right)))) for left, right in raw.t().tolist()}
    torsions: list[list[int]] = []
    categorical: list[list[int]] = []
    metadata: list[dict[str, Any]] = []
    for original_left, original_right in sorted(central_pairs):
        rd_left, rd_right = cache_to_rdkit[original_left], cache_to_rdkit[original_right]
        bond = molecule.GetBondBetweenAtoms(rd_left, rd_right)
        if bond is None or bond.IsInRing() or str(bond.GetBondType()) != "SINGLE":
            continue
        if _restricted_carbonyl_like_bond(molecule, rd_left, rd_right):
            continue
        left_key = _atom_key(molecule, rd_left, ranks, original_left)
        right_key = _atom_key(molecule, rd_right, ranks, original_right)
        if left_key <= right_key:
            center_left, center_right = original_left, original_right
            rd_center_left, rd_center_right = rd_left, rd_right
        else:
            center_left, center_right = original_right, original_left
            rd_center_left, rd_center_right = rd_right, rd_left
        left_candidates = [
            neighbor.GetIdx() for neighbor in molecule.GetAtomWithIdx(rd_center_left).GetNeighbors()
            if neighbor.GetIdx() != rd_center_right
        ]
        right_candidates = [
            neighbor.GetIdx() for neighbor in molecule.GetAtomWithIdx(rd_center_right).GetNeighbors()
            if neighbor.GetIdx() != rd_center_left
        ]
        if not left_candidates or not right_candidates:
            continue
        def terminal_key(rdkit_index: int) -> tuple[int, ...]:
            atom = molecule.GetAtomWithIdx(rdkit_index)
            cache_index = rdkit_to_cache[rdkit_index]
            return (
                int(atom.GetAtomicNum() > 1), int(ranks[rdkit_index]),
                int(atom.GetAtomicNum()), int(atom.GetFormalCharge()), int(cache_index),
            )
        rd_terminal_left = max(left_candidates, key=terminal_key)
        rd_terminal_right = max(right_candidates, key=terminal_key)
        terminal_left = rdkit_to_cache[rd_terminal_left]
        terminal_right = rdkit_to_cache[rd_terminal_right]
        quadruple = [terminal_left, center_left, center_right, terminal_right]
        torsions.append(quadruple)
        central_atom_left = molecule.GetAtomWithIdx(rd_center_left)
        central_atom_right = molecule.GetAtomWithIdx(rd_center_right)
        categorical.append([
            min(central_atom_left.GetAtomicNum(), central_atom_right.GetAtomicNum()),
            max(central_atom_left.GetAtomicNum(), central_atom_right.GetAtomicNum()),
            int(central_atom_left.GetHybridization()), int(central_atom_right.GetHybridization()),
            int(central_atom_left.GetIsAromatic() or central_atom_right.GetIsAromatic()),
        ])
        symmetric_left = len({symmetry_ranks[value] for value in left_candidates}) < len(left_candidates)
        symmetric_right = len({symmetry_ranks[value] for value in right_candidates}) < len(right_candidates)
        metadata.append({
            "central_bond": [center_left, center_right], "canonical_quadruple": quadruple,
            "central_orientation_key": [list(left_key), list(right_key)],
            "terminal_rule": "heavy_then_canonical_rank_then_atom_identity",
            "symmetric_terminal_environment": bool(symmetric_left or symmetric_right),
            "ring_bond": False, "bond_type": "SINGLE",
        })
    return (
        torch.tensor(torsions, dtype=torch.long).reshape(-1, 4),
        torch.tensor(categorical, dtype=torch.long).reshape(-1, 5),
        tuple(metadata),
    )


def dihedral_angles(coordinates: Tensor, quadruples: Tensor, eps: float = 1.0e-10) -> Tensor:
    coordinates = torch.as_tensor(coordinates)
    quadruples = torch.as_tensor(quadruples, device=coordinates.device, dtype=torch.long).reshape(-1, 4)
    if not quadruples.numel():
        return coordinates.new_empty(0)
    p0, p1, p2, p3 = (coordinates[quadruples[:, index]] for index in range(4))
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_unit = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(eps)
    v = b0 - (b0 * b1_unit).sum(-1, keepdim=True) * b1_unit
    w = b2 - (b2 * b1_unit).sum(-1, keepdim=True) * b1_unit
    x = (v * w).sum(-1)
    y = (torch.linalg.cross(b1_unit, v, dim=-1) * w).sum(-1)
    valid = (torch.linalg.vector_norm(v, dim=-1) > eps) & (torch.linalg.vector_norm(w, dim=-1) > eps)
    result = torch.atan2(y, x)
    return torch.where(valid, wrap_periodic(result), torch.zeros_like(result))


def von_mises_log_prob(value: Tensor, mean: Tensor, kappa: Tensor) -> Tensor:
    value, mean, kappa = torch.broadcast_tensors(value, mean, kappa)
    kappa = kappa.clamp_min(0.0)
    log_i0 = torch.log(torch.special.i0e(kappa).clamp_min(torch.finfo(kappa.dtype).tiny)) + kappa
    return kappa * torch.cos(circular_difference(value, mean)) - math.log(2.0 * math.pi) - log_i0


def mixture_log_prob(value: Tensor, logits: Tensor, means: Tensor, kappa: Tensor) -> Tensor:
    value = torch.as_tensor(value)
    log_weights = torch.log_softmax(logits, dim=-1)
    component = von_mises_log_prob(value[..., None], means, kappa)
    return torch.logsumexp(log_weights + component, dim=-1)


def mixture_responsibilities(value: Tensor, logits: Tensor, means: Tensor, kappa: Tensor) -> Tensor:
    joint = torch.log_softmax(logits, dim=-1) + von_mises_log_prob(value[..., None], means, kappa)
    return torch.softmax(joint, dim=-1)


class TorsionHead(nn.Module):
    """Light head over a frozen BA encoder representation."""

    def __init__(self, hidden_dim: int = 128, components: int = 3, learned_kappa: bool = False) -> None:
        super().__init__()
        self.components = int(components)
        self.learned_kappa = bool(learned_kappa)
        outputs = self.components * (4 if learned_kappa else 3)
        self.network = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, outputs),
        )

    def forward(
        self, node_embedding: Tensor, torsions: Tensor, *, fixed_kappa: float,
        kappa_floor: float = 0.25, kappa_ceiling: float = 32.0,
    ) -> dict[str, Tensor]:
        torsions = torch.as_tensor(torsions, device=node_embedding.device, dtype=torch.long).reshape(-1, 4)
        if not torsions.numel():
            empty = node_embedding.new_empty((0, self.components))
            return {"logits": empty, "weights": empty, "means": empty, "kappa": empty}
        raw = self.network(torch.cat(tuple(node_embedding[torsions[:, index]] for index in range(4)), dim=-1))
        count = self.components
        logits = raw[:, :count]
        sine = raw[:, count:2 * count]
        cosine = raw[:, 2 * count:3 * count]
        norm = torch.sqrt(sine.square() + cosine.square()).clamp_min(1.0e-8)
        means = torch.atan2(sine / norm, cosine / norm)
        if self.learned_kappa:
            kappa = (float(kappa_floor) + F.softplus(raw[:, 3 * count:4 * count])).clamp_max(float(kappa_ceiling))
        else:
            kappa = raw.new_full((raw.size(0), count), float(fixed_kappa))
        return {"logits": logits, "weights": torch.softmax(logits, -1), "means": means, "kappa": kappa}


def torsion_nll(values: Tensor, prediction: Mapping[str, Tensor]) -> Tensor:
    return -mixture_log_prob(values, prediction["logits"], prediction["means"], prediction["kappa"])


def torsion_pathology(prediction: Mapping[str, Tensor]) -> dict[str, float | bool]:
    weights, means, kappa = prediction["weights"], prediction["means"], prediction["kappa"]
    if not weights.numel():
        return {"empty": True, "component_collapse": False, "mode_duplication_fraction": 0.0}
    occupancy = weights.mean(0)
    distances = []
    for left in range(means.size(1)):
        for right in range(left + 1, means.size(1)):
            distances.append(circular_difference(means[:, left], means[:, right]).abs())
    minimum_separation = torch.stack(distances).min(0).values if distances else means.new_full((means.size(0),), math.pi)
    return {
        "empty": False,
        "occupancy_min": float(occupancy.min()), "occupancy_max": float(occupancy.max()),
        "component_collapse": bool(occupancy.min() < 0.03),
        "mode_duplication_fraction": float((minimum_separation < 0.15).float().mean()),
        "mixture_entropy_mean": float((-(weights * torch.log(weights.clamp_min(1e-12))).sum(-1)).mean()),
        "kappa_mean": float(kappa.mean()), "kappa_min": float(kappa.min()), "kappa_max": float(kappa.max()),
        "uniformization_fraction": float((kappa < 0.5).float().mean()),
        "ceiling_fraction": float((kappa >= 31.999).float().mean()),
    }

def _topology_distances(atom_count: int, bonds: Tensor, maximum: int = 3) -> dict[tuple[int, int], int]:
    neighbors = [set() for _ in range(atom_count)]
    for left, right in torch.as_tensor(bonds, dtype=torch.long).reshape(2, -1).t().tolist():
        neighbors[left].add(right)
        neighbors[right].add(left)
    result: dict[tuple[int, int], int] = {}
    for start in range(atom_count):
        frontier, seen = {start}, {start}
        for distance in range(1, maximum + 1):
            frontier = {node for value in frontier for node in neighbors[value]} - seen
            for node in frontier:
                result[tuple(sorted((start, node)))] = distance
            seen.update(frontier)
    return result


def steric_pairs(
    graph: GraphGeometry, *, safe_factor_nonbonded: float = 0.75,
    safe_factor_1_4: float = 0.60, catastrophic_factor: float = 0.50,
    include_hydrogens: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    atom_count = int(graph.atom_categorical.size(0))
    topology = _topology_distances(atom_count, graph.bonds, maximum=3)
    periodic = Chem.GetPeriodicTable()
    pairs, safe, catastrophic, relation = [], [], [], []
    for left in range(atom_count):
        z_left = int(graph.atom_categorical[left, 0])
        if z_left == 1 and not include_hydrogens:
            continue
        for right in range(left + 1, atom_count):
            z_right = int(graph.atom_categorical[right, 0])
            if z_right == 1 and not include_hydrogens:
                continue
            distance = topology.get((left, right), 0)
            if distance in (1, 2):
                continue
            radii = float(periodic.GetRvdw(z_left) + periodic.GetRvdw(z_right))
            factor = float(safe_factor_1_4 if distance == 3 else safe_factor_nonbonded)
            pairs.append((left, right)); safe.append(factor * radii)
            catastrophic.append(float(catastrophic_factor) * radii); relation.append(distance)
    return (
        torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
        torch.tensor(safe, dtype=torch.float64),
        torch.tensor(catastrophic, dtype=torch.float64),
        torch.tensor(relation, dtype=torch.long),
    )


def prepare_bat_graph(
    base: GraphGeometry, record: Any, *, safe_factor_nonbonded: float = 0.75,
    safe_factor_1_4: float = 0.60, catastrophic_factor: float = 0.50,
    include_hydrogens: bool = False,
) -> BATGraph:
    torsions, codes, metadata = canonical_rotatable_torsions(record)
    pairs, safe, catastrophic, topology = steric_pairs(
        base, safe_factor_nonbonded=safe_factor_nonbonded,
        safe_factor_1_4=safe_factor_1_4, catastrophic_factor=catastrophic_factor,
        include_hydrogens=include_hydrogens,
    )
    return BATGraph(base, torsions, codes, pairs, safe, catastrophic, topology, metadata)


def steric_barrier(coordinates: Tensor, graph: BATGraph, *, tau: float = 0.05) -> tuple[Tensor, dict[str, Tensor]]:
    coordinates = torch.as_tensor(coordinates)
    pairs = graph.nonbonded_pairs
    if not pairs.numel():
        zero = coordinates.sum() * 0.0
        return zero, {"distance": coordinates.new_empty(0), "penetration": coordinates.new_empty(0), "active": coordinates.new_empty(0, dtype=torch.bool)}
    distance = torch.linalg.vector_norm(coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]], dim=-1)
    safe = graph.safe_distances.to(device=coordinates.device, dtype=coordinates.dtype)
    penetration = safe - distance
    active = penetration > 0.0
    if not bool(active.any()):
        energy = coordinates.sum() * 0.0
    else:
        terms = float(tau) ** 2 * F.softplus(penetration / float(tau)).square()
        energy = terms.mean()
    return energy, {"distance": distance, "penetration": penetration, "active": active}


def steric_metrics(coordinates: Tensor, graph: BATGraph) -> dict[str, float | int]:
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64)
    if not graph.nonbonded_pairs.numel():
        return {"pair_count": 0, "violation_count": 0, "catastrophic_count": 0, "penetration_sum": 0.0, "penetration_max": 0.0}
    pairs = graph.nonbonded_pairs
    distance = torch.linalg.vector_norm(coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]], dim=-1)
    safe = graph.safe_distances.to(distance)
    catastrophic = graph.catastrophic_distances.to(distance)
    penetration = (safe - distance).clamp_min(0.0)
    return {
        "pair_count": int(distance.numel()), "violation_count": int((penetration > 0).sum()),
        "catastrophic_count": int((distance < catastrophic).sum()),
        "penetration_sum": float(penetration.sum()), "penetration_max": float(penetration.max()),
        "minimum_distance_ratio": float((distance / safe).min()),
    }


def _normalized_gradient(objective: Tensor, coordinates: Tensor) -> tuple[Tensor, float]:
    gradient, = torch.autograd.grad(objective, coordinates, retain_graph=True)
    gradient = remove_rigid_component(gradient, coordinates)
    rms = torch.sqrt(gradient.square().sum(-1).mean())
    if not bool(torch.isfinite(gradient).all()) or float(rms) <= 1e-14:
        return torch.zeros_like(gradient), 0.0
    return gradient / rms, float(rms)


def frozen_ba_update(
    source: Tensor, graph: BATGraph, ba_parameters: Mapping[str, Tensor], *,
    rms_budget: float, atom_cap: float,
) -> dict[str, Any]:
    proposal = direct_gradient_update(
        source, graph.base, ba_parameters, rms_budget=rms_budget, atom_cap=atom_cap, steps=1,
    )
    output, safety = safety_accept(source, proposal["coordinates"], graph.base)
    return {"coordinates": output, "safety": safety, "trace": proposal["trace"], "finite": proposal["finite"]}


def combined_gradient_update(
    source: Tensor, graph: BATGraph, ba_parameters: Mapping[str, Tensor], *,
    rms_budget: float, atom_cap: float, tau: float,
    torsion_prediction: Mapping[str, Tensor] | None = None,
    torsion_active_threshold: float | None = None,
    fallback_coordinates: Tensor | None = None,
    backtracking_fractions: Sequence[float] = (1.0, 0.5, 0.25, 0.125),
) -> dict[str, Any]:
    source = torch.as_tensor(source, dtype=torch.float64)
    coordinates = source.detach().clone().requires_grad_(True)
    ba_objective, _ = structured_objective(coordinates, graph.base, ba_parameters)
    gradients, group_rms = [], {}
    ba_gradient, ba_rms = _normalized_gradient(ba_objective, coordinates)
    if ba_rms > 0:
        gradients.append(ba_gradient); group_rms["ba"] = ba_rms
    steric_objective, steric_payload = steric_barrier(coordinates, graph, tau=tau)
    if bool(steric_payload["active"].any()):
        steric_gradient, steric_rms = _normalized_gradient(steric_objective, coordinates)
        if steric_rms > 0:
            gradients.append(steric_gradient); group_rms["steric"] = steric_rms
    active_torsion = coordinates.new_empty(0, dtype=torch.bool)
    source_torsion_nll = coordinates.new_empty(0)
    if torsion_prediction is not None and graph.torsions.numel():
        values = dihedral_angles(coordinates, graph.torsions)
        source_torsion_nll = torsion_nll(values, torsion_prediction)
        threshold = float(torsion_active_threshold)
        active_torsion = source_torsion_nll.detach() > threshold
        if bool(active_torsion.any()):
            torsion_objective = (source_torsion_nll[active_torsion] - threshold).mean()
            torsion_gradient, torsion_rms = _normalized_gradient(torsion_objective, coordinates)
            if torsion_rms > 0:
                gradients.append(torsion_gradient); group_rms["torsion"] = torsion_rms
    if not gradients:
        return {
            "coordinates": source, "fallback": False, "rejected": False, "no_op": True,
            "active_torsion_count": int(active_torsion.sum()), "active_steric_count": int(steric_payload["active"].sum()),
            "group_gradient_rms": group_rms, "backtracking_fraction": 0.0, "trust_clipped": False,
        }
    direction = -torch.stack(gradients).mean(0)
    direction_rms = torch.sqrt(direction.square().sum(-1).mean()).clamp_min(1e-14)
    candidate, trust = trust_project(
        source, direction / direction_rms * float(rms_budget),
        rms_budget=float(rms_budget), atom_cap=float(atom_cap),
    )
    before = steric_metrics(source, graph)
    accepted = None; accepted_safety = None; fraction_used = 0.0
    for fraction in backtracking_fractions:
        trial = source + float(fraction) * (candidate - source)
        after = steric_metrics(trial, graph)
        safe_candidate, safety = safety_accept(source, trial, graph.base)
        unchanged_by_guard = bool(torch.allclose(safe_candidate, trial, atol=1e-12, rtol=0.0))
        hard_ok = (
            after["catastrophic_count"] <= before["catastrophic_count"]
            and after["penetration_max"] <= before["penetration_max"] + 1e-10
        )
        if unchanged_by_guard and hard_ok:
            accepted, accepted_safety, fraction_used = trial.detach(), safety, float(fraction)
            break
    rejected = accepted is None
    if rejected:
        accepted = torch.as_tensor(fallback_coordinates if fallback_coordinates is not None else source, dtype=torch.float64)
        accepted_safety = safety_accept(source, accepted, graph.base)[1]
    delta = accepted - source
    return {
        "coordinates": accepted, "fallback": rejected, "rejected": rejected,
        "no_op": bool(torch.allclose(accepted, source, atol=1e-12, rtol=0.0)),
        "active_torsion_count": int(active_torsion.sum()),
        "active_steric_count": int(steric_payload["active"].sum()),
        "group_gradient_rms": group_rms, "backtracking_fraction": fraction_used,
        "trust_clipped": bool(trust["graph_scale"] < 1.0 or trust["atom_scale_min"] < 1.0),
        "graph_rms_movement": float(torch.sqrt(delta.square().sum(-1).mean())),
        "max_atom_movement": float(torch.linalg.vector_norm(delta, dim=-1).max()),
        "safety": accepted_safety,
        "steric_before": before, "steric_after": steric_metrics(accepted, graph),
        "source_torsion_nll": source_torsion_nll.detach(), "active_torsion_mask": active_torsion.detach(),
    }
