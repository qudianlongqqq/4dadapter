"""Basin-preserving real-error targets for ECIR-Flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import networkx as nx
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from torch import Tensor

from etflow.commons.kabsch_utils import kabsch_align, kabsch_rmsd

from .geometry import (
    bond_angles,
    bond_lengths,
    circular_difference,
    dihedral_angles,
    torsion_quads,
    unique_bonds,
    angle_triplets,
)


def _field(record: Any, name: str, default=None):
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


@dataclass
class RelaxationResult:
    coordinates: Tensor | None
    method: str
    supported: bool
    optimization_success: bool
    accepted: bool
    status_code: int | None
    steps: int
    energy_before: float | None
    energy_after: float | None
    energy_drop: float | None
    energy_per_heavy_atom_before: float | None
    energy_per_heavy_atom_after: float | None
    relaxation_rmsd: float | None
    rejection_reason: str | None

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("coordinates", None)
        return value


def _record_to_rdkit_mapping(record: Any) -> tuple[Chem.Mol, dict[int, int]]:
    smiles = str(_field(record, "smiles", ""))
    if not smiles:
        raise ValueError("SMILES is required for force-field targets")
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError("RDKit could not parse SMILES")
    mol = Chem.AddHs(base)
    atomic_numbers = torch.as_tensor(_field(record, "atomic_numbers"), dtype=torch.long)
    edge_index = torch.as_tensor(_field(record, "edge_index"), dtype=torch.long)
    if mol.GetNumAtoms() != atomic_numbers.numel():
        raise ValueError("RDKit/cache atom counts differ")

    cache_graph = nx.Graph()
    for index, atomic_number in enumerate(atomic_numbers.tolist()):
        cache_graph.add_node(index, z=int(atomic_number))
    for left, right in unique_bonds(edge_index).t().tolist():
        cache_graph.add_edge(int(left), int(right))
    rdkit_graph = nx.Graph()
    for atom in mol.GetAtoms():
        rdkit_graph.add_node(atom.GetIdx(), z=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        rdkit_graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        cache_graph,
        rdkit_graph,
        node_match=lambda left, right: left["z"] == right["z"],
    )
    try:
        mapping = next(matcher.isomorphisms_iter())
    except StopIteration as exc:
        raise ValueError("RDKit/cache molecular graphs are not isomorphic") from exc
    return mol, {int(cache): int(rdkit) for cache, rdkit in mapping.items()}


def _set_conformer(mol: Chem.Mol, mapping: Mapping[int, int], coordinates: Tensor) -> int:
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for cache_index, rdkit_index in mapping.items():
        x, y, z = coordinates[cache_index].detach().cpu().double().tolist()
        conformer.SetAtomPosition(rdkit_index, Point3D(x, y, z))
    mol.RemoveAllConformers()
    return int(mol.AddConformer(conformer, assignId=True))


def _coordinates_from_conformer(
    mol: Chem.Mol, mapping: Mapping[int, int], conformer_id: int, template: Tensor
) -> Tensor:
    conformer = mol.GetConformer(conformer_id)
    result = torch.empty_like(template)
    for cache_index, rdkit_index in mapping.items():
        point = conformer.GetAtomPosition(rdkit_index)
        result[cache_index] = template.new_tensor((point.x, point.y, point.z))
    return result


def restrained_force_field_relaxation(
    record: Any,
    coordinates: Tensor,
    *,
    max_steps: int = 50,
    max_position_displacement: float = 0.25,
    force_constant: float = 100.0,
    max_relaxation_rmsd: float = 0.35,
    allow_uff_fallback: bool = True,
) -> RelaxationResult:
    """Run anchored MMFF94s, with explicitly separated UFF fallback."""

    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    try:
        mol, mapping = _record_to_rdkit_mapping(record)
        conformer_id = _set_conformer(mol, mapping, coordinates)
    except Exception as exc:
        return RelaxationResult(
            None, "unsupported", False, False, False, None, 0,
            None, None, None, None, None, None, f"molecule_build:{exc}"
        )
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    method = "MMFF94s"
    force_field = None
    if properties is not None:
        force_field = AllChem.MMFFGetMoleculeForceField(
            mol, properties, confId=conformer_id
        )
    elif allow_uff_fallback and AllChem.UFFHasAllMoleculeParams(mol):
        method = "UFF"
        force_field = AllChem.UFFGetMoleculeForceField(mol, confId=conformer_id)
    if force_field is None:
        return RelaxationResult(
            None, "unsupported", False, False, False, None, 0,
            None, None, None, None, None, None, "force_field_unsupported"
        )
    for atom in range(mol.GetNumAtoms()):
        if method == "MMFF94s":
            force_field.MMFFAddPositionConstraint(
                atom, float(max_position_displacement), float(force_constant)
            )
        else:
            force_field.UFFAddPositionConstraint(
                atom, float(max_position_displacement), float(force_constant)
            )
    force_field.Initialize()
    heavy_atoms = max(1, sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()))
    try:
        energy_before = float(force_field.CalcEnergy())
        status = int(force_field.Minimize(maxIts=int(max_steps)))
        energy_after = float(force_field.CalcEnergy())
        relaxed = _coordinates_from_conformer(mol, mapping, conformer_id, coordinates)
        drift = float(kabsch_rmsd(relaxed, coordinates))
    except Exception as exc:
        return RelaxationResult(
            None, method, True, False, False, None, int(max_steps),
            None, None, None, None, None, None, f"optimization:{exc}"
        )
    finite = bool(torch.isfinite(relaxed).all())
    energy_nonincreasing = energy_after <= energy_before + 1.0e-5
    accepted = finite and drift <= float(max_relaxation_rmsd) and energy_nonincreasing
    reason = None
    if not finite:
        reason = "nonfinite_coordinates"
    elif drift > float(max_relaxation_rmsd):
        reason = "basin_drift"
    elif not energy_nonincreasing:
        reason = "energy_increase"
    return RelaxationResult(
        relaxed if accepted else None,
        method,
        True,
        status == 0,
        accepted,
        status,
        int(max_steps),
        energy_before,
        energy_after,
        energy_before - energy_after,
        energy_before / heavy_atoms,
        energy_after / heavy_atoms,
        drift,
        reason,
    )


def multi_reference_soft_coupling(
    x_input: Tensor,
    references: Tensor,
    record: Any,
    *,
    alpha: float = 1.0,
    beta: float = 0.25,
    gamma: float = 0.25,
    temperature: float = 0.25,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Sample one aligned reference from a soft distribution; never average coordinates."""

    x_input = torch.as_tensor(x_input, dtype=torch.float32)
    references = torch.as_tensor(references, dtype=x_input.dtype)
    if references.ndim == 2:
        references = references.unsqueeze(0)
    if references.ndim != 3 or references.shape[1:] != x_input.shape:
        raise ValueError("reference conformers must be [C, N, 3]")
    if references.size(0) < 1:
        raise ValueError("at least one reference conformer is required")
    edge_index = torch.as_tensor(_field(record, "edge_index"), dtype=torch.long)
    rotatable = torch.as_tensor(
        _field(record, "rotatable_bond_index", torch.empty(2, 0)), dtype=torch.long
    )
    bonds = unique_bonds(edge_index)
    angles = angle_triplets(edge_index, x_input.size(0))
    torsions = torsion_quads(edge_index, rotatable, x_input.size(0))
    aligned = torch.stack([kabsch_align(reference, x_input) for reference in references])
    rmsd = ((aligned - x_input).square().sum(-1).mean(-1)).sqrt()

    input_torsion = dihedral_angles(x_input, torsions)
    input_bond = bond_lengths(x_input, bonds)
    input_angle = bond_angles(x_input, angles)
    torsion_cost = []
    internal_cost = []
    for reference in aligned:
        torsion_cost.append(
            circular_difference(dihedral_angles(reference, torsions), input_torsion).abs().mean()
            if torsions.numel() else x_input.new_zeros(())
        )
        terms = []
        if bonds.numel():
            terms.append((bond_lengths(reference, bonds) - input_bond).abs().mean())
        if angles.numel():
            terms.append((bond_angles(reference, angles) - input_angle).abs().mean())
        internal_cost.append(torch.stack(terms).mean() if terms else x_input.new_zeros(()))
    torsion_cost = torch.stack(torsion_cost)
    internal_cost = torch.stack(internal_cost)
    costs = float(alpha) * rmsd + float(beta) * torsion_cost + float(gamma) * internal_cost
    probabilities = torch.softmax(-costs / float(temperature), dim=0)
    selected = int(torch.multinomial(probabilities, 1, generator=generator).item())
    order = torch.argsort(costs)
    second = int(order[1]) if order.numel() > 1 else int(order[0])
    return {
        "target": aligned[selected],
        "selected_reference_index": selected,
        "selected_reference_probability": float(probabilities[selected]),
        "nearest_reference_index": int(order[0]),
        "nearest_reference_cost": float(costs[order[0]]),
        "second_nearest_reference_cost": float(costs[second]),
        "reference_costs": costs,
        "reference_probabilities": probabilities,
        "single_reference": references.size(0) == 1,
        "cartesian_average_used": False,
    }


def build_real_error_target(
    record: Any,
    *,
    coordinates: Tensor | None = None,
    generator: torch.Generator | None = None,
    relaxation_kwargs: Mapping[str, Any] | None = None,
    coupling_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    x_input = torch.as_tensor(
        coordinates if coordinates is not None else _field(record, "x_init"),
        dtype=torch.float32,
    )
    relaxation = restrained_force_field_relaxation(
        record, x_input, **dict(relaxation_kwargs or {})
    )
    if relaxation.accepted and relaxation.coordinates is not None:
        return {
            "x_target": relaxation.coordinates,
            "target_source": "restrained_relaxation",
            "relaxation": relaxation.metadata(),
            "soft_coupling": None,
        }
    references = _field(record, "x_ref_candidates", _field(record, "x_ref_aligned"))
    if references is None:
        raise ValueError(
            f"Relaxation target rejected ({relaxation.rejection_reason}) and no references exist"
        )
    coupling = multi_reference_soft_coupling(
        x_input,
        torch.as_tensor(references),
        record,
        generator=generator,
        **dict(coupling_kwargs or {}),
    )
    target = coupling.pop("target")
    serializable = {
        key: value
        for key, value in coupling.items()
        if key not in {"reference_costs", "reference_probabilities"}
    }
    serializable["reference_costs"] = coupling["reference_costs"].tolist()
    serializable["reference_probabilities"] = coupling["reference_probabilities"].tolist()
    return {
        "x_target": target,
        "target_source": "multi_reference_soft_coupling",
        "relaxation": relaxation.metadata(),
        "soft_coupling": serializable,
    }
