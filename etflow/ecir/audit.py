"""Shared, read-only audit metrics for ECIR/MCVR diagnostics."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from etflow.commons.kabsch_utils import kabsch_align, kabsch_rmsd

from .geometry import (
    angle_triplets,
    bond_angles,
    bond_lengths,
    circular_difference,
    dihedral_angles,
    geometry_error_vector,
    torsion_quads,
    unique_bonds,
)


def field(record: Any, name: str, default=None):
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flexibility_group(record: Any) -> str:
    count = int(field(record, "num_rotatable_bonds", 0))
    return "low" if count <= 2 else ("medium" if count <= 5 else "high")


def displacement_metrics(source: Tensor, candidate: Tensor) -> dict[str, float]:
    source = torch.as_tensor(source, dtype=torch.float32)
    candidate = torch.as_tensor(candidate, dtype=source.dtype)
    if source.shape != candidate.shape or source.ndim != 2 or source.size(1) != 3:
        raise ValueError("source and candidate must have matching [N, 3] shapes")
    if not bool(torch.isfinite(source).all() and torch.isfinite(candidate).all()):
        raise ValueError("displacement coordinates must be finite")
    aligned = kabsch_align(candidate, source)
    norms = torch.linalg.vector_norm(aligned - source, dim=-1)
    return {
        "aligned_rms_displacement": float((norms.square().mean()).sqrt()),
        "mean_atom_displacement": float(norms.mean()),
        "max_atom_displacement": float(norms.max()),
    }


def nearest_reference_rmsd(coordinates: Tensor, references: Tensor) -> float:
    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    references = torch.as_tensor(references, dtype=coordinates.dtype)
    if references.ndim == 2:
        references = references.unsqueeze(0)
    return float(torch.stack([kabsch_rmsd(coordinates, ref) for ref in references]).min())


def internal_metrics(coordinates: Tensor, target: Tensor, record: Any) -> dict[str, float]:
    values = geometry_error_vector(
        torch.as_tensor(coordinates, dtype=torch.float32),
        torch.as_tensor(target, dtype=torch.float32),
        record,
    )
    names = (
        "bond_violation",
        "angle_violation",
        "torsion_circular_error",
        "ring_invalidity",
        "clash_score",
        "chirality_error",
    )
    return {name: float(value) for name, value in zip(names, values)}


def torsion_change_metrics(source: Tensor, target: Tensor, record: Any) -> dict[str, float]:
    source = torch.as_tensor(source, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=source.dtype)
    edge_index = torch.as_tensor(field(record, "edge_index"), dtype=torch.long)
    rotatable = torch.as_tensor(
        field(record, "rotatable_bond_index", torch.empty((2, 0))), dtype=torch.long
    )
    quads = torsion_quads(edge_index, rotatable, source.size(0))
    if not quads.numel():
        return {"torsion_circular_change": 0.0, "max_rotatable_torsion_change": 0.0}
    delta = circular_difference(
        dihedral_angles(target, quads), dihedral_angles(source, quads)
    ).abs()
    return {
        "torsion_circular_change": float(delta.mean()),
        "max_rotatable_torsion_change": float(delta.max()),
    }


def validity_gains(source: Tensor, target: Tensor, record: Any) -> dict[str, float]:
    # The persisted ECIR target is also the metric reference. Its bond/angle/
    # torsion/ring errors are therefore zero; clash is an absolute penalty.
    before = internal_metrics(source, target, record)
    after = internal_metrics(target, target, record)
    return {
        "bond_validity_gain": before["bond_violation"] - after["bond_violation"],
        "angle_validity_gain": before["angle_violation"] - after["angle_violation"],
        "ring_validity_gain": before["ring_invalidity"] - after["ring_invalidity"],
        "clash_validity_gain": before["clash_score"] - after["clash_score"],
    }


def classify_relaxation(target_metadata: Mapping[str, Any]) -> tuple[str, str]:
    source = str(target_metadata.get("target_source", "unknown"))
    relaxation = dict(target_metadata.get("relaxation") or {})
    method = str(relaxation.get("method", "unsupported"))
    accepted = bool(relaxation.get("accepted", False))
    converged = bool(relaxation.get("optimization_success", False))
    reason = relaxation.get("rejection_reason")
    if source == "multi_reference_soft_coupling":
        return "fallback_to_soft_reference", str(reason or "relaxation_rejected")
    if method == "unsupported" or not bool(relaxation.get("supported", False)):
        return "MMFF_unsupported", str(reason or "force_field_unsupported")
    if method == "UFF":
        return "UFF_fallback", "converged" if converged else "iteration_limit"
    if not accepted:
        return "rejected", str(reason or "relaxation_rejected")
    if converged:
        return "converged", "converged"
    return "accepted_but_not_converged", "iteration_limit"


def safe_relative_delta(candidate: float, baseline: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return math.nan
    return (candidate - baseline) / max(abs(baseline), 1.0e-12)
