"""Label-free trust region, geometry backtracking, and rejection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SafeUpdateResult:
    coordinates: Tensor
    delta_raw: Tensor
    delta_safe: Tensor
    accepted_delta: Tensor
    accepted: bool
    reject_reason: str | None
    backtracking_count: int
    atom_clipped: bool
    graph_rms_clipped: bool
    internal_norm_clipped: bool


def trust_region_clip(
    delta: Tensor,
    atom_batch: Tensor,
    *,
    max_atom_displacement: float | None,
    max_graph_rms_displacement: float | None,
    max_internal_velocity_norm: float | None,
) -> tuple[Tensor, dict[str, bool]]:
    """Clip only by predicted update magnitude, never by a reference label."""

    if delta.ndim != 2 or delta.size(-1) != 3:
        raise ValueError("delta must have shape [N, 3]")
    atom_batch = torch.as_tensor(atom_batch, device=delta.device, dtype=torch.long)
    if atom_batch.shape != (delta.size(0),):
        raise ValueError("atom_batch must have shape [N]")
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("delta contains NaN or Inf")
    result = delta.clone()
    atom_clipped = graph_clipped = norm_clipped = False
    if max_internal_velocity_norm is not None:
        norm = torch.linalg.vector_norm(result)
        if norm > float(max_internal_velocity_norm):
            result = result * (float(max_internal_velocity_norm) / norm)
            norm_clipped = True
    if max_atom_displacement is not None:
        atom_norm = torch.linalg.vector_norm(result, dim=-1, keepdim=True)
        scale = (float(max_atom_displacement) / atom_norm.clamp_min(1.0e-12)).clamp_max(
            1.0
        )
        atom_clipped = bool((scale < 1.0).any())
        result = result * scale
    if max_graph_rms_displacement is not None:
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        squared = result.square().sum(-1)
        energy = result.new_zeros(graphs)
        energy.index_add_(0, atom_batch, squared)
        counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1)
        rms = (energy / counts.to(result.dtype)).sqrt()
        graph_scale = (
            float(max_graph_rms_displacement) / rms.clamp_min(1.0e-12)
        ).clamp_max(1.0)
        graph_clipped = bool((graph_scale < 1.0).any())
        result = result * graph_scale[atom_batch, None]
    return result, {
        "atom_clipped": atom_clipped,
        "graph_rms_clipped": graph_clipped,
        "internal_norm_clipped": norm_clipped,
    }


def geometry_guard(
    current: Tensor,
    candidate: Tensor,
    edge_index: Tensor,
    atom_batch: Tensor,
    *,
    max_step_displacement: float,
    min_bond_ratio: float = 0.5,
    max_bond_ratio: float = 1.5,
    min_nonbond_distance: float = 0.5,
) -> tuple[bool, str | None]:
    """Check finite coordinates, bonded distortion, and nonbond collisions."""

    if current.shape != candidate.shape or current.ndim != 2 or current.size(-1) != 3:
        return False, "shape_mismatch"
    if not bool(torch.isfinite(candidate).all()):
        return False, "nonfinite_coordinates"
    displacement = torch.linalg.vector_norm(candidate - current, dim=-1)
    if bool((displacement > float(max_step_displacement) + 1.0e-8).any()):
        return False, "step_displacement"
    edge_index = torch.as_tensor(edge_index, device=current.device, dtype=torch.long)
    atom_batch = torch.as_tensor(atom_batch, device=current.device, dtype=torch.long)
    if edge_index.numel():
        src, dst = edge_index
        unique = src < dst
        src, dst = src[unique], dst[unique]
        before = torch.linalg.vector_norm(current[src] - current[dst], dim=-1)
        after = torch.linalg.vector_norm(candidate[src] - candidate[dst], dim=-1)
        ratio = after / before.clamp_min(1.0e-8)
        if bool((ratio < float(min_bond_ratio)).any()):
            return False, "bond_collapse"
        if bool((ratio > float(max_bond_ratio)).any()):
            return False, "bond_stretch"
    graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
    for graph in range(graphs):
        atoms = torch.nonzero(atom_batch == graph, as_tuple=False).reshape(-1)
        if atoms.numel() < 2:
            continue
        distances = torch.cdist(candidate[atoms], candidate[atoms])
        excluded = torch.eye(atoms.numel(), dtype=torch.bool, device=current.device)
        if edge_index.numel():
            local = {int(atom): index for index, atom in enumerate(atoms.tolist())}
            for source, target in edge_index.t().tolist():
                if source in local and target in local:
                    excluded[local[source], local[target]] = True
        collision = (distances < float(min_nonbond_distance)) & ~excluded
        if bool(collision.any()):
            return False, "nonbond_collision"
    return True, None


def safe_serial_update(
    current: Tensor,
    delta_raw: Tensor,
    edge_index: Tensor,
    atom_batch: Tensor,
    *,
    gate: Tensor | None = None,
    gate_accept_threshold: float = 0.0,
    max_atom_displacement: float | None = 0.1,
    max_graph_rms_displacement: float | None = 0.05,
    max_internal_velocity_norm: float | None = None,
    max_backtracks: int = 4,
    backtrack_factor: float = 0.5,
    min_bond_ratio: float = 0.5,
    max_bond_ratio: float = 1.5,
    min_nonbond_distance: float = 0.5,
) -> SafeUpdateResult:
    if max_backtracks < 0 or not 0.0 < float(backtrack_factor) < 1.0:
        raise ValueError("invalid backtracking configuration")
    atom_batch = torch.as_tensor(atom_batch, device=current.device, dtype=torch.long)
    if gate is not None and gate_accept_threshold > 0:
        gate = torch.as_tensor(gate, device=current.device).reshape(-1)
        delta_raw = delta_raw.clone()
        delta_raw[gate[atom_batch] < float(gate_accept_threshold)] = 0
    delta_safe, clip = trust_region_clip(
        delta_raw,
        atom_batch,
        max_atom_displacement=max_atom_displacement,
        max_graph_rms_displacement=max_graph_rms_displacement,
        max_internal_velocity_norm=max_internal_velocity_norm,
    )
    trial = delta_safe
    reason = None
    for backtracks in range(max_backtracks + 1):
        candidate = current + trial
        accepted, reason = geometry_guard(
            current,
            candidate,
            edge_index,
            atom_batch,
            max_step_displacement=(
                float(max_atom_displacement)
                if max_atom_displacement is not None
                else float("inf")
            ),
            min_bond_ratio=min_bond_ratio,
            max_bond_ratio=max_bond_ratio,
            min_nonbond_distance=min_nonbond_distance,
        )
        if accepted:
            return SafeUpdateResult(
                candidate,
                delta_raw,
                delta_safe,
                trial,
                True,
                None,
                backtracks,
                clip["atom_clipped"],
                clip["graph_rms_clipped"],
                clip["internal_norm_clipped"],
            )
        trial = trial * float(backtrack_factor)
    return SafeUpdateResult(
        current.clone(),
        delta_raw,
        delta_safe,
        torch.zeros_like(delta_safe),
        False,
        reason,
        max_backtracks,
        clip["atom_clipped"],
        clip["graph_rms_clipped"],
        clip["internal_norm_clipped"],
    )
