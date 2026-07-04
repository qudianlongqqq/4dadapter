from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import torch


def _fragment_type_lookup(
    fragment_types: Mapping[int, str] | Sequence[str],
    fragment_id: int,
) -> str:
    if isinstance(fragment_types, Mapping):
        return fragment_types[int(fragment_id)]
    return fragment_types[int(fragment_id)]


def _cross_matrix_for_omega_cross_r(r: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(r[:, 0])
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    return torch.stack(
        [
            torch.stack([zeros, rz, -ry], dim=1),
            torch.stack([-rz, zeros, rx], dim=1),
            torch.stack([ry, -rx, zeros], dim=1),
        ],
        dim=1,
    ).reshape(-1, 3)


def _fit_angular_velocity(r: torch.Tensor, v_rel: torch.Tensor) -> tuple[torch.Tensor, int]:
    if r.size(0) < 2:
        return torch.zeros(3, dtype=r.dtype, device=r.device), 0

    a = _cross_matrix_for_omega_cross_r(r)
    b = v_rel.reshape(-1)
    rank = int(torch.linalg.matrix_rank(a).item())

    try:
        omega = torch.linalg.lstsq(a, b).solution
    except RuntimeError:
        omega = torch.linalg.pinv(a) @ b

    return omega, rank


def decompose_velocity_by_fragment(
    x: torch.Tensor,
    v: torch.Tensor,
    fragment_ids: torch.Tensor,
    fragment_types: Mapping[int, str] | Sequence[str],
    eps: float = 1.0e-8,
) -> Dict[str, object]:
    """Fit translation + angular velocity per fragment and compute residuals."""

    if x.ndim != 2 or x.size(-1) != 3:
        raise ValueError(f"x must have shape [num_atoms, 3], got {tuple(x.shape)}")
    if v.shape != x.shape:
        raise ValueError(f"v must have the same shape as x, got {tuple(v.shape)}")
    if fragment_ids.ndim != 1 or fragment_ids.numel() != x.size(0):
        raise ValueError(
            "fragment_ids must have shape [num_atoms], "
            f"got {tuple(fragment_ids.shape)} for {x.size(0)} atoms"
        )

    original_dtype = x.dtype
    x_fit = x.detach().to(dtype=torch.float64)
    v_fit = v.detach().to(dtype=torch.float64)
    fragment_ids = fragment_ids.to(device=x.device, dtype=torch.long)

    rigid_velocity = torch.zeros_like(v_fit)
    residual_velocity = torch.zeros_like(v_fit)
    rows: List[Dict[str, object]] = []

    for fragment_id in sorted(torch.unique(fragment_ids).tolist()):
        mask = fragment_ids == int(fragment_id)
        x_frag = x_fit[mask]
        v_frag = v_fit[mask]
        num_atoms = int(mask.sum().item())

        center = x_frag.mean(dim=0)
        translation = v_frag.mean(dim=0)
        r = x_frag - center
        v_rel = v_frag - translation

        omega, rank = _fit_angular_velocity(r, v_rel)
        rotation_velocity = torch.cross(
            omega.expand_as(r),
            r,
            dim=1,
        )
        rigid_frag = translation.expand_as(v_frag) + rotation_velocity
        residual_frag = v_frag - rigid_frag

        rigid_velocity[mask] = rigid_frag
        residual_velocity[mask] = residual_frag

        velocity_norm = torch.linalg.norm(v_frag)
        translation_norm = torch.linalg.norm(translation.expand_as(v_frag))
        rotation_norm = torch.linalg.norm(rotation_velocity)
        residual_norm = torch.linalg.norm(residual_frag)
        residual_ratio = residual_norm / (velocity_norm + eps)
        rigid_explain_ratio = 1.0 - residual_norm.pow(2) / (velocity_norm.pow(2) + eps)

        status = "ok"
        if num_atoms < 2:
            status = "too_small"
        elif rank < 3:
            status = "rank_deficient"

        rows.append(
            {
                "fragment_id": int(fragment_id),
                "fragment_type": _fragment_type_lookup(fragment_types, int(fragment_id)),
                "num_fragment_atoms": num_atoms,
                "center": center.to(dtype=original_dtype),
                "translation_velocity": translation.to(dtype=original_dtype),
                "omega": omega.to(dtype=original_dtype),
                "velocity_norm": float(velocity_norm.item()),
                "translation_norm": float(translation_norm.item()),
                "rotation_norm": float(rotation_norm.item()),
                "residual_norm": float(residual_norm.item()),
                "residual_ratio": float(residual_ratio.item()),
                "rigid_explain_ratio": float(rigid_explain_ratio.item()),
                "omega_norm": float(torch.linalg.norm(omega).item()),
                "fit_rank": rank,
                "fit_status": status,
            }
        )

    return {
        "fragments": rows,
        "rigid_velocity": rigid_velocity.to(dtype=original_dtype),
        "residual_velocity": residual_velocity.to(dtype=original_dtype),
    }


def iter_atom_decomposition_rows(
    velocity_source: str,
    x: torch.Tensor,
    v: torch.Tensor,
    rigid_velocity: torch.Tensor,
    residual_velocity: torch.Tensor,
    fragment_ids: torch.Tensor,
    fragment_types: Mapping[int, str] | Sequence[str],
    eps: float = 1.0e-8,
) -> Iterable[Dict[str, object]]:
    for atom_idx in range(x.size(0)):
        fragment_id = int(fragment_ids[atom_idx].item())
        velocity_norm = torch.linalg.norm(v[atom_idx]).item()
        rigid_norm = torch.linalg.norm(rigid_velocity[atom_idx]).item()
        residual_norm = torch.linalg.norm(residual_velocity[atom_idx]).item()
        yield {
            "atom_index": atom_idx,
            "fragment_id": fragment_id,
            "fragment_type": _fragment_type_lookup(fragment_types, fragment_id),
            "velocity_source": velocity_source,
            "velocity_norm": float(velocity_norm),
            "rigid_velocity_norm": float(rigid_norm),
            "residual_norm": float(residual_norm),
            "residual_ratio": float(residual_norm / (velocity_norm + eps)),
        }
