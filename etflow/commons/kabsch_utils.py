"""Torch-native Kabsch alignment helpers for conformer refinement datasets."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _validate_conformer(x: Tensor, name: str) -> Tensor:
    x = torch.as_tensor(x)
    if not x.is_floating_point():
        x = x.to(dtype=torch.get_default_dtype())
    if x.ndim != 2 or x.size(-1) != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {tuple(x.shape)}.")
    if x.size(0) < 1:
        raise ValueError(f"{name} must contain at least one atom.")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return x


def kabsch_align(mobile: Tensor, target: Tensor) -> Tensor:
    """Rigidly align ``mobile`` onto ``target`` without changing atom order.

    Reflections are explicitly rejected, so the returned transformation is a
    proper SO(3) rotation plus translation.
    """

    mobile = _validate_conformer(mobile, "mobile")
    target = _validate_conformer(target, "target").to(
        device=mobile.device, dtype=mobile.dtype
    )
    if mobile.shape != target.shape:
        raise ValueError(
            f"mobile and target shapes differ: {tuple(mobile.shape)} vs "
            f"{tuple(target.shape)}."
        )

    mobile_center = mobile.mean(dim=0, keepdim=True)
    target_center = target.mean(dim=0, keepdim=True)
    mobile_zero = mobile - mobile_center
    target_zero = target - target_center
    covariance = mobile_zero.transpose(0, 1) @ target_zero
    u, _, vh = torch.linalg.svd(covariance)
    rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
    if torch.linalg.det(rotation) < 0:
        vh = vh.clone()
        vh[-1] *= -1
        rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
    return mobile_zero @ rotation.transpose(0, 1) + target_center


def kabsch_rmsd(mobile: Tensor, target: Tensor) -> Tensor:
    """Return atom-wise RMSD after aligning ``mobile`` onto ``target``."""

    aligned = kabsch_align(mobile, target)
    target = torch.as_tensor(target, device=aligned.device, dtype=aligned.dtype)
    return torch.sqrt((aligned - target).square().sum(dim=-1).mean())


def select_best_reference_conformer(
    x_init: Tensor, reference_conformers: Tensor
) -> Tuple[Tensor, Tensor, int, Tensor]:
    """Select and align the reference conformer closest to ``x_init``.

    Returns ``(x_ref, x_ref_aligned, index, aligned_rmsds)``.
    """

    x_init = _validate_conformer(x_init, "x_init")
    references = torch.as_tensor(
        reference_conformers, device=x_init.device, dtype=x_init.dtype
    )
    if references.ndim == 2:
        references = references.unsqueeze(0)
    if references.ndim != 3 or references.shape[1:] != x_init.shape:
        raise ValueError(
            "reference_conformers must have shape [C, N, 3] matching x_init, "
            f"got {tuple(references.shape)} and {tuple(x_init.shape)}."
        )
    if references.size(0) == 0:
        raise ValueError("reference_conformers is empty.")
    aligned = torch.stack([kabsch_align(ref, x_init) for ref in references])
    rmsds = torch.sqrt(
        (aligned - x_init.unsqueeze(0)).square().sum(dim=-1).mean(dim=-1)
    )
    best_index = int(torch.argmin(rmsds).item())
    return references[best_index], aligned[best_index], best_index, rmsds


def kabsch_sanity_check(
    x_init: Tensor, x_ref: Tensor, *, atol: float = 1.0e-5
) -> dict[str, float | bool]:
    """Return basic centering and non-increasing-RMSD diagnostics."""

    x_init = _validate_conformer(x_init, "x_init")
    x_ref = _validate_conformer(x_ref, "x_ref").to(x_init)
    aligned = kabsch_align(x_ref, x_init)
    before = torch.sqrt((x_ref - x_init).square().sum(dim=-1).mean())
    after = torch.sqrt((aligned - x_init).square().sum(dim=-1).mean())
    center_error = torch.linalg.norm(aligned.mean(0) - x_init.mean(0))
    return {
        "rmsd_before": float(before),
        "rmsd_after": float(after),
        "center_error": float(center_error),
        "rmsd_non_increasing": bool(after <= before + atol),
        "center_aligned": bool(center_error <= atol),
    }
