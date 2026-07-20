"""Per-graph, per-type normalization for MCVR V8 constraints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class FrozenResidualScales:
    """Train-only robust residual scales used by every V8 batch."""

    bond: float
    angle: float
    clash: float = 1.0
    ring: float = 1.0
    chirality: float = 1.0
    identity_sha256: str = ""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FrozenResidualScales":
        payload = values.get("scales", values)
        result = cls(
            bond=float(payload["bond"]),
            angle=float(payload["angle"]),
            clash=float(payload.get("clash", 1.0)),
            ring=float(payload.get("ring", 1.0)),
            chirality=float(payload.get("chirality", 1.0)),
            identity_sha256=str(values.get("identity_sha256", "")),
        )
        result.validate()
        return result

    @classmethod
    def load(
        cls, path: str | Path, *, expected_sha256: str | None = None
    ) -> "FrozenResidualScales":
        source = Path(path)
        raw = source.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and actual != str(expected_sha256):
            raise RuntimeError("V8 train-scale file SHA256 changed")
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("split") != "train":
            raise RuntimeError("V8 residual scales must be estimated from train only")
        if any(
            bool(payload.get(key, False))
            for key in ("validation_used", "test_used", "frozen_holdout_used")
        ):
            raise RuntimeError("V8 residual-scale payload violates data isolation")
        result = cls.from_mapping(payload)
        if result.identity_sha256 and result.identity_sha256 != actual:
            stable = {key: value for key, value in payload.items() if key != "identity_sha256"}
            canonical = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
            if hashlib.sha256(canonical).hexdigest() != result.identity_sha256:
                raise RuntimeError("V8 residual-scale canonical identity changed")
        return result

    def validate(self) -> None:
        values = {
            "bond": self.bond,
            "angle": self.angle,
            "clash": self.clash,
            "ring": self.ring,
            "chirality": self.chirality,
        }
        if any(
            not torch.isfinite(torch.tensor(value)) or value <= 0.0 for value in values.values()
        ):
            raise ValueError(f"all frozen residual scales must be finite and positive: {values}")


def normalize_constraint_type(
    residual: Tensor,
    jacobian: Tensor,
    active_weight: Tensor,
    *,
    scale: float,
    normalize_by_active_count: bool,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Apply the same soft activity and type scale to residual and Jacobian."""

    if residual.ndim != 1 or jacobian.ndim != 2 or jacobian.size(0) != residual.numel():
        raise ValueError("constraint residual/Jacobian shapes differ")
    active_weight = torch.as_tensor(
        active_weight, device=residual.device, dtype=residual.dtype
    ).reshape(-1)
    if active_weight.numel() != residual.numel():
        raise ValueError("constraint activity count differs from residual count")
    count = active_weight.sum()
    denominator = residual.new_tensor(float(scale))
    if normalize_by_active_count:
        denominator = denominator * torch.sqrt(count + float(epsilon))
    weighted = active_weight / denominator.clamp_min(float(epsilon))
    normalized_residual = weighted * residual
    normalized_jacobian = weighted[:, None] * jacobian
    diagnostics = {
        "soft_active_count": count,
        "raw_residual_rms": torch.sqrt(residual.square().mean())
        if residual.numel()
        else residual.new_zeros(()),
        "normalized_residual_rms": torch.sqrt(normalized_residual.square().mean())
        if residual.numel()
        else residual.new_zeros(()),
        "raw_jacobian_norm": torch.linalg.vector_norm(jacobian),
        "normalized_jacobian_norm": torch.linalg.vector_norm(normalized_jacobian),
    }
    return normalized_residual, normalized_jacobian, diagnostics
