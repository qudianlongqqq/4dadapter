"""Fail-closed BAC proposal checks and deterministic backtracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from .acceptance import displacement_metrics


@dataclass(frozen=True)
class BACSafetyConfig:
    max_atom_displacement: float = 0.12
    max_molecule_rms_displacement: float = 0.06
    epsilon_bond: float = 0.0
    epsilon_angle: float = 0.0
    epsilon_clash: float = 0.0
    epsilon_ring: float = 0.0
    minimum_bac_gain: float = 1.0e-8
    backtracking_scales: tuple[float, ...] = (1.0, 0.5, 0.25)


def _bac_values(values: dict[str, float]) -> dict[str, float]:
    return {
        "bond": float(values["bond_outlier_rate"]),
        "angle": float(values["angle_outlier_rate"]),
        "clash": max(
            float(values["severe_clash_rate"]),
            float(values["clash_penetration"]),
        ),
        "ring": max(
            float(values["ring_bond_outlier_rate"]),
            float(values["ring_planarity_outlier_rate"]),
        ),
    }


def evaluate_bac_proposal(
    source: Tensor,
    proposal: Tensor,
    record: Any,
    validity: Any,
    config: BACSafetyConfig,
) -> dict[str, Any]:
    source = torch.as_tensor(source, dtype=torch.float32)
    proposal = torch.as_tensor(proposal, dtype=torch.float32)
    reasons: list[str] = []
    if source.shape != proposal.shape:
        reasons.append("identity_shape_changed")
        return {"accepted": False, "reasons": reasons}
    if not bool(torch.isfinite(proposal).all()):
        reasons.append("nonfinite")
        return {"accepted": False, "reasons": reasons}
    before = validity.evaluate(source, record, baseline_coordinates=source)
    after = validity.evaluate(proposal, record, baseline_coordinates=source)
    before_bac = _bac_values(before)
    after_bac = _bac_values(after)
    deltas = {name: after_bac[name] - before_bac[name] for name in before_bac}
    displacement = displacement_metrics(source, proposal)
    if displacement["max_atom_displacement"] > config.max_atom_displacement:
        reasons.append("atom_trust_radius")
    if (
        displacement["aligned_rms_displacement"]
        > config.max_molecule_rms_displacement
    ):
        reasons.append("molecule_trust_radius")
    if deltas["bond"] > config.epsilon_bond:
        reasons.append("new_bond_violation")
    if deltas["angle"] > config.epsilon_angle:
        reasons.append("new_angle_violation")
    if deltas["clash"] > config.epsilon_clash:
        reasons.append("new_clash")
    if deltas["ring"] > config.epsilon_ring:
        reasons.append("new_ring_violation")
    if float(after["chirality_preserved"]) < 1.0:
        reasons.append("chirality_changed")
    if (
        float(after["stereocenter_degenerate_rate"])
        > float(before["stereocenter_degenerate_rate"]) + 1.0e-12
    ):
        reasons.append("stereocenter_degenerated")
    gain = sum(before_bac[name] - after_bac[name] for name in ("bond", "angle", "clash"))
    if gain < config.minimum_bac_gain:
        reasons.append("no_bac_gain")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "before": before,
        "after": after,
        "bac_deltas": deltas,
        "bac_gain": float(gain),
        "displacement": displacement,
        "safety_config": asdict(config),
    }


def select_safe_bac_proposal(
    source: Tensor,
    delta: Tensor,
    record: Any,
    validity: Any,
    config: BACSafetyConfig,
) -> tuple[Tensor, dict[str, Any]]:
    source = torch.as_tensor(source, dtype=torch.float32)
    delta = torch.as_tensor(delta, dtype=source.dtype, device=source.device)
    attempts = []
    for scale in config.backtracking_scales:
        proposal = source + float(scale) * delta
        result = evaluate_bac_proposal(source, proposal, record, validity, config)
        attempts.append({"scale": float(scale), **result})
        if result["accepted"]:
            return proposal, {
                **result,
                "selected_scale": float(scale),
                "rolled_back": False,
                "attempts": attempts,
            }
    return source.clone(), {
        "accepted": False,
        "selected_scale": 0.0,
        "rolled_back": True,
        "reasons": attempts[-1]["reasons"] if attempts else ["no_attempt"],
        "attempts": attempts,
    }
