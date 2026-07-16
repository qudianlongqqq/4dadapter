"""Minimal-displacement targets that repair only thresholded validity excess.

This module deliberately has no reference-coordinate or force-field fallback.
If optimization cannot find a safe, improving candidate, the exact input is
returned as an explicitly labelled identity target.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from etflow.commons.kabsch_utils import kabsch_align

from .audit import displacement_metrics, field, torsion_change_metrics
from .geometry import (
    bond_angles,
    bond_lengths,
    circular_difference,
    clash_score,
    dihedral_angles,
)


def thresholded_excess(value: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Distance outside ``[lower, upper]``; exactly zero inside the envelope."""

    value = torch.as_tensor(value)
    lower = torch.as_tensor(lower, device=value.device, dtype=value.dtype)
    upper = torch.as_tensor(upper, device=value.device, dtype=value.dtype)
    return torch.maximum(lower - value, value - upper).clamp_min(0.0)


def periodic_delta(current: Tensor, baseline: Tensor) -> Tensor:
    """Signed periodic difference using atan2(sin(delta), cos(delta))."""

    return circular_difference(torch.as_tensor(current), torch.as_tensor(baseline))


def _tensor_sha256(value: Tensor) -> str:
    array = torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _ring_planarity_tensor(coordinates: Tensor, indices: tuple[int, ...]) -> Tensor:
    if len(indices) < 4:
        return coordinates.new_zeros(())
    points = coordinates[list(indices)]
    centered = points - points.mean(0, keepdim=True)
    return torch.linalg.svdvals(centered)[-1] / math.sqrt(len(indices))


def _chirality_volumes(coordinates: Tensor, centers) -> Tensor:
    values = []
    for center, first, second, third in centers:
        values.append(torch.linalg.det(torch.stack([
            coordinates[first] - coordinates[center],
            coordinates[second] - coordinates[center],
            coordinates[third] - coordinates[center],
        ])))
    return torch.stack(values) if values else coordinates.new_empty(0)


@dataclass(frozen=True)
class MinimalValidityConfig:
    optimizer: str = "Adam"
    max_steps: int = 40
    learning_rate: float = 0.001
    early_stop_patience: int = 5
    min_improvement: float = 1.0e-5
    lambda_anchor: float = 2.0
    lambda_bond: float = 1.0
    lambda_angle: float = 0.5
    lambda_ring: float = 1.0
    lambda_clash: float = 2.0
    lambda_chiral: float = 2.0
    lambda_torsion_anchor: float = 0.05
    high_flex_torsion_anchor_scale: float = 2.0
    lambda_score_displacement: float = 0.25
    lambda_score_torsion: float = 0.05
    lambda_score_risk: float = 2.0
    max_molecule_rms_displacement: float = 0.15
    max_atom_displacement: float = 0.35
    max_torsion_change_rad: float = 0.70
    max_high_flex_torsion_change_rad: float = 0.35

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "MinimalValidityConfig":
        if config is None:
            return cls()
        unknown = set(config) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown minimal-target settings: {sorted(unknown)}")
        return cls(**dict(config))


class MinimalValidityTargetBuilder:
    """Small-step Cartesian validity repair with trajectory-safe selection."""

    def __init__(self, validity, config: Mapping[str, Any] | None = None) -> None:
        self.validity = validity
        self.config = MinimalValidityConfig.from_mapping(config)
        if self.config.optimizer != "Adam":
            raise ValueError("the Stage C implementation currently supports optimizer=Adam")
        if self.config.max_steps < 1 or self.config.early_stop_patience < 1:
            raise ValueError("max_steps and early_stop_patience must be positive")
        if self.config.max_molecule_rms_displacement <= 0 or self.config.max_atom_displacement <= 0:
            raise ValueError("trust limits must be positive")

    def _penalties(self, coordinates: Tensor, x_input: Tensor, record: Any) -> dict[str, Tensor]:
        prepared = self.validity._prepare(record)  # one frozen, train-derived environment map
        bonds = prepared["bonds"].to(coordinates.device)
        angles = prepared["angles"].to(coordinates.device)
        torsions = prepared["torsions"].to(coordinates.device)
        bond_stats = prepared["bond_stats"].to(coordinates)
        angle_stats = prepared["angle_stats"].to(coordinates)

        lengths = bond_lengths(coordinates, bonds)
        bond_excess = thresholded_excess(lengths, bond_stats[:, 0], bond_stats[:, 1])
        bond_scale = bond_stats[:, 2].clamp_min(1.0e-6)
        bond = (bond_excess / bond_scale).square().mean() if lengths.numel() else coordinates.new_zeros(())

        values = bond_angles(coordinates, angles)
        angle_excess = thresholded_excess(values, angle_stats[:, 0], angle_stats[:, 1])
        angle_scale = angle_stats[:, 2].clamp_min(1.0e-6)
        angle = (angle_excess / angle_scale).square().mean() if values.numel() else coordinates.new_zeros(())

        ring_terms = []
        ring_mask = prepared["ring_mask"].to(coordinates.device)
        if bool(ring_mask.any()):
            ring_terms.append((bond_excess[ring_mask] / bond_scale[ring_mask]).square().mean())
        for ring, stat in zip(prepared["rings"], prepared["planarity_stats"]):
            planarity = _ring_planarity_tensor(coordinates, tuple(ring))
            excess = thresholded_excess(
                planarity,
                coordinates.new_tensor(float(stat["lower"])),
                coordinates.new_tensor(float(stat["upper"])),
            )
            ring_terms.append((excess / max(float(stat["robust_scale"]), 1.0e-6)).square())
        ring = torch.stack(ring_terms).mean() if ring_terms else coordinates.new_zeros(())

        edge_index = prepared["edge_index"].to(coordinates.device)
        clash = clash_score(
            coordinates, edge_index, float(self.validity.config["clash_distance_angstrom"])
        ).square()

        centers = prepared["centers"]
        input_volumes = _chirality_volumes(x_input, centers)
        current_volumes = _chirality_volumes(coordinates, centers)
        if input_volumes.numel():
            direction = torch.sign(input_volumes).detach()
            scale = input_volumes.abs().detach().clamp_min(1.0e-4)
            # A positive signed fraction preserves both sign and non-degeneracy.
            chiral = torch.relu(0.05 - direction * current_volumes / scale).square().mean()
        else:
            chiral = coordinates.new_zeros(())

        aligned = kabsch_align(coordinates, x_input)
        anchor = (aligned - x_input).square().sum(-1).mean()
        if torsions.numel():
            torsion = periodic_delta(
                dihedral_angles(coordinates, torsions), dihedral_angles(x_input, torsions)
            ).square().mean()
        else:
            torsion = coordinates.new_zeros(())
        return {
            "anchor": anchor,
            "bond": bond,
            "angle": angle,
            "ring": ring,
            "clash": clash,
            "chirality": chiral,
            "torsion_anchor": torsion,
        }

    def _objective(self, penalties: Mapping[str, Tensor], record: Any) -> Tensor:
        c = self.config
        high_flex = int(field(record, "num_rotatable_bonds", 0)) >= 6
        torsion_weight = c.lambda_torsion_anchor * (
            c.high_flex_torsion_anchor_scale if high_flex else 1.0
        )
        return (
            c.lambda_anchor * penalties["anchor"]
            + c.lambda_bond * penalties["bond"]
            + c.lambda_angle * penalties["angle"]
            + c.lambda_ring * penalties["ring"]
            + c.lambda_clash * penalties["clash"]
            + c.lambda_chiral * penalties["chirality"]
            + torsion_weight * penalties["torsion_anchor"]
        )

    def _project_trust(self, coordinates: Tensor, x_input: Tensor) -> tuple[Tensor, bool]:
        c = self.config
        aligned = kabsch_align(coordinates, x_input)
        delta = aligned - x_input
        norms = torch.linalg.vector_norm(delta, dim=-1)
        atom_scale = torch.clamp(c.max_atom_displacement / norms.clamp_min(1.0e-12), max=1.0)
        clipped = delta * atom_scale[:, None]
        rms = torch.sqrt(clipped.square().sum(-1).mean())
        graph_scale = min(1.0, c.max_molecule_rms_displacement / max(float(rms), 1.0e-12))
        projected = x_input + clipped * graph_scale
        hit = bool((atom_scale < 1.0).any()) or graph_scale < 1.0
        return projected, hit

    @staticmethod
    def _mode_values(validity: Mapping[str, float]) -> dict[str, float]:
        return {
            "bond": float(validity["bond_outlier_rate"]),
            "angle": float(validity["angle_outlier_rate"]),
            "ring": max(
                float(validity["ring_bond_outlier_rate"]),
                float(validity["ring_planarity_outlier_rate"]),
            ),
            "clash": max(
                float(validity["severe_clash_rate"]),
                float(validity["clash_penetration"]),
            ),
        }

    def _candidate(self, x_input: Tensor, candidate: Tensor, record: Any, initial, step: int):
        c = self.config
        current = self.validity.evaluate(candidate, record, baseline_coordinates=x_input)
        displacement = displacement_metrics(x_input, candidate)
        torsion = torsion_change_metrics(x_input, candidate, record)
        high_flex = int(field(record, "num_rotatable_bonds", 0)) >= 6
        torsion_limit = (
            c.max_high_flex_torsion_change_rad if high_flex else c.max_torsion_change_rad
        )
        new_risk = 0.0
        new_risk += max(0.0, current["severe_clash_rate"] - initial["severe_clash_rate"])
        new_risk += max(0.0, 1.0 - current["chirality_preserved"])
        new_risk += max(
            0.0,
            current["stereocenter_degenerate_rate"] - initial["stereocenter_degenerate_rate"],
        )
        new_risk += max(0.0, current["ring_bond_outlier_rate"] - initial["ring_bond_outlier_rate"])
        new_risk += max(
            0.0,
            current["ring_planarity_outlier_rate"] - initial["ring_planarity_outlier_rate"],
        )
        safe = (
            new_risk <= 1.0e-12
            and displacement["aligned_rms_displacement"] <= c.max_molecule_rms_displacement + 1.0e-6
            and displacement["max_atom_displacement"] <= c.max_atom_displacement + 1.0e-6
            and torsion["max_rotatable_torsion_change"] <= torsion_limit + 1.0e-6
            and bool(torch.isfinite(candidate).all())
        )
        gain = (
            initial["total_thresholded_validity_score"]
            - current["total_thresholded_validity_score"]
        )
        score = (
            gain
            - c.lambda_score_displacement * displacement["aligned_rms_displacement"]
            - c.lambda_score_torsion * torsion["max_rotatable_torsion_change"]
            - c.lambda_score_risk * new_risk
        )
        return {
            "step": int(step),
            "coordinates": candidate.detach().clone(),
            "validity": current,
            "validity_gain": float(gain),
            "score": float(score),
            "safe": bool(safe),
            "new_risk": float(new_risk),
            "displacement": displacement,
            "torsion": torsion,
        }

    def build(self, coordinates: Tensor, record: Any) -> dict[str, Any]:
        """Build one target and return coordinates plus complete audit metadata."""

        c = self.config
        x_input = torch.as_tensor(coordinates, dtype=torch.float32).detach().clone()
        initial = self.validity.evaluate(x_input, record, baseline_coordinates=x_input)
        active = {key: value > 0.0 for key, value in self._mode_values(initial).items()}
        if not any(active.values()):
            return self._identity_result(x_input, initial, "already_valid", "identity_clean", [])

        x = x_input.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=c.learning_rate)
        candidates = []
        summary = []
        best_gain = 0.0
        stale = 0
        stop_reason = "max_steps"
        for step in range(1, c.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            penalties = self._penalties(x, x_input, record)
            objective = self._objective(penalties, record)
            if not bool(torch.isfinite(objective)):
                stop_reason = "numerical_anomaly"
                break
            objective.backward()
            if x.grad is None or not bool(torch.isfinite(x.grad).all()):
                stop_reason = "numerical_anomaly"
                break
            optimizer.step()
            with torch.no_grad():
                projected, trust_hit = self._project_trust(x, x_input)
                x.copy_(projected)
            try:
                candidate = self._candidate(x_input, x.detach(), record, initial, step)
            except Exception:
                stop_reason = "numerical_anomaly"
                break
            candidates.append(candidate)
            summary.append({
                "step": step,
                "objective": float(objective.detach()),
                "validity_gain": candidate["validity_gain"],
                "score": candidate["score"],
                "safe": candidate["safe"],
                "new_risk": candidate["new_risk"],
                "aligned_rms_displacement": candidate["displacement"]["aligned_rms_displacement"],
                "max_atom_displacement": candidate["displacement"]["max_atom_displacement"],
                "max_rotatable_torsion_change": candidate["torsion"]["max_rotatable_torsion_change"],
            })
            if candidate["safe"] and candidate["validity_gain"] > best_gain + c.min_improvement:
                best_gain = candidate["validity_gain"]
                stale = 0
            else:
                stale += 1
            current_modes = self._mode_values(candidate["validity"])
            if all(not enabled or current_modes[name] <= 0.0 for name, enabled in active.items()):
                stop_reason = "all_active_violations_resolved"
                break
            if candidate["new_risk"] > 0.0:
                stop_reason = "new_safety_risk"
                break
            if trust_hit:
                stop_reason = "trust_radius_reached"
                break
            if stale >= c.early_stop_patience:
                stop_reason = "improvement_plateau"
                break

        improving = [
            item for item in candidates
            if item["safe"] and item["validity_gain"] >= c.min_improvement
        ]
        if not improving:
            return self._identity_result(
                x_input, initial, stop_reason, "identity_fallback", summary
            )
        selected = max(improving, key=lambda item: (item["score"], -item["step"]))
        target = selected["coordinates"]
        final = selected["validity"]
        displacement = selected["displacement"]
        torsion = selected["torsion"]
        return {
            "x_target": target,
            "target_metadata": {
                "target_status": "minimal_validity_success",
                "initial_validity": initial,
                "final_validity": final,
                "validity_gain": selected["validity_gain"],
                "bond_gain": initial["bond_outlier_rate"] - final["bond_outlier_rate"],
                "angle_gain": initial["angle_outlier_rate"] - final["angle_outlier_rate"],
                "ring_gain": (
                    initial["ring_bond_outlier_rate"] + initial["ring_planarity_outlier_rate"]
                    - final["ring_bond_outlier_rate"] - final["ring_planarity_outlier_rate"]
                ),
                "clash_gain": (
                    initial["clash_penetration"] + initial["severe_clash_rate"]
                    - final["clash_penetration"] - final["severe_clash_rate"]
                ),
                "initial_to_target_rmsd": displacement["aligned_rms_displacement"],
                "mean_atom_displacement": displacement["mean_atom_displacement"],
                "max_atom_displacement": displacement["max_atom_displacement"],
                "torsion_change": torsion["torsion_circular_change"],
                "max_rotatable_torsion_change": torsion["max_rotatable_torsion_change"],
                "chirality_status": "preserved" if final["chirality_preserved"] >= 1.0 else "changed",
                "ring_status": "nonworse",
                "selected_step": selected["step"],
                "stop_reason": stop_reason,
                "trajectory_summary": summary,
                "target_sha256": _tensor_sha256(target),
                "active_mode_mask": active,
                "optimizer_config": asdict(c),
                "reference_fallback_used": False,
                "force_field_fallback_used": False,
            },
        }

    def _identity_result(self, x_input, initial, stop_reason, status, summary):
        return {
            "x_target": x_input.clone(),
            "target_metadata": {
                "target_status": status,
                "initial_validity": initial,
                "final_validity": dict(initial),
                "validity_gain": 0.0,
                "bond_gain": 0.0,
                "angle_gain": 0.0,
                "ring_gain": 0.0,
                "clash_gain": 0.0,
                "initial_to_target_rmsd": 0.0,
                "mean_atom_displacement": 0.0,
                "max_atom_displacement": 0.0,
                "torsion_change": 0.0,
                "max_rotatable_torsion_change": 0.0,
                "chirality_status": "preserved",
                "ring_status": "unchanged",
                "selected_step": 0,
                "stop_reason": stop_reason,
                "trajectory_summary": summary,
                "target_sha256": _tensor_sha256(x_input),
                "active_mode_mask": {
                    key: value > 0.0 for key, value in self._mode_values(initial).items()
                },
                "optimizer_config": asdict(self.config),
                "reference_fallback_used": False,
                "force_field_fallback_used": False,
            },
        }
