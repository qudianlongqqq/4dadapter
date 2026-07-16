"""Deterministic, label-free acceptance for conservative ECIR/MCVR inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .audit import displacement_metrics, field, torsion_change_metrics


DEFAULT_ACCEPTANCE_CONFIG = {
    "min_validity_gain": 1.0e-4,
    "ring_outlier_margin": 0.0,
    "max_atom_displacement": 0.12,
    "max_molecule_rms_displacement": 0.06,
    "max_torsion_change_rad": 0.70,
    "max_high_flex_torsion_change_rad": 0.35,
    "max_uncertainty": 1.0,
    "score_displacement_weight": 0.25,
    "score_torsion_weight": 0.05,
    "score_uncertainty_weight": 0.05,
}


@dataclass
class AcceptanceDecision:
    accepted: bool
    selected_step: int
    reject_reasons: list[str]
    score: float
    validity_gain: float
    input_validity: dict[str, float]
    candidate_validity: dict[str, float]
    displacement: dict[str, float]
    torsion_change: dict[str, float]
    uncertainty: float

    def metadata(self) -> dict[str, Any]:
        result = asdict(self)
        result["reject_reason"] = list(self.reject_reasons)
        result["score_breakdown"] = {
            "total": self.score,
            "validity_gain": self.validity_gain,
            "aligned_rms_displacement": self.displacement.get("aligned_rms_displacement", 0.0),
            "max_atom_displacement": self.displacement.get("max_atom_displacement", 0.0),
            "max_rotatable_torsion_change": self.torsion_change.get(
                "max_rotatable_torsion_change", 0.0
            ),
            "uncertainty": self.uncertainty,
        }
        return result


def evaluate_candidate(
    input_coordinates: Tensor,
    candidate: Tensor,
    record: Any,
    validity,
    *,
    step: int,
    uncertainty: float = 0.0,
    config: Mapping[str, float] | None = None,
    input_validity_override: Mapping[str, float] | None = None,
    candidate_validity_override: Mapping[str, float] | None = None,
) -> AcceptanceDecision:
    settings = {**DEFAULT_ACCEPTANCE_CONFIG, **dict(config or {})}
    input_coordinates = torch.as_tensor(input_coordinates, dtype=torch.float32)
    candidate = torch.as_tensor(candidate, dtype=input_coordinates.dtype)
    input_validity = dict(input_validity_override or validity.evaluate(
        input_coordinates, record, baseline_coordinates=input_coordinates
    ))
    candidate_validity = dict(candidate_validity_override or validity.evaluate(
        candidate, record, baseline_coordinates=input_coordinates
    ))
    gain = (
        input_validity["total_thresholded_validity_score"]
        - candidate_validity["total_thresholded_validity_score"]
    )
    displacement = displacement_metrics(input_coordinates, candidate)
    torsion = torsion_change_metrics(input_coordinates, candidate, record)
    local_improved = any(
        candidate_validity[name] < input_validity[name] - 1.0e-12
        for name in (
            "bond_outlier_rate", "bond_outlier_magnitude", "angle_outlier_rate",
            "angle_outlier_magnitude", "ring_bond_outlier_rate",
            "ring_planarity_outlier_rate", "severe_clash_rate",
        )
    )
    reasons = []
    if gain < float(settings["min_validity_gain"]): reasons.append("insufficient_validity_gain")
    if not local_improved: reasons.append("no_thresholded_local_improvement")
    if candidate_validity["severe_clash_rate"] > input_validity["severe_clash_rate"]: reasons.append("severe_clash_increased")
    if candidate_validity["chirality_preserved"] < 1.0: reasons.append("chirality_flip")
    if candidate_validity["stereocenter_degenerate_rate"] > input_validity["stereocenter_degenerate_rate"]: reasons.append("stereocenter_degeneracy_increased")
    if candidate_validity["ring_bond_outlier_rate"] > input_validity["ring_bond_outlier_rate"] + float(settings["ring_outlier_margin"]): reasons.append("ring_outlier_increased")
    if candidate_validity["ring_planarity_outlier_rate"] > input_validity["ring_planarity_outlier_rate"] + float(settings["ring_outlier_margin"]): reasons.append("ring_planarity_outlier_increased")
    if displacement["aligned_rms_displacement"] > float(settings["max_molecule_rms_displacement"]): reasons.append("molecule_trust_radius")
    if displacement["max_atom_displacement"] > float(settings["max_atom_displacement"]): reasons.append("atom_trust_radius")
    rotatable = int(field(record, "num_rotatable_bonds", 0))
    torsion_limit = float(
        settings["max_high_flex_torsion_change_rad"]
        if rotatable >= 6 else settings["max_torsion_change_rad"]
    )
    if torsion["max_rotatable_torsion_change"] > torsion_limit: reasons.append("torsion_trust_radius")
    if float(uncertainty) > float(settings["max_uncertainty"]): reasons.append("uncertainty_limit")
    score = (
        gain
        - float(settings["score_displacement_weight"]) * displacement["aligned_rms_displacement"]
        - float(settings["score_torsion_weight"]) * torsion["max_rotatable_torsion_change"]
        - float(settings["score_uncertainty_weight"]) * float(uncertainty)
    )
    return AcceptanceDecision(
        accepted=not reasons, selected_step=int(step), reject_reasons=reasons,
        score=float(score), validity_gain=float(gain), input_validity=input_validity,
        candidate_validity=candidate_validity, displacement=displacement,
        torsion_change=torsion, uncertainty=float(uncertainty),
    )


def select_trajectory_candidate(
    input_coordinates: Tensor,
    trajectory: Sequence[Tensor],
    record: Any,
    validity,
    *,
    mode: str = "best_of_trajectory",
    uncertainties: Sequence[float] | None = None,
    config: Mapping[str, float] | None = None,
) -> tuple[Tensor, AcceptanceDecision]:
    if mode not in {"best_of_trajectory", "final_step"}:
        raise ValueError("acceptance mode must be best_of_trajectory or final_step")
    if not trajectory:
        raise ValueError("trajectory must contain at least one candidate")
    uncertainties = list(uncertainties or [0.0] * len(trajectory))
    if len(uncertainties) != len(trajectory):
        raise ValueError("uncertainties must match trajectory length")
    indices = [len(trajectory) - 1] if mode == "final_step" else list(range(len(trajectory)))
    decisions = [
        evaluate_candidate(
            input_coordinates, trajectory[index], record, validity,
            step=index + 1, uncertainty=uncertainties[index], config=config,
        )
        for index in indices
    ]
    accepted = [decision for decision in decisions if decision.accepted]
    if accepted:
        selected = max(accepted, key=lambda decision: (decision.score, -decision.selected_step))
        return torch.as_tensor(trajectory[selected.selected_step - 1]).clone(), selected
    input_validity = validity.evaluate(input_coordinates, record, baseline_coordinates=input_coordinates)
    rejected = max(decisions, key=lambda decision: decision.score)
    rejected.accepted = False
    rejected.selected_step = 0
    rejected.reject_reasons = sorted(set(rejected.reject_reasons + ["return_input"]))
    rejected.candidate_validity = input_validity
    rejected.validity_gain = 0.0
    rejected.score = 0.0
    rejected.displacement = {key: 0.0 for key in rejected.displacement}
    rejected.torsion_change = {key: 0.0 for key in rejected.torsion_change}
    return torch.as_tensor(input_coordinates).clone(), rejected
