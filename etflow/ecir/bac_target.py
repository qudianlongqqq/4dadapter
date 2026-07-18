"""Deterministic unified Bond-Angle-Clash minimal-target construction."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from .bac_constraints import (
    CONSTRAINT_FEATURE_VERSION,
    CONSTRAINT_SCHEMA_VERSION,
    canonical_constraint_fields,
    sparse_clash_edges,
    standardized_interval_residual,
)
from .geometry import bond_angles, bond_lengths
from .minimal_validity_target import MinimalValidityTargetBuilder


BAC_TARGET_SCHEMA_VERSION = "mcvr-v2-bac-minimal-target-v1"
BAC_SOLVER_VERSION = "projected-adam-sparse-bac-v1"


@dataclass(frozen=True)
class BACMinimalTargetConfig:
    optimizer: str = "Adam"
    max_steps: int = 40
    learning_rate: float = 0.001
    early_stop_patience: int = 5
    min_improvement: float = 1.0e-5
    lambda_anchor: float = 2.0
    lambda_bond: float = 1.0
    lambda_angle: float = 1.0
    lambda_clash: float = 1.0
    lambda_preserve: float = 1.0
    lambda_torsion_anchor: float = 0.05
    high_flex_torsion_anchor_scale: float = 2.0
    lambda_score_displacement: float = 0.25
    lambda_score_torsion: float = 0.05
    lambda_score_risk: float = 2.0
    max_molecule_rms_displacement: float = 0.15
    max_atom_displacement: float = 0.35
    max_torsion_change_rad: float = 0.70
    max_high_flex_torsion_change_rad: float = 0.35
    clash_cutoff: float = 2.0
    clash_allowed_contact: float = 1.0
    clash_exclude_topology_distance: int = 2
    max_clash_edges: int = 128
    epsilon_bond: float = 0.0
    epsilon_angle: float = 0.0
    epsilon_clash: float = 0.0
    epsilon_ring: float = 0.0

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, Any] | None
    ) -> "BACMinimalTargetConfig":
        if config is None:
            return cls()
        unknown = set(config) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown BAC target settings: {sorted(unknown)}")
        return cls(**dict(config))


class BACMinimalTargetBuilder(MinimalValidityTargetBuilder):
    """One projected solve for BAC with ring/chirality as hard protection."""

    def __init__(
        self,
        validity: Any,
        config: Mapping[str, Any] | None = None,
        *,
        source_identity_sha256: str,
    ) -> None:
        self.validity = validity
        self.config = BACMinimalTargetConfig.from_mapping(config)
        self.source_identity_sha256 = str(source_identity_sha256)
        if self.config.optimizer != "Adam":
            raise ValueError("BAC target solver currently supports optimizer=Adam")
        if self.config.max_steps < 1 or self.config.early_stop_patience < 1:
            raise ValueError("max_steps and early_stop_patience must be positive")
        if (
            self.config.max_molecule_rms_displacement <= 0
            or self.config.max_atom_displacement <= 0
        ):
            raise ValueError("trust limits must be positive")

    def _static(self, record: Any) -> dict[str, Any]:
        return canonical_constraint_fields(
            self.validity,
            record,
            source_identity_sha256=self.source_identity_sha256,
        )

    def _penalties(
        self, coordinates: Tensor, x_input: Tensor, record: Any
    ) -> dict[str, Tensor]:
        static = self._static(record)
        bonds = static["active_bond_constraint_index"].to(coordinates.device)
        bond_ranges = static["bond_allowed_range"].to(coordinates)
        angles = static["active_angle_constraint_index"].to(coordinates.device).t()
        angle_ranges = static["angle_allowed_range"].to(coordinates)
        bond_residual, _ = standardized_interval_residual(
            bond_lengths(coordinates, bonds), bond_ranges
        )
        angle_residual, _ = standardized_interval_residual(
            bond_angles(coordinates, angles), angle_ranges
        )
        clash = sparse_clash_edges(
            coordinates,
            bonds,
            cutoff=self.config.clash_cutoff,
            allowed_contact=self.config.clash_allowed_contact,
            exclude_topology_distance=self.config.clash_exclude_topology_distance,
            max_edges_per_graph=self.config.max_clash_edges,
        )
        active_bond = bond_residual.abs() > float(self.config.epsilon_bond)
        active_angle = angle_residual.abs() > float(self.config.epsilon_angle)
        active_clash = clash["penetration"] > float(self.config.epsilon_clash)

        def normalized_square(values: Tensor, mask: Tensor) -> Tensor:
            if not values.numel() or not bool(mask.any()):
                return coordinates.new_zeros(())
            return values[mask].square().mean()

        bond = normalized_square(bond_residual, active_bond)
        angle = normalized_square(angle_residual, active_angle)
        clash_penetration = clash["penetration"] / max(
            float(self.config.clash_allowed_contact), 1.0e-8
        )
        clash_loss = normalized_square(clash_penetration, active_clash)

        source_bonds = bond_lengths(x_input, bonds)
        current_bonds = bond_lengths(coordinates, bonds)
        source_angles = bond_angles(x_input, angles)
        current_angles = bond_angles(coordinates, angles)
        normal_bond = ~active_bond
        normal_angle = ~active_angle
        preservation_terms = []
        if bool(normal_bond.any()):
            preservation_terms.append(
                (current_bonds[normal_bond] - source_bonds[normal_bond]).square().mean()
            )
        if bool(normal_angle.any()):
            preservation_terms.append(
                (current_angles[normal_angle] - source_angles[normal_angle])
                .square()
                .mean()
            )
        preservation = (
            torch.stack(preservation_terms).mean()
            if preservation_terms
            else coordinates.new_zeros(())
        )
        aligned = coordinates - coordinates.mean(0, keepdim=True)
        input_aligned = x_input - x_input.mean(0, keepdim=True)
        anchor = (aligned - input_aligned).square().sum(-1).mean()
        return {
            "anchor": anchor,
            "bond": bond,
            "angle": angle,
            "clash": clash_loss,
            "preserve": preservation,
            # The base solver expects this key. Ring remains a hard safety
            # constraint and therefore has no active optimization gradient.
            "ring": coordinates.new_zeros(()),
            "chirality": coordinates.new_zeros(()),
            "torsion_anchor": coordinates.new_zeros(()),
        }

    def _objective(self, penalties: Mapping[str, Tensor], record: Any) -> Tensor:
        del record
        config = self.config
        return (
            config.lambda_anchor * penalties["anchor"]
            + config.lambda_bond * penalties["bond"]
            + config.lambda_angle * penalties["angle"]
            + config.lambda_clash * penalties["clash"]
            + config.lambda_preserve * penalties["preserve"]
        )

    def _candidate(
        self,
        x_input: Tensor,
        candidate: Tensor,
        record: Any,
        initial: Mapping[str, float],
        step: int,
    ) -> dict[str, Any]:
        result = super()._candidate(x_input, candidate, record, initial, step)
        current = result["validity"]
        config = self.config
        deltas = {
            "bond": float(current["bond_outlier_rate"])
            - float(initial["bond_outlier_rate"]),
            "angle": float(current["angle_outlier_rate"])
            - float(initial["angle_outlier_rate"]),
            "clash": max(
                float(current["severe_clash_rate"])
                - float(initial["severe_clash_rate"]),
                float(current["clash_penetration"])
                - float(initial["clash_penetration"]),
            ),
            "ring": max(
                float(current["ring_bond_outlier_rate"])
                - float(initial["ring_bond_outlier_rate"]),
                float(current["ring_planarity_outlier_rate"])
                - float(initial["ring_planarity_outlier_rate"]),
            ),
        }
        hard_safe = (
            deltas["bond"] <= config.epsilon_bond + 1.0e-12
            and deltas["angle"] <= config.epsilon_angle + 1.0e-12
            and deltas["clash"] <= config.epsilon_clash + 1.0e-12
            and deltas["ring"] <= config.epsilon_ring + 1.0e-12
            and float(current["chirality_preserved"]) >= 1.0
            and float(current["stereocenter_degenerate_rate"])
            <= float(initial["stereocenter_degenerate_rate"]) + 1.0e-12
        )
        result["safe"] = bool(result["safe"] and hard_safe)
        result["bac_safety_deltas"] = deltas
        return result

    def build(self, coordinates: Tensor, record: Any) -> dict[str, Any]:
        result = super().build(coordinates, record)
        metadata = result["target_metadata"]
        static = self._static(record)
        source = inspect.getsource(BACMinimalTargetBuilder).encode("utf-8")
        config_payload = asdict(self.config)
        metadata.update(
            {
                "target_schema_version": BAC_TARGET_SCHEMA_VERSION,
                "solver": "projected Adam unified BAC",
                "solver_version": BAC_SOLVER_VERSION,
                "solver_parameters": config_payload,
                "statistics_identity_sha256": self.validity.statistics[
                    "identity_sha256"
                ],
                "source_identity_sha256": self.source_identity_sha256,
                "constraint_schema_version": CONSTRAINT_SCHEMA_VERSION,
                "constraint_feature_version": CONSTRAINT_FEATURE_VERSION,
                "constraint_identity_sha256": static[
                    "constraint_identity_sha256"
                ],
                "builder_code_sha256": hashlib.sha256(source).hexdigest(),
                "convergence_state": metadata["stop_reason"],
                "unified_delta": True,
                "independent_target_sum": False,
                "ring_is_active_target": False,
                "test_records_read": 0,
                "test_assets_opened": False,
                "validation_only": True,
            }
        )
        metadata["target_identity_sha256"] = hashlib.sha256(
            result["x_target"].detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        return result
