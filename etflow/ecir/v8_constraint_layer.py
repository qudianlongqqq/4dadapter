"""Type-balanced differentiable Bond/Angle constraint layer for MCVR V8."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .audit import field
from .bac_jacobian import bond_residual_jacobian, cosine_angle_residual_jacobian
from .geometry import bond_angles, bond_lengths
from .v8_constraint_normalization import FrozenResidualScales, normalize_constraint_type
from .v8_solver import DifferentiableSolveConfig, solve_unified_delta


@dataclass(frozen=True)
class ConstraintLayerConfig:
    enabled: bool = True
    use_bond: bool = True
    use_angle: bool = True
    normalize_by_active_count: bool = True
    solver_lambda_bond: float = 1.0
    solver_lambda_angle: float = 1.0
    solver_lambda_move: float = 0.0
    damping: float = 1.0e-6
    solver_backend: str = "cholesky"
    solve_dtype: str = "float64"
    soft_active_temperature_bond: float = 0.01
    soft_active_temperature_angle: float = 0.05
    minimum_distance: float = 1.0e-8
    near_linear_sine_threshold: float = 0.05
    fail_closed: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "ConstraintLayerConfig":
        if values is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        payload = {key: value for key, value in values.items() if key in allowed}
        return cls(**payload)

    def validate(self) -> None:
        if self.solve_dtype not in {"float32", "float64"}:
            raise ValueError("solve_dtype must be float32 or float64")
        if self.soft_active_temperature_bond <= 0 or self.soft_active_temperature_angle <= 0:
            raise ValueError("soft active temperatures must be positive")
        if self.minimum_distance <= 0:
            raise ValueError("minimum_distance must be positive")
        if self.near_linear_sine_threshold <= 0:
            raise ValueError("near-linear sine threshold must be positive")
        DifferentiableSolveConfig(
            lambda_bond=self.solver_lambda_bond,
            lambda_angle=self.solver_lambda_angle,
            lambda_move=self.solver_lambda_move,
            damping=self.damping,
            backend=self.solver_backend,
            fail_closed=self.fail_closed,
        ).validate()


def _interval_boundary(values: Tensor, ranges: Tensor) -> tuple[Tensor, Tensor]:
    lower, upper = ranges[:, 0], ranges[:, 1]
    below = lower - values
    above = values - upper
    violation = torch.maximum(torch.maximum(below, above), torch.zeros_like(values))
    boundary = torch.where(values < lower, lower, torch.where(values > upper, upper, values))
    return boundary, violation


def _soft_activity(violation: Tensor, temperature: float) -> Tensor:
    # Exactly zero for valid constraints, smooth and bounded for every positive excess.
    return -torch.expm1(-violation / float(temperature))


def _graph_ptr(batch: Any, atom_count: int, device: torch.device) -> Tensor:
    value = field(batch, "ptr")
    if value is not None:
        return torch.as_tensor(value, device=device, dtype=torch.long)
    assignment = field(batch, "batch")
    if assignment is None:
        return torch.tensor([0, atom_count], device=device, dtype=torch.long)
    assignment = torch.as_tensor(assignment, device=device, dtype=torch.long)
    graphs = int(assignment.max()) + 1 if assignment.numel() else 1
    counts = torch.bincount(assignment, minlength=graphs)
    return torch.cat((counts.new_zeros(1), counts.cumsum(0)))


def _empty_rows(coordinates: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        coordinates.new_empty(0),
        coordinates.new_empty((0, coordinates.numel())),
        coordinates.new_empty(0),
        coordinates.new_empty(0),
    )


class DifferentiableMolecularConstraintLayer(nn.Module):
    """Solve one unified Cartesian delta independently for every molecular graph."""

    def __init__(
        self,
        config: ConstraintLayerConfig | Mapping[str, Any] | None = None,
        *,
        scales: FrozenResidualScales | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, ConstraintLayerConfig)
            else ConstraintLayerConfig.from_mapping(config)
        )
        self.config.validate()
        self.scales = (
            scales
            if isinstance(scales, FrozenResidualScales)
            else FrozenResidualScales.from_mapping(scales or {"bond": 1.0, "angle": 1.0})
        )
        self.scales.validate()

    def _local_constraints(
        self, batch: Any, left: int, right: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        bonds_value = field(batch, "active_bond_constraint_index")
        bond_ranges_value = field(batch, "bond_allowed_range")
        angles_value = field(batch, "active_angle_constraint_index")
        angle_ranges_value = field(batch, "angle_allowed_range")
        if bonds_value is None or bond_ranges_value is None:
            bonds = torch.empty((2, 0), device=device, dtype=torch.long)
            bond_ranges = torch.empty((0, 3), device=device, dtype=dtype)
        else:
            all_bonds = torch.as_tensor(bonds_value, device=device, dtype=torch.long).reshape(2, -1)
            all_ranges = torch.as_tensor(bond_ranges_value, device=device, dtype=dtype).reshape(
                -1, 3
            )
            if all_ranges.size(0) != all_bonds.size(1):
                raise ValueError("Bond constraints and ranges are not aligned")
            keep = (
                (all_bonds[0] >= left)
                & (all_bonds[0] < right)
                & (all_bonds[1] >= left)
                & (all_bonds[1] < right)
            )
            bonds, bond_ranges = all_bonds[:, keep] - left, all_ranges[keep]
        if angles_value is None or angle_ranges_value is None:
            angles = torch.empty((0, 3), device=device, dtype=torch.long)
            angle_ranges = torch.empty((0, 3), device=device, dtype=dtype)
        else:
            all_angles = torch.as_tensor(angles_value, device=device, dtype=torch.long)
            if all_angles.ndim == 2 and all_angles.size(0) == 3:
                all_angles = all_angles.t()
            all_angles = all_angles.reshape(-1, 3)
            all_ranges = torch.as_tensor(angle_ranges_value, device=device, dtype=dtype).reshape(
                -1, 3
            )
            if all_ranges.size(0) != all_angles.size(0):
                raise ValueError("Angle constraints and ranges are not aligned")
            keep = ((all_angles >= left) & (all_angles < right)).all(dim=1)
            angles, angle_ranges = all_angles[keep] - left, all_ranges[keep]
        return bonds, bond_ranges, angles, angle_ranges

    def _bond_rows(
        self, coordinates: Tensor, bonds: Tensor, ranges: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        if not self.config.use_bond or not bonds.numel():
            residual, jacobian, activity, violation = _empty_rows(coordinates)
        else:
            values = bond_lengths(coordinates, bonds)
            target, violation = _interval_boundary(values, ranges)
            residual, jacobian, degenerate = bond_residual_jacobian(
                coordinates, bonds, target, minimum_distance=self.config.minimum_distance
            )
            activity = _soft_activity(violation, self.config.soft_active_temperature_bond)
            activity = activity * (~degenerate).to(activity.dtype)
        residual_n, jacobian_n, diagnostics = normalize_constraint_type(
            residual,
            jacobian,
            activity,
            scale=self.scales.bond,
            normalize_by_active_count=self.config.normalize_by_active_count,
        )
        diagnostics["hard_active_count"] = (violation > 0).sum().to(coordinates.dtype)
        diagnostics["total_count"] = coordinates.new_tensor(residual.numel())
        keep = activity > 0
        return residual_n[keep], jacobian_n[keep], activity, diagnostics

    def _angle_rows(
        self, coordinates: Tensor, angles: Tensor, ranges: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        if not self.config.use_angle or not angles.numel():
            residual, jacobian, activity, violation = _empty_rows(coordinates)
            sine = coordinates.new_empty(0)
        else:
            values = bond_angles(coordinates, angles)
            target, violation = _interval_boundary(values, ranges)
            residual, jacobian, degenerate, sine = cosine_angle_residual_jacobian(
                coordinates,
                angles,
                torch.cos(target),
                minimum_distance=self.config.minimum_distance,
            )
            activity = _soft_activity(violation, self.config.soft_active_temperature_angle)
            activity = activity * (~degenerate).to(activity.dtype)
            activity = activity * (sine / float(self.config.near_linear_sine_threshold)).clamp(
                max=1.0
            )
        residual_n, jacobian_n, diagnostics = normalize_constraint_type(
            residual,
            jacobian,
            activity,
            scale=self.scales.angle,
            normalize_by_active_count=self.config.normalize_by_active_count,
        )
        diagnostics["hard_active_count"] = (violation > 0).sum().to(coordinates.dtype)
        diagnostics["total_count"] = coordinates.new_tensor(residual.numel())
        diagnostics["near_linear_count"] = (sine < 1.0e-3).sum().to(coordinates.dtype)
        keep = activity > 0
        return residual_n[keep], jacobian_n[keep], activity, diagnostics

    def forward(
        self,
        coordinates: Tensor,
        delta_prior: Tensor,
        prior_confidence: Tensor,
        batch: Any,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {
                "delta_final": delta_prior,
                "solver_status": ("DISABLED",),
                "solver_failure": coordinates.new_zeros(1),
            }
        solve_dtype = torch.float64 if self.config.solve_dtype == "float64" else torch.float32
        ptr = _graph_ptr(batch, coordinates.size(0), coordinates.device)
        outputs = []
        statuses: list[str] = []
        failures = []
        rows: list[dict[str, Tensor]] = []
        solver_config = DifferentiableSolveConfig(
            lambda_bond=self.config.solver_lambda_bond,
            lambda_angle=self.config.solver_lambda_angle,
            lambda_move=self.config.solver_lambda_move,
            damping=self.config.damping,
            backend=self.config.solver_backend,
            fail_closed=self.config.fail_closed,
        )
        solve_started = time.perf_counter()
        for graph_index in range(ptr.numel() - 1):
            left, right = int(ptr[graph_index]), int(ptr[graph_index + 1])
            local = coordinates[left:right].to(solve_dtype)
            prior = delta_prior[left:right].to(solve_dtype)
            confidence = prior_confidence[left:right].to(solve_dtype)
            bonds, bond_ranges, angles, angle_ranges = self._local_constraints(
                batch, left, right, coordinates.device, solve_dtype
            )
            bond_r, bond_j, _, bond_diag = self._bond_rows(local, bonds, bond_ranges)
            angle_r, angle_j, _, angle_diag = self._angle_rows(local, angles, angle_ranges)
            solved, solve_diag = solve_unified_delta(
                prior,
                confidence,
                bond_r,
                bond_j,
                angle_r,
                angle_j,
                solver_config,
            )
            outputs.append(solved.to(coordinates.dtype))
            statuses.append(str(solve_diag["status"]))
            failures.append(float(bool(solve_diag["fallback"])))
            rows.append(
                {
                    **{f"bond_{key}": value for key, value in bond_diag.items()},
                    **{f"angle_{key}": value for key, value in angle_diag.items()},
                    **{
                        key: value for key, value in solve_diag.items() if isinstance(value, Tensor)
                    },
                }
            )
        result: dict[str, Any] = {
            "delta_final": torch.cat(outputs, dim=0) if outputs else torch.zeros_like(coordinates),
            "solver_status": tuple(statuses),
            "solver_failure": coordinates.new_tensor(failures),
            "solver_duration_seconds": coordinates.new_tensor(time.perf_counter() - solve_started),
        }
        if rows:
            for key in rows[0]:
                result[key] = torch.stack([row[key].to(coordinates.dtype) for row in rows])
        return result
