"""Monotonic post-hoc confidence calibration for frozen MCVR bond proposals."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .mvr_model import MCVRModel


SCHEMA_VERSION = "ecir-mvr-stage-e0-confidence-calibrator-v1"
CALIBRATION_DATA_SCHEMA = "ecir-mvr-stage-e0-calibration-data-v1"
MANIFEST_SCHEMA = "ecir-mvr-stage-e0-calibration-manifest-v1"
DIAGNOSTIC_ALL_ONE = "DIAGNOSTIC_ORACLE_ONLY"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


class MonotonicConfidenceCalibrator(nn.Module):
    """Two-parameter monotonic map ``sigmoid(a * z + b)`` with ``a > 0``."""

    def __init__(self, *, epsilon: float = 1.0e-8, dtype: torch.dtype = torch.float64):
        super().__init__()
        if not 0.0 < float(epsilon) < 1.0:
            raise ValueError("epsilon must be in (0, 1)")
        self.epsilon = float(epsilon)
        self.raw_a = nn.Parameter(torch.tensor(_inverse_softplus(1.0 - epsilon), dtype=dtype))
        self.b = nn.Parameter(torch.zeros((), dtype=dtype))

    @property
    def a(self) -> Tensor:
        return F.softplus(self.raw_a) + self.epsilon

    def calibrated_logit(self, confidence_logit: Tensor) -> Tensor:
        values = torch.as_tensor(confidence_logit)
        return self.a.to(values) * values + self.b.to(values)

    def forward(self, confidence_logit: Tensor) -> Tensor:
        return torch.sigmoid(self.calibrated_logit(confidence_logit))


def optimal_scale_targets(
    predicted_residual: Tensor | Sequence[float],
    target_residual: Tensor | Sequence[float],
    *,
    epsilon: float = 1.0e-8,
) -> Tensor:
    predicted = torch.as_tensor(predicted_residual)
    target = torch.as_tensor(target_residual, device=predicted.device, dtype=predicted.dtype)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target residuals must have identical shapes")
    valid = (predicted.abs() > float(epsilon)) & (torch.sign(predicted) == torch.sign(target))
    ratio = torch.where(valid, target / predicted, torch.zeros_like(predicted))
    return ratio.clamp(0.0, 1.0)


def calibrated_bond_residual(
    unattenuated_residual: Tensor,
    confidence_logit: Tensor,
    calibrator: MonotonicConfidenceCalibrator,
) -> Tensor:
    return torch.as_tensor(unattenuated_residual) * calibrator(confidence_logit)


def confidence_for_mode(
    confidence_logit: Tensor,
    *,
    mode: str,
    calibrator: MonotonicConfidenceCalibrator | None = None,
    diagnostic_oracle_only: bool = False,
) -> Tensor:
    if mode == "deployed":
        return torch.sigmoid(confidence_logit)
    if mode == "calibrated":
        if calibrator is None:
            raise ValueError("calibrated mode requires a calibrator")
        return calibrator(confidence_logit)
    if mode == "confidence_all_one":
        if not diagnostic_oracle_only:
            raise PermissionError("confidence-all-one is DIAGNOSTIC_ORACLE_ONLY")
        return torch.ones_like(confidence_logit)
    raise ValueError(f"unknown confidence mode: {mode}")


def split_calibration_molecules(
    molecule_ids: Sequence[str], *, seed: int = 42, fit_fraction: float = 0.80
) -> tuple[list[str], list[str]]:
    identifiers = sorted({str(value) for value in molecule_ids})
    if len(identifiers) < 2:
        raise ValueError("calibration split requires at least two molecules")
    if not 0.0 < float(fit_fraction) < 1.0:
        raise ValueError("fit_fraction must be in (0, 1)")
    rng = np.random.default_rng(int(seed))
    order = np.asarray(identifiers, dtype=object)[rng.permutation(len(identifiers))].tolist()
    fit_count = min(len(order) - 1, max(1, int(math.floor(len(order) * float(fit_fraction)))))
    return sorted(order[:fit_count]), sorted(order[fit_count:])


def build_calibration_manifest(
    molecule_ids: Sequence[str],
    *,
    checkpoint_sha256: str,
    frozen_identities: Mapping[str, Any],
    seed: int = 42,
    fit_fraction: float = 0.80,
    smoke: bool = False,
) -> dict[str, Any]:
    fit, check = split_calibration_molecules(molecule_ids, seed=seed, fit_fraction=fit_fraction)
    identity_payload = {
        "seed": int(seed), "fit_fraction": float(fit_fraction),
        "fit_molecule_ids": fit, "internal_check_molecule_ids": check,
        "checkpoint_sha256": str(checkpoint_sha256),
        "frozen_identities": dict(frozen_identities),
    }
    return {
        "schema_version": MANIFEST_SCHEMA, "training_only": True,
        "validation_records_read": 0, "test_records_read": 0, "smoke": bool(smoke),
        **identity_payload,
        "training_molecule_count": len(fit) + len(check),
        "fit_molecule_count": len(fit), "internal_check_molecule_count": len(check),
        "training_molecule_identity_sha256": canonical_sha256(sorted(fit + check)),
        "manifest_identity_sha256": canonical_sha256(identity_payload),
    }


def validate_calibration_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unexpected Stage E0 manifest schema")
    if not manifest.get("training_only") or manifest.get("test_records_read") != 0:
        raise ValueError("calibration manifest is not training-only and test-free")
    fit = [str(value) for value in manifest["fit_molecule_ids"]]
    check = [str(value) for value in manifest["internal_check_molecule_ids"]]
    if set(fit) & set(check):
        raise ValueError("calibration fit and internal-check molecules overlap")
    if manifest["training_molecule_identity_sha256"] != canonical_sha256(sorted(fit + check)):
        raise ValueError("training molecule identity mismatch")
    payload = {
        "seed": int(manifest["seed"]), "fit_fraction": float(manifest["fit_fraction"]),
        "fit_molecule_ids": fit, "internal_check_molecule_ids": check,
        "checkpoint_sha256": str(manifest["checkpoint_sha256"]),
        "frozen_identities": dict(manifest["frozen_identities"]),
    }
    if manifest["manifest_identity_sha256"] != canonical_sha256(payload):
        raise ValueError("calibration manifest identity mismatch")


def validate_calibration_frame(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "split", "molecule_id", "record_id", "rollout_step",
        "bond_index", "confidence_logit", "unattenuated_residual", "target_residual",
        "optimal_scale", "weight", "active_target", "outlier", "severe_outlier",
        "ring", "zero_target", "training_only", "test_records_read",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"calibration data missing columns: {sorted(required - set(frame.columns))}")
    if set(frame.schema_version.astype(str)) != {CALIBRATION_DATA_SCHEMA}:
        raise ValueError("unexpected calibration data schema")
    if not bool(frame.training_only.astype(bool).all()) or set(frame.test_records_read.astype(int)) != {0}:
        raise ValueError("calibration data is not training-only and test-free")
    if not set(frame.split.astype(str)).issubset({"fit", "internal_check"}):
        raise ValueError("calibration data contains a forbidden split")
    expected = set(manifest["fit_molecule_ids"]) | set(manifest["internal_check_molecule_ids"])
    if set(frame.molecule_id.astype(str)) != expected:
        raise ValueError("calibration data molecule identity differs from manifest")
    numeric = frame[[
        "confidence_logit", "unattenuated_residual", "target_residual",
        "optimal_scale", "weight",
    ]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("calibration data contains NaN or Inf")
    if not frame.optimal_scale.between(0.0, 1.0).all():
        raise ValueError("optimal-scale labels are outside [0, 1]")


def severity_weights(
    target_residual: Tensor,
    *,
    active_threshold: float = 0.005,
    severe_threshold: float = 0.05,
    active_weight: float = 2.0,
    severe_weight: float = 4.0,
) -> Tensor:
    magnitude = torch.as_tensor(target_residual).abs()
    weight = torch.ones_like(magnitude)
    weight = torch.where(magnitude > float(active_threshold), weight * float(active_weight), weight)
    weight = torch.where(magnitude > float(severe_threshold), weight * float(severe_weight), weight)
    return weight


def weighted_soft_bce(
    calibrator: MonotonicConfidenceCalibrator,
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
) -> Tensor:
    transformed = calibrator.calibrated_logit(logits)
    loss = F.binary_cross_entropy_with_logits(
        transformed, targets.to(transformed), reduction="none"
    )
    weights = weights.to(loss)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0e-12)


def fit_monotonic_calibrator(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    epsilon: float = 1.0e-8,
    max_iter: int = 100,
) -> tuple[MonotonicConfidenceCalibrator, dict[str, float]]:
    validate_calibration_manifest(manifest)
    validate_calibration_frame(frame, manifest)
    fit = frame[frame.split.eq("fit")]
    check = frame[frame.split.eq("internal_check")]
    calibrator = MonotonicConfidenceCalibrator(epsilon=epsilon)

    def tensors(values: pd.DataFrame) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(torch.as_tensor(values[column].to_numpy(dtype=np.float64, copy=True), dtype=torch.float64) for column in (
            "confidence_logit", "optimal_scale", "weight"
        ))

    fit_tensors = tensors(fit)
    check_tensors = tensors(check)
    initial_fit = float(weighted_soft_bce(calibrator, *fit_tensors).detach())
    initial_check = float(weighted_soft_bce(calibrator, *check_tensors).detach())
    optimizer = torch.optim.LBFGS(
        calibrator.parameters(), max_iter=int(max_iter), line_search_fn="strong_wolfe",
        tolerance_grad=1.0e-12, tolerance_change=1.0e-14,
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = weighted_soft_bce(calibrator, *fit_tensors)
        loss.backward()
        return loss

    optimizer.step(closure)
    metrics = {
        "initial_fit_loss": initial_fit,
        "final_fit_loss": float(weighted_soft_bce(calibrator, *fit_tensors).detach()),
        "initial_internal_check_loss": initial_check,
        "final_internal_check_loss": float(weighted_soft_bce(calibrator, *check_tensors).detach()),
    }
    return calibrator, metrics


def calibrator_payload(
    calibrator: MonotonicConfidenceCalibrator,
    *,
    checkpoint_sha256: str,
    training_molecule_identity_sha256: str,
    manifest_identity_sha256: str,
    fit_metrics: Mapping[str, float],
    smoke: bool = False,
) -> dict[str, Any]:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "raw_a": float(calibrator.raw_a.detach()), "a": float(calibrator.a.detach()),
        "b": float(calibrator.b.detach()), "epsilon": float(calibrator.epsilon),
        "checkpoint_sha256": str(checkpoint_sha256),
        "training_molecule_identity_sha256": str(training_molecule_identity_sha256),
        "manifest_identity_sha256": str(manifest_identity_sha256),
    }
    return {
        **identity, "calibration_identity_sha256": canonical_sha256(identity),
        "fit_metrics": {name: float(value) for name, value in fit_metrics.items()},
        "validation_used_for_fit": False, "test_records_read": 0, "smoke": bool(smoke),
    }


def load_calibrator(payload: Mapping[str, Any]) -> MonotonicConfidenceCalibrator:
    identity = {name: payload[name] for name in (
        "schema_version", "raw_a", "a", "b", "epsilon", "checkpoint_sha256",
        "training_molecule_identity_sha256", "manifest_identity_sha256",
    )}
    if payload.get("calibration_identity_sha256") != canonical_sha256(identity):
        raise ValueError("calibrator identity mismatch")
    calibrator = MonotonicConfidenceCalibrator(epsilon=float(payload["epsilon"]))
    calibrator.raw_a.data.fill_(float(payload["raw_a"]))
    calibrator.b.data.fill_(float(payload["b"]))
    if not math.isclose(float(calibrator.a.detach()), float(payload["a"]), rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError("persisted calibrator slope mismatch")
    calibrator.eval()
    return calibrator


def strict_load_frozen_model(
    checkpoint: str | Path, *, expected_sha256: str, device: torch.device
) -> tuple[MCVRModel, dict[str, Any]]:
    if file_sha256(checkpoint) != str(expected_sha256):
        raise RuntimeError("Stage E0 frozen checkpoint identity changed")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = MCVRModel(**payload["config"]["model"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not getattr(model, "torsion_gate_fixed_zero", False):
        raise RuntimeError("Stage E0 requires the torsion gate to remain fixed at zero")
    return model, payload


def molecule_paired_bootstrap(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    metrics: Sequence[str],
    draws: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        pivot = frame.pivot(index="molecule_id", columns="method", values=metric).dropna()
        delta = pivot[candidate].to_numpy(float) - pivot[baseline].to_numpy(float)
        if not len(delta):
            result[metric] = {"mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
            continue
        rng = np.random.default_rng(int(seed))
        means = np.empty(int(draws), dtype=np.float64)
        for start in range(0, int(draws), 256):
            count = min(256, int(draws) - start)
            indices = rng.integers(0, len(delta), size=(count, len(delta)))
            means[start:start + count] = delta[indices].mean(axis=1)
        result[metric] = {
            "mean": float(delta.mean()),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
        }
    return result


def stage_e0_decision(criteria: Mapping[str, bool], *, harms: bool) -> str:
    if all(bool(value) for value in criteria.values()):
        return "STAGE_E0_CONFIDENCE_CALIBRATION_PASS"
    if harms:
        return "STAGE_E0_HARMS"
    return "STAGE_E0_NO_ADDED_VALUE"
