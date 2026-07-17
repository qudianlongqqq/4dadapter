"""Bounded residual, sign-safe confidence calibration for MCVR Stage G."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .confidence_calibration import (
    build_calibration_manifest,
    canonical_sha256,
    file_sha256,
    validate_calibration_manifest,
)
from .feature_conditioned_confidence import (
    BOND_TYPE_IDS,
    CATEGORICAL_FEATURES,
    DEPLOYMENT_FEATURES,
    NUMERIC_FEATURES,
)


SCHEMA_VERSION = "ecir-mvr-stage-g-bounded-residual-v1"
DATA_SCHEMA = "ecir-mvr-stage-g-calibration-data-v1"
MANIFEST_SCHEMA = "ecir-mvr-stage-g-calibration-manifest-v1"
STAGE_F_DECISION = "STAGE_F_HARMS"
STAGE_G_METHOD = "bounded_residual_sign_safe_stage_g"
STAGE_G_DEPLOYMENT_FEATURES = (*DEPLOYMENT_FEATURES, "original_confidence")


def verify_stage_f_identity(config: Mapping[str, Any]) -> None:
    """Fail closed if any formal Stage F result used to motivate Stage G changed."""

    frozen = config["frozen_stage_f"]
    result_path = Path(frozen["validation_result"])
    import json

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("decision") != STAGE_F_DECISION:
        raise RuntimeError("Stage F formal decision changed")
    if int(result.get("selected_step", -1)) != 5000 or int(result.get("test_records_read", -1)) != 0:
        raise RuntimeError("Stage F formal identity changed")
    abstention = result.get("activation_audit", {}).get("feature_conditioned_sign_safe", {}).get(
        "abstention_fraction"
    )
    if float(abstention) != 1.0:
        raise RuntimeError("Stage F abstention identity changed")
    for name, expected in frozen["sha256"].items():
        if file_sha256(frozen[name]) != str(expected):
            raise RuntimeError(f"Stage F protected artifact changed: {name}")
    for path, expected in frozen["checkpoints_sha256"].items():
        if file_sha256(path) != str(expected):
            raise RuntimeError(f"Stage F protected checkpoint changed: {path}")


class BoundedResidualSignSafeCalibrator(nn.Module):
    """Learn a bounded multiplier around the immutable sign-safe D1-B baseline."""

    def __init__(
        self,
        *,
        hidden_dim: int = 24,
        num_layers: int = 2,
        bond_type_embedding_dim: int = 4,
        element_pair_embedding_dim: int = 4,
        element_pair_buckets: int = 32,
        time_embedding_dim: int = 4,
        min_multiplier: float = 0.5,
        max_multiplier: float = 1.5,
        epsilon: float = 1.0e-8,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if not 1 <= int(num_layers) <= 2:
            raise ValueError("Stage G MLP must contain one or two linear layers")
        if not 1 <= int(hidden_dim) <= 64:
            raise ValueError("Stage G hidden dimension must be in [1, 64]")
        if float(dropout) != 0.0:
            raise ValueError("Stage G dropout is fixed to zero")
        if int(time_embedding_dim) != 4:
            raise ValueError("Stage G uses the fixed four-component time embedding")
        if not 0.0 < float(min_multiplier) < 1.0 < float(max_multiplier):
            raise ValueError("Stage G multiplier bounds must satisfy 0 < min < 1 < max")
        if float(max_multiplier) <= float(min_multiplier):
            raise ValueError("Stage G max_multiplier must exceed min_multiplier")
        self.min_multiplier = float(min_multiplier)
        self.max_multiplier = float(max_multiplier)
        self.epsilon = float(epsilon)
        self.num_layers = int(num_layers)
        self.element_pair_buckets = int(element_pair_buckets)
        self.anchor_logit = math.log(
            (1.0 - self.min_multiplier)
            / (self.max_multiplier - 1.0)
        )
        self.anchor = nn.Parameter(torch.tensor(self.anchor_logit, dtype=dtype))
        self.bond_type_embedding = nn.Embedding(
            len(BOND_TYPE_IDS), int(bond_type_embedding_dim), dtype=dtype
        )
        self.element_pair_embedding = nn.Embedding(
            self.element_pair_buckets, int(element_pair_embedding_dim), dtype=dtype
        )
        input_dim = (
            len(NUMERIC_FEATURES)
            + 2
            + int(bond_type_embedding_dim)
            + int(element_pair_embedding_dim)
            + int(time_embedding_dim)
        )
        if self.num_layers == 1:
            self.feature_mlp = nn.Sequential(nn.Linear(input_dim, 1, dtype=dtype))
        else:
            self.feature_mlp = nn.Sequential(
                nn.Linear(input_dim, int(hidden_dim), dtype=dtype),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), 1, dtype=dtype),
            )
        final = self.feature_mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def _time_embedding(time_value: Tensor) -> Tensor:
        return torch.stack(
            [
                torch.sin(math.pi * time_value),
                torch.cos(math.pi * time_value),
                torch.sin(2.0 * math.pi * time_value),
                torch.cos(2.0 * math.pi * time_value),
            ],
            dim=-1,
        )

    def residual_logit(self, features: Mapping[str, Tensor]) -> Tensor:
        dtype, device = self.anchor.dtype, self.anchor.device
        numeric = torch.stack(
            [
                torch.nan_to_num(
                    torch.as_tensor(features[name], device=device, dtype=dtype),
                    nan=0.0,
                    posinf=1.0e6,
                    neginf=-1.0e6,
                )
                for name in NUMERIC_FEATURES
            ],
            dim=-1,
        )
        flags = torch.stack(
            [torch.as_tensor(features[name], device=device, dtype=dtype) for name in ("ring", "aromatic")],
            dim=-1,
        )
        bond_type = self.bond_type_embedding(
            torch.as_tensor(features["bond_type_id"], device=device, dtype=torch.long).clamp(
                0, len(BOND_TYPE_IDS) - 1
            )
        )
        element_pair = self.element_pair_embedding(
            torch.as_tensor(features["element_pair_id"], device=device, dtype=torch.long).remainder(
                self.element_pair_buckets
            )
        )
        time = self._time_embedding(
            torch.nan_to_num(
                torch.as_tensor(features["time_value"], device=device, dtype=dtype), nan=0.0
            )
        )
        value = self.feature_mlp(
            torch.cat([numeric, flags, bond_type, element_pair, time], dim=-1)
        ).squeeze(-1)
        return torch.nan_to_num(value, nan=0.0, posinf=20.0, neginf=-20.0) + self.anchor

    def multiplier(self, features: Mapping[str, Tensor]) -> Tensor:
        unit = torch.sigmoid(self.residual_logit(features))
        value = self.min_multiplier + (self.max_multiplier - self.min_multiplier) * unit
        return torch.nan_to_num(
            value,
            nan=1.0,
            posinf=self.max_multiplier,
            neginf=self.min_multiplier,
        ).clamp(self.min_multiplier, self.max_multiplier)

    def base_confidence(self, features: Mapping[str, Tensor]) -> Tensor:
        dtype, device = self.anchor.dtype, self.anchor.device
        if "original_confidence" in features:
            original = torch.as_tensor(features["original_confidence"], device=device, dtype=dtype)
        else:
            logit = torch.as_tensor(features["confidence_logit"], device=device, dtype=dtype)
            original = torch.sigmoid(torch.nan_to_num(logit, nan=-100.0, posinf=100.0, neginf=-100.0))
        original = torch.nan_to_num(original, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        stable = original.clamp(self.epsilon, 1.0 - self.epsilon)
        original = torch.where(
            original <= 0.0,
            torch.zeros_like(original),
            torch.where(original >= 1.0, torch.ones_like(original), stable),
        )
        mask = torch.as_tensor(features["sign_safe_mask"], device=device, dtype=torch.bool)
        return torch.where(mask, original, torch.zeros_like(original))

    def forward_components(self, features: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        base = self.base_confidence(features)
        multiplier = self.multiplier(features)
        final = torch.nan_to_num(base * multiplier, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return final, multiplier, base

    def forward(self, features: Mapping[str, Tensor]) -> Tensor:
        return self.forward_components(features)[0]


def build_stage_g_manifest(
    molecule_ids: Sequence[str],
    *,
    checkpoint_sha256: str,
    frozen_identities: Mapping[str, Any],
    builder_batch_size: int,
    seed: int = 42,
    fit_fraction: float = 0.80,
    smoke: bool = False,
) -> dict[str, Any]:
    manifest = build_calibration_manifest(
        molecule_ids,
        checkpoint_sha256=checkpoint_sha256,
        frozen_identities=frozen_identities,
        seed=seed,
        fit_fraction=fit_fraction,
        smoke=smoke,
    )
    manifest.update(
        {
            "schema_version": MANIFEST_SCHEMA,
            "deployment_features": list(STAGE_G_DEPLOYMENT_FEATURES),
            "source_feature_used": False,
            "builder_batch_size": int(builder_batch_size),
            "stage_f_formal_decision_unchanged": STAGE_F_DECISION,
        }
    )
    identity = {
        "base_manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "deployment_features": manifest["deployment_features"],
        "builder_batch_size": manifest["builder_batch_size"],
        "stage_f_formal_decision_unchanged": manifest[
            "stage_f_formal_decision_unchanged"
        ],
    }
    manifest["stage_g_manifest_identity_sha256"] = canonical_sha256(identity)
    return manifest


def validate_stage_g_manifest(manifest: Mapping[str, Any]) -> None:
    adapted = dict(manifest)
    adapted["schema_version"] = "ecir-mvr-stage-e0-calibration-manifest-v1"
    validate_calibration_manifest(adapted)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unexpected Stage G manifest schema")
    if manifest.get("deployment_features") != list(STAGE_G_DEPLOYMENT_FEATURES):
        raise ValueError("Stage G deployment feature identity changed")
    if manifest.get("source_feature_used") is not False:
        raise ValueError("Stage G source-label leakage detected")
    if int(manifest.get("builder_batch_size", 0)) < 1:
        raise ValueError("Stage G builder batch size must be positive")
    stage_g_identity = {
        "base_manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "deployment_features": manifest["deployment_features"],
        "builder_batch_size": manifest["builder_batch_size"],
        "stage_f_formal_decision_unchanged": manifest[
            "stage_f_formal_decision_unchanged"
        ],
    }
    if manifest.get("stage_g_manifest_identity_sha256") != canonical_sha256(stage_g_identity):
        raise ValueError("Stage G manifest identity mismatch")


def validate_stage_g_frame(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> None:
    validate_stage_g_manifest(manifest)
    required = {
        "schema_version",
        "split",
        "molecule_id",
        "molecule_code",
        "record_id",
        "rollout_step",
        "bond_index",
        *STAGE_G_DEPLOYMENT_FEATURES,
        "sign_safe_mask",
        "target_residual",
        "optimal_scale",
        "scale_weight",
        "wrong_sign",
        "zero_target",
        "already_valid_unsafe",
        "beneficial",
        "training_only",
        "validation_records_read",
        "test_records_read",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Stage G calibration data missing columns: {sorted(missing)}")
    if set(frame.schema_version.astype(str)) != {DATA_SCHEMA}:
        raise ValueError("unexpected Stage G calibration data schema")
    if not set(frame.split.astype(str)).issubset({"fit", "internal_check"}):
        raise ValueError("Stage G calibration data contains validation/test rows")
    if set(frame.validation_records_read.astype(int)) != {0} or set(
        frame.test_records_read.astype(int)
    ) != {0}:
        raise ValueError("Stage G calibration data is not validation/test isolated")
    if "source" in frame.columns or "severity" in frame.columns:
        raise ValueError("Stage G calibration data persists a forbidden source label")
    expected = set(manifest["fit_molecule_ids"]) | set(manifest["internal_check_molecule_ids"])
    if set(frame.molecule_id.astype(str)) != expected:
        raise ValueError("Stage G calibration molecule identity differs from manifest")
    numeric_columns = [
        *NUMERIC_FEATURES,
        "time_value",
        "original_confidence",
        "target_residual",
        "optimal_scale",
        "scale_weight",
    ]
    numeric = frame[numeric_columns].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Stage G calibration data contains invalid numeric values")
    if not frame.optimal_scale.between(0.0, 1.0).all() or not frame.original_confidence.between(0.0, 1.0).all():
        raise ValueError("Stage G calibration confidence/scale is out of range")


def _continuous_tensor(frame: pd.DataFrame, name: str, dtype: np.dtype) -> Tensor:
    array = np.ascontiguousarray(frame[name].to_numpy(dtype=dtype, copy=True))
    return torch.from_numpy(array).contiguous()


def dataframe_stage_g_tensors(frame: pd.DataFrame) -> dict[str, Tensor]:
    """Convert the complete frame once; no pandas work is needed inside training steps."""

    tensors: dict[str, Tensor] = {
        name: _continuous_tensor(frame, name, np.float32)
        for name in (*NUMERIC_FEATURES, "time_value", "original_confidence")
    }
    tensors.update(
        {
            name: _continuous_tensor(frame, name, np.int64)
            for name in ("bond_type_id", "element_pair_id", "molecule_code")
        }
    )
    tensors.update(
        {
            name: _continuous_tensor(frame, name, np.bool_)
            for name in (
                "ring",
                "aromatic",
                "sign_safe_mask",
                "wrong_sign",
                "zero_target",
                "already_valid_unsafe",
                "beneficial",
            )
        }
    )
    tensors.update(
        {
            name: _continuous_tensor(frame, name, np.float32)
            for name in ("optimal_scale", "scale_weight")
        }
    )
    return tensors


def tensor_bundle_nbytes(tensors: Mapping[str, Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in tensors.values())


def move_tensor_bundle(
    tensors: Mapping[str, Tensor], device: torch.device, *, non_blocking: bool = False
) -> dict[str, Tensor]:
    return {
        name: value.to(device=device, non_blocking=non_blocking).contiguous()
        for name, value in tensors.items()
    }


def pin_tensor_bundle(tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.pin_memory() if value.device.type == "cpu" else value for name, value in tensors.items()}


def feature_view(tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensors[name] for name in (*STAGE_G_DEPLOYMENT_FEATURES, "sign_safe_mask")}


def sampled_molecule_ranking_loss(
    confidence: Tensor,
    optimal_scale: Tensor,
    molecule_ids: Tensor,
    *,
    margin: float = 0.05,
) -> Tensor:
    """Vectorized molecule-local ranking using adjacent rows after a stable group sort."""

    if confidence.numel() < 2:
        return confidence.sum() * 0.0
    order = torch.argsort(molecule_ids, stable=True)
    ids = molecule_ids[order]
    values = confidence[order]
    targets = optimal_scale[order]
    same = ids[1:] == ids[:-1]
    difference = targets[1:] - targets[:-1]
    useful = same & (difference.abs() > float(margin))
    if not bool(useful.any()):
        return confidence.sum() * 0.0
    signed = torch.sign(difference[useful]) * (values[1:][useful] - values[:-1][useful])
    return F.softplus(-signed).mean()


def stage_g_loss(
    confidence: Tensor,
    multiplier: Tensor,
    *,
    optimal_scale: Tensor,
    scale_weight: Tensor,
    wrong_sign: Tensor,
    false_positive: Tensor,
    beneficial: Tensor,
    molecule_ids: Tensor,
    lambda_wrong_sign: float,
    lambda_false_positive: float,
    lambda_overactivation: float,
    lambda_rank: float,
    lambda_beneficial_recall: float,
    lambda_multiplier_identity: float,
    beneficial_confidence_floor: float,
    smooth_l1_beta: float = 0.1,
    rank_margin: float = 0.05,
) -> tuple[Tensor, dict[str, Tensor]]:
    target = optimal_scale.to(confidence)
    weights = scale_weight.to(confidence)
    scale = F.smooth_l1_loss(
        confidence, target, reduction="none", beta=float(smooth_l1_beta)
    )
    scale = (scale * weights).sum() / weights.sum().clamp_min(1.0e-12)
    wrong_mask = wrong_sign.to(device=confidence.device, dtype=torch.bool)
    false_mask = false_positive.to(device=confidence.device, dtype=torch.bool)
    beneficial_mask = beneficial.to(device=confidence.device, dtype=torch.bool)
    wrong = confidence[wrong_mask].mean() if bool(wrong_mask.any()) else confidence.sum() * 0.0
    false = confidence[false_mask].mean() if bool(false_mask.any()) else confidence.sum() * 0.0
    over = F.relu(confidence - target).mean()
    beneficial_recall = (
        F.relu(float(beneficial_confidence_floor) - confidence[beneficial_mask]).mean()
        if bool(beneficial_mask.any())
        else confidence.sum() * 0.0
    )
    identity = (multiplier - 1.0).square().mean()
    rank = sampled_molecule_ranking_loss(
        confidence, target, molecule_ids.to(confidence.device), margin=rank_margin
    )
    total = (
        scale
        + float(lambda_wrong_sign) * wrong
        + float(lambda_false_positive) * false
        + float(lambda_overactivation) * over
        + float(lambda_rank) * rank
        + float(lambda_beneficial_recall) * beneficial_recall
        + float(lambda_multiplier_identity) * identity
    )
    return total, {
        "scale": scale,
        "wrong_sign": wrong,
        "false_positive": false,
        "overactivation": over,
        "rank": rank,
        "beneficial_recall_loss": beneficial_recall,
        "multiplier_identity": identity,
        "total": total,
    }


def selection_metrics(
    frame: pd.DataFrame,
    confidence: np.ndarray,
    multiplier: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | bool]:
    wrong = frame.wrong_sign.to_numpy(bool)
    false = frame.zero_target.to_numpy(bool) | frame.already_valid_unsafe.to_numpy(bool)
    beneficial = frame.beneficial.to_numpy(bool)
    active = confidence >= float(threshold)
    cosine = frame.cartesian_bond_cosine.to_numpy(float)
    metrics: dict[str, float | bool] = {
        "beneficial_repair_recall": float((active & beneficial).sum() / max(beneficial.sum(), 1)),
        "optimal_scale_mae": float(np.abs(confidence - frame.optimal_scale.to_numpy(float)).mean()),
        "wrong_sign_activation": float(confidence[wrong].mean()) if wrong.any() else 0.0,
        "false_positive_activation": float(confidence[false].mean()) if false.any() else 0.0,
        "multiplier_identity_error": float(np.square(multiplier - 1.0).mean()),
        "cancellation_proxy": float(confidence[cosine < 0.0].mean()) if (cosine < 0.0).any() else 0.0,
        "abstention_fraction": float((~active).mean()),
        "mean_final_confidence": float(confidence.mean()),
        "multiplier_mean": float(multiplier.mean()),
    }
    for prefix, values in (("confidence", confidence), ("multiplier", multiplier)):
        for percentile in (5, 25, 50, 75, 95):
            metrics[f"{prefix}_p{percentile:02d}"] = float(np.percentile(values, percentile))
    metrics["collapsed"] = bool(metrics["abstention_fraction"] >= 0.95)
    return metrics


def checkpoint_selection_priority(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool(metrics["collapsed"]),
        -float(metrics["beneficial_repair_recall"]),
        float(metrics["optimal_scale_mae"]),
        float(metrics["wrong_sign_activation"]),
        float(metrics["false_positive_activation"]),
        float(metrics["multiplier_identity_error"]),
        float(metrics["cancellation_proxy"]),
    )


def select_stage_g_checkpoint(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [row for row in rows if not bool(row["collapsed"])]
    if not eligible:
        return None
    return min(eligible, key=checkpoint_selection_priority)


def stage_g_decision(criteria: Mapping[str, bool], *, collapsed: bool, harms: bool) -> str:
    if collapsed:
        return "STAGE_G_COLLAPSED"
    if all(bool(value) for value in criteria.values()):
        return "STAGE_G_BOUNDED_RESIDUAL_PASS"
    if harms:
        return "STAGE_G_HARMS"
    return "STAGE_G_NO_ADDED_VALUE"


def calibrator_identity_payload(
    calibrator: BoundedResidualSignSafeCalibrator,
    *,
    model_config: Mapping[str, Any],
    checkpoint_sha256: str,
    training_molecule_identity_sha256: str,
    manifest_identity_sha256: str,
    selected_step: int,
    selection: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    state_hash = canonical_sha256(
        {
            name: value.detach().cpu().tolist()
            for name, value in sorted(calibrator.state_dict().items())
        }
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "model_config": dict(model_config),
        "state_sha256": state_hash,
        "checkpoint_sha256": str(checkpoint_sha256),
        "training_molecule_identity_sha256": str(training_molecule_identity_sha256),
        "manifest_identity_sha256": str(manifest_identity_sha256),
        "selected_step": int(selected_step),
    }
    return {
        **identity,
        "calibrator_identity_sha256": canonical_sha256(identity),
        "selection_metrics": dict(selection),
        "validation_used_for_fit_or_selection": False,
        "test_records_read": 0,
        "smoke": bool(smoke),
    }


def load_stage_g_calibrator(
    checkpoint_path: str | Path,
    payload: Mapping[str, Any],
    *,
    device: torch.device,
) -> BoundedResidualSignSafeCalibrator:
    identity = {
        name: payload[name]
        for name in (
            "schema_version",
            "model_config",
            "state_sha256",
            "checkpoint_sha256",
            "training_molecule_identity_sha256",
            "manifest_identity_sha256",
            "selected_step",
        )
    }
    if payload.get("calibrator_identity_sha256") != canonical_sha256(identity):
        raise ValueError("Stage G calibrator identity mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint.get("step", -1)) != int(payload["selected_step"]):
        raise ValueError("Stage G selected checkpoint step mismatch")
    calibrator = BoundedResidualSignSafeCalibrator(**payload["model_config"]).to(device)
    calibrator.load_state_dict(checkpoint["calibrator_state_dict"], strict=True)
    state_hash = canonical_sha256(
        {
            name: value.detach().cpu().tolist()
            for name, value in sorted(calibrator.state_dict().items())
        }
    )
    if state_hash != payload["state_sha256"]:
        raise ValueError("Stage G calibrator state identity mismatch")
    calibrator.eval()
    return calibrator
