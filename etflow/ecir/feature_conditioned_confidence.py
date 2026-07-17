"""Feature-conditioned, sign-safe confidence calibration for frozen MCVR proposals."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .confidence_calibration import (
    DIAGNOSTIC_ALL_ONE,
    build_calibration_manifest,
    canonical_sha256,
    optimal_scale_targets,
    validate_calibration_manifest,
)
from .geometry import bond_lengths


SCHEMA_VERSION = "ecir-mvr-stage-f-feature-confidence-v1"
DATA_SCHEMA = "ecir-mvr-stage-f-calibration-data-v1"
MANIFEST_SCHEMA = "ecir-mvr-stage-f-calibration-manifest-v1"
DIAGNOSTIC_ORACLE_ONLY = DIAGNOSTIC_ALL_ONE
NUMERIC_FEATURES = (
    "confidence_logit",
    "uncertainty",
    "absolute_predicted_residual",
    "current_bond_length",
    "signed_lower_margin",
    "signed_upper_margin",
    "distance_to_valid_interval",
    "endpoint_degree",
    "adjacent_residual_mean",
    "adjacent_residual_max",
    "adjacent_confidence_mean",
    "adjacent_confidence_max",
    "cartesian_bond_cosine",
)
CATEGORICAL_FEATURES = ("ring", "aromatic", "bond_type_id", "element_pair_id")
DEPLOYMENT_FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "time_value")
FORBIDDEN_DEPLOYMENT_FEATURES = {
    "source", "severity", "target_residual", "optimal_scale", "minimal_target",
    "reference", "validation_label", "test_label",
}
BOND_TYPE_IDS = {
    "UNKNOWN": 0,
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
    "QUADRUPLE": 5,
}


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


def distance_to_valid_interval(length: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    length = torch.as_tensor(length)
    lower = torch.as_tensor(lower, device=length.device, dtype=length.dtype)
    upper = torch.as_tensor(upper, device=length.device, dtype=length.dtype)
    return torch.maximum(lower - length, length - upper).clamp_min(0.0)


def sign_validity_safe_mask(
    current_length: Tensor | Sequence[float],
    lower: Tensor | Sequence[float],
    upper: Tensor | Sequence[float],
    predicted_residual: Tensor | Sequence[float],
) -> Tensor:
    """Authorize only proposals that reduce invalidity or keep valid bonds valid."""

    length = torch.as_tensor(current_length)
    lower = torch.as_tensor(lower, device=length.device, dtype=length.dtype)
    upper = torch.as_tensor(upper, device=length.device, dtype=length.dtype)
    predicted = torch.as_tensor(predicted_residual, device=length.device, dtype=length.dtype)
    proposed = length + predicted
    before = distance_to_valid_interval(length, lower, upper)
    after = distance_to_valid_interval(proposed, lower, upper)
    currently_valid = (length >= lower) & (length <= upper)
    proposed_valid = (proposed >= lower) & (proposed <= upper)
    return torch.where(currently_valid, proposed_valid, after < before)


def encode_bond_type(value: str) -> int:
    return BOND_TYPE_IDS.get(str(value).upper(), BOND_TYPE_IDS["UNKNOWN"])


def encode_element_pair(left_atomic_number: int, right_atomic_number: int, *, buckets: int = 32) -> int:
    pair = f"{min(int(left_atomic_number), int(right_atomic_number))}-{max(int(left_atomic_number), int(right_atomic_number))}"
    return int.from_bytes(hashlib.sha256(pair.encode("ascii")).digest()[:8], "big") % int(buckets)


class FeatureConditionedConfidenceCalibrator(nn.Module):
    """Small bounded residual calibrator with a strictly positive global slope."""

    def __init__(
        self,
        *,
        hidden_dim: int = 24,
        num_layers: int = 2,
        bond_type_embedding_dim: int = 4,
        element_pair_embedding_dim: int = 4,
        element_pair_buckets: int = 32,
        time_embedding_dim: int = 4,
        max_bias: float = 1.0,
        epsilon: float = 1.0e-8,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if not 1 <= int(num_layers) <= 2:
            raise ValueError("Stage F MLP must contain one or two linear layers")
        if not 1 <= int(hidden_dim) <= 32:
            raise ValueError("Stage F hidden dimension must be in [1, 32]")
        if float(dropout) != 0.0:
            raise ValueError("Stage F dropout is fixed to zero")
        if int(time_embedding_dim) != 4:
            raise ValueError("Stage F uses the fixed four-component time embedding")
        self.epsilon = float(epsilon)
        self.max_bias = float(max_bias)
        self.num_layers = int(num_layers)
        self.element_pair_buckets = int(element_pair_buckets)
        self.raw_a = nn.Parameter(torch.tensor(_inverse_softplus(1.0 - epsilon), dtype=dtype))
        self.bond_type_embedding = nn.Embedding(len(BOND_TYPE_IDS), int(bond_type_embedding_dim), dtype=dtype)
        self.element_pair_embedding = nn.Embedding(
            self.element_pair_buckets, int(element_pair_embedding_dim), dtype=dtype
        )
        input_dim = (
            len(NUMERIC_FEATURES) + 2 + int(bond_type_embedding_dim)
            + int(element_pair_embedding_dim) + int(time_embedding_dim)
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

    @property
    def a(self) -> Tensor:
        return F.softplus(self.raw_a) + self.epsilon

    @staticmethod
    def _time_embedding(time_value: Tensor) -> Tensor:
        time_value = torch.as_tensor(time_value)
        return torch.stack([
            torch.sin(math.pi * time_value), torch.cos(math.pi * time_value),
            torch.sin(2.0 * math.pi * time_value), torch.cos(2.0 * math.pi * time_value),
        ], dim=-1)

    def feature_bias(self, features: Mapping[str, Tensor]) -> Tensor:
        dtype, device = self.raw_a.dtype, self.raw_a.device
        numeric = torch.stack([
            torch.as_tensor(features[name], device=device, dtype=dtype)
            for name in NUMERIC_FEATURES
        ], dim=-1)
        flags = torch.stack([
            torch.as_tensor(features[name], device=device, dtype=dtype)
            for name in ("ring", "aromatic")
        ], dim=-1)
        bond_type = self.bond_type_embedding(
            torch.as_tensor(features["bond_type_id"], device=device, dtype=torch.long)
        )
        element_pair = self.element_pair_embedding(
            torch.as_tensor(features["element_pair_id"], device=device, dtype=torch.long)
        )
        time = self._time_embedding(
            torch.as_tensor(features["time_value"], device=device, dtype=dtype)
        )
        raw_bias = self.feature_mlp(torch.cat([numeric, flags, bond_type, element_pair, time], dim=-1)).squeeze(-1)
        return self.max_bias * torch.tanh(raw_bias)

    def forward(self, features: Mapping[str, Tensor]) -> Tensor:
        logit = torch.as_tensor(
            features["confidence_logit"], device=self.raw_a.device, dtype=self.raw_a.dtype
        )
        mask = torch.as_tensor(
            features["sign_safe_mask"], device=self.raw_a.device, dtype=self.raw_a.dtype
        )
        return mask * torch.sigmoid(self.a * logit + self.feature_bias(features))


def dataframe_feature_tensors(frame: pd.DataFrame, *, device: torch.device | str = "cpu") -> dict[str, Tensor]:
    result = {
        name: torch.as_tensor(frame[name].to_numpy(dtype=np.float64, copy=True), device=device)
        for name in (*NUMERIC_FEATURES, "time_value")
    }
    result.update({
        name: torch.as_tensor(frame[name].to_numpy(dtype=np.int64, copy=True), device=device)
        for name in ("bond_type_id", "element_pair_id")
    })
    result.update({
        name: torch.as_tensor(frame[name].to_numpy(dtype=bool, copy=True), device=device)
        for name in ("ring", "aromatic", "sign_safe_mask")
    })
    return result


def build_stage_f_manifest(
    molecule_ids: Sequence[str], *, checkpoint_sha256: str,
    frozen_identities: Mapping[str, Any], seed: int = 42,
    fit_fraction: float = 0.80, smoke: bool = False,
) -> dict[str, Any]:
    manifest = build_calibration_manifest(
        molecule_ids, checkpoint_sha256=checkpoint_sha256,
        frozen_identities=frozen_identities, seed=seed,
        fit_fraction=fit_fraction, smoke=smoke,
    )
    manifest["schema_version"] = MANIFEST_SCHEMA
    manifest["deployment_features"] = list(DEPLOYMENT_FEATURES)
    manifest["source_feature_used"] = False
    identity = {
        name: manifest[name] for name in (
            "seed", "fit_fraction", "fit_molecule_ids", "internal_check_molecule_ids",
            "checkpoint_sha256", "frozen_identities",
        )
    }
    manifest["manifest_identity_sha256"] = canonical_sha256(identity)
    return manifest


def validate_stage_f_manifest(manifest: Mapping[str, Any]) -> None:
    adapted = dict(manifest)
    adapted["schema_version"] = "ecir-mvr-stage-e0-calibration-manifest-v1"
    validate_calibration_manifest(adapted)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unexpected Stage F manifest schema")
    if manifest.get("deployment_features") != list(DEPLOYMENT_FEATURES):
        raise ValueError("Stage F deployment feature identity changed")
    if manifest.get("source_feature_used") is not False:
        raise ValueError("Stage F source-label leakage detected")


def validate_stage_f_frame(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> None:
    validate_stage_f_manifest(manifest)
    required = {
        "schema_version", "split", "molecule_id", "record_id", "rollout_step",
        "bond_index", *DEPLOYMENT_FEATURES, "sign_safe_mask", "target_residual",
        "optimal_scale", "scale_weight", "wrong_sign", "zero_target",
        "already_valid_unsafe", "beneficial", "training_only", "validation_records_read",
        "test_records_read",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Stage F calibration data missing columns: {sorted(missing)}")
    if set(frame.schema_version.astype(str)) != {DATA_SCHEMA}:
        raise ValueError("unexpected Stage F calibration data schema")
    if not set(frame.split.astype(str)).issubset({"fit", "internal_check"}):
        raise ValueError("Stage F calibration data contains validation/test rows")
    if set(frame.validation_records_read.astype(int)) != {0} or set(frame.test_records_read.astype(int)) != {0}:
        raise ValueError("Stage F calibration data is not validation/test isolated")
    if "source" in frame.columns or "severity" in frame.columns:
        raise ValueError("Stage F calibration data persists a forbidden source label")
    expected = set(manifest["fit_molecule_ids"]) | set(manifest["internal_check_molecule_ids"])
    if set(frame.molecule_id.astype(str)) != expected:
        raise ValueError("Stage F calibration molecule identity differs from manifest")
    numeric = frame[[*NUMERIC_FEATURES, "time_value", "target_residual", "optimal_scale", "scale_weight"]].to_numpy(float)
    if not np.isfinite(numeric).all() or not frame.optimal_scale.between(0.0, 1.0).all():
        raise ValueError("Stage F calibration data contains invalid numeric values")


def molecule_pairwise_ranking_loss(
    confidence: Tensor, optimal_scale: Tensor, molecule_ids: Tensor,
    *, margin: float = 0.05, max_pairs_per_molecule: int = 256,
) -> Tensor:
    losses = []
    for molecule in torch.unique(molecule_ids, sorted=True):
        keep = torch.nonzero(molecule_ids == molecule, as_tuple=False).reshape(-1)
        if keep.numel() < 2:
            continue
        target = optimal_scale[keep]
        difference = target[:, None] - target[None, :]
        pairs = torch.nonzero(difference > float(margin), as_tuple=False)
        if not pairs.numel():
            continue
        pairs = pairs[:int(max_pairs_per_molecule)]
        high = confidence[keep[pairs[:, 0]]]
        low = confidence[keep[pairs[:, 1]]]
        losses.append(F.softplus(-(high - low)).mean())
    return torch.stack(losses).mean() if losses else confidence.sum() * 0.0


def stage_f_loss(
    confidence: Tensor, *, optimal_scale: Tensor, scale_weight: Tensor,
    wrong_sign: Tensor, false_positive: Tensor, molecule_ids: Tensor,
    lambda_wrong_sign: float, lambda_false_positive: float,
    lambda_overactivation: float, lambda_rank: float,
    smooth_l1_beta: float = 0.1, rank_margin: float = 0.05,
    max_rank_pairs_per_molecule: int = 256,
) -> tuple[Tensor, dict[str, Tensor]]:
    optimal_scale = optimal_scale.to(confidence)
    weights = scale_weight.to(confidence)
    scale = F.smooth_l1_loss(
        confidence, optimal_scale, reduction="none", beta=float(smooth_l1_beta)
    )
    scale = (scale * weights).sum() / weights.sum().clamp_min(1.0e-12)
    wrong_mask = wrong_sign.to(device=confidence.device, dtype=torch.bool)
    false_mask = false_positive.to(device=confidence.device, dtype=torch.bool)
    wrong = confidence[wrong_mask].mean() if wrong_mask.any() else confidence.sum() * 0.0
    false = confidence[false_mask].mean() if false_mask.any() else confidence.sum() * 0.0
    over = F.relu(confidence - optimal_scale).mean()
    rank = molecule_pairwise_ranking_loss(
        confidence, optimal_scale, molecule_ids.to(confidence.device),
        margin=rank_margin, max_pairs_per_molecule=max_rank_pairs_per_molecule,
    )
    total = (
        scale + float(lambda_wrong_sign) * wrong
        + float(lambda_false_positive) * false
        + float(lambda_overactivation) * over
        + float(lambda_rank) * rank
    )
    return total, {
        "scale": scale, "wrong_sign": wrong, "false_positive": false,
        "overactivation": over, "rank": rank, "total": total,
    }


def internal_check_priority(metrics: Mapping[str, float]) -> tuple[float, float, float, float]:
    return (
        float(metrics["wrong_sign_activation"]),
        float(metrics["false_positive_activation"]),
        float(metrics["optimal_scale_mae"]),
        -float(metrics["beneficial_correction_capture"]),
    )


def stage_f_decision(
    criteria: Mapping[str, bool], *, sign_safe_only_better: bool, harms: bool,
) -> str:
    if sign_safe_only_better:
        return "STAGE_F_SIGN_SAFE_ONLY_BETTER"
    if all(bool(value) for value in criteria.values()):
        return "STAGE_F_FEATURE_CONFIDENCE_PASS"
    if harms:
        return "STAGE_F_HARMS"
    return "STAGE_F_NO_ADDED_VALUE"


def calibrator_identity_payload(
    calibrator: FeatureConditionedConfidenceCalibrator, *, model_config: Mapping[str, Any],
    checkpoint_sha256: str, training_molecule_identity_sha256: str,
    manifest_identity_sha256: str, selected_step: int, selection_metrics: Mapping[str, float],
    smoke: bool,
) -> dict[str, Any]:
    state_hash = canonical_sha256({
        name: value.detach().cpu().tolist()
        for name, value in sorted(calibrator.state_dict().items())
    })
    identity = {
        "schema_version": SCHEMA_VERSION,
        "model_config": dict(model_config), "state_sha256": state_hash,
        "checkpoint_sha256": str(checkpoint_sha256),
        "training_molecule_identity_sha256": str(training_molecule_identity_sha256),
        "manifest_identity_sha256": str(manifest_identity_sha256),
        "selected_step": int(selected_step),
    }
    return {
        **identity, "calibrator_identity_sha256": canonical_sha256(identity),
        "a": float(calibrator.a.detach()),
        "selection_metrics": {name: float(value) for name, value in selection_metrics.items()},
        "validation_used_for_fit_or_selection": False, "test_records_read": 0,
        "smoke": bool(smoke),
    }


def load_feature_calibrator(
    checkpoint_path: str, payload: Mapping[str, Any], *, device: torch.device,
) -> FeatureConditionedConfidenceCalibrator:
    identity = {name: payload[name] for name in (
        "schema_version", "model_config", "state_sha256", "checkpoint_sha256",
        "training_molecule_identity_sha256", "manifest_identity_sha256", "selected_step",
    )}
    if payload.get("calibrator_identity_sha256") != canonical_sha256(identity):
        raise ValueError("Stage F calibrator identity mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("step") != payload["selected_step"]:
        raise ValueError("Stage F selected checkpoint step mismatch")
    calibrator = FeatureConditionedConfidenceCalibrator(**payload["model_config"]).to(device)
    calibrator.load_state_dict(checkpoint["calibrator_state_dict"], strict=True)
    state_hash = canonical_sha256({
        name: value.detach().cpu().tolist()
        for name, value in sorted(calibrator.state_dict().items())
    })
    if state_hash != payload["state_sha256"]:
        raise ValueError("Stage F calibrator state identity mismatch")
    calibrator.eval()
    return calibrator


def inference_feature_batch(
    *, current: Tensor, output: Mapping[str, Tensor], batch: Any,
    items: Sequence[Mapping[str, Any]], ptr: Sequence[int], validity: Any,
    time_value: float,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Build deployment features without target, reference, source, or validation labels."""

    bonds = output["bond_indices"]
    bond_graph = batch.batch[bonds[0]]
    logit = output["bond_confidence_logit"]
    predicted = output["bond_unattenuated_residual"]
    confidence = torch.sigmoid(logit)
    result: dict[str, list[Tensor]] = defaultdict(list)
    metadata: list[dict[str, Any]] = []
    for local, item in enumerate(items):
        keep = bond_graph == local
        indices = torch.nonzero(keep, as_tuple=False).reshape(-1)
        left_offset, right_offset = int(ptr[local]), int(ptr[local + 1])
        local_bonds = bonds[:, keep] - left_offset
        local_coordinates = current[left_offset:right_offset]
        lengths = bond_lengths(local_coordinates, local_bonds)
        prepared = validity._prepare(item["record"])
        atom, environments, _ = validity._environment(item["record"])
        stats_by_bond = {
            tuple(sorted(pair)): stat for pair, stat in zip(
                prepared["bonds"].t().tolist(), prepared["bond_stats"].tolist()
            )
        }
        ring_by_bond = {
            tuple(sorted(pair)): bool(value) for pair, value in zip(
                prepared["bonds"].t().tolist(), prepared["ring_mask"].tolist()
            )
        }
        degree = torch.bincount(local_bonds.reshape(-1), minlength=local_coordinates.shape[0])
        local_cartesian = output["v_cartesian_raw"][left_offset:right_offset]
        local_bond_correction = output["v_bond_correction"][left_offset:right_offset]
        local_predicted = predicted[keep]
        local_confidence = confidence[keep]
        pairs = [tuple(map(int, pair)) for pair in local_bonds.t().tolist()]
        for bond_index, global_index in enumerate(indices.tolist()):
            left, right = pairs[bond_index]
            key = tuple(sorted((left, right)))
            lower, upper = (float(value) for value in stats_by_bond[key][:2])
            length = lengths[bond_index]
            adjacent = [
                index for index, pair in enumerate(pairs)
                if index != bond_index and (left in pair or right in pair)
            ]
            if adjacent:
                adjacent_residual = local_predicted[adjacent].abs()
                adjacent_confidence = local_confidence[adjacent]
                residual_mean, residual_max = adjacent_residual.mean(), adjacent_residual.max()
                confidence_mean, confidence_max = adjacent_confidence.mean(), adjacent_confidence.max()
            else:
                residual_mean = residual_max = length.new_zeros(())
                confidence_mean = confidence_max = length.new_zeros(())
            bond_type, aromatic, metadata_ring = environments[key]
            left_z, right_z = int(atom[left][0]), int(atom[right][0])
            lower_tensor = length.new_tensor(lower)
            upper_tensor = length.new_tensor(upper)
            safe = sign_validity_safe_mask(
                length, lower_tensor, upper_tensor, local_predicted[bond_index]
            )
            cartesian_local_vector = local_cartesian[local_bonds[:, bond_index]].reshape(-1)
            bond_local_vector = local_bond_correction[local_bonds[:, bond_index]].reshape(-1)
            local_cosine = F.cosine_similarity(
                cartesian_local_vector.unsqueeze(0), bond_local_vector.unsqueeze(0), dim=-1,
                eps=1.0e-12,
            ).squeeze(0)
            values = {
                "confidence_logit": logit[global_index],
                "uncertainty": output["bond_uncertainty"][global_index],
                "absolute_predicted_residual": local_predicted[bond_index].abs(),
                "current_bond_length": length,
                "signed_lower_margin": length - lower_tensor,
                "signed_upper_margin": upper_tensor - length,
                "distance_to_valid_interval": distance_to_valid_interval(length, lower_tensor, upper_tensor),
                "endpoint_degree": torch.maximum(degree[left], degree[right]).to(length),
                "adjacent_residual_mean": residual_mean,
                "adjacent_residual_max": residual_max,
                "adjacent_confidence_mean": confidence_mean,
                "adjacent_confidence_max": confidence_max,
                "cartesian_bond_cosine": local_cosine,
                "ring": length.new_tensor(bool(ring_by_bond[key] or metadata_ring)),
                "aromatic": length.new_tensor(bool(aromatic)),
                "bond_type_id": torch.tensor(encode_bond_type(str(bond_type)), device=length.device),
                "element_pair_id": torch.tensor(encode_element_pair(left_z, right_z), device=length.device),
                "time_value": length.new_tensor(float(time_value)),
                "sign_safe_mask": safe,
            }
            for name, value in values.items():
                result[name].append(value)
            metadata.append({
                "molecule_id": str(item["row"].molecule_id),
                "record_id": str(item["row"].sample_id), "bond_index": bond_index,
                "atom_i": left, "atom_j": right, "valid_lower": lower,
                "valid_upper": upper, "bond_type": str(bond_type),
                "element_pair": f"{min(left_z, right_z)}-{max(left_z, right_z)}",
            })
    tensors = {
        name: torch.stack(values) if values else current.new_empty(0)
        for name, values in result.items()
    }
    return tensors, metadata
