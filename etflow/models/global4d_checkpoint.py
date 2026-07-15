"""Fail-closed checkpoint loading and explicit Global4D warm starts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from etflow.models.global_coupled_4d_flow import (
    GlobalCoupled4DFlowLightningModule,
)


def resolved_model_arguments(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve constructor arguments while preserving legacy strict semantics."""

    arguments = {
        **dict(config.get("model") or {}),
        **dict(config.get("loss") or {}),
        **dict(config.get("optimizer") or {}),
        **dict(config.get("time_sampling") or {}),
    }
    arguments.pop("scheduler", None)
    arguments.setdefault("fusion_mode", "strict_orthogonal")
    arguments.setdefault("internal_beta", 1.0)
    arguments.setdefault("joint_mode", "full_4d")
    return arguments


def checkpoint_hyperparameters(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hparams = payload.get("hyper_parameters")
    if not isinstance(hparams, Mapping):
        raise ValueError(f"Checkpoint has no hyper_parameters mapping: {path}")
    result = dict(hparams)
    result.setdefault("fusion_mode", "strict_orthogonal")
    result.setdefault("internal_beta", 1.0)
    return result


def checkpoint_fusion_identity(path: str | Path) -> dict[str, Any]:
    hparams = checkpoint_hyperparameters(path)
    return {
        "fusion_mode": str(hparams["fusion_mode"]),
        "internal_beta": float(hparams["internal_beta"]),
        "has_gate_head": str(hparams["fusion_mode"]) == "gated_additive",
    }


def _load_compatible_state(
    model: GlobalCoupled4DFlowLightningModule,
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    target = model.state_dict()
    loaded: dict[str, Any] = {}
    unexpected = []
    shape_mismatches = []
    for key, value in state_dict.items():
        if key not in target:
            unexpected.append(key)
        elif tuple(target[key].shape) != tuple(value.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(target[key].shape),
                }
            )
        else:
            loaded[key] = value
    missing = sorted(set(target).difference(loaded))
    if unexpected or shape_mismatches:
        raise RuntimeError(
            "Warm-start checkpoint is incompatible: "
            f"unexpected={unexpected}, shape_mismatches={shape_mismatches}"
        )
    model.load_state_dict(loaded, strict=False)
    return {
        "loaded_keys": sorted(loaded),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }


def warm_start_global4d(
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    *,
    initialize_missing_gate: bool,
    map_location: str | torch.device = "cpu",
) -> tuple[GlobalCoupled4DFlowLightningModule, dict[str, Any]]:
    """Instantiate target config and explicitly load reusable model weights."""

    arguments = resolved_model_arguments(config)
    target_mode = str(arguments["fusion_mode"])
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    source_hparams = dict(payload.get("hyper_parameters") or {})
    source_mode = str(source_hparams.get("fusion_mode", "strict_orthogonal"))
    model = GlobalCoupled4DFlowLightningModule(**arguments)
    report = _load_compatible_state(model, payload.get("state_dict") or {})
    gate_missing = [key for key in report["missing_keys"] if key.startswith("gate_head.")]
    other_missing = sorted(set(report["missing_keys"]).difference(gate_missing))
    if other_missing:
        raise RuntimeError(f"Warm start has non-gate missing keys: {other_missing}")
    if gate_missing and not initialize_missing_gate:
        raise RuntimeError(
            "Target gated_additive model requires a gate head absent from the "
            "checkpoint; pass --initialize_missing_gate explicitly"
        )
    if target_mode != "gated_additive" and gate_missing:
        raise RuntimeError("Non-gated target unexpectedly has missing gate parameters")
    report.update(
        {
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "source_fusion_mode": source_mode,
            "target_fusion_mode": target_mode,
            "initialized_gate_keys": gate_missing,
            "initialize_missing_gate": bool(initialize_missing_gate),
        }
    )
    return model, report


def load_global4d_for_inference(
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    *,
    map_location: str | torch.device,
    initialize_missing_gate: bool = False,
) -> tuple[GlobalCoupled4DFlowLightningModule, dict[str, Any]]:
    """Load a sampler model and reject config/checkpoint semantic mismatches."""

    requested = resolved_model_arguments(config)
    checkpoint_identity = checkpoint_fusion_identity(checkpoint_path)
    requested_mode = str(requested["fusion_mode"])
    requested_beta = float(requested["internal_beta"])
    same = (
        checkpoint_identity["fusion_mode"] == requested_mode
        and checkpoint_identity["internal_beta"] == requested_beta
    )
    if same:
        model = GlobalCoupled4DFlowLightningModule.load_from_checkpoint(
            checkpoint_path, map_location=map_location
        )
        return model, {
            **checkpoint_identity,
            "requested_fusion_mode": requested_mode,
            "requested_internal_beta": requested_beta,
            "warm_started_in_memory": False,
        }
    if not (
        initialize_missing_gate
        and requested_mode == "gated_additive"
        and not checkpoint_identity["has_gate_head"]
    ):
        raise RuntimeError(
            "Resolved config and checkpoint fusion semantics differ: "
            f"config=({requested_mode}, beta={requested_beta}), "
            f"checkpoint=({checkpoint_identity['fusion_mode']}, "
            f"beta={checkpoint_identity['internal_beta']})"
        )
    model, report = warm_start_global4d(
        checkpoint_path,
        config,
        initialize_missing_gate=True,
        map_location=map_location,
    )
    report["warm_started_in_memory"] = True
    return model, report
