"""Fail-closed factory for applying frozen V7 to formal D1-B checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from .mvr_v2_bac import MCVRBACModel
from .mvr_v7_constraint_specific import MCVRConstraintSpecificHybrid


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_v7_formal_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "mcvr-v7-formal-large-wrapper-v1":
        raise RuntimeError("V7 formal wrapper schema changed")
    isolation = {
        "checkpoint_or_config_selected_from_test": False,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
    }
    for key, expected in isolation.items():
        if config.get(key) != expected:
            raise RuntimeError(f"V7 formal wrapper isolation field changed: {key}")
    return config


def build_v7_formal_model(
    checkpoint: Mapping[str, Any],
    wrapper_config: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> MCVRConstraintSpecificHybrid:
    prior_contract = wrapper_config["prior"]
    if checkpoint.get("schema_version") not in set(
        prior_contract["allowed_checkpoint_schemas"]
    ):
        raise RuntimeError("V7 formal prior checkpoint schema changed")
    if checkpoint.get("model_type") != prior_contract["model_type"]:
        raise RuntimeError("V7 formal prior model type changed")
    if int(checkpoint.get("step", -1)) != int(prior_contract["completed_steps"]):
        raise RuntimeError("V7 formal prior is not the completed checkpoint")
    model_config = dict(checkpoint["config"]["model"])
    forbidden = {
        "bac_mode",
        "bac_constraint_scale",
        "bac_active_constraint_normalization",
    }
    if forbidden & set(model_config):
        raise RuntimeError("V7 formal prior unexpectedly contains learned BAC modules")
    prior = MCVRBACModel(**model_config)
    if prior.has_bac_modules:
        raise RuntimeError("V7 formal prior constructed learned BAC modules")
    incompatible = prior.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("V7 formal prior strict-load failed")
    settings = dict(wrapper_config["v7"])
    jacobian = settings.pop("jacobian_config")
    model = MCVRConstraintSpecificHybrid(
        prior, jacobian_config=jacobian, **settings
    ).to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("V7 formal wrapper unexpectedly has trainable parameters")
    return model
