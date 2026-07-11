#!/usr/bin/env python
"""Read the completed small FlexBond-4D run without retraining it.

The five fixed fields define this round's matched budget. Optional fields are
copied from the old resolved config when present; missing optional evidence is
reported and never blocks the new 5k experiment.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "logs_flexbond_formal_small/flexbond4d_hybrid_5k"
FIXED = {
    "max_steps": 5000,
    "batch_size": 4,
    "accumulate_grad_batches": 2,
    "effective_batch_size": 8,
    "learning_rate": 0.0002,
}


def _nested(data, *paths, default=None):
    for path in paths:
        value = data
        try:
            for key in path.split("."):
                value = value[key]
            return value
        except (KeyError, TypeError):
            continue
    return default


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"step[=_-]?(\d+)", path.name, re.I)
    filename_step = int(match.group(1)) if match else 0
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload.get("global_step", filename_step)) if isinstance(payload, dict) else filename_step
    except Exception:
        return filename_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output_json", type=Path, default=ROOT / "reports/reference_4d_training_budget.json")
    parser.add_argument("--output_md", type=Path, default=ROOT / "reports/reference_4d_training_budget.md")
    args = parser.parse_args()
    run = args.reference_run.expanduser().resolve()
    config_path = run / "config.resolved.yaml"
    config = {}
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    checkpoints = sorted((run / "checkpoints").glob("*.ckpt"), key=_checkpoint_step) if run.is_dir() else []
    latest = checkpoints[-1] if checkpoints else None

    optional_sources = {
        "t_min": ("time_sampling.t_min", "time.t_min"),
        "t_max": ("time_sampling.t_max", "time.t_max"),
        "hidden_dim": ("model.hidden_dim", "model_args.hidden_dim"),
        "edge_hidden_dim": ("model.edge_hidden_dim", "model_args.edge_hidden_dim"),
        "num_layers": ("model.num_layers", "model_args.num_layers"),
        "optimizer": ("optimizer.type", "model.optimizer", "model_args.optimizer_type"),
        "scheduler": ("optimizer.scheduler", "model.scheduler", "model_args.lr_scheduler_type"),
        "precision": ("trainer.precision", "trainer_args.precision"),
        "train_data": ("data.cache_dir", "datamodule_args.cache_dir"),
        "val_data": ("data.cache_dir", "datamodule_args.cache_dir"),
        "seed": ("seed",),
        "validation_frequency": ("trainer.val_check_interval", "trainer_args.val_check_interval"),
    }
    fallback_config = yaml.safe_load(
        (ROOT / "configs/global_coupled_4d_local025_matched.yaml").read_text(encoding="utf-8")
    )
    fallback = {
        "t_min": fallback_config["time_sampling"]["t_min"],
        "t_max": fallback_config["time_sampling"]["t_max"],
        "hidden_dim": fallback_config["model"]["hidden_dim"],
        "edge_hidden_dim": fallback_config["model"]["edge_hidden_dim"],
        "num_layers": fallback_config["model"]["num_layers"],
        "optimizer": "AdamW",
        "scheduler": fallback_config["optimizer"].get("scheduler", "none"),
        "precision": str(fallback_config["trainer"].get("precision", "32-true")),
        "train_data": fallback_config["data"]["cache_dir"],
        "val_data": fallback_config["data"]["cache_dir"],
        "seed": fallback_config["seed"],
        "validation_frequency": fallback_config["trainer"]["val_check_interval"],
    }
    optional, missing = {}, []
    for field, paths in optional_sources.items():
        value = _nested(config, *paths)
        if value is None:
            value = fallback[field]
            missing.append(field)
        optional[field] = value
    payload = {
        "reference_run": str(run),
        "reference_status": "FOUND" if run.is_dir() else "OLD_RESULT_MISSING",
        "config_path": str(config_path) if config_path.is_file() else "",
        "checkpoint_path": str(latest or ""),
        "checkpoint_global_step": _checkpoint_step(latest) if latest else 0,
        **FIXED,
        **optional,
        "optional_fields_using_global4d_fallback": missing,
        "formal_training_allowed": True,
        "old_model_retraining_allowed": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Reference FlexBond-4D small-run budget", "",
             f"Reference status: **{payload['reference_status']}**", "",
             "> The old model is read-only and is never retrained by the Global Coupled 4D pipeline.", "",
             "## Fixed matched budget", ""]
    lines.extend(f"- `{key}`: `{value}`" for key, value in FIXED.items())
    lines.extend(["", "## Optional fields", ""])
    for key, value in optional.items():
        suffix = " (Global4D fallback; old field missing)" if key in missing else " (old resolved config)"
        lines.append(f"- `{key}`: `{value}`{suffix}")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("reference_status", "max_steps", "batch_size", "accumulate_grad_batches", "learning_rate", "formal_training_allowed")}, indent=2))


if __name__ == "__main__":
    main()
