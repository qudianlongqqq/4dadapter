#!/usr/bin/env python
"""Validate only the fixed, non-negotiable 5k budget fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


EXPECTED = {
    "max_steps": 5000,
    "batch_size": 4,
    "accumulate_grad_batches": 2,
    "effective_batch_size": 8,
    "learning_rate": 0.0002,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("reports/reference_4d_training_budget.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/global_coupled_4d_local025_matched.yaml"))
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    actual = {
        "max_steps": config["trainer"]["max_steps"],
        "batch_size": config["data"]["batch_size"],
        "accumulate_grad_batches": config["trainer"]["accumulate_grad_batches"],
        "effective_batch_size": config["data"]["batch_size"] * config["trainer"]["accumulate_grad_batches"],
        "learning_rate": config["optimizer"]["lr"],
    }
    differences = [(key, EXPECTED[key], actual[key]) for key in EXPECTED if actual[key] != EXPECTED[key]]
    reference_differences = [(key, EXPECTED[key], reference.get(key)) for key in EXPECTED if reference.get(key) != EXPECTED[key]]
    if differences or reference_differences:
        print("TRAINING_BUDGET_MISMATCH")
        for key, expected, value in differences:
            print(f"- config {key}: expected={expected!r}, actual={value!r}")
        for key, expected, value in reference_differences:
            print(f"- extracted {key}: expected={expected!r}, actual={value!r}")
        raise SystemExit(2)
    print("TRAINING_BUDGET_MATCH_5K")
    optional_actual = {
        "t_min": config["time_sampling"]["t_min"],
        "t_max": config["time_sampling"]["t_max"],
        "hidden_dim": config["model"]["hidden_dim"],
        "edge_hidden_dim": config["model"]["edge_hidden_dim"],
        "num_layers": config["model"]["num_layers"],
        "optimizer": "AdamW",
        "scheduler": config["optimizer"].get("scheduler", "none"),
        "precision": str(config["trainer"].get("precision", "32-true")),
        "train_data": config["data"]["cache_dir"],
        "val_data": config["data"]["cache_dir"],
        "seed": config["seed"],
        "validation_frequency": config["trainer"]["val_check_interval"],
    }
    optional_differences = [
        (field, reference.get(field), value)
        for field, value in optional_actual.items()
        if str(reference.get(field)) != str(value)
    ]
    if optional_differences:
        print("Optional comparison differences (reported, non-blocking for this small experiment):")
        for field, old, new in optional_differences:
            print(f"- {field}: old={old!r}, global4d={new!r}")
    if reference.get("optional_fields_using_global4d_fallback"):
        print("Optional old fields missing; using documented Global4D fair-config fallbacks:")
        for field in reference["optional_fields_using_global4d_fallback"]:
            print(f"- {field}")


if __name__ == "__main__":
    main()
