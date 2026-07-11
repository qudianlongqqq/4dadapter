#!/usr/bin/env python
"""Fail-closed formal training budget validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("reports/reference_4d_training_budget.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/global_coupled_4d_local025_matched.yaml"))
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if reference.get("confidence") != "high" or reference.get("ambiguous"):
        print("TRAINING_BUDGET_MISMATCH")
        print("Reference formal run is not uniquely established with high confidence.")
        raise SystemExit(2)
    actual = {
        "max_steps": config["trainer"]["max_steps"],
        "batch_size": config["data"]["batch_size"],
        "accumulate_grad_batches": config["trainer"]["accumulate_grad_batches"],
        "effective_batch_size": config["data"]["batch_size"] * config["trainer"]["accumulate_grad_batches"],
        "learning_rate": config["optimizer"]["lr"],
        "scheduler": config["optimizer"].get("scheduler", "none"),
        "t_min": config["time_sampling"]["t_min"],
        "t_max": config["time_sampling"]["t_max"],
        "seed": config["seed"],
        "precision": str(config["trainer"].get("precision", "unknown")),
    }
    differences = []
    for key, value in actual.items():
        expected = reference.get(key)
        if expected in (None, "unknown", 0, 0.0) and key not in {"t_min", "t_max"}:
            differences.append((key, expected, value, "reference field unavailable"))
        elif str(expected).lower() != str(value).lower():
            try:
                same = abs(float(expected) - float(value)) < 1e-12
            except (TypeError, ValueError):
                same = False
            if not same:
                differences.append((key, expected, value, "different"))
    if differences:
        print("TRAINING_BUDGET_MISMATCH")
        for key, expected, value, reason in differences:
            print(f"- {key}: reference={expected!r}, global4d={value!r} ({reason})")
        raise SystemExit(2)
    print("TRAINING_BUDGET_MATCH")


if __name__ == "__main__":
    main()

