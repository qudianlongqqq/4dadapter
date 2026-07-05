#!/usr/bin/env python
"""Print resolved FlexBond hyperparameters and validation-loss checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def _last_and_best_metric(run_dir: Path, metric_name: str) -> dict[str, float | int | None]:
    values = []
    for path in run_dir.rglob("metrics.csv"):
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = row.get(metric_name)
                if value not in (None, ""):
                    values.append((int(float(row.get("step") or -1)), float(value)))
    if not values:
        return {"last_step": None, "last_value": None, "best_step": None, "best_value": None}
    values.sort()
    best_step, best_value = min(values, key=lambda item: item[1])
    last_step, last_value = values[-1]
    return {
        "last_step": last_step,
        "last_value": last_value,
        "best_step": best_step,
        "best_value": best_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs_flexbond_formal_small", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configs = sorted(args.root.rglob("config.resolved.yaml")) if args.root.exists() else []
    if not configs:
        raise SystemExit(f"No config.resolved.yaml files found under {args.root}")
    rows = []
    for path in configs:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        model = config.get("model", {})
        data = config.get("data", {})
        trainer = config.get("trainer", {})
        validation = _last_and_best_metric(path.parent, "val/final_loss")
        row = {
            "run_dir": str(path.parent),
            "mode": model.get("mode"),
            "correction_scale": model.get("correction_scale"),
            "q_loss_weight": model.get("q_loss_weight"),
            "corr_reg_weight": model.get("corr_reg_weight"),
            "ridge_eps": model.get("ridge_eps"),
            "max_q_norm": model.get("max_q_norm"),
            "max_condition": model.get("max_condition"),
            "learning_rate": model.get("lr"),
            "batch_size": data.get("batch_size"),
            "accumulate_grad_batches": trainer.get("accumulate_grad_batches"),
            "effective_batch_size": (
                int(data.get("batch_size", 0))
                * int(trainer.get("accumulate_grad_batches", 1))
            ),
            "max_steps": trainer.get("max_steps"),
            **{f"val_final_{key}": value for key, value in validation.items()},
        }
        rows.append(row)
    rendered = json.dumps(rows, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
