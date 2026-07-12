#!/usr/bin/env python
"""Write matched-budget and final-loss comparison for formal-large training."""

from __future__ import annotations

import csv
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import yaml

from etflow.formal_large import assert_matched_training_budgets


RUNS = {
    "cartesian": Path("logs_formal_large/cartesian_seed42_200k"),
    "global4d": Path("logs_formal_large/global4d_seed42_200k"),
}


def _latest_metrics(path: Path) -> dict:
    latest = {}
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            latest.update({key: value for key, value in row.items() if value not in (None, "")})
    return latest


def main() -> None:
    configs = {
        method: yaml.safe_load(
            (run / "config.resolved.yaml").read_text(encoding="utf-8")
        )
        for method, run in RUNS.items()
    }
    budget = assert_matched_training_budgets(configs)
    rows = []
    for method, run in RUNS.items():
        metrics = _latest_metrics(run / "metrics.csv")
        rows.append({
            "method": method,
            "train_loss": metrics.get("train/loss_epoch", metrics.get("train/loss_step")),
            "val_loss": metrics.get("val/loss", metrics.get("val/final_loss")),
            "config": str((run / "config.resolved.yaml").resolve()),
            "checkpoint": str((run / "checkpoints/step200000.ckpt").resolve()),
        })
    output = {"budget": budget, "methods": rows}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/formal_large_training_comparison.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    lines = ["# Formal-large training comparison", "", f"Budget: `{json.dumps(budget)}`", ""]
    lines += [
        f"- {row['method']}: train={row['train_loss']}, val={row['val_loss']}"
        for row in rows
    ]
    Path("reports/formal_large_training_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
