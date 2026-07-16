#!/usr/bin/env python
"""Select the preregistered Medium Seed42 Schedule V4 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.commons.global_coupled_4d_sampling import atomic_json_save


PREREGISTERED_STEPS = (500, 1000, 1500, 2000, 3000, 5000, 7500, 10000)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(row) -> tuple:
    return (
        round(float(row.validity_delta), 6),
        -float(row.max_core_relative_improvement),
        float(row.mean_displacement),
        float(row.unseen_validity_delta),
        int(row.step),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = pd.read_csv(args.comparison)
    rows["preregistered_checkpoint"] = rows.step.isin(PREREGISTERED_STEPS)
    present = tuple(sorted(int(value) for value in rows.loc[rows.preregistered_checkpoint, "step"]))
    if present != PREREGISTERED_STEPS:
        raise RuntimeError(f"preregistered checkpoint set incomplete: {present}")
    eligible = rows[
        rows.preregistered_checkpoint
        & rows.accuracy_noninferior.astype(bool)
        & rows.safety_qualified.astype(bool)
    ]
    if eligible.empty:
        raise RuntimeError("no accuracy-, identity-, safety-, and high-flex-qualified checkpoint exists")

    metadata = json.loads(args.run_metadata.read_text(encoding="utf-8"))
    training_completed = metadata["status"] == "COMPLETED" and int(metadata["completed_steps"]) == 10000
    if not training_completed:
        raise RuntimeError("Schedule V4 did not complete all 10000 optimizer steps")
    selected = min((row for _, row in eligible.iterrows()), key=_key)
    selected_path = Path(selected.checkpoint)
    if not selected_path.is_file():
        raise RuntimeError(f"selected checkpoint missing: {selected_path}")

    result = {
        "schema_version": "ecir-mvr-medium-schedule-v4-checkpoint-selection-v1",
        "policy": "all_preregistered_accuracy_safety_high_flex_then_validity",
        "validity_closeness_rounding_decimals": 6,
        "training_completed": True,
        "preregistered_steps": list(PREREGISTERED_STEPS),
        "qualified_checkpoint_count": int(len(eligible)),
        "selected_step": int(selected.step),
        "selected_checkpoint": str(selected_path.resolve()),
        "selected_checkpoint_sha256": _sha(selected_path),
        "selected_metrics": {name: float(selected[name]) for name in (
            "learning_rate", "validity_delta", "max_core_relative_improvement",
            "mean_displacement", "identity_fraction", "rmsd_delta", "mat_p_delta",
            "mat_r_delta", "high_flex_validity_delta", "high_flex_rmsd_delta",
            "unseen_validity_delta", "unseen_rmsd_delta",
        )},
        "test_records_read": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_save(result, args.output)
    rows.to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
