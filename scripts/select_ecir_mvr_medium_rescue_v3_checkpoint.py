#!/usr/bin/env python
"""Select the frozen formal V3 checkpoint while reporting any earlier global best."""

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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(row) -> tuple:
    return (
        round(float(row.validity_delta), 6), float(row.mean_displacement),
        -float(row.identity_fraction), float(row.high_flex_validity_delta),
        float(row.unseen_validity_delta),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-comparison", type=Path, required=True)
    parser.add_argument("--v3-comparison", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v2 = pd.read_csv(args.v2_comparison); v2["segment"] = "V2"
    v3 = pd.read_csv(args.v3_comparison) if args.v3_comparison.is_file() else v2.iloc[0:0].copy()
    v3["segment"] = "V3"
    rows = pd.concat([v2, v3], ignore_index=True)
    rows["formal_checkpoint"] = rows.step.isin([5000, 10000, 15000, 20000])
    eligible = rows[rows.accuracy_noninferior.astype(bool)]
    if eligible.empty:
        raise RuntimeError("no accuracy-noninferior checkpoint exists")
    best_overall = min((row for _, row in eligible.iterrows()), key=_key)
    formal = eligible[eligible.formal_checkpoint]
    metadata = json.loads(args.run_metadata.read_text(encoding="utf-8"))
    training_completed = metadata["status"] == "COMPLETED" and int(metadata["completed_steps"]) == 20000
    selected = min((row for _, row in formal.iterrows()), key=_key) if not formal.empty else best_overall
    selected_path = Path(selected.checkpoint)
    if not selected_path.is_file():
        raise RuntimeError(f"selected checkpoint missing: {selected_path}")
    result = {
        "schema_version": "ecir-mvr-medium-rescue-v3-checkpoint-selection-v1",
        "policy": "formal_5000_10000_15000_20000_preferred",
        "training_completed": training_completed,
        "qualified_formal_checkpoint_exists": not formal.empty,
        "selected_step": int(selected.step),
        "selected_segment": str(selected.segment),
        "selected_checkpoint": str(selected_path.resolve()),
        "selected_checkpoint_sha256": _sha(selected_path),
        "selected_metrics": {name: float(selected[name]) for name in (
            "validity_delta", "mean_displacement", "identity_fraction", "rmsd_delta",
            "mat_p_delta", "mat_r_delta", "high_flex_validity_delta", "unseen_validity_delta",
        )},
        "best_overall_step": int(best_overall.step),
        "best_overall_segment": str(best_overall.segment),
        "best_overall_checkpoint": str(Path(best_overall.checkpoint).resolve()),
        "best_overall_is_early": int(best_overall.step) < 5000,
        "best_overall_validity_delta": float(best_overall.validity_delta),
        "test_records_read": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_save(result, args.output)
    rows.to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
