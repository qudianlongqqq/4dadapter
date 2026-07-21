#!/usr/bin/env python
"""Create the pre-registered Seeds 12/43/48 MCVR V8 summary artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SEEDS = (12, 43, 48)
METRICS = (
    "weighted_bac",
    "bond",
    "angle",
    "active_angle",
    "ring",
    "clash",
    "acceptance",
    "accepted",
    "weighted_bac_delta",
    "bond_delta",
    "angle_delta",
    "active_angle_delta",
    "ring_delta",
    "clash_delta",
    "mean_displacement",
    "RMSD",
    "COV_P",
    "COV_R",
    "MAT_P",
    "MAT_R",
)
MAPPING = {
    "weighted_bac": "weighted_bac_delta",
    "bond": "bond_delta",
    "angle": "angle_delta",
    "active_angle": "active_angle_delta",
    "ring": "ring_delta",
    "clash": "clash_delta",
    "acceptance": "accepted",
    "mean_displacement": "mean_displacement",
    "RMSD": "rmsd",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(seed: int, path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("mode") != "FULL" or int(report.get("records", -1)) != 10000:
        raise RuntimeError(f"Seed{seed} is not a FULL10K report")
    if int(report.get("formal_test_records_read", -1)) != 0:
        raise RuntimeError(f"Seed{seed} read formal test records")
    if int(report.get("frozen_holdout_records_read", -1)) != 0:
        raise RuntimeError(f"Seed{seed} read frozen holdout records")
    metrics = report["metrics"]
    set_metrics = report["set_metrics"]
    result = {
        "method": "MCVR V8 Full v1",
        "seed": seed,
        **{target: float(metrics[source]) for target, source in MAPPING.items()},
        **{
            name: float(metrics[name])
            for name in (
                "accepted",
                "weighted_bac_delta",
                "bond_delta",
                "angle_delta",
                "active_angle_delta",
                "ring_delta",
                "clash_delta",
            )
        },
        **{name: float(set_metrics[name]) for name in ("COV_P", "COV_R", "MAT_P", "MAT_R")},
        "clash_interpretation": "low-power natural cohort",
        "validation_sha256": _sha(path),
    }
    return result


def build_summary(paths: dict[int, Path]) -> tuple[pd.DataFrame, dict]:
    if set(paths) != set(SEEDS):
        raise RuntimeError("summary requires exactly Seeds 12, 43 and 48")
    rows = [_row(seed, paths[seed]) for seed in SEEDS]
    numeric = pd.DataFrame(rows).set_index("seed")[list(METRICS)]
    mean = numeric.mean(axis=0)
    std = numeric.std(axis=0, ddof=1)
    aggregate_rows = [
        {
            "method": "MCVR V8 Full v1",
            "seed": label,
            **values.to_dict(),
            "clash_interpretation": "low-power natural cohort",
            "validation_sha256": "",
        }
        for label, values in (("mean", mean), ("sample_std_ddof1", std))
    ]
    frame = pd.DataFrame(rows + aggregate_rows)
    payload = {
        "schema_version": "mcvr-v8-multiseed-summary-v1",
        "status": "MCVR_V8_MULTI_SEED_12_43_48_COMPLETED",
        "seeds": list(SEEDS),
        "standard_deviation": "sample",
        "ddof": 1,
        "metric_mapping": MAPPING,
        "cov_mat_reporting": ["COV_P", "COV_R", "MAT_P", "MAT_R"],
        "single_direction_posthoc_selection": False,
        "clash_interpretation": "low-power natural cohort",
        "runs": rows,
        "mean": mean.to_dict(),
        "sample_std_ddof1": std.to_dict(),
    }
    return frame, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    for seed in SEEDS:
        parser.add_argument(f"--seed{seed}-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {seed: getattr(args, f"seed{seed}_evaluation") for seed in SEEDS}
    frame, payload = build_summary(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "V8_MULTI_SEED_SUMMARY"
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    display = frame.drop(columns=["validation_sha256"])
    lines = [
        "# V8 Multi-Seed Summary",
        "",
        "Seeds: 12, 43, 48. Standard deviation is sample std (`ddof=1`).",
        "COV-P/COV-R/MAT-P/MAT-R are all reported; no direction is selected post hoc.",
        "Clash is a low-power natural-cohort metric.",
        "",
        display.to_markdown(index=False),
        "",
    ]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
