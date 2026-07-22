#!/usr/bin/env python
"""Create the preregistered V8 Full versus single-factor ablation table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = {
    "V8 Full Seed43": "formal_large_200k/full_seed43",
    "No-Constraint": "ablations/no_constraint_seed43",
    "No-Confidence": "ablations/no_confidence_seed43",
    "No-Error-State": "ablations/no_error_state_seed43",
    "No-Type-Normalization": "ablations/no_type_normalization_seed43",
}
METRICS = (
    "accepted",
    "weighted_bac_delta",
    "bond_delta",
    "angle_delta",
    "active_angle_delta",
    "ring_delta",
    "clash_delta",
    "mean_displacement",
    "rmsd",
    "MAT_P",
    "MAT_R",
    "COV_P",
    "COV_R",
)


def _load(root: Path, relative: str) -> dict:
    path = root / relative / "validation_cache/step012500/full/evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "FULL" or int(payload.get("records", -1)) != 10000:
        raise RuntimeError(f"not a FULL10K result: {path}")
    if int(payload.get("formal_test_records_read", -1)) != 0:
        raise RuntimeError(f"formal test reads are nonzero: {path}")
    if int(payload.get("frozen_holdout_records_read", -1)) != 0:
        raise RuntimeError(f"frozen holdout reads are nonzero: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    baseline = None
    for method, relative in RUNS.items():
        payload = _load(args.run_root, relative)
        values = {**payload["metrics"], **payload.get("set_metrics", {})}
        row = {"method": method, **{key: values.get(key) for key in METRICS}}
        if baseline is None:
            baseline = row
        for key in METRICS:
            row[f"delta_vs_full__{key}"] = (
                None if baseline[key] is None or row[key] is None else row[key] - baseline[key]
            )
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "V8_ABLATION_SUMMARY.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "V8_ABLATION_SUMMARY.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "mcvr-v8-ablation-summary-v1",
                "rows": rows,
                "formal_test_records_read": 0,
                "frozen_holdout_records_read": 0,
                "result_conditioned_tuning": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    headers = list(rows[0])
    lines = [
        "# MCVR V8 single-factor ablations",
        "",
        "Seed43, original-200K schedule checkpoint at step12500, FULL10K frozen evaluator.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                "" if row[key] is None else (f"{row[key]:.10g}" if isinstance(row[key], float) else str(row[key]))
                for key in headers
            )
            + " |"
        )
    lines.extend(["", "formal test reads=0; frozen holdout reads=0.", ""])
    (args.output_dir / "V8_ABLATION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "MCVR_V8_ABLATION_SUMMARY_COMPLETED", "rows": len(rows)}))


if __name__ == "__main__":
    main()
