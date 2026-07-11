#!/usr/bin/env python
"""Collect rollout evaluations and select the best checkpoint deterministically."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--pattern", default="**/summary.csv")
    args = parser.parse_args()
    rows = []
    all_subset_rows = []
    for path in sorted(args.root.glob(args.pattern)):
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("method") != "global_coupled_4d_adapter":
                    continue
                text = str(path)
                step = re.search(r"step(\d+)", text)
                alpha = re.search(r"alpha(0?\d+)", text)
                mode = next((name for name in ("full_4d", "torsion_only", "bending_torsion", "angular_only", "stretch_only", "internal_zero") if name in text), "full_4d")
                enriched = {
                    "checkpoint_step": int(step.group(1)) if step else 0,
                    "alpha": float("0." + alpha.group(1).lstrip("0")) if alpha else float(row.get("alpha", 0) or 0),
                    "joint_mode": mode,
                    "summary_path": str(path),
                    **row,
                }
                all_subset_rows.append(enriched)
                if row.get("subset") == "all":
                    rows.append(enriched)
    if not rows:
        raise RuntimeError(f"no Global Coupled 4D all-subset summaries below {args.root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = args.output_dir / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "comparison_all_subset.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_subset_rows[0]))
        writer.writeheader(); writer.writerows(all_subset_rows)
    # Required priority: failure, RMSD/MAT, COV-R, COV-P.
    best = min(rows, key=lambda row: (
        float(row["failure_rate"]), float(row["rmsd_mean"]),
        float(row["MAT-R"]), float(row["MAT-P"]),
        -float(row["COV-R"]), -float(row["COV-P"]),
    ))
    best_payload = {
        "checkpoint_step": best["checkpoint_step"], "alpha": best["alpha"],
        "joint_mode": best["joint_mode"], "summary_path": best["summary_path"],
        "failure_rate": float(best["failure_rate"]), "rmsd_mean": float(best["rmsd_mean"]),
        "COV-R": float(best["COV-R"]), "COV-P": float(best["COV-P"]),
        "MAT-R": float(best["MAT-R"]), "MAT-P": float(best["MAT-P"]),
    }
    (args.output_dir / "best_checkpoint.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    summary = ["Global Coupled 4D checkpoint sweep", json.dumps(best_payload, indent=2)]
    (args.output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
