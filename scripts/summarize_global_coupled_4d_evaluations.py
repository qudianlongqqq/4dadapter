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
    parser.add_argument("--checkpoint_dir", type=Path)
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
                group_name = path.parent.parent.name
                checkpoint_name = group_name.split("_alpha", 1)[0]
                reuse_path = path.parent.parent / "reused_from.json"
                reused_from = ""
                if reuse_path.is_file():
                    try:
                        reused_from = str(json.loads(
                            reuse_path.read_text(encoding="utf-8")
                        ).get("reused_from", ""))
                    except Exception:
                        reused_from = "invalid_reuse_metadata"
                checkpoint_step = int(step.group(1)) if step else 0
                if checkpoint_step == 0 and args.checkpoint_dir:
                    checkpoint_file = args.checkpoint_dir / f"{checkpoint_name}.ckpt"
                    if checkpoint_file.is_file():
                        try:
                            import torch
                            checkpoint_step = int(torch.load(checkpoint_file, map_location="cpu", weights_only=False).get("global_step", 0))
                        except Exception:
                            checkpoint_step = 0
                mode = next((name for name in ("full_4d", "torsion_only", "bending_torsion", "angular_only", "stretch_only", "internal_zero") if name in text), "full_4d")
                enriched = {
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_name": checkpoint_name,
                    "alpha": float("0." + alpha.group(1).lstrip("0")) if alpha else float(row.get("alpha", 0) or 0),
                    "joint_mode": mode,
                    "reused_from": reused_from,
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
    high_rows = {}
    for row in all_subset_rows:
        if row.get("subset", "all") == "all":
            continue
        key = (row["checkpoint_name"], row["alpha"], row["joint_mode"])
        threshold = int(re.search(r"(\d+)$", row["subset"]).group(1)) if re.search(r"(\d+)$", row["subset"]) else 0
        if key not in high_rows or threshold > high_rows[key][0]:
            high_rows[key] = (threshold, row)
    # Required priority: failure, RMSD/MAT, COV, then the most-flexible subset.
    best = min(rows, key=lambda row: (
        float(row["failure_rate"]), float(row["rmsd_mean"]),
        float(row["MAT-R"]), float(row["MAT-P"]),
        -float(row["COV-R"]), -float(row["COV-P"]),
        float(high_rows.get((row["checkpoint_name"], row["alpha"], row["joint_mode"]), (0, row))[1]["failure_rate"]),
        float(high_rows.get((row["checkpoint_name"], row["alpha"], row["joint_mode"]), (0, row))[1]["rmsd_mean"]),
    ))
    checkpoint_path = ""
    if args.checkpoint_dir:
        candidate = args.checkpoint_dir / f"{best['checkpoint_name']}.ckpt"
        checkpoint_path = str(candidate.resolve())
    best_payload = {
        "checkpoint_step": best["checkpoint_step"], "alpha": best["alpha"],
        "checkpoint_name": best["checkpoint_name"], "checkpoint_path": checkpoint_path,
        "joint_mode": best["joint_mode"], "summary_path": best["summary_path"],
        "failure_rate": float(best["failure_rate"]), "rmsd_mean": float(best["rmsd_mean"]),
        "COV-R": float(best["COV-R"]), "COV-P": float(best["COV-P"]),
        "MAT-R": float(best["MAT-R"]), "MAT-P": float(best["MAT-P"]),
    }
    (args.output_dir / "best_checkpoint.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    summary = ["Global Coupled 4D checkpoint sweep", json.dumps(best_payload, indent=2)]
    (args.output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (args.output_dir / "comparison_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
