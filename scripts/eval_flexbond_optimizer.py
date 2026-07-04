#!/usr/bin/env python
"""Evaluate upstream and refined coordinates, including flexible subsets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from etflow.commons.kabsch_utils import kabsch_rmsd


SUBSETS = {
    "all": 0,
    "rotatable_ge_3": 3,
    "rotatable_ge_5": 5,
    "rotatable_ge_6": 6,
}


def _evaluate(records: list[dict], coordinate_key: str, threshold: float) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        source_id = str(record.get("source_mol_id", record["mol_id"]))
        grouped.setdefault(source_id, []).append(record)
    rows = []
    for source_id, group in grouped.items():
        generated = torch.stack([row[coordinate_key] for row in group])
        refs = group[0]["x_ref_candidates"]
        for row in group[1:]:
            if row["x_ref_candidates"].shape != refs.shape:
                raise ValueError(f"Reference-set shape mismatch for {source_id}.")
        distances = torch.stack(
            [
                torch.stack([kabsch_rmsd(gen, ref) for gen in generated])
                for ref in refs
            ]
        )
        best_for_reference = distances.min(dim=1).values
        best_for_generated = distances.min(dim=0).values
        rows.append(
            {
                "mol_id": source_id,
                "num_rotatable_bonds": group[0]["num_rotatable_bonds"],
                "rmsd": float(best_for_generated.mean()),
                "cov_r": float((best_for_reference < threshold).float().mean()),
                "cov_p": float((best_for_generated < threshold).float().mean()),
                "mat_r": float(best_for_reference.mean()),
                "mat_p": float(best_for_generated.mean()),
            }
        )
    return rows


def _summaries(rows: list[dict], method: str) -> list[dict]:
    output = []
    for subset, minimum in SUBSETS.items():
        chosen = [row for row in rows if row["num_rotatable_bonds"] >= minimum]
        if not chosen:
            continue
        metric = lambda name: np.asarray([row[name] for row in chosen], dtype=float)
        output.append(
            {
                "method": method,
                "subset": subset,
                "num_molecules": len(chosen),
                "rmsd_mean": float(metric("rmsd").mean()),
                "rmsd_median": float(np.median(metric("rmsd"))),
                "COV-R": float(metric("cov_r").mean()),
                "COV-P": float(metric("cov_p").mean()),
                "MAT-R": float(metric("mat_r").mean()),
                "MAT-P": float(metric("mat_p").mean()),
                "AMR": float(metric("mat_p").mean()),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=1.25)
    args = parser.parse_args()
    payload = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = payload["results"] if isinstance(payload, dict) else payload
    if not records:
        raise SystemExit("No stable sampled records to evaluate.")
    method_names = {str(record.get("method_name", "refined")) for record in records}
    if len(method_names) != 1:
        raise ValueError(f"Sample file mixes refinement methods: {sorted(method_names)}")
    refined_method = next(iter(method_names))
    summaries = []
    for method, key in (
        ("upstream_only", "x_init"),
        (refined_method, "x_refined"),
    ):
        summaries.extend(_summaries(_evaluate(records, key, args.threshold), method))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(summaries[0])
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"threshold": args.threshold, "metrics": summaries}, handle, indent=2)
    with (args.output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# FlexBond optimizer evaluation\n\n")
        handle.write(f"Kabsch RMSD coverage threshold: {args.threshold:.2f} Å\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in summaries:
            handle.write("| " + " | ".join(str(row[column]) for column in columns) + " |\n")
    print(f"Wrote summary.md, summary.csv, and summary.json to {args.output_dir}")


if __name__ == "__main__":
    main()
