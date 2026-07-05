#!/usr/bin/env python
"""Measure local-geometry distortion along the Cartesian adapter training path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from etflow.commons.geometry_diagnostics import path_geometry_metrics
from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_molecules", type=int, default=100)
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--times", nargs="+", type=float, default=(0.25, 0.5, 0.75))
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    if any(time < 0.0 or time > 1.0 for time in args.times):
        raise ValueError("All interpolation times must be in [0, 1].")

    dataset = FlexBondOptimizerDataset(
        args.cache_dir,
        args.split,
        max_molecules=args.max_molecules,
        validate=True,
    )
    num_records = min(len(dataset), args.max_records)
    rows = []
    for index in range(num_records):
        data = dataset[index]
        for time in args.times:
            x_t = (1.0 - time) * data.x_init + time * data.x_ref_aligned
            metrics = path_geometry_metrics(
                data.x_init,
                data.x_ref_aligned,
                x_t,
                time,
                data.edge_index,
                data.rotatable_bond_index,
            )
            rows.append(
                {
                    "sample_id": str(data.mol_id),
                    "mol_id": str(data.source_mol_id),
                    "time": time,
                    **metrics,
                }
            )

    metric_names = [
        key
        for key in rows[0]
        if key not in {"sample_id", "mol_id", "time", "num_bonds", "num_angles", "num_torsions"}
    ]
    summary = []
    for time in args.times:
        chosen = [row for row in rows if row["time"] == time]
        result = {"time": time, "num_records": len(chosen)}
        for name in metric_names:
            values = np.asarray([row[name] for row in chosen], dtype=float)
            finite = values[np.isfinite(values)]
            result[f"{name}_mean"] = float(finite.mean()) if finite.size else float("nan")
            result[f"{name}_median"] = (
                float(np.median(finite)) if finite.size else float("nan")
            )
            result[f"{name}_max"] = float(finite.max()) if finite.size else float("nan")
        summary.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "path_geometry_records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "path_geometry_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(f"Analyzed {num_records} records; wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
