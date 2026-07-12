#!/usr/bin/env python
"""Merge fair-cohort metrics, timing metadata, and diversity diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--diversity", required=True, type=Path)
    parser.add_argument("--cartesian_samples", required=True, type=Path)
    parser.add_argument("--global4d_samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8-sig") as handle:
        metrics = list(csv.DictReader(handle))
    with args.diversity.open(encoding="utf-8-sig") as handle:
        diversity = {row["method"]: row for row in csv.DictReader(handle)}
    payloads = {
        "cartesian_adapter": torch.load(
            args.cartesian_samples, map_location="cpu", weights_only=False
        ),
        "global_coupled_4d_adapter": torch.load(
            args.global4d_samples, map_location="cpu", weights_only=False
        ),
    }
    states = {
        "cartesian_adapter": json.loads(
            (args.cartesian_samples.parent / "sampling_state.json").read_text(encoding="utf-8")
        ),
        "global_coupled_4d_adapter": json.loads(
            (args.global4d_samples.parent / "sampling_state.json").read_text(encoding="utf-8")
        ),
    }
    rows = []
    for row in metrics:
        method = row["method"]
        payload = payloads.get(method, {})
        records = payload.get("records", [])
        timing = sum(
            sum(float(value) for value in record.get("mean_timing", {}).values())
            for record in records
        )
        backends = payload.get("solver_backend_counts", {})
        rows.append({
            **row,
            "inference_seconds": states.get(method, {}).get("total_seconds", timing),
            "peak_memory_bytes": max(
                [
                    float(record.get("mean_timing", {}).get("peak_gpu_memory", 0))
                    for record in records
                ]
                or [0]
            ),
            "solver_backend": str(backends),
            "pairwise_diversity_before": diversity.get(method, {}).get(
                "pairwise_rmsd_mean_before_mean"
            ),
            "pairwise_diversity_after": diversity.get(method, {}).get(
                "pairwise_rmsd_mean_after_mean"
            ),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    columns = list(rows[0])
    lines = ["# Formal-large final test", "", "| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(str(row[key]) for key in columns) + " |" for row in rows]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
