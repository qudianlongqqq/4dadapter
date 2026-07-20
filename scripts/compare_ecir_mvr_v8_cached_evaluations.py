#!/usr/bin/env python
"""Build frozen paired V8-versus-baseline statistics from cached evaluations."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np

from etflow.ecir.v8_validation_cache import ISOLATION, atomic_json, file_sha256


METRICS = (
    "accepted",
    "bond_delta",
    "angle_delta",
    "active_angle_delta",
    "clash_delta",
    "ring_delta",
    "weighted_bac_delta",
    "mean_displacement",
    "max_atom_displacement",
    "target_loss",
    "chirality_preserved",
    "rmsd",
)


def _paired_all(values: np.ndarray, draws: int, seed: int = 43) -> np.ndarray:
    """Bootstrap every method/metric with one frozen paired index stream."""
    rng = np.random.default_rng(seed)
    sampled = np.empty((*values.shape[:2], draws), dtype=np.float64)
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        indices = rng.integers(0, values.shape[-1], size=(count, values.shape[-1]))
        for baseline_index in range(values.shape[0]):
            for metric_index in range(values.shape[1]):
                sampled[baseline_index, metric_index, start : start + count] = values[
                    baseline_index, metric_index, indices
                ].mean(axis=1)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8", type=Path, required=True)
    parser.add_argument("--d1", type=Path, required=True)
    parser.add_argument("--v5-b", type=Path, required=True)
    parser.add_argument("--v7", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    paths = {"V8": args.v8, "D1": args.d1, "V5-B": args.v5_b, "V7": args.v7}
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    rows = {name: report.get("per_record_metrics") for name, report in reports.items()}
    if any(value is None for value in rows.values()):
        raise RuntimeError("paired comparison requires FULL per-record metrics")
    identity = [(row["record_index"], row["sample_id"]) for row in rows["V8"]]
    for method, values in rows.items():
        if [(row["record_index"], row["sample_id"]) for row in values] != identity:
            raise RuntimeError(f"paired evaluation identity/order changed: {method}")
    baselines = ("D1", "V5-B", "V7")
    differences = np.asarray(
        [
            [
                [
                    float(v8[metric]) - float(base[metric])
                    for v8, base in zip(rows["V8"], rows[baseline], strict=True)
                ]
                for metric in METRICS
            ]
            for baseline in baselines
        ],
        dtype=np.float64,
    )
    sampled = _paired_all(differences, args.bootstrap_draws)
    comparisons = {
        f"V8-minus-{baseline}": {
            metric: {
                "mean": float(differences[baseline_index, metric_index].mean()),
                "ci95_low": float(
                    np.quantile(sampled[baseline_index, metric_index], 0.025)
                ),
                "ci95_high": float(
                    np.quantile(sampled[baseline_index, metric_index], 0.975)
                ),
                "draws": args.bootstrap_draws,
            }
            for metric_index, metric in enumerate(METRICS)
        }
        for baseline_index, baseline in enumerate(baselines)
    }
    output = {
        "schema_version": "mcvr-v8-full-paired-cache-comparison-v1",
        "status": "COMPLETED",
        "records": len(identity),
        "record_identity_equal": True,
        "methods": list(paths),
        "report_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "paired": comparisons,
        **ISOLATION,
    }
    atomic_json(args.output, output)


if __name__ == "__main__":
    main()
