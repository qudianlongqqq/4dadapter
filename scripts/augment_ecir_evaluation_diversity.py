#!/usr/bin/env python
"""Add frozen conformer diversity metrics to a cached evaluation copy."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.ecir.run_a_evaluation import rmsd_matrix
from etflow.ecir.v8_validation_cache import ISOLATION, atomic_json, file_sha256, iter_prediction_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duplicate-rmsd-threshold", type=float, default=0.1)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.resolve().read_text(encoding="utf-8"))
    records = list(iter_prediction_records(args.prediction_manifest.resolve()))
    expected = [(int(row["record_index"]), str(row["sample_id"])) for row in evaluation["per_record_metrics"]]
    observed = [(int(row["record_index"]), str(row["sample_id"])) for row in records]
    if observed != expected:
        raise RuntimeError("diversity augmentation prediction/evaluation identity changed")
    by_molecule = defaultdict(list)
    for row in records:
        by_molecule[str(row["molecule_id"])].append(torch.as_tensor(row["safe_coordinates"]).cpu())
    diversity = []
    duplicate_rates = []
    pair_count = 0
    duplicate_count = 0
    for coordinates in by_molecule.values():
        if len(coordinates) < 2:
            continue
        matrix = rmsd_matrix(coordinates, torch.stack(coordinates))
        mask = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
        pairs = matrix[mask]
        if not pairs.numel():
            continue
        duplicates = pairs < float(args.duplicate_rmsd_threshold)
        diversity.append(float(pairs.mean()))
        duplicate_rates.append(float(duplicates.float().mean()))
        pair_count += int(pairs.numel())
        duplicate_count += int(duplicates.sum())
    evaluation.setdefault("set_metrics", {})["conformer_diversity"] = sum(diversity) / len(diversity) if diversity else 0.0
    evaluation["set_metrics"]["duplicate_conformer_rate"] = sum(duplicate_rates) / len(duplicate_rates) if duplicate_rates else 0.0
    evaluation["diversity_definition"] = {
        "alignment": "frozen_Kabsch_rmsd_matrix",
        "aggregation": "mean_per_molecule_then_mean_across_molecules",
        "duplicate_rmsd_threshold_angstrom": float(args.duplicate_rmsd_threshold),
        "molecules_with_pairs": len(diversity),
        "pair_count": pair_count,
        "duplicate_pair_count": duplicate_count,
        "prediction_manifest_sha256": file_sha256(args.prediction_manifest),
    }
    evaluation.update(ISOLATION)
    atomic_json(args.output.resolve(), evaluation)
    print(json.dumps({"status": "COMPLETED", **evaluation["diversity_definition"], **{key: evaluation["set_metrics"][key] for key in ("conformer_diversity", "duplicate_conformer_rate")}}, indent=2))


if __name__ == "__main__":
    main()
