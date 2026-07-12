#!/usr/bin/env python
"""Measure whether post-generation refinement collapses conformer diversity."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from etflow.commons.kabsch_utils import kabsch_rmsd
from etflow.data.flexbond_eval_manifest import (
    load_eval_manifest,
    validate_dataset_against_manifest,
)
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset
from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset


def _references(data) -> torch.Tensor:
    pointers = data.reference_conformer_ptr.tolist()
    return torch.stack(
        [
            data.x_ref_candidates[pointers[index] : pointers[index + 1]]
            for index in range(len(pointers) - 1)
        ]
    )


def _load_samples(path: Path, method: str, manifest: dict) -> dict[str, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("manifest", {}).get("records") != manifest["records"]:
        raise ValueError(f"Sample payload uses a different manifest: {path}")
    records = {}
    for record in payload.get("records", []):
        sample_id = str(record["sample_id"])
        if record.get("method_name") != method:
            raise ValueError(f"Unexpected method in {path}: {record.get('method_name')}")
        if sample_id in records:
            raise ValueError(f"Duplicate sample id {sample_id!r} in {path}")
        records[sample_id] = record
    return records


def _pairwise(coordinates: torch.Tensor) -> tuple[float, float]:
    values = [
        float(kabsch_rmsd(coordinates[left], coordinates[right]))
        for left in range(coordinates.size(0))
        for right in range(left + 1, coordinates.size(0))
    ]
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(np.median(array))


def _coverage(
    coordinates: torch.Tensor, references: torch.Tensor, threshold: float
) -> dict[str, float | int]:
    distances = torch.stack(
        [
            torch.stack([kabsch_rmsd(coordinate, reference) for coordinate in coordinates])
            for reference in references
        ]
    )
    reference_best = distances.min(dim=1).values
    generated_best, assignments = distances.min(dim=0)
    return {
        "unique_reference_coverage": int(assignments.unique().numel()),
        "unique_reference_coverage_fraction": float(assignments.unique().numel())
        / max(references.size(0), 1),
        "COV-R": float((reference_best < threshold).float().mean()),
        "COV-P": float((generated_best < threshold).float().mean()),
    }


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inference_cache", required=True)
    parser.add_argument("--reference_cache", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cartesian_samples", type=Path)
    parser.add_argument("--flexbond_samples", type=Path)
    parser.add_argument("--global_coupled_4d_samples", type=Path)
    parser.add_argument("--threshold", type=float, default=1.25)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.cartesian_samples is None
        and args.flexbond_samples is None
        and args.global_coupled_4d_samples is None
    ):
        raise ValueError("At least one refined sample payload is required")

    manifest = load_eval_manifest(args.manifest)
    inference = validate_dataset_against_manifest(
        FlexBondInferenceDataset(args.inference_cache, args.split), manifest
    )
    reference_by_sample = {
        str(data.mol_id): data
        for data in FlexBondOptimizerDataset(
            args.reference_cache, args.split, validate=True
        )
    }
    methods = {}
    if args.cartesian_samples is not None:
        methods["cartesian_adapter"] = _load_samples(
            args.cartesian_samples, "cartesian_adapter", manifest
        )
    if args.flexbond_samples is not None:
        methods["flexbond4d_adapter"] = _load_samples(
            args.flexbond_samples, "flexbond4d_adapter", manifest
        )
    if args.global_coupled_4d_samples is not None:
        methods["global_coupled_4d_adapter"] = _load_samples(
            args.global_coupled_4d_samples,
            "global_coupled_4d_adapter",
            manifest,
        )

    manifest_groups: dict[str, list[dict]] = {}
    for row in manifest["records"]:
        manifest_groups.setdefault(str(row["mol_id"]), []).append(row)
    rows = []
    for method, samples in methods.items():
        for mol_id, group in manifest_groups.items():
            initial, refined = [], []
            references = None
            failure_count = 0
            for manifest_row in group:
                sample_id = str(manifest_row["sample_id"])
                data = inference[sample_id]
                initial.append(data.x_init.cpu())
                sample = samples.get(sample_id)
                failed = (
                    sample is None
                    or sample.get("status") != "success"
                    or sample.get("x_refined") is None
                )
                failure_count += int(failed)
                refined.append(
                    data.x_init.cpu()
                    if failed
                    else torch.as_tensor(sample["x_refined"]).cpu()
                )
                current_references = _references(reference_by_sample[sample_id])
                if references is None:
                    references = current_references
                elif not torch.equal(references, current_references):
                    raise ValueError(f"Reference set mismatch within molecule {mol_id!r}")
            initial_tensor, refined_tensor = torch.stack(initial), torch.stack(refined)
            before_mean, before_median = _pairwise(initial_tensor)
            after_mean, after_median = _pairwise(refined_tensor)
            before_coverage = _coverage(initial_tensor, references, args.threshold)
            after_coverage = _coverage(refined_tensor, references, args.threshold)
            rows.append(
                {
                    "method": method,
                    "mol_id": mol_id,
                    "num_samples": len(group),
                    "num_rotatable_bonds": int(group[0]["num_rotatable_bonds"]),
                    "failure_count": failure_count,
                    "pairwise_rmsd_mean_before": before_mean,
                    "pairwise_rmsd_median_before": before_median,
                    "pairwise_rmsd_mean_after": after_mean,
                    "pairwise_rmsd_median_after": after_median,
                    "diversity_ratio": after_mean / max(before_mean, 1.0e-12),
                    "unique_reference_coverage_before": before_coverage[
                        "unique_reference_coverage"
                    ],
                    "unique_reference_coverage_after": after_coverage[
                        "unique_reference_coverage"
                    ],
                    "unique_reference_coverage_fraction_before": before_coverage[
                        "unique_reference_coverage_fraction"
                    ],
                    "unique_reference_coverage_fraction_after": after_coverage[
                        "unique_reference_coverage_fraction"
                    ],
                    "COV-R_before": before_coverage["COV-R"],
                    "COV-R_after": after_coverage["COV-R"],
                    "COV-R_change": after_coverage["COV-R"] - before_coverage["COV-R"],
                    "COV-P_before": before_coverage["COV-P"],
                    "COV-P_after": after_coverage["COV-P"],
                    "COV-P_change": after_coverage["COV-P"] - before_coverage["COV-P"],
                }
            )

    summary = []
    numeric = [key for key in rows[0] if key not in {"method", "mol_id"}]
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        output = {"method": method, "num_molecules": len(selected)}
        for key in numeric:
            values = np.asarray([row[key] for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            output[f"{key}_mean"] = float(finite.mean()) if finite.size else float("nan")
            output[f"{key}_median"] = (
                float(np.median(finite)) if finite.size else float("nan")
            )
        summary.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "diversity_diagnostics.csv", rows)
    _write(args.output_dir / "diversity_summary.csv", summary)
    print(f"Wrote diversity diagnostics for {len(rows)} molecule/method pairs.")


if __name__ == "__main__":
    main()
