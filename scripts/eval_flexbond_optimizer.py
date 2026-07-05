#!/usr/bin/env python
"""Fair three-way evaluation over one frozen FlexBond cohort."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from etflow.commons.kabsch_utils import kabsch_rmsd
from etflow.commons.provenance import collect_run_provenance
from etflow.data.flexbond_cache_schema import tensor_sha256
from etflow.data.flexbond_eval_manifest import (
    load_eval_manifest,
    validate_dataset_against_manifest,
)
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset
from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset


SUBSETS = {
    "all": 0,
    "rotatable_ge_3": 3,
    "rotatable_ge_5": 5,
    "rotatable_ge_6": 6,
}


def _reference_candidates(data) -> torch.Tensor:
    ptr = data.reference_conformer_ptr.tolist()
    return torch.stack(
        [data.x_ref_candidates[ptr[index] : ptr[index + 1]] for index in range(len(ptr) - 1)]
    )


def _load_method_records(
    path: Path, method: str, manifest: dict
) -> tuple[dict[str, dict], list[str], list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path} is not a manifest-aware sample payload.")
    payload_rows = payload.get("manifest", {}).get("records")
    if payload_rows != manifest["records"]:
        raise ValueError(f"Sample payload {path} was not produced from the requested manifest.")
    records: dict[str, dict] = {}
    for record in payload["records"]:
        sample_id = str(record["sample_id"])
        if sample_id in records:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}.")
        if str(record.get("method_name")) != method:
            raise ValueError(
                f"Expected method {method!r}, got {record.get('method_name')!r} in {path}."
            )
        records[sample_id] = record
    expected = {str(row["sample_id"]) for row in manifest["records"]}
    unexpected = sorted(set(records).difference(expected))
    if unexpected:
        raise ValueError(f"Unexpected sample ids in {path}: {unexpected[:20]}.")
    missing = sorted(expected.difference(records))
    failed = sorted(
        sample_id
        for sample_id, record in records.items()
        if record.get("status") != "success" or record.get("x_refined") is None
    )
    manifest_by_id = {str(row["sample_id"]): row for row in manifest["records"]}
    for sample_id, record in records.items():
        if str(record.get("x_init_hash")) != str(manifest_by_id[sample_id]["x_init_hash"]):
            raise ValueError(f"x_init_hash mismatch for {sample_id!r} in {path}.")
    return records, missing, failed


def _evaluate(records: list[dict], threshold: float) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["mol_id"]), []).append(record)
    rows = []
    for mol_id, group in grouped.items():
        generated = torch.stack([row["coordinates"] for row in group])
        refs = group[0]["references"]
        reference_hash = tensor_sha256(refs)
        if any(tensor_sha256(row["references"]) != reference_hash for row in group[1:]):
            raise ValueError(f"Reference-set content mismatch for molecule {mol_id!r}.")
        distances = torch.stack(
            [torch.stack([kabsch_rmsd(gen, ref) for gen in generated]) for ref in refs]
        )
        best_for_reference = distances.min(dim=1).values
        best_for_generated = distances.min(dim=0).values
        rows.append(
            {
                "mol_id": mol_id,
                "num_rotatable_bonds": int(group[0]["num_rotatable_bonds"]),
                "rmsd": float(best_for_generated.mean()),
                "cov_r": float((best_for_reference < threshold).float().mean()),
                "cov_p": float((best_for_generated < threshold).float().mean()),
                "mat_r": float(best_for_reference.mean()),
                "mat_p": float(best_for_generated.mean()),
            }
        )
    return rows


def _summaries(
    rows: list[dict], method: str, failure_rate: float, missing_count: int
) -> list[dict]:
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
                "failure_rate": failure_rate,
                "missing_count": missing_count,
                "rmsd_mean": float(metric("rmsd").mean()),
                "rmsd_median": float(np.median(metric("rmsd"))),
                "COV-R": float(metric("cov_r").mean()),
                "COV-P": float(metric("cov_p").mean()),
                "MAT-R": float(metric("mat_r").mean()),
                "MAT-P": float(metric("mat_p").mean()),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inference_cache", required=True)
    parser.add_argument("--reference_cache", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cartesian_samples", required=True, type=Path)
    parser.add_argument("--flexbond_samples", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=1.25)
    args = parser.parse_args()

    manifest = load_eval_manifest(args.manifest)
    inference_dataset = FlexBondInferenceDataset(args.inference_cache, args.split)
    inference = validate_dataset_against_manifest(inference_dataset, manifest)
    reference_dataset = FlexBondOptimizerDataset(args.reference_cache, args.split, validate=True)
    references = {str(data.mol_id): data for data in reference_dataset}
    cartesian, cart_missing, cart_failed = _load_method_records(
        args.cartesian_samples, "cartesian_adapter", manifest
    )
    flexbond, flex_missing, flex_failed = _load_method_records(
        args.flexbond_samples, "flexbond4d_adapter", manifest
    )

    methods = {
        "upstream_only": ({}, [], []),
        "cartesian_adapter": (cartesian, cart_missing, cart_failed),
        "flexbond4d_adapter": (flexbond, flex_missing, flex_failed),
    }
    summaries = []
    diagnostics = {}
    denominator = len(manifest["records"])
    for method, (sample_records, missing, failed) in methods.items():
        evaluation_records = []
        for row in manifest["records"]:
            sample_id = str(row["sample_id"])
            data = inference[sample_id]
            reference = references.get(sample_id)
            if reference is None:
                raise ValueError(f"Reference cache is missing manifest sample {sample_id!r}.")
            if method == "upstream_only":
                coordinates = data.x_init.cpu()
            else:
                sampled = sample_records.get(sample_id)
                # Failed/missing refinements fall back to upstream coordinates so every
                # method retains the exact same sample and molecule denominator.
                coordinates = (
                    sampled["x_refined"]
                    if sampled is not None
                    and sampled.get("status") == "success"
                    and sampled.get("x_refined") is not None
                    else data.x_init.cpu()
                )
            evaluation_records.append(
                {
                    "mol_id": str(row["mol_id"]),
                    "sample_id": sample_id,
                    "num_rotatable_bonds": int(row["num_rotatable_bonds"]),
                    "coordinates": torch.as_tensor(coordinates),
                    "references": _reference_candidates(reference),
                }
            )
        failures = sorted(set(missing).union(failed))
        failure_rate = len(failures) / denominator if denominator else 0.0
        rows = _evaluate(evaluation_records, args.threshold)
        summaries.extend(_summaries(rows, method, failure_rate, len(missing)))
        diagnostics[method] = {
            "failure_rate": failure_rate,
            "failed_ids": failed,
            "missing_ids": missing,
            "failure_policy": "upstream_fallback",
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(summaries[0])
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    provenance = collect_run_provenance(
        cache_path=args.inference_cache,
        checkpoint_path=None,
        config_path=None,
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "threshold": args.threshold,
                "manifest": str(args.manifest.resolve()),
                "metrics": summaries,
                "diagnostics": diagnostics,
                "provenance": provenance,
            },
            handle,
            indent=2,
        )
    with (args.output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# FlexBond adapter fair-cohort evaluation\n\n")
        handle.write("Failed or missing refinements use upstream fallback.\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in summaries:
            handle.write("| " + " | ".join(str(row[column]) for column in columns) + " |\n")
    print(f"Evaluated {denominator} common samples; wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
