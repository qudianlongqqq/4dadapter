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


def _evaluate(records: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["mol_id"]), []).append(record)
    rows = []
    sample_rows = []
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
        initial = torch.stack([row["x_init"] for row in group])
        initial_distances = torch.stack(
            [torch.stack([kabsch_rmsd(pos, ref) for pos in initial]) for ref in refs]
        )
        best_for_initial = initial_distances.min(dim=0).values
        for index, record in enumerate(group):
            update_norm = torch.linalg.norm(
                record["coordinates"] - record["x_init"], dim=-1
            )
            initial_rmsd = float(best_for_initial[index])
            refined_rmsd = float(best_for_generated[index])
            delta = initial_rmsd - refined_rmsd
            sample_rows.append(
                {
                    "method": record["method"],
                    "mol_id": mol_id,
                    "sample_id": record["sample_id"],
                    "num_rotatable_bonds": int(record["num_rotatable_bonds"]),
                    "status": record["status"],
                    "initial_rmsd": initial_rmsd,
                    "refined_rmsd": refined_rmsd,
                    "rmsd_improvement": delta,
                    "improved": delta > 1.0e-6,
                    "worsened": delta < -1.0e-6,
                    "mean_update_norm": float(update_norm.mean()),
                    "median_update_norm": float(update_norm.median()),
                    "max_update_norm": float(update_norm.max()),
                    "update_to_initial_rmsd_ratio": float(update_norm.mean())
                    / max(initial_rmsd, 1.0e-8),
                }
            )
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
    return rows, sample_rows


def _initial_rmsd_bin(value: float) -> str:
    boundaries = (0.5, 1.0, 1.5, 2.0)
    lower = 0.0
    for upper in boundaries:
        if value < upper:
            return f"[{lower:g},{upper:g})"
        lower = upper
    return "[2,inf)"


def _rotatable_bin(value: int) -> str:
    if value < 3:
        return "0-2"
    if value < 5:
        return "3-4"
    if value < 6:
        return "5"
    return "6+"


def _diagnostic_summaries(sample_rows: list[dict], method: str) -> list[dict]:
    groups: list[tuple[str, str, list[dict]]] = [("all", "all", sample_rows)]
    for label in ("0-2", "3-4", "5", "6+"):
        groups.append(
            (
                "rotatable_bonds",
                label,
                [
                    row
                    for row in sample_rows
                    if _rotatable_bin(row["num_rotatable_bonds"]) == label
                ],
            )
        )
    for label in ("[0,0.5)", "[0.5,1)", "[1,1.5)", "[1.5,2)", "[2,inf)"):
        groups.append(
            (
                "initial_rmsd",
                label,
                [row for row in sample_rows if _initial_rmsd_bin(row["initial_rmsd"]) == label],
            )
        )

    output = []
    for group_type, group_name, chosen in groups:
        if not chosen:
            continue
        values = lambda name: np.asarray([row[name] for row in chosen], dtype=float)
        output.append(
            {
                "method": method,
                "group_type": group_type,
                "group": group_name,
                "num_samples": len(chosen),
                "mean_update_norm": float(values("mean_update_norm").mean()),
                "median_update_norm": float(np.median(values("median_update_norm"))),
                "mean_update_to_initial_rmsd_ratio": float(
                    values("update_to_initial_rmsd_ratio").mean()
                ),
                "fraction_improved": float(values("improved").mean()),
                "fraction_worsened": float(values("worsened").mean()),
                "fraction_unchanged": float(
                    1.0 - values("improved").mean() - values("worsened").mean()
                ),
                "initial_rmsd_mean": float(values("initial_rmsd").mean()),
                "refined_rmsd_mean": float(values("refined_rmsd").mean()),
                "rmsd_improvement_mean": float(values("rmsd_improvement").mean()),
            }
        )
    return output


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
    parser.add_argument("--cartesian_samples", type=Path)
    parser.add_argument("--flexbond_samples", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=1.25)
    args = parser.parse_args()

    manifest = load_eval_manifest(args.manifest)
    inference_dataset = FlexBondInferenceDataset(args.inference_cache, args.split)
    inference = validate_dataset_against_manifest(inference_dataset, manifest)
    reference_dataset = FlexBondOptimizerDataset(args.reference_cache, args.split, validate=True)
    references = {str(data.mol_id): data for data in reference_dataset}
    methods = {
        "upstream_only": ({}, [], []),
    }
    if args.cartesian_samples is not None:
        methods["cartesian_adapter"] = _load_method_records(
            args.cartesian_samples, "cartesian_adapter", manifest
        )
    if args.flexbond_samples is not None:
        methods["flexbond4d_adapter"] = _load_method_records(
            args.flexbond_samples, "flexbond4d_adapter", manifest
        )
    if len(methods) == 1:
        raise ValueError("At least one adapter sample file is required.")
    summaries = []
    diagnostics = {}
    sample_diagnostics = []
    update_diagnostics = []
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
                status = "upstream"
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
                status = (
                    "success"
                    if sampled is not None
                    and sampled.get("status") == "success"
                    and sampled.get("x_refined") is not None
                    else "upstream_fallback"
                )
            evaluation_records.append(
                {
                    "mol_id": str(row["mol_id"]),
                    "sample_id": sample_id,
                    "num_rotatable_bonds": int(row["num_rotatable_bonds"]),
                    "coordinates": torch.as_tensor(coordinates),
                    "x_init": data.x_init.cpu(),
                    "references": _reference_candidates(reference),
                    "method": method,
                    "status": status,
                }
            )
        failures = sorted(set(missing).union(failed))
        failure_rate = len(failures) / denominator if denominator else 0.0
        rows, method_sample_diagnostics = _evaluate(evaluation_records, args.threshold)
        summaries.extend(_summaries(rows, method, failure_rate, len(missing)))
        sample_diagnostics.extend(method_sample_diagnostics)
        update_diagnostics.extend(
            _diagnostic_summaries(method_sample_diagnostics, method)
        )
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
    with (args.output_dir / "sample_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_diagnostics[0]))
        writer.writeheader()
        writer.writerows(sample_diagnostics)
    with (args.output_dir / "update_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(update_diagnostics[0]))
        writer.writeheader()
        writer.writerows(update_diagnostics)
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
                "update_diagnostics": update_diagnostics,
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
