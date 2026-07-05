#!/usr/bin/env python
"""Per-sample diagnosis for upstream, Cartesian, and FlexBond refinements."""

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


def _minimum_rmsd(coordinates: torch.Tensor, references: torch.Tensor) -> float:
    return min(float(kabsch_rmsd(coordinates, reference)) for reference in references)


def _load_samples(path: Path | None, manifest: dict, expected_method: str) -> dict[str, dict]:
    if path is None:
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("manifest", {}).get("records") != manifest["records"]:
        raise ValueError(f"Sample payload uses a different manifest: {path}")
    records = {}
    for record in payload.get("records", []):
        sample_id = str(record["sample_id"])
        if sample_id in records:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}")
        if record.get("method_name") != expected_method:
            raise ValueError(f"Unexpected method in {path}: {record.get('method_name')}")
        records[sample_id] = record
    return records


def _sample_result(
    record: dict | None, x_init: torch.Tensor, references: torch.Tensor
) -> tuple[float, float, float, bool]:
    failed = (
        record is None
        or record.get("status") != "success"
        or record.get("x_refined") is None
    )
    coordinates = x_init if failed else torch.as_tensor(record["x_refined"])
    displacement = torch.linalg.norm(coordinates - x_init, dim=-1)
    mean_update = (
        float(record.get("mean_update_norm", displacement.mean()))
        if record is not None
        else 0.0
    )
    max_update = (
        float(record.get("max_update_norm", displacement.max()))
        if record is not None
        else 0.0
    )
    return _minimum_rmsd(coordinates, references), mean_update, max_update, failed


def _initial_bin(value: float) -> str:
    if value < 0.5:
        return "0-0.5"
    if value < 1.0:
        return "0.5-1.0"
    if value < 1.5:
        return "1.0-1.5"
    if value < 2.0:
        return "1.5-2.0"
    return ">2.0"


def _update_bin(value: float) -> str:
    if value < 0.01:
        return "0-0.01"
    if value < 0.05:
        return "0.01-0.05"
    if value < 0.1:
        return "0.05-0.1"
    if value < 0.2:
        return "0.1-0.2"
    if value < 0.5:
        return "0.2-0.5"
    return ">=0.5"


def _method_summary(rows: list[dict], method: str, group: str = "all") -> dict:
    prefix = "cartesian" if method == "cartesian_adapter" else "flexbond"
    if method == "upstream_only":
        delta = np.zeros(len(rows), dtype=float)
        failure = np.zeros(len(rows), dtype=float)
        update_mean = np.zeros(len(rows), dtype=float)
        update_max = np.zeros(len(rows), dtype=float)
    else:
        delta = np.asarray([row[f"delta_{prefix}"] for row in rows], dtype=float)
        failure = np.asarray([row[f"{prefix}_failure"] for row in rows], dtype=float)
        update_mean = np.asarray(
            [row[f"{prefix}_update_norm_mean"] for row in rows], dtype=float
        )
        update_max = np.asarray(
            [row[f"{prefix}_max_atom_displacement"] for row in rows], dtype=float
        )
    return {
        "method": method,
        "group": group,
        "num_samples": len(rows),
        "fraction_improved": float((delta < -1.0e-6).mean()),
        "fraction_worsened": float((delta > 1.0e-6).mean()),
        "mean_delta_rmsd": float(delta.mean()),
        "median_delta_rmsd": float(np.median(delta)),
        "failure_rate": float(failure.mean()),
        "mean_update_norm": float(update_mean.mean()),
        "median_update_norm": float(np.median(update_mean)),
        "max_update_norm": float(update_max.max()),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
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
    parser.add_argument("--upstream_only", action="store_true")
    parser.add_argument("--cartesian_samples", type=Path)
    parser.add_argument("--flexbond_samples", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    if args.cartesian_samples is None and args.flexbond_samples is None:
        raise ValueError("At least one adapter sample file is required.")

    manifest = load_eval_manifest(args.manifest)
    inference_dataset = FlexBondInferenceDataset(args.inference_cache, args.split)
    inference = validate_dataset_against_manifest(inference_dataset, manifest)
    reference_dataset = FlexBondOptimizerDataset(
        args.reference_cache, args.split, validate=True
    )
    reference_by_id = {str(data.mol_id): data for data in reference_dataset}
    cartesian = _load_samples(
        args.cartesian_samples, manifest, "cartesian_adapter"
    )
    flexbond = _load_samples(args.flexbond_samples, manifest, "flexbond4d_adapter")

    rows = []
    for manifest_row in manifest["records"]:
        sample_id = str(manifest_row["sample_id"])
        data = inference[sample_id]
        reference_data = reference_by_id.get(sample_id)
        if reference_data is None:
            raise ValueError(f"Reference cache is missing sample {sample_id!r}")
        references = _references(reference_data)
        x_init = data.x_init.cpu()
        upstream_rmsd = _minimum_rmsd(x_init, references)
        cart_rmsd, cart_update, cart_max, cart_failed = _sample_result(
            cartesian.get(sample_id), x_init, references
        )
        flex_rmsd, flex_update, flex_max, flex_failed = _sample_result(
            flexbond.get(sample_id), x_init, references
        )
        if args.cartesian_samples is None:
            cart_rmsd, cart_update, cart_max, cart_failed = (
                float("nan"),
                float("nan"),
                float("nan"),
                True,
            )
        if args.flexbond_samples is None:
            flex_rmsd, flex_update, flex_max, flex_failed = (
                float("nan"),
                float("nan"),
                float("nan"),
                True,
            )
        rows.append(
            {
                "sample_id": sample_id,
                "mol_id": str(manifest_row["mol_id"]),
                "num_rotatable_bonds": int(manifest_row["num_rotatable_bonds"]),
                "initial_rmsd_to_ref": upstream_rmsd,
                "upstream_rmsd": upstream_rmsd,
                "cartesian_rmsd": cart_rmsd,
                "flexbond_rmsd": flex_rmsd,
                "delta_cartesian": cart_rmsd - upstream_rmsd,
                "delta_flexbond": flex_rmsd - upstream_rmsd,
                "cartesian_improved": cart_rmsd < upstream_rmsd - 1.0e-6,
                "flexbond_improved": flex_rmsd < upstream_rmsd - 1.0e-6,
                "cartesian_update_norm_mean": cart_update,
                "flexbond_update_norm_mean": flex_update,
                "cartesian_max_atom_displacement": cart_max,
                "flexbond_max_atom_displacement": flex_max,
                "cartesian_failure": cart_failed,
                "flexbond_failure": flex_failed,
            }
        )

    methods = []
    if args.upstream_only:
        methods.append("upstream_only")
    if args.cartesian_samples is not None:
        methods.append("cartesian_adapter")
    if args.flexbond_samples is not None:
        methods.append("flexbond4d_adapter")
    summary = [_method_summary(rows, method) for method in methods]

    by_rotatable = []
    subsets = {"all": 0, "rotatable_ge_3": 3, "rotatable_ge_5": 5, "rotatable_ge_6": 6}
    for method in methods:
        for name, minimum in subsets.items():
            chosen = [row for row in rows if row["num_rotatable_bonds"] >= minimum]
            if chosen:
                by_rotatable.append(_method_summary(chosen, method, name))

    by_initial = []
    for method in methods:
        for name in ("0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", ">2.0"):
            chosen = [row for row in rows if _initial_bin(row["upstream_rmsd"]) == name]
            if chosen:
                by_initial.append(_method_summary(chosen, method, name))

    by_update = []
    for method, prefix in (
        ("cartesian_adapter", "cartesian"),
        ("flexbond4d_adapter", "flexbond"),
    ):
        if method not in methods:
            continue
        for name in ("0-0.01", "0.01-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.5", ">=0.5"):
            chosen = [
                row
                for row in rows
                if _update_bin(row[f"{prefix}_update_norm_mean"]) == name
            ]
            if chosen:
                by_update.append(_method_summary(chosen, method, name))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "diagnostics_per_sample.csv", rows)
    _write_csv(args.output_dir / "diagnostics_summary.csv", summary)
    _write_csv(args.output_dir / "diagnostics_by_rotatable.csv", by_rotatable)
    _write_csv(args.output_dir / "diagnostics_by_initial_rmsd_bin.csv", by_initial)
    _write_csv(args.output_dir / "diagnostics_by_update_norm_bin.csv", by_update)
    print(f"Wrote diagnostics for {len(rows)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
