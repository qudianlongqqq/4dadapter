#!/usr/bin/env python
"""Audit restrained-relaxation labels used by the frozen ECIR error atlas."""

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
import pandas as pd
import torch

from etflow.ecir.audit import (
    classify_relaxation,
    displacement_metrics,
    flexibility_group,
    torsion_change_metrics,
    validity_gains,
)


def _summary_row(name: str, value: str, frame: pd.DataFrame) -> dict:
    numeric = frame.select_dtypes(include=[np.number])
    row = {"dimension": name, "group": value, "records": int(len(frame))}
    for column in numeric.columns:
        row[f"mean_{column}"] = float(numeric[column].mean())
        row[f"p50_{column}"] = float(numeric[column].quantile(0.50))
        row[f"p90_{column}"] = float(numeric[column].quantile(0.90))
        row[f"p95_{column}"] = float(numeric[column].quantile(0.95))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--output_dir", type=Path, default=Path("diagnostics/ecir_mvr/target_audit"))
    args = parser.parse_args()

    rows = []
    for split in args.splits:
        atlas = pd.read_parquet(args.atlas_dir / f"{split}.parquet")
        for atlas_row in atlas.itertuples(index=False):
            record = torch.load(Path(atlas_row.source_path), map_location="cpu", weights_only=False)
            target_payload = torch.load(Path(atlas_row.target_cache_path), map_location="cpu", weights_only=False)
            coordinate_key = str(atlas_row.coordinate_key)
            source = torch.as_tensor(record[coordinate_key], dtype=torch.float32)
            target = torch.as_tensor(target_payload["x_target"], dtype=torch.float32)
            metadata = dict(target_payload.get("target_metadata") or {})
            relaxation = dict(metadata.get("relaxation") or {})
            status, stop_reason = classify_relaxation(metadata)
            displacement = displacement_metrics(source, target)
            torsion = torsion_change_metrics(source, target, record)
            gains = validity_gains(source, target, record)
            rows.append({
                "split": split,
                "molecule_id": str(atlas_row.molecule_id),
                "sample_id": str(atlas_row.sample_id),
                "source_type": str(atlas_row.source_type),
                "flexibility_group": flexibility_group(record),
                "rotatable_bond_count": int(record.get("num_rotatable_bonds", 0)),
                "input_to_target_aligned_RMSD": displacement["aligned_rms_displacement"],
                **displacement,
                **torsion,
                **gains,
                "total_validity_gain": sum(gains.values()),
                "energy_drop": relaxation.get("energy_drop"),
                "optimization_iterations": relaxation.get("steps"),
                "optimizer_status_code": relaxation.get("status_code"),
                "convergence_status": status,
                "acceptance_status": bool(relaxation.get("accepted", False)),
                "acceptance_reason": relaxation.get("rejection_reason") or (
                    "finite_within_drift_and_nonincreasing_energy"
                    if bool(relaxation.get("accepted", False)) else "not_accepted"
                ),
                "stop_reason": stop_reason,
                "target_source": metadata.get("target_source"),
                "force_field_method": relaxation.get("method"),
                "force_field_supported": relaxation.get("supported"),
            })

    frame = pd.DataFrame(rows)
    gain_cutoff = float(frame.total_validity_gain.quantile(0.25))
    displacement_cutoff = float(frame.aligned_rms_displacement.quantile(0.90))
    frame["small_gain_large_displacement"] = (
        (frame.total_validity_gain <= gain_cutoff)
        & (frame.aligned_rms_displacement >= displacement_cutoff)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_target.csv", index=False)

    summary_rows = [_summary_row("all", "all", frame)]
    for dimension in ("source_type", "convergence_status", "flexibility_group"):
        for value, selected in frame.groupby(dimension):
            summary_rows.append(_summary_row(dimension, str(value), selected))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    accepted = frame[frame.acceptance_status]
    correlations = accepted[[
        "aligned_rms_displacement", "torsion_circular_change", "total_validity_gain"
    ]].corr(method="spearman").to_dict()
    machine = {
        "records": int(len(frame)),
        "splits": list(args.splits),
        "status_counts": frame.convergence_status.value_counts().to_dict(),
        "source_counts": frame.source_type.value_counts().to_dict(),
        "small_gain_large_displacement_records": int(frame.small_gain_large_displacement.sum()),
        "small_gain_cutoff": gain_cutoff,
        "large_displacement_cutoff": displacement_cutoff,
        "spearman_correlations_accepted": correlations,
        "optimization_iterations_semantics": "persisted maximum iteration budget; RDKit does not expose actual iteration count in this cache",
        "test_used": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")
    print(json.dumps(machine, indent=2))


if __name__ == "__main__":
    main()
