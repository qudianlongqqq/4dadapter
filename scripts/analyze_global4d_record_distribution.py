#!/usr/bin/env python
"""Explain manifest expansion and Global 4D sampling long-tail structure."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.global4d_performance import (
    PROFILE_SCHEMA_VERSION,
    compact_json,
    numeric_summary,
    pearson_correlation,
    write_csv,
)


def _load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError(f"Invalid manifest: {path}")
    return value


def _scan_cache(root: Path, split: str) -> dict[str, dict[str, Any]]:
    directory = root / split if (root / split).is_dir() else root
    output = {}
    for path in sorted(directory.glob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        sample_id = str(record.get("sample_id", record.get("mol_id", path.stem)))
        output[sample_id] = record
    return output


def _reference_counts(root: Path | None, split: str) -> dict[str, int]:
    if root is None:
        return {}
    records = _scan_cache(root, split)
    counts = defaultdict(int)
    for record in records.values():
        molecule = str(record.get("source_mol_id", record.get("mol_id")))
        refs = torch.as_tensor(record.get("x_ref_candidates", []))
        count = int(refs.size(0)) if refs.ndim == 3 else int(bool(refs.numel()))
        counts[molecule] = max(counts[molecule], count)
    return dict(counts)


def _profile_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row["sample_id"]): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--reference_cache", type=Path)
    parser.add_argument("--profile_csv", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", type=Path, default=Path("reports"))
    parser.add_argument("--top_molecules", type=int, default=20)
    parser.add_argument("--top_records", type=int, default=100)
    args = parser.parse_args()
    manifest = _load_manifest(args.manifest)
    cache = _scan_cache(args.cache_dir, args.split)
    reference_counts = _reference_counts(args.reference_cache, args.split)
    timing = _profile_rows(args.profile_csv)
    missing = [str(row["sample_id"]) for row in manifest["records"] if str(row["sample_id"]) not in cache]
    if missing:
        raise ValueError(f"Cache misses {len(missing)} manifest records; first={missing[:5]}")

    per_molecule: dict[str, dict[str, Any]] = {}
    per_record = []
    for row in manifest["records"]:
        sample_id = str(row["sample_id"])
        molecule = str(row["mol_id"])
        record = cache[sample_id]
        atoms = int(torch.as_tensor(record["atomic_numbers"]).numel())
        rotatable = int(record.get("num_rotatable_bonds", row.get("num_rotatable_bonds", 0)))
        jacobian_columns = 4 * rotatable
        profile = timing.get(sample_id, {})
        seconds = float(profile.get("record_seconds", 0.0) or 0.0)
        rank = float(profile.get("effective_rank_mean", 0.0) or 0.0)
        partial_bytes = int(float(profile.get("partial_file_bytes", 0) or 0))
        item = {
            "sample_id": sample_id,
            "mol_id": molecule,
            "num_atoms": atoms,
            "num_rotatable_bonds": rotatable,
            "jacobian_columns": jacobian_columns,
            "record_seconds": seconds,
            "effective_rank": rank,
            "partial_file_bytes": partial_bytes,
        }
        per_record.append(item)
        summary = per_molecule.setdefault(
            molecule,
            {
                "mol_id": molecule,
                "generated_record_count": 0,
                "reference_conformer_count": reference_counts.get(molecule),
                "num_atoms": atoms,
                "num_rotatable_bonds": rotatable,
                "jacobian_columns": jacobian_columns,
                "measured_records": 0,
                "total_record_seconds": 0.0,
                "mean_record_seconds": 0.0,
            },
        )
        summary["generated_record_count"] += 1
        if seconds:
            summary["measured_records"] += 1
            summary["total_record_seconds"] += seconds

    for summary in per_molecule.values():
        measured = int(summary["measured_records"])
        summary["mean_record_seconds"] = (
            float(summary["total_record_seconds"]) / measured if measured else 0.0
        )
    molecule_rows = sorted(per_molecule.values(), key=lambda row: row["mol_id"])
    measured = [row for row in per_record if row["record_seconds"] > 0]
    slow_molecules = sorted(
        molecule_rows, key=lambda row: row["total_record_seconds"], reverse=True
    )[: args.top_molecules]
    slow_records = sorted(
        measured, key=lambda row: row["record_seconds"], reverse=True
    )[: args.top_records]
    payload = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "cache_dir": str(args.cache_dir.resolve()),
            "reference_cache": str(args.reference_cache.resolve()) if args.reference_cache else None,
            "profile_csv": str(args.profile_csv.resolve()) if args.profile_csv else None,
        },
        "counts": {
            "independent_molecules": len(molecule_rows),
            "generated_records": len(per_record),
            "sum_generated_records_by_molecule": sum(row["generated_record_count"] for row in molecule_rows),
            "measured_timing_records": len(measured),
        },
        "record_count_explanation": (
            "Sampling records are manifest sample_id rows (generated conformers). "
            "The total is the sum of generated records per source molecule; reference "
            "conformers are compared later by the evaluator and do not multiply rollout count."
        ),
        "distributions": {
            "records_per_molecule": numeric_summary([row["generated_record_count"] for row in molecule_rows]),
            "reference_conformers_per_molecule": numeric_summary([row["reference_conformer_count"] for row in molecule_rows if row["reference_conformer_count"] is not None]),
            "atom_count": numeric_summary([row["num_atoms"] for row in per_record]),
            "rotatable_bonds": numeric_summary([row["num_rotatable_bonds"] for row in per_record]),
            "jacobian_columns": numeric_summary([row["jacobian_columns"] for row in per_record]),
            "record_seconds": numeric_summary([row["record_seconds"] for row in measured]),
        },
        "correlations": {
            "time_vs_atoms": pearson_correlation([row["record_seconds"] for row in measured], [row["num_atoms"] for row in measured]),
            "time_vs_rotatable_bonds": pearson_correlation([row["record_seconds"] for row in measured], [row["num_rotatable_bonds"] for row in measured]),
            "time_vs_jacobian_columns": pearson_correlation([row["record_seconds"] for row in measured], [row["jacobian_columns"] for row in measured]),
            "time_vs_effective_rank": pearson_correlation([row["record_seconds"] for row in measured], [row["effective_rank"] for row in measured]),
            "time_vs_partial_file_bytes": pearson_correlation([row["record_seconds"] for row in measured], [row["partial_file_bytes"] for row in measured]),
        },
        "slowest_molecules": slow_molecules,
        "slowest_records": slow_records,
        "compact": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact_json(payload, args.output_dir / "global4d_record_distribution.json")
    write_csv(molecule_rows, args.output_dir / "global4d_record_distribution.csv")
    write_csv(slow_records, args.output_dir / "global4d_slowest_records.csv")
    lines = [
        "# Global 4D record distribution",
        "",
        f"- Independent molecules: {len(molecule_rows)}",
        f"- Generated rollout records: {len(per_record)}",
        f"- Measured timing records: {len(measured)}",
        "- Rollout count is the sum of generated conformers, not generated × reference conformers.",
        "",
        "## Distributions",
        "",
        "```json",
        json.dumps(payload["distributions"], indent=2),
        "```",
        "",
        "## Slowest molecules",
        "",
        "| Molecule | Records | Atoms | Rotatable | Total s | Mean s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in slow_molecules:
        lines.append(
            f"| {row['mol_id']} | {row['generated_record_count']} | {row['num_atoms']} | "
            f"{row['num_rotatable_bonds']} | {row['total_record_seconds']:.6f} | {row['mean_record_seconds']:.6f} |"
        )
    (args.output_dir / "global4d_record_distribution.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
