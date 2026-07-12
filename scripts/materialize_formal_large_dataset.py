#!/usr/bin/env python
"""Select and materialize frozen formal-large cache splits from ETFlow candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.global_coupled_4d_sampling import atomic_json_save, file_sha256
from etflow.formal_large import (
    SEED,
    TEST_MOLECULES,
    TRAIN_MOLECULES,
    TRAIN_PAIRS_PER_MOLECULE,
    VAL_MOLECULES,
    VAL_PAIRS_PER_MOLECULE,
    assert_disjoint_splits,
    canonical_sha256,
    flexibility_tier,
    pair_count_distribution,
    select_pair_records,
)


def _candidate_records(root: Path, split: str) -> list[dict]:
    directory = root / split
    rows = []
    for path in sorted(directory.glob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        rows.append({
            **record,
            "sample_id": str(record.get("sample_id", record["mol_id"])),
            "_source_path": str(path.resolve()),
        })
    return rows


def _manifest(split: str, records: list[dict]) -> dict:
    return {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_large_split": split,
        "selection_seed": SEED,
        "records": [
            {
                "mol_id": str(record.get("source_mol_id", record["mol_id"])),
                "sample_id": str(record["sample_id"]),
                "x_init_hash": str(record["x_init_hash"]),
                "num_rotatable_bonds": int(record["num_rotatable_bonds"]),
            }
            for record in records
        ],
    }


def _cache_hash(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["sample_id"]).encode("utf-8"))
        digest.update(file_sha256(record["_source_path"]).encode("ascii"))
    return digest.hexdigest()


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_cache", required=True, type=Path)
    parser.add_argument(
        "--output_cache",
        type=Path,
        default=Path("data/flexbond_cache_formal_large"),
    )
    parser.add_argument("--manifest_dir", type=Path, default=Path("manifests"))
    parser.add_argument(
        "--report_json",
        type=Path,
        default=Path("reports/formal_large_dataset_report.json"),
    )
    args = parser.parse_args()

    candidates = {
        split: _candidate_records(args.candidate_cache, split)
        for split in ("train", "val", "test")
    }
    targets = {
        "train": (TRAIN_MOLECULES, TRAIN_PAIRS_PER_MOLECULE),
        "val": (VAL_MOLECULES, VAL_PAIRS_PER_MOLECULE),
        "test": (TEST_MOLECULES, None),
    }
    availability = {}
    sufficient = True
    eligible_candidates = {}
    for split, (target, _) in targets.items():
        pair_cap = targets[split][1]
        counts = Counter(
            str(row.get("source_mol_id", row.get("mol_id")))
            for row in candidates[split]
        )
        eligible_ids = {
            molecule_id
            for molecule_id, count in counts.items()
            if pair_cap is None or count >= pair_cap
        }
        eligible_candidates[split] = [
            row
            for row in candidates[split]
            if str(row.get("source_mol_id", row.get("mol_id"))) in eligible_ids
        ]
        available = len(eligible_ids)
        target_pairs = target * (pair_cap or 1)
        available_pairs = sum(
            min(counts[molecule_id], pair_cap or counts[molecule_id])
            for molecule_id in eligible_ids
        )
        availability[split] = {
            "source_path": str((args.candidate_cache / split).resolve()),
            "upstream_etflow_paths": sorted({
                str(row.get("metadata", {}).get("init_path", ""))
                for row in candidates[split]
                if row.get("metadata", {}).get("init_path")
            }),
            "reference_paths": sorted({
                str(row.get("metadata", {}).get("reference_path", ""))
                for row in candidates[split]
                if row.get("metadata", {}).get("reference_path")
            }),
            "available_molecules": available,
            "available_pairs": available_pairs,
            "target_molecules": target,
            "target_pairs": target_pairs,
            "missing_molecules": max(target - available, 0),
            "missing_pairs": max(target_pairs - available_pairs, 0),
        }
        sufficient &= available >= target and available_pairs >= target_pairs

    if not sufficient:
        payload = {
            "status": "insufficient_etflow_outputs",
            "selection_seed": SEED,
            "availability": availability,
            "rdkit_random_fallback_used": False,
        }
        atomic_json_save(payload, args.report_json)
        args.report_json.with_suffix(".md").write_text(
            "# Formal-large dataset unavailable\n\n```json\n"
            + json.dumps(payload, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
        raise SystemExit(2)

    selected = {
        split: select_pair_records(
            eligible_candidates[split],
            molecule_limit=target,
            pairs_per_molecule=pair_cap,
            seed=SEED,
        )
        for split, (target, pair_cap) in targets.items()
    }
    assert_disjoint_splits(selected)
    manifests = {split: _manifest(split, rows) for split, rows in selected.items()}

    args.output_cache.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, records in selected.items():
        destination = args.output_cache / split
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty split: {destination}")
        for record in records:
            source = Path(record["_source_path"])
            target = destination / source.name
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        atomic_json_save(
            manifests[split], args.manifest_dir / f"formal_large_{split}.json"
        )

    split_reports = {}
    for split, records in selected.items():
        molecule_rows = {}
        for record in records:
            molecule_rows.setdefault(
                str(record.get("source_mol_id", record["mol_id"])), record
            )
        rotations = [int(row["num_rotatable_bonds"]) for row in molecule_rows.values()]
        atoms = [
            int(torch.as_tensor(row["atomic_numbers"]).numel())
            for row in molecule_rows.values()
        ]
        tiers = Counter(flexibility_tier(value) for value in rotations)
        split_reports[split] = {
            "molecule_count": len(molecule_rows),
            "pair_count": len(records),
            "pairs_per_molecule_distribution": pair_count_distribution(records),
            "atom_count_distribution": _distribution(atoms),
            "rotatable_bond_distribution": _distribution(rotations),
            "flexibility_counts": dict(tiers),
            "upstream_etflow_sources": sorted(
                {str(row["generator_checkpoint"]) for row in records}
            ),
            "reference_sources": sorted(
                {
                    str(row.get("metadata", {}).get("reference_path"))
                    for row in records
                }
            ),
            "manifest_sha256": canonical_sha256(manifests[split]),
            "cache_sha256": _cache_hash(records),
        }
    payload = {
        "status": "ready",
        "selection_seed": SEED,
        "splits": split_reports,
        "overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
        "rdkit_random_fallback_used": False,
    }
    atomic_json_save(payload, args.report_json)
    args.report_json.with_suffix(".md").write_text(
        "# Formal-large dataset report\n\n```json\n"
        + json.dumps(payload, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
