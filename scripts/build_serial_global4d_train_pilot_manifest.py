#!/usr/bin/env python
"""Select a deterministic, flexibility-stratified formal-train pilot cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.data.flexbond_cache_schema import x_init_sha256
from etflow.formal_large import canonical_sha256


def flexibility_tier(joints: int) -> str:
    return "low" if joints <= 2 else ("medium" if joints <= 5 else "high")


def deterministic_rank(value: str, seed: int) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(), value


def _scan_split(root: Path, *, verify_selected: set[str] | None = None) -> list[dict]:
    rows = []
    files = sorted(root.glob("*.pt"))
    for index, path in enumerate(files):
        record = torch.load(path, map_location="cpu", weights_only=False)
        sample_id = str(record.get("sample_id", record.get("mol_id")))
        mol_id = str(record.get("source_mol_id", record.get("mol_id")))
        rotatable = int(torch.as_tensor(record["rotatable_bond_index"]).size(1))
        row = {
            "mol_id": mol_id,
            "sample_id": sample_id,
            "x_init_hash": str(record["x_init_hash"]),
            "num_rotatable_bonds": rotatable,
            "num_atoms": int(torch.as_tensor(record["x_init"]).size(0)),
            "num_edges": int(torch.as_tensor(record["edge_index"]).size(1)),
            "num_joints": rotatable,
            "flexibility_cohort": flexibility_tier(rotatable),
            "source_file_name": path.name,
        }
        if verify_selected is not None and sample_id in verify_selected:
            actual = x_init_sha256(record["x_init"], record["atomic_numbers"])
            if actual != row["x_init_hash"]:
                raise ValueError(f"x_init_hash mismatch for {sample_id!r}")
        rows.append(row)
        if (index + 1) % 10000 == 0:
            print(f"scanned {index + 1}/{len(files)} {root.name} records", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal_cache", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--high_molecules", type=int, default=1000)
    parser.add_argument("--medium_molecules", type=int, default=334)
    parser.add_argument("--low_molecules", type=int, default=333)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "serial_global4d_train_pilot5000.json"
    identity_path = output / "serial_global4d_train_pilot5000.identity.json"
    if manifest_path.exists() or identity_path.exists():
        raise FileExistsError("Refusing to overwrite an existing pilot manifest")
    train_rows = _scan_split(args.formal_cache / "train")
    sample_ids = [row["sample_id"] for row in train_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Formal train cache contains duplicate sample IDs")
    source_payload = {
        "manifest_version": "1.0",
        "formal_large_split": "train",
        "records": [
            {key: row[key] for key in ("mol_id", "sample_id", "x_init_hash", "num_rotatable_bonds")}
            for row in train_rows
        ],
    }
    source_sha = canonical_sha256(source_payload)
    by_molecule: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows:
        by_molecule[row["mol_id"]].append(row)
    tier_molecules: dict[str, list[str]] = defaultdict(list)
    for mol_id, rows in by_molecule.items():
        tiers = {row["flexibility_cohort"] for row in rows}
        if len(tiers) != 1:
            raise ValueError(f"Flexibility tier changes within molecule {mol_id!r}")
        tier_molecules[tiers.pop()].append(mol_id)
    requested = {
        "high": args.high_molecules,
        "medium": args.medium_molecules,
        "low": args.low_molecules,
    }
    selected_molecules = []
    for tier in ("high", "medium", "low"):
        ordered = sorted(
            tier_molecules[tier], key=lambda value: deterministic_rank(value, args.seed)
        )
        if len(ordered) < requested[tier]:
            raise ValueError(f"Not enough {tier} molecules for requested pilot")
        selected_molecules.extend(ordered[: requested[tier]])
    selected_set = set(selected_molecules)
    selected_rows = [row for row in train_rows if row["mol_id"] in selected_set]
    selected_rows.sort(
        key=lambda row: (
            deterministic_rank(row["mol_id"], args.seed),
            row["sample_id"],
        )
    )
    # Recompute coordinate hashes for every selected record, not only trust the
    # persisted formal-cache value used during inventory selection.
    verified = {}
    train_root = args.formal_cache / "train"
    for row in selected_rows:
        record = torch.load(
            train_root / row["source_file_name"], map_location="cpu", weights_only=False
        )
        actual = x_init_sha256(record["x_init"], record["atomic_numbers"])
        if actual != row["x_init_hash"]:
            raise ValueError(f"x_init_hash mismatch for {row['sample_id']!r}")
        verified[row["sample_id"]] = row
    if len(verified) != len(selected_rows):
        raise ValueError("Pilot verification did not recover every selected record")
    # Test identities are used only for leakage rejection, never selection.
    val_rows = _scan_split(args.formal_cache / "val")
    test_rows = _scan_split(args.formal_cache / "test")
    selected_samples = {row["sample_id"] for row in selected_rows}
    selected_mols = {row["mol_id"] for row in selected_rows}
    overlaps = {
        "val_sample": sorted(selected_samples & {row["sample_id"] for row in val_rows}),
        "val_molecule": sorted(selected_mols & {row["mol_id"] for row in val_rows}),
        "test_sample": sorted(selected_samples & {row["sample_id"] for row in test_rows}),
        "test_molecule": sorted(selected_mols & {row["mol_id"] for row in test_rows}),
    }
    if any(overlaps.values()):
        raise ValueError(f"Pilot split leakage detected: {overlaps}")
    manifest = {
        "manifest_version": "1.0",
        "formal_large_split": "train",
        "selection_seed": args.seed,
        "selection_strategy": "deterministic_complete_molecules_flexibility_60_20_20",
        "test_used_for_selection": False,
        "test_used_for_leakage_check_only": True,
        "source_manifest_canonical_sha256": source_sha,
        "selected_molecule_count": len(selected_mols),
        "selected_record_count": len(selected_rows),
        "requested_molecules": requested,
        "records": selected_rows,
    }
    pilot_sha = canonical_sha256(manifest)
    tier_records = Counter(row["flexibility_cohort"] for row in selected_rows)
    identity = {
        "pilot_manifest_canonical_sha256": pilot_sha,
        "source_manifest_canonical_sha256": source_sha,
        "source_record_count": len(train_rows),
        "source_molecule_count": len(by_molecule),
        "selected_record_count": len(selected_rows),
        "selected_molecule_count": len(selected_mols),
        "record_tier_counts": dict(tier_records),
        "molecule_tier_counts": requested,
        "atom_count": {
            "min": min(row["num_atoms"] for row in selected_rows),
            "max": max(row["num_atoms"] for row in selected_rows),
            "mean": sum(row["num_atoms"] for row in selected_rows) / len(selected_rows),
        },
        "edge_count": {
            "min": min(row["num_edges"] for row in selected_rows),
            "max": max(row["num_edges"] for row in selected_rows),
            "mean": sum(row["num_edges"] for row in selected_rows) / len(selected_rows),
        },
        "joint_count": dict(Counter(row["num_joints"] for row in selected_rows)),
        "overlaps": overlaps,
    }
    atomic_json_save(manifest, manifest_path)
    atomic_json_save(identity, identity_path)
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
