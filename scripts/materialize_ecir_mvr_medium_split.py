#!/usr/bin/env python
"""Materialize the frozen 5000/500 MCVR medium molecule split without test access."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch


SCHEMA_VERSION = "ecir-mvr-medium-split-v1"


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_listing(split_dir: Path) -> dict[str, Any]:
    names = sorted(path.name for path in split_dir.glob("*.pt"))
    return {"records": len(names), "listing_sha256": _canonical_sha(names)}


def _group(rotatable: int) -> str:
    if rotatable <= 2:
        return "rotatable_le_2"
    if rotatable <= 5:
        return "rotatable_3_5"
    return "rotatable_ge_6"


def _size_group(atoms: int) -> str:
    if atoms <= 20:
        return "size_small_le_20"
    if atoms <= 40:
        return "size_medium_21_40"
    return "size_large_ge_41"


def _record_summary(path: Path, split: str) -> dict[str, Any]:
    record = torch.load(path, map_location="cpu", weights_only=False)
    molecule = str(record.get("source_mol_id", record.get("mol_id")))
    rotatable = int(record.get("num_rotatable_bonds", 0))
    atoms = int(record.get("num_atoms", torch.as_tensor(record["x_init"]).size(0)))
    ring = bool(torch.as_tensor(record.get("bond_is_in_ring", [])).any())
    return {
        "split": split,
        "molecule_id": molecule,
        "source_id": str(record.get("sample_id", record.get("source_record_id", path.stem))),
        "source_path": str(path.resolve()),
        "num_rotatable_bonds": rotatable,
        "rotatable_group": _group(rotatable),
        "ring_group": "ring" if ring else "non_ring",
        "num_atoms": atoms,
        "size_group": _size_group(atoms),
    }


def _catalog(
    split_dir: Path, split: str, target: int, seed: int, *, scan_all: bool = False
) -> list[dict[str, Any]]:
    paths = sorted(split_dir.glob("*.pt"))
    rng = random.Random(seed)
    rng.shuffle(paths)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    # A 4x candidate pool makes the proportional strata allocation stable while
    # avoiding any result-dependent selection.
    candidate_target = len(paths) if scan_all else min(len(paths), target * 4)
    for path in paths:
        item = _record_summary(path, split)
        if item["molecule_id"] in seen:
            continue
        seen.add(item["molecule_id"])
        result.append(item)
        if len(result) >= candidate_target:
            break
    if len(result) < target:
        raise ValueError(f"{split} has only {len(result)} unique molecules; need {target}")
    return result


def _stratum(item: dict[str, Any]) -> tuple[str, str, str]:
    return item["rotatable_group"], item["ring_group"], item["size_group"]


def _allocate(
    catalog: list[dict[str, Any]], target: int, seed: int, *, min_non_ring: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in catalog:
        groups[_stratum(item)].append(item)
    total = len(catalog)
    raw = {key: target * len(values) / total for key, values in groups.items()}
    quotas = {key: min(len(groups[key]), int(value)) for key, value in raw.items()}
    remaining = target - sum(quotas.values())
    ordering = sorted(
        groups,
        key=lambda key: (-(raw[key] - int(raw[key])), _canonical_sha([seed, key])),
    )
    while remaining:
        progressed = False
        for key in ordering:
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("unable to complete proportional stratum allocation")
    selected = []
    for key in sorted(groups):
        values = sorted(groups[key], key=lambda item: _canonical_sha([seed, item["molecule_id"]]))
        selected.extend(values[: quotas[key]])
    # Non-ring molecules are genuinely rare in formal-large. Freeze an explicit
    # rare-stratum floor without using any generated-source or model result. If
    # the parent candidate pool contains fewer than the floor, include all of it.
    rare = sorted(
        (item for item in catalog if item["ring_group"] == "non_ring"),
        key=lambda item: _canonical_sha([seed, "non_ring", item["molecule_id"]]),
    )
    desired = min(min_non_ring, len(rare))
    selected_ids = {item["molecule_id"] for item in selected}
    missing = [item for item in rare if item["molecule_id"] not in selected_ids][:desired]
    current_rare = sum(item["ring_group"] == "non_ring" for item in selected)
    missing = missing[: max(0, desired - current_rare)]
    for addition in missing:
        same_stratum = [
            item for item in selected
            if item["ring_group"] == "ring"
            and item["rotatable_group"] == addition["rotatable_group"]
            and item["size_group"] == addition["size_group"]
        ]
        removable = same_stratum or [item for item in selected if item["ring_group"] == "ring"]
        removal = max(removable, key=lambda item: _canonical_sha([seed, "replace", item["molecule_id"]]))
        selected.remove(removal)
        selected.append(addition)
    return sorted(selected, key=lambda item: item["molecule_id"])


def _assign_roles(items: list[dict[str, Any]], cartesian_count: int, seed: int) -> None:
    if not 0 < cartesian_count < len(items):
        raise ValueError("each split requires ETFlow and Cartesian molecules")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[_stratum(item)].append(item)
    raw = {key: cartesian_count * len(values) / len(items) for key, values in groups.items()}
    quotas = {key: min(len(groups[key]), int(value)) for key, value in raw.items()}
    remaining = cartesian_count - sum(quotas.values())
    ordering = sorted(groups, key=lambda key: (-(raw[key] - int(raw[key])), _canonical_sha([seed, key])))
    for key in ordering:
        if not remaining:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("unable to assign requested Cartesian roles")
    for key, values in groups.items():
        ranked = sorted(values, key=lambda item: _canonical_sha([seed, "role", item["molecule_id"]]))
        cartesian = {item["molecule_id"] for item in ranked[: quotas[key]]}
        for item in values:
            item["source_role"] = "cartesian" if item["molecule_id"] in cartesian else "etflow"


def _counts(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "molecules": len(items),
        "source_roles": dict(Counter(item["source_role"] for item in items)),
        "rotatable_groups": dict(Counter(item["rotatable_group"] for item in items)),
        "ring_groups": dict(Counter(item["ring_group"] for item in items)),
        "size_groups": dict(Counter(item["size_group"] for item in items)),
        "joint_strata": {
            "|".join(key): value
            for key, value in sorted(Counter(_stratum(item) for item in items).items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/ecir_mvr/medium"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_molecules", type=int, default=5000)
    parser.add_argument("--val_molecules", type=int, default=500)
    parser.add_argument("--train_cartesian_molecules", type=int, default=2500)
    parser.add_argument("--val_cartesian_molecules", type=int, default=200)
    args = parser.parse_args()
    if args.seed != 42 or args.train_molecules != 5000 or args.val_molecules != 500:
        raise ValueError("medium split is frozen to seed42 and 5000/500 molecules")
    if "test" in {part.lower() for part in args.formal_cache.resolve().parts}:
        raise ValueError("formal cache root may not be a test path")
    train_dir, val_dir = args.formal_cache / "train", args.formal_cache / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError("formal train/val cache directories are required")

    parent = {
        "formal_cache": str(args.formal_cache.resolve()),
        "train": _formal_listing(train_dir),
        "val": _formal_listing(val_dir),
        "test_paths_opened": 0,
    }
    parent["identity_sha256"] = _canonical_sha(parent)
    train_catalog = _catalog(train_dir, "train", args.train_molecules, args.seed)
    val_catalog = _catalog(val_dir, "val", args.val_molecules, args.seed + 1, scan_all=True)
    train = _allocate(train_catalog, args.train_molecules, args.seed, min_non_ring=50)
    val = _allocate(val_catalog, args.val_molecules, args.seed + 1, min_non_ring=20)
    _assign_roles(train, args.train_cartesian_molecules, args.seed)
    # Validation source roles are assigned only after every selected molecule
    # has undergone the same safe Cartesian candidate protocol. This allows the
    # pre-registered severity floors to include every naturally available severe
    # source without changing the molecule cohort.
    for item in val:
        item["source_role"] = "candidate"
    leakage = sorted({item["molecule_id"] for item in train} & {item["molecule_id"] for item in val})
    if leakage:
        raise ValueError(f"formal train/val molecule leakage: {leakage[:5]}")
    if not any(item["ring_group"] == "non_ring" for item in val):
        raise ValueError("validation selection contains no non-ring molecules")
    if sum(item["rotatable_group"] == "rotatable_ge_6" for item in val) < 20:
        raise ValueError("validation selection contains too few high-flex molecules")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, items in (("train", train), ("val", val)):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "seed": args.seed,
            "selection_method": "seeded proportional joint-stratum allocation before model training",
            "parent_formal_split_identity_sha256": parent["identity_sha256"],
            "records": items,
        }
        path = args.output_dir / f"split_{split}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        outputs[split] = {"path": str(path.resolve()), "sha256": _file_sha(path), **_counts(items)}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "parent_formal_split": parent,
        "splits": outputs,
        "train_val_overlap": 0,
        "test_paths_opened": 0,
        "stratification_frozen_before_training": True,
        "stratification": ["rotatable_group", "ring_group", "size_group", "source_role"],
        "cartesian_severity_rule": "safe rollout severity is graded later using train-only frozen quantiles before training",
        "validation_source_assignment": {
            "candidate_molecules": args.val_molecules,
            "final_cartesian_molecules": args.val_cartesian_molecules,
            "final_etflow_molecules": args.val_molecules - args.val_cartesian_molecules,
            "frozen_severity_floors": {"severe": 20, "medium": 50, "mild": 50},
            "selection_timing": "after safe source generation and before MCVR training",
        },
        "rare_stratum_policy": {
            "train_non_ring_floor": 50,
            "validation_non_ring_floor": 20,
            "policy": "include the floor or every available non-ring candidate when the parent pool has fewer",
            "train_non_ring_candidates": sum(item["ring_group"] == "non_ring" for item in train_catalog),
            "validation_non_ring_candidates": sum(item["ring_group"] == "non_ring" for item in val_catalog),
        },
    }
    metadata["identity_sha256"] = _canonical_sha(metadata)
    (args.output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
