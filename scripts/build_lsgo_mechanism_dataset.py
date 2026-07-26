#!/usr/bin/env python3
"""Freeze a TRAIN-only, exposed-identity-disjoint mechanism cohort."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import yaml
from rdkit import Chem

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.bat_refinement import canonical_rotatable_torsions
from etflow.ecir.lsgo_io import atomic_json, atomic_torch_save, file_sha256
from etflow.ecir.target_building import _record_to_rdkit_mapping

OUT = ROOT / "reports/ecir_mvr/lsgo_mechanism"
CONFIG_PATH = ROOT / "configs/ecir_mvr_lsgo_mechanism.yaml"
COMPACT_PATH = OUT / "manifests/MECHANISM_CONFIRM_COMPACT.pt"


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def chemical_features(item):
    molecule, _ = _record_to_rdkit_mapping(item["record"])
    torsions, _, _ = canonical_rotatable_torsions(item["record"])
    amide = molecule.HasSubstructMatch(Chem.MolFromSmarts("[NX3][CX3](=[OX1])"))
    return {
        "heavy_atom_count": int(sum(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms())),
        "rotatable_bond_count": int(torsions.size(0)), "aromatic": any(atom.GetIsAromatic() for atom in molecule.GetAtoms()),
        "ring": molecule.GetRingInfo().NumRings() > 0, "amide_like": bool(amide),
    }


def bin_name(count, bins):
    for name, definition in bins.items():
        maximum = definition["maximum"]
        if count >= definition["minimum"] and (maximum is None or count <= maximum): return name
    raise RuntimeError(f"rotor count outside bins: {count}")


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")); OUT.mkdir(parents=True, exist_ok=True)
    train_path = Path(config["dataset"]["training_compact"])
    if file_sha256(train_path) != config["dataset"]["training_compact_sha256"]: raise RuntimeError("training compact SHA mismatch")
    payload = torch.load(train_path, map_location="cpu", weights_only=False)
    if payload.get("formal_test_records_read") != 0 or payload.get("frozen_holdout_records_read") != 0: raise RuntimeError("protected split access")
    bat = json.loads(Path(config["dataset"]["bat_identity"]).read_text(encoding="utf-8"))
    exposed = set(bat["cohorts"]["BAT_EXTERNAL_CONFIRM"]["molecule_ids"])
    train_ids = set(bat["cohorts"]["BAT_TRAIN"]["molecule_ids"])
    if train_ids & exposed: raise RuntimeError("BAT train/external overlap")
    candidates = []
    for item in payload["items"]:
        if item["partition"] != "train" or len(item["references"]) < 2: continue
        features = chemical_features(item); flex = bin_name(features["rotatable_bond_count"], config["dataset"]["flex_bins"])
        candidates.append((item, features, flex))
    selected = []
    for flex, definition in config["dataset"]["flex_bins"].items():
        rows = [row for row in candidates if row[2] == flex]
        rows.sort(key=lambda row: hashlib.sha256(f"{config['dataset']['selection_seed']}|{row[0]['molecule_id']}".encode()).hexdigest())
        chosen = rows[: int(definition["molecules"])]
        if len(chosen) != int(definition["molecules"]): raise RuntimeError(f"insufficient {flex}")
        selected.extend(chosen)
    selected.sort(key=lambda row: hashlib.sha256(str(row[0]["molecule_id"]).encode()).hexdigest())
    items, entries = [], []
    for item, features, flex in selected:
        copied = dict(item); copied["partition"] = "mechanism_confirm"; copied["mechanism_features"] = features; copied["flex_bin"] = flex
        items.append(copied); entries.append({"molecule_id": str(item["molecule_id"]), "references": len(item["references"]), "sources": len(item["sources"]), "flex_bin": flex, **features})
    identities = sorted(row["molecule_id"] for row in entries)
    compact = {"schema_version": "mcvr-lsgo-mechanism-compact-v1", "items": items, "formal_test_records_read": 0, "frozen_holdout_records_read": 0}
    atomic_torch_save(COMPACT_PATH, compact)
    identity = {
        "schema_version": "mcvr-lsgo-mechanism-identity-v1", "status": "FROZEN", "selection_seed": config["dataset"]["selection_seed"],
        "selection_rule": "TRAIN-only; >=2 References; SHA-ranked within preregistered free-rotor bins 0-2/3-4/>=5",
        "molecule_count": len(items), "source_record_count": sum(len(item["sources"]) for item in items), "reference_count": sum(len(item["references"]) for item in items),
        "flex_histogram": dict(Counter(row["flex_bin"] for row in entries)), "molecule_ids": identities, "entries": entries,
        "molecule_identity_sha256": canonical_sha(identities), "compact_path": str(COMPACT_PATH), "compact_sha256": file_sha256(COMPACT_PATH),
        "training_compact_sha256": file_sha256(train_path), "bat_external_overlap": len(set(identities) & exposed),
        "historical_exposed_union_count": bat["historical_exposed_union_count"] + len(bat["cohorts"]["BAT_EXTERNAL_CONFIRM"]["molecule_ids"]),
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    }
    atomic_json(OUT / "DATASET_IDENTITY.json", identity)
    print("LSGO_MECHANISM_DATASET_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
