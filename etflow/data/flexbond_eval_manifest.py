"""Frozen evaluation-cohort manifests for fair adapter comparisons."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EVAL_MANIFEST_VERSION = "1.0"


def data_manifest_row(data: Any) -> dict[str, Any]:
    return {
        "mol_id": str(data.source_mol_id),
        "sample_id": str(data.sample_id),
        "x_init_hash": str(data.x_init_hash),
        "num_rotatable_bonds": int(data.num_rotatable_bonds.item()),
    }


def build_eval_manifest(dataset: Iterable[Any]) -> dict[str, Any]:
    rows = [data_manifest_row(data) for data in dataset]
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Evaluation manifest contains duplicate sample_id values.")
    per_molecule = {}
    for row in rows:
        previous = per_molecule.setdefault(row["mol_id"], row["num_rotatable_bonds"])
        if previous != row["num_rotatable_bonds"]:
            raise ValueError("num_rotatable_bonds differs within one molecule cohort.")
    return {
        "manifest_version": EVAL_MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": rows,
    }


def limit_manifest_molecules(manifest: dict[str, Any], limit: int) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("Molecule limit must be positive.")
    selected = []
    molecule_ids: set[str] = set()
    for row in manifest["records"]:
        mol_id = str(row["mol_id"])
        if mol_id in molecule_ids:
            selected.append(row)
        elif len(molecule_ids) < limit:
            molecule_ids.add(mol_id)
            selected.append(row)
    return {**manifest, "records": selected}


def load_eval_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if str(manifest.get("manifest_version")) != EVAL_MANIFEST_VERSION:
        raise ValueError("Unsupported or missing evaluation manifest version.")
    rows = manifest.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation manifest has no records.")
    required = {"mol_id", "sample_id", "x_init_hash", "num_rotatable_bonds"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Manifest row is missing fields: {sorted(missing)}.")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation manifest contains duplicate sample_id values.")
    return manifest


def validate_dataset_against_manifest(dataset: Iterable[Any], manifest: dict) -> dict[str, Any]:
    by_id = {str(data.sample_id): data for data in dataset}
    expected_ids = {str(row["sample_id"]) for row in manifest["records"]}
    missing = sorted(expected_ids.difference(by_id))
    if missing:
        raise ValueError(f"Inference cache is missing manifest sample ids: {missing[:20]}.")
    for row in manifest["records"]:
        data = by_id[str(row["sample_id"])]
        actual = data_manifest_row(data)
        if actual != {
            "mol_id": str(row["mol_id"]),
            "sample_id": str(row["sample_id"]),
            "x_init_hash": str(row["x_init_hash"]),
            "num_rotatable_bonds": int(row["num_rotatable_bonds"]),
        }:
            raise ValueError(f"Manifest mismatch for sample {row['sample_id']!r}.")
    return by_id


def write_eval_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
