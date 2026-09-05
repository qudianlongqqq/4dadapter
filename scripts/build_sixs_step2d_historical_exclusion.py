#!/usr/bin/env python3
"""Build the STEP 2D canonical historical-exclusion identity union.

The script reads identity/topology fields only. It does not load scientific result
tables, run any model, or compute any geometry/performance metric.
"""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rdkit import Chem, RDLogger


IDENTITY_DEFINITION = "sixs-canonical-isomeric-atom-map-free-h-normalized-v1"


def canonical_identity(mol: Chem.Mol) -> str:
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    copy = Chem.RemoveHs(copy, sanitize=True)
    Chem.SanitizeMol(copy)
    value = Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)
    if not value:
        raise ValueError("Empty canonical molecular identity")
    return value


def identity_from_cache(path: Path) -> str:
    record = torch.load(path, map_location="cpu", weights_only=False)
    smiles = record.get("smiles")
    mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
    if mol is None:
        raise ValueError(f"No valid molecular identity in {path}")
    return canonical_identity(mol)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def add(
    rows: dict[str, dict[str, set[str]]],
    identity: str,
    *,
    reason: str,
    source: str,
    role: str,
) -> None:
    row = rows[identity]
    row["reason_excluded"].add(reason)
    row["source_manifest"].add(source)
    row["historical_role"].add(role)


def build(args: argparse.Namespace) -> None:
    RDLogger.DisableLog("rdApp.*")
    provenance: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {
            "reason_excluded": set(),
            "source_manifest": set(),
            "historical_role": set(),
        }
    )
    source_to_identity: dict[str, str] = {}

    dev_payload = json.loads(args.dev_manifest.read_text(encoding="utf-8"))
    dev_source_ids = {str(row["molecule_id"]) for row in dev_payload["rows"]}

    for split, parquet_path, cache_dir in (
        ("train", args.train_parquet, args.train_cache),
        ("val", args.val_parquet, args.val_cache),
    ):
        frame = pd.read_parquet(
            parquet_path, columns=["molecule_id", "source_path"]
        ).drop_duplicates("molecule_id")
        expected = 50_000 if split == "train" else 5_000
        if len(frame) != expected:
            raise RuntimeError(f"Unexpected {split} molecule count: {len(frame)}")
        items = [
            (str(row.molecule_id), cache_dir / Path(str(row.source_path)).name)
            for row in frame.itertuples(index=False)
        ]
        missing_paths = [str(path) for _, path in items if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(missing_paths[0])
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            identities = executor.map(identity_from_cache, (path for _, path in items))
            resolved = zip(items, identities)
            for (source_id, cache_path), identity in resolved:
                prior = source_to_identity.setdefault(f"{split}\0{source_id}", identity)
                if prior != identity:
                    raise RuntimeError(f"Conflicting identity for {split}:{source_id}")
                add(
                    provenance,
                    identity,
                    reason="CURRENT_REFINER_TRAIN"
                    if split == "train"
                    else "HISTORICAL_DEVELOPMENT_VAL",
                    source=str(parquet_path.resolve()),
                    role="CURRENT_TRAIN_AND_INDEXED_LEGACY_COHORTS"
                    if split == "train"
                    else "V2_VAL_AND_V2_DEV_TEST_FULL_VAL_UNION",
                )
                if split == "val" and source_id in dev_source_ids:
                    add(
                        provenance,
                        identity,
                        reason="CURRENT_DEV_AND_OUTCOME_EXPOSED_MODEL_DEVELOPMENT",
                        source=str(args.dev_manifest.resolve()),
                        role=(
                            "CURRENT_DEV_FACTORIAL_ABLATIONS_SEED307_SEED331_SEED353_"
                            "MODEL_SELECTION"
                        ),
                    )

    formal = json.loads(args.formal_manifest.read_text(encoding="utf-8"))
    formal_ids = sorted({str(row["mol_id"]) for row in formal["records"]})
    if len(formal_ids) != 100:
        raise RuntimeError(f"Unexpected Formal100 identity count: {len(formal_ids)}")
    formal_files = list(args.formal_cache.glob("*.pt"))
    for opaque_id in formal_ids:
        matches = [path for path in formal_files if f"__{opaque_id}__" in path.name]
        if not matches:
            raise FileNotFoundError(f"No Formal100 cache record for {opaque_id}")
        identity = identity_from_cache(sorted(matches)[0])
        add(
            provenance,
            identity,
            reason="FORMAL100_OUTCOME_EXPOSED",
            source=str(args.formal_manifest.resolve()),
            role="FORMAL100_PROTECTED_HISTORICAL_READ",
        )

    large_manifest = json.loads(args.large_manifest.read_text(encoding="utf-8"))
    wanted = {str(row["molecule_id"]) for row in large_manifest["records"]}
    large_selection = json.loads(
        args.large_selection_manifest.read_text(encoding="utf-8")
    )
    selection_rows = large_selection["molecules"]
    found: dict[str, str] = {}
    for row in selection_rows:
        molecule_id = str(row["molecule_id"])
        mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
        if molecule_id in wanted and mol is not None:
            found[molecule_id] = canonical_identity(mol)
    if len(wanted) != 783 or set(found) != wanted or len(set(found.values())) != 783:
        raise RuntimeError("Large-holdout canonical identity manifest is incomplete")
    for identity in found.values():
        add(
            provenance,
            identity,
            reason="OUTCOME_EXPOSED_SECONDARY_COHORT",
            source=str(args.large_selection_manifest.resolve()),
            role="DITMC_LARGE_HOLDOUT_V1",
        )

    ordered = sorted(provenance)
    union_payload: dict[str, Any] = {
        "schema_version": "sixs-step2d-historical-exclusion-union-v1",
        "identity_definition": IDENTITY_DEFINITION,
        "molecule_identities": ordered,
    }
    atomic_text(
        args.union_json,
        json.dumps(union_payload, indent=2, sort_keys=True) + "\n",
    )

    csv_lines: list[str] = []
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "molecule_identity",
            "molecule_identity_sha256",
            "reason_excluded",
            "source_manifest",
            "historical_role",
            "historical_use_status",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for identity in ordered:
        row = provenance[identity]
        writer.writerow(
            {
                "molecule_identity": identity,
                "molecule_identity_sha256": hashlib.sha256(
                    f"{IDENTITY_DEFINITION}\0{identity}".encode("utf-8")
                ).hexdigest(),
                "reason_excluded": ";".join(sorted(row["reason_excluded"])),
                "source_manifest": ";".join(sorted(row["source_manifest"])),
                "historical_role": ";".join(sorted(row["historical_role"])),
                "historical_use_status": "PROVEN_USED",
            }
        )
    atomic_text(args.provenance_csv, buffer.getvalue())
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "molecules": len(ordered),
                "union_sha256": sha256_file(args.union_json),
                "provenance_sha256": sha256_file(args.provenance_csv),
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--val-parquet", required=True, type=Path)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--val-cache", required=True, type=Path)
    parser.add_argument("--dev-manifest", required=True, type=Path)
    parser.add_argument("--formal-manifest", required=True, type=Path)
    parser.add_argument("--formal-cache", required=True, type=Path)
    parser.add_argument("--large-manifest", required=True, type=Path)
    parser.add_argument("--large-selection-manifest", required=True, type=Path)
    parser.add_argument("--union-json", required=True, type=Path)
    parser.add_argument("--provenance-csv", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
