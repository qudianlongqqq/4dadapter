#!/usr/bin/env python3
"""Freeze an outcome-blind prospective SIXS primary cohort from identity metadata.

This utility performs no model inference, coordinate generation, metric evaluation,
MMFF, or xTB calculation. It must only be run after the source universe and the
historical exclusion union have independently been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


IDENTITY_DEFINITION = "sixs-canonical-isomeric-atom-map-free-h-normalized-v1"
SELECTION_DOMAIN = "sixs-prospective-primary-final-v1"
TARGET_MOLECULES = 2_500
RECORDS_PER_MOLECULE = 2
ETFLOW_MAX_Z_EXCLUSIVE = 100
XTB_GFN2_MAX_Z_INCLUSIVE = 86

REQUIRED_FIELDS = {
    "molecule_id",
    "molecule_identity_sha256",
    "identity_definition",
    "source_dataset_identity",
    "reference_identity",
    "valid_graph",
    "single_component",
    "reference_available",
    "topology_compatible",
    "etflow_compatible",
    "mmff94s_compatible",
    "xtb_compatible",
    "atomic_numbers",
    "formal_charge",
    "heavy_atom_count",
    "rotatable_bond_count",
    "ring_count",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_sha256(molecule_id: str) -> str:
    payload = f"{IDENTITY_DEFINITION}\0{molecule_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selection_key(molecule_id: str) -> str:
    payload = f"{SELECTION_DOMAIN}\0{molecule_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Universe line {line_number} is not an object")
            rows.append(row)
    return rows


def load_exclusion(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity_definition") != IDENTITY_DEFINITION:
        raise ValueError("Historical exclusion identity definition mismatch")
    identities = payload.get("molecule_identities")
    if not isinstance(identities, list) or not all(
        isinstance(value, str) and value for value in identities
    ):
        raise ValueError("Historical exclusion union is not a string identity list")
    if len(set(identities)) != len(identities):
        raise ValueError("Historical exclusion union contains duplicate identities")
    return set(identities)


def is_eligible(row: Mapping[str, Any], exclusion: set[str]) -> tuple[bool, list[str]]:
    missing = sorted(REQUIRED_FIELDS.difference(row))
    if missing:
        return False, ["missing_fields:" + ",".join(missing)]

    molecule_id = row["molecule_id"]
    reasons: list[str] = []
    if not isinstance(molecule_id, str) or not molecule_id:
        reasons.append("invalid_molecule_id")
    elif row["identity_definition"] != IDENTITY_DEFINITION:
        reasons.append("identity_definition_mismatch")
    elif row["molecule_identity_sha256"] != identity_sha256(molecule_id):
        reasons.append("identity_hash_mismatch")
    elif molecule_id in exclusion:
        reasons.append("historical_exclusion_overlap")

    for field in (
        "valid_graph",
        "single_component",
        "reference_available",
        "topology_compatible",
        "etflow_compatible",
        "mmff94s_compatible",
        "xtb_compatible",
    ):
        if row.get(field) is not True:
            reasons.append(f"{field}_not_true")

    atomic_numbers = row.get("atomic_numbers")
    if not isinstance(atomic_numbers, list) or not atomic_numbers:
        reasons.append("atomic_numbers_missing")
    elif not all(
        isinstance(z, int) and 1 <= z < ETFLOW_MAX_Z_EXCLUSIVE
        for z in atomic_numbers
    ):
        reasons.append("atomic_number_outside_frozen_etflow_embedding")
    elif not all(z <= XTB_GFN2_MAX_Z_INCLUSIVE for z in atomic_numbers):
        reasons.append("atomic_number_outside_frozen_gfn2_xtb_support")

    if not isinstance(row.get("formal_charge"), int):
        reasons.append("formal_charge_not_integer")
    for field in ("heavy_atom_count", "rotatable_bond_count", "ring_count"):
        if not isinstance(row.get(field), int) or int(row[field]) < 0:
            reasons.append(f"invalid_{field}")
    if isinstance(row.get("heavy_atom_count"), int) and row["heavy_atom_count"] < 1:
        reasons.append("no_heavy_atoms")

    return not reasons, reasons


def freeze(
    *, source_universe: Path, historical_exclusion: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {output}")

    exclusion = load_exclusion(historical_exclusion)
    universe = load_jsonl(source_universe)
    seen: set[str] = set()
    eligible: list[dict[str, Any]] = []
    for row in universe:
        molecule_id = row.get("molecule_id")
        if isinstance(molecule_id, str):
            if molecule_id in seen:
                raise ValueError(f"Duplicate molecule identity: {molecule_id}")
            seen.add(molecule_id)
        accepted, _ = is_eligible(row, exclusion)
        if accepted:
            eligible.append(dict(row))

    eligible.sort(key=lambda row: (selection_key(row["molecule_id"]), row["molecule_id"]))
    if len(eligible) < TARGET_MOLECULES:
        raise RuntimeError(
            f"Eligible pool has {len(eligible)} molecules; frozen target is "
            f"{TARGET_MOLECULES}. Do not downsize or replacement-sample."
        )

    selected = eligible[:TARGET_MOLECULES]
    rows = []
    for rank, row in enumerate(selected, start=1):
        rows.append(
            {
                "molecule_id": row["molecule_id"],
                "molecule_identity_sha256": row["molecule_identity_sha256"],
                "source_dataset_identity": row["source_dataset_identity"],
                "reference_identity": row["reference_identity"],
                "eligibility_status": "PASS",
                "selection_key_sha256": selection_key(row["molecule_id"]),
                "selection_rank": rank,
                "historical_exclusion_overlap": False,
                "heavy_atom_count": row["heavy_atom_count"],
                "rotatable_bond_count": row["rotatable_bond_count"],
                "ring_count": row["ring_count"],
                "formal_charge": row["formal_charge"],
                "atomic_numbers": row["atomic_numbers"],
            }
        )

    payload = {
        "schema_version": "sixs-prospective-primary-final-manifest-v1",
        "identity_definition": IDENTITY_DEFINITION,
        "selection_algorithm": "ascending_sha256(domain_NUL_canonical_identity), tie_break_identity",
        "selection_domain": SELECTION_DOMAIN,
        "selection_seed": None,
        "target_molecules": TARGET_MOLECULES,
        "records_per_molecule": RECORDS_PER_MOLECULE,
        "target_records": TARGET_MOLECULES * RECORDS_PER_MOLECULE,
        "eligible_pool_molecules": len(eligible),
        "source_universe_sha256": sha256_file(source_universe),
        "historical_exclusion_sha256": sha256_file(historical_exclusion),
        "selection_code_sha256": sha256_file(Path(__file__).resolve()),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-universe", required=True, type=Path)
    parser.add_argument("--historical-exclusion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = freeze(
        source_universe=args.source_universe,
        historical_exclusion=args.historical_exclusion,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "molecules": payload["target_molecules"],
                "records": payload["target_records"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
