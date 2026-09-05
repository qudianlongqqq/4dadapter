#!/usr/bin/env python
"""Outcome-blind completion audit for the frozen Step 3A ETFlow source asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from rdkit import Chem


EXPECTED_PRIMARY_SHA256 = "2a1d07af8c9e3150d1f2f3719d0bd43bd33819ca7674c364d0770c010cb86ee1"
EXPECTED_PROTOCOL_SHA256 = "4bbb748286b40aac3c956b2f502ab42b093a4155ff02b0b5bab79e0b3af0f084"
EXPECTED_CHECKPOINT_SHA256 = "a24ae9a1fed2708696929308ed1dc10ab167fd66a2d51c44a4afb6c11badccb2"
TARGET_MOLECULES = 2500
TARGET_RECORDS = 5000
RECORDS_PER_MOLECULE = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def coordinate_sha256(coordinates: torch.Tensor) -> str:
    array = np.ascontiguousarray(coordinates.detach().cpu().numpy(), dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def molecule_identity_sha256(identity: str) -> str:
    definition = "sixs-canonical-isomeric-atom-map-free-h-normalized-v1"
    return hashlib.sha256(f"{definition}\0{identity}".encode("utf-8")).hexdigest()


def record_seed(protocol_seed: int, identity_sha256: str, record_index: int) -> int:
    token = f"{protocol_seed}\0{identity_sha256}\0{record_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & (2**63 - 1)


def canonical_identity(mol: Chem.Mol) -> str:
    normalized = Chem.Mol(mol)
    for atom in normalized.GetAtoms():
        atom.SetAtomMapNum(0)
    normalized = Chem.RemoveHs(normalized, sanitize=True)
    Chem.SanitizeMol(normalized)
    return Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=True)


def atom_order_identity(mol: Chem.Mol) -> str:
    atoms = [
        {
            "index": atom.GetIdx(),
            "atomic_number": atom.GetAtomicNum(),
            "formal_charge": atom.GetFormalCharge(),
            "isotope": atom.GetIsotope(),
            "chiral_tag": str(atom.GetChiralTag()),
        }
        for atom in mol.GetAtoms()
    ]
    bonds = [
        {
            "begin": min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            "end": max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            "bond_type": str(bond.GetBondType()),
            "stereo": str(bond.GetStereo()),
        }
        for bond in mol.GetBonds()
    ]
    bonds.sort(key=lambda row: (row["begin"], row["end"], row["bond_type"], row["stereo"]))
    return canonical_json_sha256({"atoms": atoms, "bonds": bonds})


def load_reference(source_root: Path, locator: str, cache: dict[Path, Mapping[str, Any]]) -> Chem.Mol:
    relative, separator, raw_key = locator.partition("::")
    if not separator:
        raise ValueError(f"invalid reference locator: {locator}")
    path = (source_root / relative).resolve()
    if source_root.resolve() not in path.parents:
        raise ValueError(f"reference locator escapes source root: {locator}")
    if path not in cache:
        # The standardized corpus is tens of GiB.  Rows are audited grouped by
        # pickle path, so retaining only the active shard gives one load per
        # referenced shard without retaining the full corpus in RAM.
        cache.clear()
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, Mapping):
            raise TypeError(f"reference pickle is not a mapping: {path}")
        cache[path] = payload
    entry = cache[path].get(raw_key)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("conformers"), list) or not entry["conformers"]:
        raise KeyError(f"reference entry/conformers missing: {locator}")
    mol = entry["conformers"][0].get("rd_mol")
    if mol is None:
        raise KeyError(f"reference rd_mol missing: {locator}")
    return Chem.Mol(mol)


def main(args: argparse.Namespace) -> int:
    primary_path = args.primary.resolve()
    protocol_path = args.protocol.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    source_root = args.source_root.resolve()
    report_dir = args.report_dir.resolve()

    hashes = {
        "primary_manifest": sha256_file(primary_path),
        "protocol_manifest": sha256_file(protocol_path),
        "checkpoint": sha256_file(checkpoint_path),
    }
    expected_hashes = {
        "primary_manifest": EXPECTED_PRIMARY_SHA256,
        "protocol_manifest": EXPECTED_PROTOCOL_SHA256,
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
    }
    if hashes != expected_hashes:
        raise RuntimeError(f"frozen hash mismatch: observed={hashes}")

    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = list(primary["rows"])
    if len(rows) != TARGET_MOLECULES or primary.get("target_records") != TARGET_RECORDS:
        raise RuntimeError("primary manifest cardinality mismatch")
    if protocol["primary_final_manifest_sha256"] != hashes["primary_manifest"]:
        raise RuntimeError("protocol is not bound to the frozen primary manifest")
    if protocol["checkpoint_sha256"] != hashes["checkpoint"]:
        raise RuntimeError("protocol is not bound to the frozen ETFlow checkpoint")
    protocol_seed = int(protocol["randomness"]["protocol_seed"])

    generation_status = json.loads((output_dir / "FINAL_STATUS.json").read_text(encoding="utf-8"))
    failures = json.loads((output_dir / "GENERATION_FAILURES.json").read_text(encoding="utf-8"))
    if generation_status.get("status") != "PASS" or failures:
        raise RuntimeError("source generation did not finish cleanly")
    manifest_path = output_dir / "SOURCE_RECORD_MANIFEST.jsonl"
    record_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if len(record_rows) != TARGET_RECORDS:
        raise RuntimeError("source record manifest cardinality mismatch")

    primary_by_rank = {int(row["selection_rank"]): row for row in rows}
    if len(primary_by_rank) != TARGET_MOLECULES:
        raise RuntimeError("duplicate primary selection rank")
    records_by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in record_rows:
        records_by_rank[int(row["final_molecule_index"])].append(row)

    errors: list[dict[str, Any]] = []
    reference_cache: dict[Path, Mapping[str, Any]] = {}
    record_ids: set[str] = set()
    record_assets: set[str] = set()
    atom_counts: list[int] = []
    hydrogen_counts: list[int] = []
    formal_charges: list[int] = []
    engineering_attempts: Counter[int] = Counter()
    asset_hash_lines: list[str] = []

    audit_ranks = sorted(
        range(1, TARGET_MOLECULES + 1),
        key=lambda rank: (
            primary_by_rank[rank]["reference_identity"].partition("::")[0],
            rank,
        ),
    )
    reference_pickle_files_read: set[str] = set()
    for rank in audit_ranks:
        primary_row = primary_by_rank[rank]
        try:
            reference_pickle_files_read.add(
                primary_row["reference_identity"].partition("::")[0]
            )
            reference = load_reference(source_root, primary_row["reference_identity"], reference_cache)
            identity = canonical_identity(reference)
            identity_hash = molecule_identity_sha256(identity)
            atomic_numbers = torch.tensor([a.GetAtomicNum() for a in reference.GetAtoms()], dtype=torch.int32)
            topology_hash = atom_order_identity(reference)
            total_charge = sum(a.GetFormalCharge() for a in reference.GetAtoms())
            heavy_atoms = sum(a.GetAtomicNum() > 1 for a in reference.GetAtoms())
            hydrogens = sum(a.GetAtomicNum() == 1 for a in reference.GetAtoms())
            if identity != primary_row["molecule_id"]:
                raise ValueError("reference canonical identity mismatch")
            if identity_hash != primary_row["molecule_identity_sha256"]:
                raise ValueError("reference molecule identity hash mismatch")
            if total_charge != int(primary_row["formal_charge"]):
                raise ValueError("reference formal charge mismatch")
            if heavy_atoms != int(primary_row["heavy_atom_count"]):
                raise ValueError("reference heavy atom count mismatch")
            atom_counts.append(int(reference.GetNumAtoms()))
            hydrogen_counts.append(hydrogens)
            formal_charges.append(total_charge)

            manifest_records = records_by_rank.get(rank, [])
            if sorted(int(row["etflow_record_index"]) for row in manifest_records) != [0, 1]:
                raise ValueError("record indices are not exactly [0, 1]")
            for manifest_row in manifest_records:
                record_index = int(manifest_row["etflow_record_index"])
                expected_record_id = f"primary-final-{rank:06d}-etflow-{record_index}"
                if manifest_row["record_id"] != expected_record_id:
                    raise ValueError("record id mismatch")
                asset = Path(manifest_row["record_asset"]).resolve()
                if manifest_row["record_id"] in record_ids or str(asset) in record_assets:
                    raise ValueError("duplicate record id or asset path")
                record_ids.add(manifest_row["record_id"])
                record_assets.add(str(asset))
                observed_asset_hash = sha256_file(asset)
                if observed_asset_hash != manifest_row["record_asset_sha256"]:
                    raise ValueError("record asset SHA256 mismatch")
                payload = torch.load(asset, map_location="cpu", weights_only=False)
                expected = {
                    "record_id": expected_record_id,
                    "final_molecule_index": rank,
                    "canonical_identity": identity,
                    "molecule_identity_sha256": identity_hash,
                    "etflow_record_index": record_index,
                    "record_seed": record_seed(protocol_seed, identity_hash, record_index),
                    "atom_order_identity_sha256": topology_hash,
                    "checkpoint_sha256": hashes["checkpoint"],
                    "primary_manifest_sha256": hashes["primary_manifest"],
                    "protocol_sha256": hashes["protocol_manifest"],
                    "coordinate_dtype": "float32",
                    "coordinate_units": "angstrom",
                }
                for key, value in expected.items():
                    if payload.get(key) != value:
                        raise ValueError(f"record payload mismatch: {key}")
                if not torch.equal(torch.as_tensor(payload["atomic_numbers"], dtype=torch.int32), atomic_numbers):
                    raise ValueError("record atom order/atomic numbers mismatch")
                coordinates = torch.as_tensor(payload["source_coordinates"], dtype=torch.float32)
                if tuple(coordinates.shape) != (reference.GetNumAtoms(), 3):
                    raise ValueError("record coordinate shape mismatch")
                if not bool(torch.isfinite(coordinates).all()):
                    raise ValueError("record has NaN/Inf coordinates")
                observed_coordinate_hash = coordinate_sha256(coordinates)
                if observed_coordinate_hash != payload.get("coordinate_sha256") or observed_coordinate_hash != manifest_row["coordinate_sha256"]:
                    raise ValueError("record coordinate SHA256 mismatch")
                for key in ("record_id", "canonical_identity", "molecule_identity_sha256", "record_seed", "atom_order_identity_sha256", "checkpoint_sha256", "protocol_sha256"):
                    if manifest_row.get(key) != payload.get(key):
                        raise ValueError(f"record manifest/payload mismatch: {key}")
                attempt = int(payload["engineering_attempt"])
                if attempt not in (1, 2):
                    raise ValueError("engineering attempt is outside frozen finite-retry policy")
                engineering_attempts[attempt] += 1
                asset_hash_lines.append(
                    f"{expected_record_id}\0{observed_asset_hash}\0{observed_coordinate_hash}\n"
                )
        except Exception as exc:
            errors.append({"final_molecule_index": rank, "error_type": type(exc).__name__, "error": str(exc)})

    collection_hash = hashlib.sha256("".join(sorted(asset_hash_lines)).encode("utf-8")).hexdigest()
    checks = {
        "frozen_hashes": hashes == expected_hashes,
        "generation_status_pass": generation_status.get("status") == "PASS",
        "generation_failures_zero": len(failures) == 0,
        "primary_molecules_exact": len(rows) == TARGET_MOLECULES,
        "record_manifest_records_exact": len(record_rows) == TARGET_RECORDS,
        "unique_record_ids_exact": len(record_ids) == TARGET_RECORDS,
        "unique_record_assets_exact": len(record_assets) == TARGET_RECORDS,
        "two_records_per_molecule": len(records_by_rank) == TARGET_MOLECULES and all(len(v) == 2 for v in records_by_rank.values()),
        "reference_identity_atom_order_charge_hydrogen_pass": not errors,
        "coordinates_finite_shape_hash_pass": not errors,
        "protocol_seed_binding_pass": not errors,
        "finite_engineering_retry_policy_pass": not errors,
    }
    status = "PASS" if all(checks.values()) and not errors else "FAIL"
    audit = {
        "schema_version": "sixs-step3a-source-completion-audit-v1",
        "status": status,
        "scientific_outcome_read": False,
        "primary_membership_changed": False,
        "source_generation_rerun": False,
        "target_molecules": TARGET_MOLECULES,
        "target_records": TARGET_RECORDS,
        "validated_molecules": TARGET_MOLECULES - len(errors),
        "validated_records": len(record_ids),
        "reference_pickle_files_read": len(reference_pickle_files_read),
        "atom_count_min": min(atom_counts) if atom_counts else None,
        "atom_count_max": max(atom_counts) if atom_counts else None,
        "explicit_hydrogen_count_min": min(hydrogen_counts) if hydrogen_counts else None,
        "explicit_hydrogen_count_max": max(hydrogen_counts) if hydrogen_counts else None,
        "formal_charge_distribution": dict(sorted(Counter(formal_charges).items())),
        "engineering_attempt_distribution": dict(sorted(engineering_attempts.items())),
        "checks": checks,
        "errors": errors,
    }
    asset_freeze = {
        "schema_version": "sixs-step3a-source-asset-freeze-v1",
        "status": status,
        "source_record_manifest": str(manifest_path),
        "source_record_manifest_sha256": sha256_file(manifest_path),
        "source_coordinate_asset_collection_sha256": collection_hash,
        "collection_hash_definition": "SHA256 of UTF-8 concatenation of sorted record_id NUL record_asset_sha256 NUL coordinate_sha256 LF lines",
        "records": len(asset_hash_lines),
        "molecules": TARGET_MOLECULES if status == "PASS" else None,
        "primary_manifest_sha256": hashes["primary_manifest"],
        "protocol_manifest_sha256": hashes["protocol_manifest"],
        "etflow_checkpoint_sha256": hashes["checkpoint"],
        "scientific_outcome_read": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "06_SOURCE_COMPLETION_INTEGRITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report_dir / "07_SOURCE_ASSET_FREEZE.json").write_text(
        json.dumps(asset_freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": status,
        "validated_molecules": audit["validated_molecules"],
        "validated_records": audit["validated_records"],
        "source_record_manifest_sha256": asset_freeze["source_record_manifest_sha256"],
        "source_coordinate_asset_collection_sha256": collection_hash,
        "errors": len(errors),
    }, indent=2))
    return 0 if status == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
