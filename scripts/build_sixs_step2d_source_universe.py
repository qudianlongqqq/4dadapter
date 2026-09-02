#!/usr/bin/env python3
"""Build a resumable identity-only index of the official GEOM-DRUGS assets.

Only one standardized pickle chunk is resident at a time. Per-chunk JSONL
caches contain identity/topology metadata and no coordinates, so interrupted
runs resume without reloading completed conformer chunks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import pickle
from typing import Any, Iterable

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Lipinski


IDENTITY_DEFINITION = "sixs-canonical-isomeric-atom-map-free-h-normalized-v1"
CACHE_SCHEMA = "sixs-step2d-source-chunk-identity-cache-v2"
SOURCE_DATASET = "GEOM-DRUGS-WITH-TORSIONAL-DIFFUSION-NATIVE-SPLITS"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_hash(identity: str) -> str:
    return hashlib.sha256(
        f"{IDENTITY_DEFINITION}\0{identity}".encode("utf-8")
    ).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def canonical_identity(mol: Chem.Mol) -> tuple[str, Chem.Mol]:
    normalized = Chem.Mol(mol)
    for atom in normalized.GetAtoms():
        atom.SetAtomMapNum(0)
    normalized = Chem.RemoveHs(normalized, sanitize=True)
    Chem.SanitizeMol(normalized)
    identity = Chem.MolToSmiles(
        normalized, canonical=True, isomericSmiles=True
    )
    if not identity:
        raise ValueError("empty canonical identity")
    return identity, normalized


def load_raw_listing(
    *, filename_index: Path | None, member_map: Path | None
) -> tuple[list[str], str]:
    if filename_index is not None:
        names = [
            line.strip()
            for line in filename_index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        stems = [name[:-7] if name.endswith(".pickle") else name for name in names]
        source_sha = sha256_file(filename_index)
    elif member_map is not None:
        stems = []
        with member_map.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    stems.append(str(json.loads(line)["source_stem"]))
        stems.sort()
        source_sha = sha256_file(member_map)
    else:
        raise ValueError("One of --raw-filename-index or --raw-member-map is required")
    if len(stems) != len(set(stems)):
        raise RuntimeError("Raw filename listing contains duplicate source stems")
    if stems != sorted(stems):
        raise RuntimeError("Raw filename listing is not in deterministic lexical order")
    return stems, source_sha


def load_split_lookup(path: Path) -> tuple[dict[int, str], dict[str, int], int]:
    payload = np.load(path, allow_pickle=True)
    if len(payload) != 3:
        raise RuntimeError(f"Expected three native splits, got {len(payload)}")
    lookup: dict[int, str] = {}
    counts: dict[str, int] = {}
    for split, values in zip(("train", "val", "test"), payload):
        counts[split] = len(values)
        for raw in values:
            index = int(raw)
            if index < 0 or index in lookup:
                raise RuntimeError(f"Invalid or duplicate native split index: {index}")
            lookup[index] = split
    if not lookup or set(lookup) != set(range(max(lookup) + 1)):
        raise RuntimeError("Native split indices do not form a complete zero-based universe")
    return lookup, counts, max(lookup) + 1


def _chunk_fingerprint(
    path: Path,
    *,
    split_sha256: str,
    raw_prefix_sha256: str,
    raw_index_status: str,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "schema_version": CACHE_SCHEMA,
        "source_path": str(path.resolve()),
        "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "split_sha256": split_sha256,
        "raw_prefix_sha256": raw_prefix_sha256,
        "raw_index_status": raw_index_status,
        "identity_definition": IDENTITY_DEFINITION,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _chunk_rows(
    path: Path,
    *,
    raw_index: dict[str, int],
    split_lookup: dict[int, str],
    raw_index_status: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        chunk_id = int(path.stem)
    except ValueError as exc:
        raise RuntimeError(f"Standardized chunk filename is not numeric: {path}") from exc
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Standardized chunk is not a dictionary: {path}")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for raw_stem, mol_dict in payload.items():
        stem = str(raw_stem)
        listed_index = raw_index.get(stem)
        native_index = listed_index if raw_index_status == "complete" else None
        if native_index is not None and native_index // 1000 != chunk_id:
            raise RuntimeError(
                "Raw prefix/chunk lineage mismatch: "
                f"chunk={chunk_id}, native_index={native_index}, stem={stem}"
            )
        native_split = (
            split_lookup[native_index] if native_index is not None else "unassigned"
        )
        conformers = (
            mol_dict.get("conformers", []) if isinstance(mol_dict, dict) else []
        )
        try:
            if not conformers:
                raise ValueError("reference conformer unavailable")
            first = conformers[0]
            mol = first.get("rd_mol") if isinstance(first, dict) else None
            if mol is None:
                raise ValueError("reference rd_mol unavailable")
            identity, normalized = canonical_identity(mol)
            atomic_numbers = sorted(
                {int(atom.GetAtomicNum()) for atom in normalized.GetAtoms()}
            )
            single_component = len(Chem.GetMolFrags(normalized)) == 1
            heavy_atoms = int(normalized.GetNumHeavyAtoms())
            try:
                mmff_compatible = bool(AllChem.MMFFHasAllMoleculeParams(normalized))
            except Exception:
                mmff_compatible = False
            locator = f"DRUGS/standardized_pickles/{path.name}::{stem}"
            rows.append(
                {
                    "molecule_id": identity,
                    "molecule_identity_sha256": identity_hash(identity),
                    "identity_definition": IDENTITY_DEFINITION,
                    "source_dataset_identity": f"{SOURCE_DATASET}::{locator}",
                    "reference_identity": locator,
                    "source_chunk": path.name,
                    "source_stem": stem,
                    "native_index": native_index,
                    "raw_archive_member_present": listed_index is not None,
                    "native_split": native_split,
                    "native_split_linkage": (
                        "EXACT_COMPLETE_RAW_INDEX" if native_index is not None else "UNKNOWN"
                    ),
                    "valid_graph": True,
                    "single_component": single_component,
                    "reference_available": True,
                    "topology_compatible": single_component and heavy_atoms >= 1,
                    "etflow_compatible": bool(atomic_numbers)
                    and all(1 <= z < 100 for z in atomic_numbers),
                    "mmff94s_compatible": mmff_compatible,
                    "xtb_compatible": bool(atomic_numbers)
                    and all(1 <= z <= 86 for z in atomic_numbers),
                    "atomic_numbers": atomic_numbers,
                    "formal_charge": int(Chem.GetFormalCharge(normalized)),
                    "heavy_atom_count": heavy_atoms,
                    "rotatable_bond_count": int(
                        Lipinski.NumRotatableBonds(normalized)
                    ),
                    "ring_count": int(Lipinski.RingCount(normalized)),
                    "reference_conformer_count": len(conformers),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "source_chunk": path.name,
                    "source_stem": stem,
                    "native_index": native_index,
                    "native_split": native_split,
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc)[:500],
                    "policy": "INELIGIBLE_AND_EXPLICITLY_REPORTED",
                }
            )
    return rows, failures


def cached_chunk_rows(
    path: Path,
    *,
    cache_dir: Path,
    raw_index: dict[str, int],
    split_lookup: dict[int, str],
    raw_index_status: str,
    split_sha256: str,
    raw_prefix_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows_path = cache_dir / f"{path.stem}.identity.jsonl"
    failures_path = cache_dir / f"{path.stem}.failures.jsonl"
    meta_path = cache_dir / f"{path.stem}.meta.json"
    fingerprint = _chunk_fingerprint(
        path,
        split_sha256=split_sha256,
        raw_prefix_sha256=raw_prefix_sha256,
        raw_index_status=raw_index_status,
    )
    if rows_path.is_file() and failures_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta == fingerprint:
            return _read_jsonl(rows_path), _read_jsonl(failures_path), True

    rows, failures = _chunk_rows(
        path,
        raw_index=raw_index,
        split_lookup=split_lookup,
        raw_index_status=raw_index_status,
    )
    atomic_jsonl(rows_path, rows)
    atomic_jsonl(failures_path, failures)
    atomic_text(meta_path, json.dumps(fingerprint, indent=2, sort_keys=True) + "\n")
    return rows, failures, False


def _compatible(row: dict[str, Any]) -> bool:
    return all(
        row[field] is True
        for field in (
            "valid_graph",
            "single_component",
            "reference_available",
            "topology_compatible",
            "etflow_compatible",
            "mmff94s_compatible",
            "xtb_compatible",
        )
    ) and row["heavy_atom_count"] >= 1


def build(args: argparse.Namespace) -> dict[str, Any]:
    raw_stems, raw_prefix_sha256 = load_raw_listing(
        filename_index=args.raw_filename_index, member_map=args.raw_member_map
    )
    raw_index = {stem: index for index, stem in enumerate(raw_stems)}
    split_sha256 = sha256_file(args.split)
    split_lookup, split_index_counts, native_universe_size = load_split_lookup(
        args.split
    )
    if args.raw_index_status == "complete" and len(raw_stems) != native_universe_size:
        raise RuntimeError(
            "A complete raw index must have one filename for every split index: "
            f"raw={len(raw_stems)}, split={native_universe_size}"
        )
    chunks = sorted(args.standardized_dir.glob("*.pickle"))
    if not chunks:
        raise RuntimeError("No standardized pickle chunks found")

    RDLogger.DisableLog("rdApp.*")
    grouped: dict[str, dict[str, Any]] = {}
    all_failures: list[dict[str, Any]] = []
    standardized_entries = 0
    cache_hits = 0

    for number, path in enumerate(chunks, start=1):
        rows, failures, hit = cached_chunk_rows(
            path,
            cache_dir=args.cache_dir,
            raw_index=raw_index,
            split_lookup=split_lookup,
            raw_index_status=args.raw_index_status,
            split_sha256=split_sha256,
            raw_prefix_sha256=raw_prefix_sha256,
        )
        cache_hits += int(hit)
        all_failures.extend(failures)
        standardized_entries += len(rows) + len(failures)
        for row in rows:
            identity = row["molecule_id"]
            current = grouped.get(identity)
            occurrence = {
                "source_dataset_identity": row["source_dataset_identity"],
                "reference_identity": row["reference_identity"],
                "source_chunk": row["source_chunk"],
                "source_stem": row["source_stem"],
                "native_index": row["native_index"],
                "raw_archive_member_present": row["raw_archive_member_present"],
                "native_split": row["native_split"],
                "native_split_linkage": row["native_split_linkage"],
                "reference_conformer_count": row["reference_conformer_count"],
            }
            if current is None:
                current = {key: value for key, value in row.items() if key not in occurrence}
                current["_occurrences"] = [occurrence]
                grouped[identity] = current
            else:
                for key in (
                    "atomic_numbers",
                    "formal_charge",
                    "heavy_atom_count",
                    "valid_graph",
                    "single_component",
                ):
                    if current[key] != row[key]:
                        raise RuntimeError(
                            f"Canonical duplicate metadata mismatch for {identity}: {key}"
                        )
                current["_occurrences"].append(occurrence)
        if number % 10 == 0 or number == len(chunks):
            print(
                f"SOURCE_INDEX chunks={number}/{len(chunks)} entries={standardized_entries} "
                f"identities={len(grouped)} cache_hits={cache_hits}",
                flush=True,
            )

    source_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for identity in sorted(grouped):
        row = grouped[identity]
        occurrences = row.pop("_occurrences")
        splits = sorted({item["native_split"] for item in occurrences})
        all_exact = all(
            item["native_split_linkage"] == "EXACT_COMPLETE_RAW_INDEX"
            for item in occurrences
        )
        row["source_dataset_identity"] = min(
            item["source_dataset_identity"] for item in occurrences
        )
        row["reference_identity"] = min(
            item["reference_identity"] for item in occurrences
        )
        row["source_chunk"] = min(item["source_chunk"] for item in occurrences)
        row["native_split"] = "|".join(splits)
        row["native_split_linkage"] = (
            "EXACT_COMPLETE_RAW_INDEX" if all_exact else "UNKNOWN_OR_MIXED"
        )
        row["raw_source_count"] = len(occurrences)
        row["raw_archive_member_present"] = any(
            item["raw_archive_member_present"] for item in occurrences
        )
        row["raw_source_identity_sha256"] = hashlib.sha256(
            "\n".join(
                sorted(str(item["source_stem"]) for item in occurrences)
            ).encode("utf-8")
        ).hexdigest()
        row["reference_conformer_count"] = sum(
            int(item["reference_conformer_count"]) for item in occurrences
        )
        row["history_status"] = (
            "PROVEN_NATIVE_TEST"
            if all_exact and splits == ["test"]
            else "CONSERVATIVE_EXCLUSION_NATIVE_TRAIN_VAL_OR_UNKNOWN"
        )
        row["eligibility_status"] = (
            "PASS_INPUT_ONLY" if _compatible(row) else "FAIL_INPUT_ONLY"
        )
        source_rows.append(row)
        if len(occurrences) > 1:
            duplicate_rows.append(
                {
                    "molecule_id": identity,
                    "molecule_identity_sha256": row["molecule_identity_sha256"],
                    "source_occurrences": len(occurrences),
                    "native_splits": row["native_split"],
                    "resolution": "CANONICAL_DEDUPLICATION_SINGLE_IDENTITY",
                }
            )

    fields = [
        "molecule_id",
        "molecule_identity_sha256",
        "identity_definition",
        "source_dataset_identity",
        "reference_identity",
        "source_chunk",
        "native_split",
        "native_split_linkage",
        "raw_archive_member_present",
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
        "raw_source_count",
        "raw_source_identity_sha256",
        "reference_conformer_count",
        "history_status",
        "eligibility_status",
    ]
    atomic_jsonl(
        args.jsonl_output,
        ({field: row[field] for field in fields} for row in source_rows),
    )
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = args.csv_output.with_suffix(args.csv_output.suffix + ".tmp")
    with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in source_rows:
            output = {field: row[field] for field in fields}
            output["atomic_numbers"] = ";".join(map(str, row["atomic_numbers"]))
            writer.writerow(output)
    os.replace(csv_tmp, args.csv_output)
    atomic_jsonl(args.failure_output, all_failures)
    atomic_jsonl(args.duplicate_output, duplicate_rows)

    split_counts = Counter(row["native_split"] for row in source_rows)
    exact_test = [
        row
        for row in source_rows
        if row["history_status"] == "PROVEN_NATIVE_TEST"
    ]
    summary: dict[str, Any] = {
        "schema_version": "sixs-step2d-source-universe-summary-v2",
        "status": "COMPLETE",
        "source_dataset": SOURCE_DATASET,
        "identity_definition": IDENTITY_DEFINITION,
        "native_split_index_counts": split_index_counts,
        "native_split_index_universe": native_universe_size,
        "raw_filename_listing_count": len(raw_stems),
        "raw_filename_listing_sha256": raw_prefix_sha256,
        "raw_index_status": args.raw_index_status.upper(),
        "raw_listing_native_linkage": (
            "PASS_COMPLETE_INDEX"
            if args.raw_index_status == "complete"
            else "SUBSET_ONLY__NO_GLOBAL_NATIVE_INDEX_LINKAGE"
        ),
        "native_linkage_outside_complete_raw_index": (
            "NOT_APPLICABLE"
            if args.raw_index_status == "complete"
            else "UNRECOVERABLE_FROM_DRUGS_TAR_GZ"
        ),
        "standardized_pickle_chunks": len(chunks),
        "standardized_entries": standardized_entries,
        "source_universe_n_molecules": len(source_rows),
        "source_universe_native_split_counts": dict(sorted(split_counts.items())),
        "canonicalization_failures": len(all_failures),
        "duplicate_canonical_identities": len(duplicate_rows),
        "input_only_eligible": sum(_compatible(row) for row in source_rows),
        "proven_native_test": len(exact_test),
        "proven_native_test_input_only_eligible": sum(
            _compatible(row) for row in exact_test
        ),
        "cache_hits": cache_hits,
        "cache_misses": len(chunks) - cache_hits,
        "prefix_lineage_violations": 0,
        "source_universe_jsonl": str(args.jsonl_output.resolve()),
        "source_universe_jsonl_sha256": sha256_file(args.jsonl_output),
        "source_universe_csv": str(args.csv_output.resolve()),
        "source_universe_csv_sha256": sha256_file(args.csv_output),
        "canonical_failure_report": str(args.failure_output.resolve()),
        "canonical_failure_report_sha256": sha256_file(args.failure_output),
        "duplicate_identity_report": str(args.duplicate_output.resolve()),
        "duplicate_identity_report_sha256": sha256_file(args.duplicate_output),
    }
    atomic_text(
        args.summary_output, json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    raw = parser.add_mutually_exclusive_group(required=True)
    raw.add_argument("--raw-filename-index", type=Path)
    raw.add_argument("--raw-member-map", type=Path)
    parser.add_argument(
        "--raw-index-status", choices=("complete", "subset"), default="subset"
    )
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    parser.add_argument("--jsonl-output", required=True, type=Path)
    parser.add_argument("--failure-output", required=True, type=Path)
    parser.add_argument("--duplicate-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
