#!/usr/bin/env python3
"""Freeze molecule-disjoint train-only LSGO cohorts without protected-split access."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

from etflow.ecir.lsgo_io import (
    atomic_json,
    atomic_torch_save,
    center_coordinates,
    file_sha256,
    validate_record_identity,
)

V8 = Path(r"E:\3dconformergenerationcode\4dadapter-v8")
SOURCE_MANIFEST = V8 / "data/ecir_mvr/formal_large/real_sources/train.parquet"
SOURCE_CACHE = V8 / "data/flexbond_cache_formal_large/train"
TRAIN_COMPACT = Path(
    r"E:\3dconformergenerationcode\4dadapter-label-free-score"
    r"\reports\ecir_mvr\label_free_score_pilot\manifests\TRAIN_ONLY_COMPACT_DATASET.pt"
)
NSSM_IDENTITY = Path(
    r"E:\3dconformergenerationcode\4dadapter-narrow-sigma"
    r"\reports\ecir_mvr\narrow_sigma\NSSM_EXTERNAL_CONFIRM_IDENTITY.json"
)
OUT = ROOT / "reports/ecir_mvr/learned_geometry"
EXTERNAL_COMPACT = OUT / "manifests/LSGO_EXTERNAL_CONFIRM_COMPACT.pt"
SELECTION_SEED = "lsgo-external-confirm-20260726"
EXTERNAL_MOLECULES = 200


def tensor_sha256(value: torch.Tensor) -> str:
    array = torch.as_tensor(value, dtype=torch.float32).cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compact_record(record: dict) -> dict:
    forbidden = {
        "x_init", "x_ref", "x_ref_aligned", "x_ref_candidates", "metadata",
        "DATA_DIR", "created_at",
    }
    return {key: value for key, value in record.items() if key not in forbidden}


def resolve_source(value: object) -> Path:
    path = Path(str(value))
    candidate = path if path.is_file() else SOURCE_CACHE / path.name
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _partition_summary(items: list[dict], name: str) -> dict:
    chosen = [item for item in items if str(item["partition"]) == name]
    identities = sorted(str(item["molecule_id"]) for item in chosen)
    return {
        "molecule_count": len(chosen),
        "record_count": 3 * len(chosen),
        "reference_count": int(sum(len(item["references"]) for item in chosen)),
        "molecule_identity_sha256": canonical_sha256(identities),
        "molecule_ids": identities,
    }


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    old = torch.load(TRAIN_COMPACT, map_location="cpu", weights_only=False)
    if int(old.get("formal_test_records_read", -1)) != 0 or int(old.get("frozen_holdout_records_read", -1)) != 0:
        raise RuntimeError("protected-split read detected in train compact identity")
    items = old["items"]
    cohorts = {
        "LSGO_TRAIN": _partition_summary(items, "train"),
        "LSGO_DEV_A": _partition_summary(items, "dev_a"),
        "LSGO_DEV_B": _partition_summary(items, "dev_b"),
    }
    if [cohorts[key]["molecule_count"] for key in ("LSGO_TRAIN", "LSGO_DEV_A", "LSGO_DEV_B")] != [2000, 250, 250]:
        raise RuntimeError("frozen train-only compact partition counts changed")
    used = set().union(*(set(value["molecule_ids"]) for value in cohorts.values()))
    exposed_external: set[str] = set()
    if NSSM_IDENTITY.is_file():
        nssm = json.loads(NSSM_IDENTITY.read_text(encoding="utf-8"))
        exposed_external = {str(row["molecule_id"]) for row in nssm["entries"]}

    frame = pd.read_parquet(SOURCE_MANIFEST)
    if (
        len(frame) != 150000
        or frame.molecule_id.nunique() != 50000
        or set(frame.split.astype(str)) != {"train"}
        or frame.test_record.fillna(False).astype(bool).any()
    ):
        raise RuntimeError("formal-large TRAIN source identity changed")
    grouped = {
        str(key): value.sort_values("sample_id")
        for key, value in frame.groupby("molecule_id", sort=False)
    }
    available = [name for name in grouped if name not in used and name not in exposed_external]
    selected = sorted(
        available,
        key=lambda name: hashlib.sha256(f"{SELECTION_SEED}|{name}".encode()).hexdigest(),
    )[:EXTERNAL_MOLECULES]
    if len(selected) != EXTERNAL_MOLECULES or set(selected) & (used | exposed_external):
        raise RuntimeError("fresh external-confirm selection failed")

    external_items, external_entries = [], []
    reference_histogram: Counter[int] = Counter()
    for position, molecule_id in enumerate(selected):
        rows = grouped[molecule_id]
        if len(rows) != 3:
            raise RuntimeError("expected exactly three Sources per molecule")
        records, paths = [], []
        for row in rows.itertuples():
            path = resolve_source(row.source_path)
            record = torch.load(path, map_location="cpu", weights_only=False)
            validate_record_identity(record)
            if str(record["sample_id"]) != str(row.sample_id) or str(record["source_mol_id"]) != molecule_id:
                raise RuntimeError("cache/manifest atom or molecule identity mismatch")
            records.append(record)
            paths.append(path)
        first = records[0]
        hash_lists = [
            [tensor_sha256(center_coordinates(ref)) for ref in torch.as_tensor(record["x_ref_candidates"])]
            for record in records
        ]
        if any(value != hash_lists[0] for value in hash_lists[1:]):
            raise RuntimeError("three Sources expose different Reference sets")
        references, reference_hashes = [], []
        for reference, digest in zip(
            torch.as_tensor(first["x_ref_candidates"], dtype=torch.float32),
            hash_lists[0],
            strict=True,
        ):
            if digest not in reference_hashes:
                references.append(reference.clone())
                reference_hashes.append(digest)
        sources = torch.stack([torch.as_tensor(record["x_init"], dtype=torch.float32) for record in records])
        reference_histogram[len(references)] += 1
        item = {
            "molecule_id": molecule_id,
            "partition": "external_confirm",
            "sample_ids": [str(record["sample_id"]) for record in records],
            "cache_names": [path.name for path in paths],
            "record": compact_record(first),
            "node_attr": torch.as_tensor(first["node_attr"], dtype=torch.float32),
            "edge_index": torch.as_tensor(first["edge_index"], dtype=torch.long),
            "edge_attr": torch.as_tensor(first["edge_attr"], dtype=torch.float32),
            "sources": sources,
            "references": torch.stack(references),
            "source_coordinate_sha256": [tensor_sha256(value) for value in sources],
            "reference_coordinate_sha256": reference_hashes,
        }
        external_items.append(item)
        external_entries.append({
            "molecule_id": molecule_id,
            "sample_ids": item["sample_ids"],
            "atom_count": int(sources.size(1)),
            "source_coordinate_sha256": item["source_coordinate_sha256"],
            "reference_coordinate_sha256": reference_hashes,
        })
        if (position + 1) % 50 == 0:
            print(f"prepared {position + 1}/{EXTERNAL_MOLECULES}", flush=True)

    compact_payload = {
        "schema_version": "mcvr-lsgo-external-confirm-compact-v1",
        "partition": "external_confirm",
        "selection_seed": SELECTION_SEED,
        "items": external_items,
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
        "full10k_used_for_tuning": False,
    }
    atomic_torch_save(EXTERNAL_COMPACT, compact_payload)
    cohorts["LSGO_EXTERNAL_CONFIRM"] = {
        "molecule_count": EXTERNAL_MOLECULES,
        "record_count": 3 * EXTERNAL_MOLECULES,
        "reference_count": int(sum(len(item["references"]) for item in external_items)),
        "molecule_identity_sha256": canonical_sha256(selected),
        "molecule_ids": sorted(selected),
    }
    names = list(cohorts)
    overlap = {
        left: {
            right: len(set(cohorts[left]["molecule_ids"]) & set(cohorts[right]["molecule_ids"]))
            for right in names
        }
        for left in names
    }
    identity = {
        "schema_version": "mcvr-lsgo-dataset-identity-v1",
        "status": "FROZEN",
        "split": "train_only",
        "training_compact": str(TRAIN_COMPACT),
        "training_compact_sha256": file_sha256(TRAIN_COMPACT),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": file_sha256(SOURCE_MANIFEST),
        "external_compact": str(EXTERNAL_COMPACT),
        "external_compact_sha256": file_sha256(EXTERNAL_COMPACT),
        "external_selection_seed": SELECTION_SEED,
        "external_selection_rule": "exclude the frozen 2500 compact molecules and all NSSM external molecules, then SHA-rank TRAIN molecule IDs",
        "nssm_exposed_molecule_count": len(exposed_external),
        "available_unexposed_molecules": len(available),
        "cohorts": cohorts,
        "molecule_overlap_matrix": overlap,
        "external_reference_count_histogram": {str(k): v for k, v in sorted(reference_histogram.items())},
        "external_entries": external_entries,
        "source_reference_identity_preserved": True,
        "formal_test_molecule_overlap": 0,
        "frozen_holdout_molecule_overlap": 0,
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
        "full10k_used_for_tuning": False,
        "runtime_seconds": time.time() - started,
    }
    identity_for_sha = {key: value for key, value in identity.items() if key not in {"runtime_seconds", "identity_sha256"}}
    identity["identity_sha256"] = canonical_sha256(identity_for_sha)
    atomic_json(OUT / "DATASET_IDENTITY.json", identity)
    atomic_json(OUT / "DATASET_IDENTITY_SHA256.json", {
        "identity_sha256": identity["identity_sha256"],
        "training_compact_sha256": identity["training_compact_sha256"],
        "external_compact_sha256": identity["external_compact_sha256"],
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
    })
    print("LSGO_DATASET_IDENTITY_FROZEN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
