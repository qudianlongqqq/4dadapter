#!/usr/bin/env python3
"""Freeze BAT train/dev identities and a fresh train-side external cohort."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import yaml

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.lsgo_io import atomic_json, atomic_torch_save, center_coordinates, file_sha256, validate_record_identity

OUT = ROOT / "reports/ecir_mvr/bat_refinement"
CONFIG_PATH = ROOT / "configs/ecir_mvr_bat_refinement.yaml"
V8 = Path(r"E:\3dconformergenerationcode\4dadapter-v8")
SOURCE_MANIFEST = V8 / "data/ecir_mvr/formal_large/real_sources/train.parquet"
SOURCE_CACHE = V8 / "data/flexbond_cache_formal_large/train"
UNIFIED_IDENTITY = Path(r"E:\3dconformergenerationcode\4dadapter-narrow-sigma\reports\ecir_mvr\unified_external_utility\COMMON_IDENTITY_MANIFEST.json")
EXTERNAL_COMPACT = OUT / "manifests/BAT_EXTERNAL_CONFIRM_COMPACT.pt"
SELECTION_SEED = "bat-external-confirm-20260726-v1"


def tensor_sha(value: torch.Tensor) -> str:
    array = torch.as_tensor(value, dtype=torch.float32).cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def compact_record(record: dict) -> dict:
    forbidden = {"x_init", "x_ref", "x_ref_aligned", "x_ref_candidates", "metadata", "DATA_DIR", "created_at"}
    return {key: value for key, value in record.items() if key not in forbidden}


def resolve_source(value: object) -> Path:
    path = Path(str(value))
    candidate = path if path.is_file() else SOURCE_CACHE / path.name
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def summary(items: list[dict], partition: str, label: str) -> dict:
    selected = [item for item in items if str(item["partition"]) == partition]
    identities = sorted(str(item["molecule_id"]) for item in selected)
    return {
        "name": label, "molecule_count": len(selected), "record_count": 3 * len(selected),
        "reference_count": int(sum(len(item["references"]) for item in selected)),
        "molecule_ids": identities, "molecule_identity_sha256": canonical_sha(identities),
    }


def exposed_molecules(config: dict) -> tuple[set[str], dict[str, int]]:
    result: set[str] = set(); counts = {}
    lsgo_path = Path(config["dataset"]["old_lsgo_identity"])
    lsgo = json.loads(lsgo_path.read_text(encoding="utf-8"))
    values = {str(row["molecule_id"]) for row in lsgo["external_entries"]}
    result.update(values); counts["LSGO_EXTERNAL_CONFIRM"] = len(values)
    nssm_path = Path(config["dataset"]["old_nssm_identity"])
    nssm = json.loads(nssm_path.read_text(encoding="utf-8"))
    values = {str(row["molecule_id"]) for row in nssm["entries"]}
    result.update(values); counts["NSSM_EXTERNAL_CONFIRM"] = len(values)
    if UNIFIED_IDENTITY.is_file():
        unified = json.loads(UNIFIED_IDENTITY.read_text(encoding="utf-8"))
        values = {str(row["molecule_id"]) for row in unified["identities"]}
        result.update(values); counts["UNIFIED_EXTERNAL_UTILITY"] = len(values)
    return result, counts


def main() -> int:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    training_path = Path(config["dataset"]["training_compact"])
    if file_sha256(training_path) != config["dataset"]["training_compact_sha256"]:
        raise RuntimeError("training compact SHA changed")
    training = torch.load(training_path, map_location="cpu", weights_only=False)
    if int(training.get("formal_test_records_read", -1)) != 0 or int(training.get("frozen_holdout_records_read", -1)) != 0:
        raise RuntimeError("protected split read in training identity")
    items = list(training["items"])
    cohorts = {
        "BAT_TRAIN": summary(items, "train", "BAT_TRAIN"),
        "BAT_DEV_A": summary(items, "dev_a", "BAT_DEV_A"),
        "BAT_DEV_B": summary(items, "dev_b", "BAT_DEV_B"),
    }
    if [cohorts[name]["molecule_count"] for name in cohorts] != [2000, 250, 250]:
        raise RuntimeError("train/dev cohort counts changed")
    internal_ids = set().union(*(set(value["molecule_ids"]) for value in cohorts.values()))
    exposed, exposed_counts = exposed_molecules(config)

    frame = pd.read_parquet(SOURCE_MANIFEST)
    if len(frame) != 150000 or frame.molecule_id.nunique() != 50000 or set(frame.split.astype(str)) != {"train"} or frame.test_record.fillna(False).astype(bool).any():
        raise RuntimeError("formal-large TRAIN source manifest changed")
    grouped = {str(key): value.sort_values("sample_id") for key, value in frame.groupby("molecule_id", sort=False)}
    available = [name for name in grouped if name not in internal_ids and name not in exposed]
    count = int(config["dataset"]["external_molecules"])
    selected = sorted(available, key=lambda name: hashlib.sha256(f"{SELECTION_SEED}|{name}".encode()).hexdigest())[:count]
    if len(selected) != count or set(selected) & (internal_ids | exposed):
        raise RuntimeError("fresh external selection failed")

    external_items, entries = [], []; reference_histogram: Counter[int] = Counter()
    for position, molecule_id in enumerate(selected):
        rows = grouped[molecule_id]
        if len(rows) != 3:
            raise RuntimeError("each molecule must expose three Sources")
        records, paths = [], []
        for row in rows.itertuples():
            path = resolve_source(row.source_path); record = torch.load(path, map_location="cpu", weights_only=False)
            validate_record_identity(record)
            if str(record["sample_id"]) != str(row.sample_id) or str(record["source_mol_id"]) != molecule_id:
                raise RuntimeError("source cache identity mismatch")
            records.append(record); paths.append(path)
        reference_hash_lists = [[tensor_sha(center_coordinates(ref)) for ref in torch.as_tensor(record["x_ref_candidates"])] for record in records]
        if any(values != reference_hash_lists[0] for values in reference_hash_lists[1:]):
            raise RuntimeError("three Sources expose different Reference ensembles")
        references, reference_hashes = [], []
        for reference, digest in zip(torch.as_tensor(records[0]["x_ref_candidates"], dtype=torch.float32), reference_hash_lists[0], strict=True):
            if digest not in reference_hashes:
                references.append(reference.clone()); reference_hashes.append(digest)
        sources = torch.stack([torch.as_tensor(record["x_init"], dtype=torch.float32) for record in records])
        reference_histogram[len(references)] += 1
        item = {
            "molecule_id": molecule_id, "partition": "external_confirm",
            "sample_ids": [str(record["sample_id"]) for record in records],
            "cache_names": [path.name for path in paths], "record": compact_record(records[0]),
            "sources": sources, "references": torch.stack(references),
            "source_coordinate_sha256": [tensor_sha(value) for value in sources],
            "reference_coordinate_sha256": reference_hashes,
        }
        external_items.append(item)
        entries.append({
            "molecule_id": molecule_id, "sample_ids": item["sample_ids"], "atom_count": int(sources.size(1)),
            "source_coordinate_sha256": item["source_coordinate_sha256"], "reference_coordinate_sha256": reference_hashes,
        })
        if (position + 1) % 50 == 0:
            print(f"BAT EXTERNAL PREPARE {position + 1}/{count}", flush=True)
    compact = {
        "schema_version": "mcvr-bat-external-confirm-v1", "partition": "external_confirm",
        "selection_seed": SELECTION_SEED, "items": external_items,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0, "full10k_used_for_tuning": False,
    }
    atomic_torch_save(EXTERNAL_COMPACT, compact)
    external_ids = sorted(selected)
    cohorts["BAT_EXTERNAL_CONFIRM"] = {
        "name": "BAT_EXTERNAL_CONFIRM", "molecule_count": count, "record_count": 3 * count,
        "reference_count": int(sum(len(item["references"]) for item in external_items)),
        "molecule_ids": external_ids, "molecule_identity_sha256": canonical_sha(external_ids),
    }
    names = list(cohorts)
    overlap = {left: {right: len(set(cohorts[left]["molecule_ids"]) & set(cohorts[right]["molecule_ids"])) for right in names} for left in names}
    payload = {
        "schema_version": "mcvr-bat-dataset-identity-v1", "status": "FROZEN", "split": "train_only",
        "training_compact": str(training_path), "training_compact_sha256": file_sha256(training_path),
        "source_manifest": str(SOURCE_MANIFEST), "source_manifest_sha256": file_sha256(SOURCE_MANIFEST),
        "external_compact": str(EXTERNAL_COMPACT), "external_compact_sha256": file_sha256(EXTERNAL_COMPACT),
        "external_selection_seed": SELECTION_SEED,
        "external_selection_rule": "exclude BAT train/dev and all explicitly enumerated historical exposed external identities; SHA-rank remaining TRAIN molecules",
        "historical_exposed_counts": exposed_counts, "historical_exposed_union_count": len(exposed),
        "old_lsgo_external_overlap": len(set(external_ids) & exposed),
        "cohorts": cohorts, "molecule_overlap_matrix": overlap, "external_entries": entries,
        "external_reference_count_histogram": {str(key): value for key, value in sorted(reference_histogram.items())},
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0, "full10k_used_for_tuning": False,
        "runtime_seconds": time.time() - started,
    }
    payload["identity_sha256"] = canonical_sha({key: value for key, value in payload.items() if key not in {"runtime_seconds", "identity_sha256"}})
    atomic_json(OUT / "DATASET_IDENTITY.json", payload)
    atomic_json(OUT / "DATASET_IDENTITY_SHA256.json", {
        "identity_sha256": payload["identity_sha256"], "training_compact_sha256": payload["training_compact_sha256"],
        "external_compact_sha256": payload["external_compact_sha256"], "source_manifest_sha256": payload["source_manifest_sha256"],
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    })
    atomic_json(OUT / "OLD_EXTERNAL_EXCLUSION_AUDIT.json", {
        "status": "PASS", "excluded_sources": exposed_counts, "excluded_union_count": len(exposed),
        "selected_external_molecules": count, "overlap_with_exposed_union": 0,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    })
    print("BAT_DATASET_IDENTITY_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
