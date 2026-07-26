#!/usr/bin/env python3
"""Build the train/validation-only formal-large LSGO sufficient dataset."""

from __future__ import annotations

import hashlib
import io
import json
import os
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
from etflow.ecir.learned_geometry import prepare_graph
from etflow.ecir.lsgo_io import atomic_json, atomic_torch_save, center_coordinates, file_sha256, validate_record_identity

CONFIG = ROOT / "configs/ecir_mvr_lsgo_ba_formal_large.yaml"
OUT = ROOT / "reports/ecir_mvr/lsgo_formal"


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = torch.as_tensor(value, dtype=torch.float32).cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_verified(path: Path, expected_sha256: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != str(expected_sha256):
        raise RuntimeError(f"cache SHA mismatch: {path}")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


def reference_set(record: dict) -> tuple[torch.Tensor, list[str]]:
    references, digests = [], []
    for value in torch.as_tensor(record["x_ref_candidates"], dtype=torch.float32):
        centered = center_coordinates(value)
        digest = tensor_sha256(centered)
        if digest not in digests:
            references.append(centered)
            digests.append(digest)
    if not references:
        raise RuntimeError("empty Reference ensemble")
    return torch.stack(references), digests


def validate_manifest(frame: pd.DataFrame, split: str, molecules: int, records: int, per_molecule: int) -> None:
    required = {"split", "sample_id", "molecule_id", "source_path", "source_file_sha256", "coordinate_sha256", "test_record"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{split} manifest fields missing: {sorted(required-set(frame.columns))}")
    counts = frame.groupby("molecule_id").size()
    if (
        len(frame) != records or frame.molecule_id.nunique() != molecules
        or set(frame.split.astype(str)) != {split}
        or frame.test_record.fillna(False).astype(bool).any()
        or set(counts.astype(int)) != {per_molecule}
        or frame.sample_id.duplicated().any()
    ):
        raise RuntimeError(f"formal-large {split} identity mismatch")


def build_split(frame: pd.DataFrame, split: str, cache_dir: Path, calibration: dict) -> tuple[list[dict], dict]:
    items, reference_histogram = [], Counter()
    grouped = frame.sort_values(["molecule_id", "sample_id"]).groupby("molecule_id", sort=True)
    source_binding_rows = []
    for position, (molecule_id, rows) in enumerate(grouped, start=1):
        records, reference_hash_lists, sources = [], [], []
        for row in rows.itertuples(index=False):
            path = cache_dir / Path(str(row.source_path)).name
            if not path.is_file():
                raise FileNotFoundError(path)
            record = load_verified(path, str(row.source_file_sha256))
            validate_record_identity(record)
            if str(record["sample_id"]) != str(row.sample_id) or str(record["source_mol_id"]) != str(molecule_id):
                raise RuntimeError(f"cache/manifest identity mismatch: {row.sample_id}")
            _, hashes = reference_set(record)
            records.append(record); reference_hash_lists.append(hashes)
            sources.append(center_coordinates(torch.as_tensor(record["x_init"], dtype=torch.float32)))
            source_binding_rows.append((str(row.sample_id), str(row.source_file_sha256), str(row.coordinate_sha256)))
        if any(value != reference_hash_lists[0] for value in reference_hash_lists[1:]):
            raise RuntimeError(f"Source records expose different References: {molecule_id}")
        references, reference_hashes = reference_set(records[0])
        graph = prepare_graph(records[0], calibration)
        item = {
            "molecule_id": str(molecule_id), "partition": split,
            "sample_ids": [str(value) for value in rows.sample_id],
            "graph": graph, "references": references,
            "reference_coordinate_sha256": reference_hashes,
        }
        if split == "val":
            item["sources"] = torch.stack(sources)
        items.append(item); reference_histogram[len(references)] += 1
        if position % 500 == 0:
            print(f"LSGO FORMAL PREPARE {split} {position}/{len(grouped)}", flush=True)
    summary = {
        "molecule_count": len(items), "record_count": len(frame),
        "reference_count": int(sum(len(item["references"]) for item in items)),
        "molecule_identity_sha256": canonical_sha256(sorted(str(value) for value in frame.molecule_id.unique())),
        "source_binding_sha256": canonical_sha256(source_binding_rows),
        "reference_histogram": {str(k): int(v) for k, v in sorted(reference_histogram.items())},
    }
    return items, summary


def main() -> int:
    started = time.time(); config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")); dataset = config["dataset"]
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "cache").mkdir(exist_ok=True)
    metadata_path = Path(dataset["source_metadata"]); metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("test_records_read", -1)) != 0 or int(metadata.get("train_val_overlap", -1)) != 0:
        raise RuntimeError("formal-large source metadata protection failure")
    calibration_path = Path(dataset["drcsr_calibration"]); calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if int(calibration.get("formal_test_records_read", -1)) != 0 or int(calibration.get("frozen_holdout_records_read", -1)) != 0:
        raise RuntimeError("frozen scale protected-read failure")
    train_path, val_path = Path(dataset["train_manifest"]), Path(dataset["val_manifest"])
    train, val = pd.read_parquet(train_path), pd.read_parquet(val_path)
    validate_manifest(train, "train", int(dataset["expected_train_molecules"]), int(dataset["expected_train_records"]), int(dataset["expected_train_records_per_molecule"]))
    validate_manifest(val, "val", int(dataset["expected_val_molecules"]), int(dataset["expected_val_records"]), int(dataset["expected_val_records_per_molecule"]))
    train_ids, val_ids = set(train.molecule_id.astype(str)), set(val.molecule_id.astype(str))
    if train_ids & val_ids:
        raise RuntimeError("formal-large TRAIN/VAL molecule leakage")
    train_items, train_summary = build_split(train, "train", Path(dataset["train_cache"]), calibration)
    val_items, val_summary = build_split(val, "val", Path(dataset["val_cache"]), calibration)
    prepared = {
        "schema_version": "mcvr-lsgo-ba-formal-prepared-v1", "train": train_items, "val": val_items,
        "train_manifest_sha256": file_sha256(train_path), "val_manifest_sha256": file_sha256(val_path),
        "calibration_sha256": file_sha256(calibration_path), "source_coordinates_used_for_training": False,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    }
    prepared_path = ROOT / dataset["prepared_path"]
    atomic_torch_save(prepared_path, prepared)
    identity = {
        "schema_version": "mcvr-lsgo-ba-formal-dataset-v1", "status": "FROZEN",
        "train": train_summary, "validation": val_summary, "train_val_molecule_overlap": 0,
        "train_manifest_path": str(train_path), "train_manifest_sha256": file_sha256(train_path),
        "val_manifest_path": str(val_path), "val_manifest_sha256": file_sha256(val_path),
        "source_metadata_path": str(metadata_path), "source_metadata_sha256": file_sha256(metadata_path),
        "prepared_path": str(prepared_path), "prepared_sha256": file_sha256(prepared_path),
        "drcsr_calibration_path": str(calibration_path), "drcsr_calibration_sha256": file_sha256(calibration_path),
        "drcsr_calibration_molecules": int(calibration["counts"]["molecules"]),
        "drcsr_calibration_references": int(calibration["counts"]["references"]),
        "test_overlap": "unknown/unread; protected by frozen split metadata",
        "source_coordinates_used_for_training": False,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
        "runtime_seconds": time.time() - started,
    }
    identity["identity_sha256"] = canonical_sha256({k: v for k, v in identity.items() if k not in {"runtime_seconds", "identity_sha256"}})
    atomic_json(OUT / "DATASET_IDENTITY.json", identity)
    atomic_text(OUT / "FORMAL_INHERITANCE_AUDIT.md", f"""# LSGO-BA formal-large inheritance audit

The formal method is the frozen `LearnedGeometryObjective(hidden_dim=128, layers=3, learned_sigma=false)` with a shared invariant message-passing encoder, neural Bond and cosine-Angle conditional-mean heads, and the frozen DRCSR Reference scale. It has 473,674 trainable parameters. Bond and Angle primitive Gaussian NLL means are aggregated equally. No Cartesian coordinate target, MVT teacher, learned sigma, Torsion, Ring objective, soft Clash, xTB or PoseBusters term is present.

Training inherits AdamW (`lr=3e-4`, `weight_decay=1e-6`), cosine scheduling, gradient clipping at 1.0, and uniform sampling of one Reference per sampled molecule. Formal effective batch is 64 and the frozen horizon is 12,500 optimizer steps.

Formal TRAIN contains {train_summary['molecule_count']:,} molecules / {train_summary['record_count']:,} Source identity records and VALIDATION contains {val_summary['molecule_count']:,} / {val_summary['record_count']:,}; molecule overlap is zero. LSGO does not train against those Source coordinates: it consumes Reference local Bond/Angle geometry. Therefore 800,000 draws are 16.0 molecule-equivalent epochs; 5.33 is retained only as the historical `800,000/150,000` record-accounting convention.

The Reference scales remain frozen from {identity['drcsr_calibration_molecules']:,} development TRAIN molecules ({identity['drcsr_calibration_references']:,} References), SHA `{identity['drcsr_calibration_sha256']}`. They are not refitted on formal-large.

Checkpoint payloads must contain model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, sampler generator, exposure and validation identities and must strict-resume. Selection is frozen to lowest full-validation joint BA NLL, then calibration error, then earlier step. External xTB/PoseBusters are prohibited for selection.

Formal test reads = **0**. Frozen holdout reads = **0**.
""")
    print("LSGO_BA_FORMAL_DATASET_FROZEN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
