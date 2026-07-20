#!/usr/bin/env python
"""Strict train/validation-only preflight for the MCVR V8 formal-large pilot."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.ecir.mcvr_v8_full import MCVRV8FullRefiner


ISOLATION = {
    "formal_test_records_read": 0,
    "formal_test_assets_opened": False,
    "minimal_validity_target_test_used": False,
    "frozen_holdout_records_read": 0,
    "parameter_selection_from_formal_test": False,
}


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbidden(path: Path) -> None:
    lowered = str(path.resolve()).lower().replace("\\", "/")
    if any(token in lowered for token in ("formal_test", "/test/", "holdout")):
        raise RuntimeError(f"forbidden preflight path: {path}")


def _frame(path: Path, split: str, expected: int) -> pd.DataFrame:
    _forbidden(path)
    frame = pd.read_parquet(path)
    if len(frame) != expected:
        raise RuntimeError(f"{path} row count changed: expected {expected}, got {len(frame)}")
    if set(frame.split.astype(str)) != {split}:
        raise RuntimeError(f"{path} split identity changed")
    if frame.sample_id.astype(str).duplicated().any():
        raise RuntimeError(f"{path} contains duplicate sample_id")
    for column in frame:
        name = str(column).lower()
        if ("test" in name or "holdout" in name) and bool(
            frame[column].fillna(False).astype(bool).any()
        ):
            raise RuntimeError(f"forbidden record flag is active in {path}: {column}")
    return frame


def _load_verified(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    actual = _sha256_bytes(raw)
    if actual != expected_sha256:
        raise RuntimeError(f"cache SHA256 changed: {path}")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"cache payload is not a mapping: {path}")
    return payload, actual


def _scan_split(
    sources: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    split: str,
    source_root: Path,
    target_root: Path,
    payload_samples: int,
) -> dict[str, Any]:
    source_ids = set(sources.sample_id.astype(str))
    target_ids = set(targets.sample_id.astype(str))
    if source_ids != target_ids:
        raise RuntimeError(f"{split} source-target sample identity differs")
    indexed = targets.set_index("sample_id", drop=False)
    target_molecule = indexed.molecule_id.astype(str)
    source_molecule = sources.set_index("sample_id").molecule_id.astype(str)
    if not source_molecule.sort_index().equals(target_molecule.sort_index()):
        raise RuntimeError(f"{split} source-target molecule identity differs")
    ordered_sources = sources.set_index("sample_id").sort_index()
    ordered_targets = targets.set_index("sample_id").sort_index()
    if not ordered_sources.source_file_sha256.astype(str).equals(
        ordered_targets.source_file_sha256.astype(str)
    ):
        raise RuntimeError(f"{split} source SHA linkage differs")
    if not ordered_sources.coordinate_sha256.astype(str).equals(
        ordered_targets.source_coordinate_sha256.astype(str)
    ):
        raise RuntimeError(f"{split} ordered-coordinate SHA linkage differs")
    source_files = source_root / split
    target_files = target_root / split
    expected_source_files = {Path(str(value)).name for value in sources.source_path}
    expected_target_files = {Path(str(value)).name for value in targets.target_cache_path}
    actual_source_files = {
        entry.name
        for entry in os.scandir(source_files)
        if entry.is_file() and entry.name.endswith(".pt")
    }
    actual_target_files = {
        entry.name
        for entry in os.scandir(target_files)
        if entry.is_file() and entry.name.endswith(".pt")
    }
    if actual_source_files != expected_source_files:
        raise RuntimeError(f"{split} source cache filename identity differs")
    if actual_target_files != expected_target_files:
        raise RuntimeError(f"{split} target cache filename identity differs")
    sample_count = min(max(int(payload_samples), 2), len(sources))
    sample_indices = sorted(
        {round(index * (len(sources) - 1) / (sample_count - 1)) for index in range(sample_count)}
    )
    atoms = 0
    for offset, row_index in enumerate(sample_indices, start=1):
        row = sources.iloc[row_index]
        target = indexed.loc[str(row.sample_id)]
        if str(row.source_file_sha256) != str(target.source_file_sha256):
            raise RuntimeError(f"{split} source SHA binding differs: {row.sample_id}")
        if str(row.coordinate_sha256) != str(target.source_coordinate_sha256):
            raise RuntimeError(f"{split} coordinate SHA binding differs: {row.sample_id}")
        source_path = source_files / Path(str(row.source_path)).name
        target_path = target_files / Path(str(target.target_cache_path)).name
        source, _ = _load_verified(source_path, str(row.source_file_sha256))
        target_payload, _ = _load_verified(target_path, str(target.target_file_sha256))
        source_atomic = torch.as_tensor(source.get("atomic_numbers", []), dtype=torch.long)
        source_order = torch.as_tensor(
            source.get("x_init_atomic_numbers", source_atomic), dtype=torch.long
        )
        target_atomic = torch.as_tensor(
            target_payload.get("source_atomic_numbers", []), dtype=torch.long
        )
        source_coordinates = torch.as_tensor(source.get("x_init"))
        target_input = torch.as_tensor(target_payload.get("x_input"))
        target_coordinates = torch.as_tensor(target_payload.get("x_target"))
        expected_atoms = int(row.num_atoms)
        if not (
            source_atomic.numel()
            == source_order.numel()
            == target_atomic.numel()
            == source_coordinates.shape[0]
            == target_input.shape[0]
            == target_coordinates.shape[0]
            == expected_atoms
        ):
            raise RuntimeError(f"{split} atom count differs: {row.sample_id}")
        if not torch.equal(source_atomic, source_order) or not torch.equal(
            source_order, target_atomic
        ):
            raise RuntimeError(f"{split} atom ordering differs: {row.sample_id}")
        if str(source.get("sample_id")) != str(row.sample_id):
            raise RuntimeError(f"{split} source payload sample identity differs: {row.sample_id}")
        if str(target_payload.get("sample_id")) != str(row.sample_id):
            raise RuntimeError(f"{split} target payload sample identity differs: {row.sample_id}")
        if str(source.get("source_mol_id")) != str(row.molecule_id):
            raise RuntimeError(f"{split} source payload molecule identity differs: {row.sample_id}")
        if str(target_payload.get("molecule_id")) != str(row.molecule_id):
            raise RuntimeError(f"{split} target payload molecule identity differs: {row.sample_id}")
        if str(source.get("x_init_hash")) != str(row.source_x_init_hash):
            raise RuntimeError(f"{split} source coordinate identity differs: {row.sample_id}")
        if int(target_payload.get("test_records_read", -1)) != 0:
            raise RuntimeError(f"{split} target payload test isolation changed: {row.sample_id}")
        atoms += expected_atoms
        if offset % 100 == 0 or offset == len(sample_indices):
            print(
                f"formal_large_preflight_payload_progress={split}:{offset}/{len(sample_indices)}",
                flush=True,
            )
    return {
        "records": len(sources),
        "manifest_linkages_verified": len(sources),
        "cache_filenames_verified": len(sources) * 2,
        "payload_samples_verified": len(sample_indices),
        "sampled_atoms_verified": atoms,
        "payloads_verified": len(sample_indices) * 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sources", type=Path, required=True)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--val-sources", type=Path, required=True)
    parser.add_argument("--val-targets", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-train-records", type=int, default=150000)
    parser.add_argument("--expected-val-records", type=int, default=10000)
    parser.add_argument("--payload-samples-per-split", type=int, default=512)
    args = parser.parse_args()
    for path in (
        args.train_sources,
        args.train_targets,
        args.val_sources,
        args.val_targets,
        args.source_cache_root,
        args.target_cache_root,
    ):
        _forbidden(path)
        if not path.exists():
            raise FileNotFoundError(path)
    train_sources = _frame(args.train_sources, "train", args.expected_train_records)
    train_targets = _frame(args.train_targets, "train", args.expected_train_records)
    val_sources = _frame(args.val_sources, "val", args.expected_val_records)
    val_targets = _frame(args.val_targets, "val", args.expected_val_records)
    overlap = set(train_sources.molecule_id.astype(str)) & set(val_sources.molecule_id.astype(str))
    if overlap:
        raise RuntimeError(f"train/validation molecule identity overlaps: {len(overlap)}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("isolation") != ISOLATION:
        raise RuntimeError("config isolation contract changed")
    checkpoint_sha = _sha256_file(args.checkpoint)
    if checkpoint_sha != args.checkpoint_sha256:
        raise RuntimeError("D1 checkpoint SHA256 changed")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "ecir-mvr-medium-rescue-formal-large-d1b-checkpoint-v1":
        raise RuntimeError("D1 checkpoint schema changed")
    strict_model = MCVRV8FullRefiner.from_d1_checkpoint(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        error_state={"confidence_mode": "fixed", "fixed_confidence": 1.0},
        constraint_layer={"enabled": False},
        residual_scales={"bond": 1.0, "angle": 1.0},
        unroll_steps=1,
        error_state_enabled=False,
        train_d1_backbone=False,
        train_d1_head=False,
    )
    if strict_model.d1_checkpoint_identity.get("strict_load") is not True:
        raise RuntimeError("D1 checkpoint strict load was not established")
    del strict_model
    split_results = {
        "train": _scan_split(
            train_sources,
            train_targets,
            split="train",
            source_root=args.source_cache_root,
            target_root=args.target_cache_root,
            payload_samples=args.payload_samples_per_split,
        ),
        "val": _scan_split(
            val_sources,
            val_targets,
            split="val",
            source_root=args.source_cache_root,
            target_root=args.target_cache_root,
            payload_samples=args.payload_samples_per_split,
        ),
    }
    manifests = {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in {
            "train_sources": args.train_sources,
            "train_targets": args.train_targets,
            "val_sources": args.val_sources,
            "val_targets": args.val_targets,
        }.items()
    }
    result = {
        "schema_version": "mcvr-v8-formal-large-preflight-v1",
        "status": "MCVR_V8_FORMAL_LARGE_PREFLIGHT_READY",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifests": manifests,
        "splits": split_results,
        "train_validation_molecule_overlap": 0,
        "source_target_identity": "exact",
        "atom_count_and_ordering": (
            "all records cryptographically linked by ordered-coordinate SHA; "
            "deterministic payload samples deserialized and compared"
        ),
        "source_cache_root": str(args.source_cache_root.resolve()),
        "target_cache_root": str(args.target_cache_root.resolve()),
        "config_sha256": _sha256_file(args.config),
        "d1_checkpoint": str(args.checkpoint.resolve()),
        "d1_checkpoint_sha256": checkpoint_sha,
        "d1_checkpoint_strict_load": True,
        **ISOLATION,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
