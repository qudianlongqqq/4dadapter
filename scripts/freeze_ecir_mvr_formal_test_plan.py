#!/usr/bin/env python
"""Freeze the pre-existing formal-large test split and dual-seed test plan."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.record_identity import source_record_identity
from etflow.data.flexbond_cache_schema import validate_inference_record
from etflow.data.flexbond_eval_manifest import manifest_content_sha256
from etflow.ecir.formal_target_assets import tensor_sha256
from etflow.formal_large import SEED, TEST_MOLECULES
from scripts.evaluate_ecir_mvr_formal_test import (
    LOCKED_PLAN_STATUS,
    PLAN_SCHEMA_VERSION,
    SEED42_CHECKPOINT_SHA256,
    SELECTED_STEP,
    TRAINING_COMMIT,
    _canonical_sha256,
    _file_sha256,
    _inference_record,
    _metric_references,
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_json_lf(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-cache-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("manifests/formal_large_test.json"),
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=Path("reports/ecir_mvr/D1B_FORMAL_DUAL_SEED_TEST_PLAN.json"),
    )
    parser.add_argument(
        "--seed43-checkpoint",
        type=Path,
        default=Path(
            "logs_ecir_mvr/formal_large/d1_b_seed43_windows/checkpoints/"
            "best_noninferior_validity.ckpt"
        ),
    )
    parser.add_argument(
        "--seed43-config",
        type=Path,
        default=Path("reports/ecir_mvr/D1B_FORMAL_WINDOWS_SEED43.yaml"),
    )
    parser.add_argument(
        "--seed43-run-metadata",
        type=Path,
        default=Path(
            "logs_ecir_mvr/formal_large/d1_b_seed43_windows/run_metadata.json"
        ),
    )
    parser.add_argument(
        "--seed42-checkpoint",
        default=(
            "logs_ecir_mvr/formal_large/d1_b_seed42/checkpoints/"
            "best_noninferior_validity.ckpt"
        ),
    )
    parser.add_argument(
        "--seed42-checkpoint-sha256", default=SEED42_CHECKPOINT_SHA256
    )
    return parser


def _validate_seed43(args: argparse.Namespace) -> dict[str, str | int]:
    metadata = json.loads(args.seed43_run_metadata.read_text(encoding="utf-8"))
    if (
        metadata.get("status") != "COMPLETED"
        or metadata.get("formal_large_completed") is not True
        or int(metadata.get("completed_steps", -1)) != SELECTED_STEP
        or int(metadata.get("best_noninferior_step", -1)) != SELECTED_STEP
        or metadata.get("stop_reason") is not None
        or int(metadata.get("test_records_read", -1)) != 0
        or int(metadata.get("seed", -1)) != 43
        or metadata.get("git_commit") != TRAINING_COMMIT
    ):
        raise RuntimeError("seed43 formal training is not a frozen clean completion")
    checkpoint_sha = _file_sha256(args.seed43_checkpoint)
    config_sha = _file_sha256(args.seed43_config)
    if config_sha != metadata.get("config_sha256"):
        raise RuntimeError("seed43 config SHA differs from completed training")
    payload = torch.load(args.seed43_checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("model_type") != "MCVRModel"
        or int(payload.get("step", -1)) != SELECTED_STEP
        or int(payload.get("config", {}).get("seed", -1)) != 43
        or payload.get("config", {}).get("resolved", {}).get("config_sha256")
        != config_sha
        or payload.get("config", {}).get("resolved", {}).get("git_commit")
        != TRAINING_COMMIT
    ):
        raise RuntimeError("seed43 checkpoint provenance is not frozen")
    return {
        "seed": 43,
        "checkpoint": str(args.seed43_checkpoint.expanduser().resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "selected_step": SELECTED_STEP,
        "config": str(args.seed43_config.expanduser().resolve()),
        "config_sha256": config_sha,
        "training_git_commit": TRAINING_COMMIT,
    }


def _freeze_test_cache(root: Path) -> tuple[dict, str, str, int]:
    split = root.expanduser()
    if (split / "test").is_dir():
        split = split / "test"
    paths = sorted(split.glob("*.pt"))
    if not paths:
        raise RuntimeError(f"formal test cache is empty: {split}")
    records = []
    for path in paths:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict):
            raise TypeError(f"formal test record is not a mapping: {path}")
        checked = validate_inference_record(_inference_record(raw))
        references = _metric_references(raw, checked)
        row = {
            "mol_id": source_record_identity(raw),
            "sample_id": str(raw.get("sample_id", raw["mol_id"])),
            "x_init_hash": str(checked["x_init_hash"]),
            "num_rotatable_bonds": int(checked["rotatable_bond_index"].size(1)),
        }
        records.append(
            {
                "manifest": row,
                "source": {
                    **row,
                    "atomic_numbers_sha256": tensor_sha256(
                        checked["atomic_numbers"]
                    ),
                    "topology_signature": str(raw.get("topology_signature", "")),
                },
                "reference": {
                    "sample_id": row["sample_id"],
                    "references_sha256": tensor_sha256(references),
                },
            }
        )
    records.sort(key=lambda value: str(value["manifest"]["sample_id"]))
    sample_ids = [str(value["manifest"]["sample_id"]) for value in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("formal test cache contains duplicate sample IDs")
    molecule_count = len({str(value["manifest"]["mol_id"]) for value in records})
    if molecule_count != TEST_MOLECULES:
        raise RuntimeError(
            f"formal test molecule count changed: {molecule_count} != {TEST_MOLECULES}"
        )
    manifest = {
        "manifest_version": "1.0",
        "created_at": _now(),
        "formal_large_split": "test",
        "selection_seed": SEED,
        "cohort_policy": "all_records_in_preexisting_formal_large_test_cache",
        "selection_performed_during_freeze": False,
        "records": [value["manifest"] for value in records],
    }
    source_identity = _canonical_sha256([value["source"] for value in records])
    reference_identity = _canonical_sha256(
        [value["reference"] for value in records]
    )
    return manifest, source_identity, reference_identity, molecule_count


def main() -> int:
    args = build_parser().parse_args()
    if args.manifest_output.exists() or args.plan_output.exists():
        raise FileExistsError("refusing to overwrite a frozen test manifest or plan")
    seed43 = _validate_seed43(args)
    manifest, source_sha, reference_sha, molecules = _freeze_test_cache(
        args.test_cache_root
    )
    _atomic_json_lf(manifest, args.manifest_output)
    manifest_file_sha = _file_sha256(args.manifest_output)
    manifest_content_sha = manifest_content_sha256(manifest)
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": LOCKED_PLAN_STATUS,
        "created_at": _now(),
        "training_git_commit": TRAINING_COMMIT,
        "checkpoint_or_config_selected_from_test": False,
        "cohort_selection_from_test_metrics": False,
        "paths_are_provenance_only": True,
        "checkpoints": [
            {
                "seed": 42,
                "checkpoint": args.seed42_checkpoint,
                "checkpoint_sha256": args.seed42_checkpoint_sha256,
                "selected_step": SELECTED_STEP,
                "training_git_commit": TRAINING_COMMIT,
            },
            seed43,
        ],
        "test": {
            "manifest": str(args.manifest_output.expanduser().resolve()),
            "manifest_sha256": manifest_file_sha,
            "manifest_content_sha256": manifest_content_sha,
            "cache_root": str(args.test_cache_root.expanduser().resolve()),
            "source_identity_sha256": source_sha,
            "reference_identity_sha256": reference_sha,
            "records": len(manifest["records"]),
            "molecules": molecules,
            "records_read_during_freeze": len(manifest["records"]),
            "cohort_policy": manifest["cohort_policy"],
        },
    }
    _atomic_json_lf(plan, args.plan_output)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
