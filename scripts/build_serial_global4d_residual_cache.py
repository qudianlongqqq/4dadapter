#!/usr/bin/env python
"""Build a new identity-bound Stage 2 cache from frozen Cartesian rollout."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.global_coupled_4d_sampling import (
    atomic_json_save,
    atomic_torch_save,
    file_sha256,
)
from etflow.data.flexbond_cache_schema import x_init_sha256
from etflow.data.flexbond_eval_manifest import load_eval_manifest
from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset
from etflow.formal_large import canonical_sha256
from etflow.serial_global4d.cache import (
    build_stage2_training_record,
    cartesian_sampling_identity,
    load_frozen_cartesian_teacher,
    resolve_cartesian_teacher_selection,
    rollout_frozen_cartesian,
    validate_stage2_training_record,
)
from etflow.serial_global4d.targets import materialize_stage2_targets


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _resume_identity(
    candidate: dict[str, Any], identity_path: Path
) -> tuple[dict[str, Any], list[str]]:
    """Reuse an existing sampling identity only when every command input matches."""

    if not identity_path.is_file():
        atomic_json_save(candidate, identity_path)
        return candidate, [str(candidate["code_commit"])]
    previous = json.loads(identity_path.read_text(encoding="utf-8"))

    def comparable(value: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(value)
        result.pop("identity_sha256", None)
        result.pop("code_commit", None)
        return result

    if comparable(previous) != comparable(candidate):
        raise ValueError("Existing Stage 2 cache belongs to another teacher/command")
    commits = list(
        dict.fromkeys((str(previous["code_commit"]), str(candidate["code_commit"])))
    )
    return previous, commits


def _record_manifest_row(
    record: Mapping[str, Any], manifest_index: int, dataset_index: int, path: Path
) -> dict[str, Any]:
    return {
        "index": manifest_index,
        "source_dataset_index": dataset_index,
        "sample_id": str(record.get("sample_id", manifest_index)),
        "mol_id": str(record.get("mol_id", "")),
        "path": path.name,
        "x_init_hash": str(record["x_init_hash"]),
        "x_cart_sha256": str(record["x_cart_sha256"]),
    }


def _audit_partial_cache(
    split_dir: Path,
    manifest_rows: list[Mapping[str, Any]],
    manifest_indices: list[int],
    *,
    limit: int,
    split: str,
    identity: Mapping[str, Any],
    manifest_sha: str,
    target_times: list[float],
) -> list[dict[str, Any]]:
    """Prove that an existing cache is an atomic, identity-complete prefix."""

    unexpected = sorted(
        path.name
        for path in split_dir.iterdir()
        if not path.is_file() or re.fullmatch(r"\d{8}\.pt", path.name) is None
    )
    if unexpected:
        raise ValueError(f"Partial cache contains unexpected files: {unexpected[:20]}")
    files = sorted(split_dir.glob("*.pt"))
    if len(files) > limit:
        raise ValueError(
            "Partial cache contains more records than the requested cohort"
        )
    expected_names = [f"{index:08d}.pt" for index in range(len(files))]
    if [path.name for path in files] != expected_names:
        raise ValueError("Partial cache is not a contiguous manifest-order prefix")

    teacher_sha = str(identity["identity_sha256"])
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(files):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(record, Mapping):
            raise TypeError(f"Partial cache record is not a mapping: {path}")
        validate_stage2_training_record(record, require_targets=True)
        expected = manifest_rows[index]
        sample_id = str(record.get("sample_id", ""))
        hard_checks = {
            "sample_id": sample_id == str(expected["sample_id"]),
            "mol_id": str(record.get("mol_id", "")) == str(expected["mol_id"]),
            "x_init_hash": str(record.get("x_init_hash", ""))
            == str(expected["x_init_hash"]),
            "num_atoms": int(record.get("num_atoms", -1)) == int(expected["num_atoms"]),
            "num_edges": int(record.get("num_edges", -1)) == int(expected["num_edges"]),
            "num_joints": int(record.get("num_joints", -1))
            == int(expected["num_joints"]),
            "flexibility_cohort": str(record.get("flexibility_cohort", ""))
            == str(expected["flexibility_cohort"]),
            "split": record.get("split") == split
            and record.get("source_split") == split,
            "teacher_identity": record.get("teacher_sampling_identity") == identity,
            "teacher_identity_sha256": str(
                record.get("teacher_sampling_identity_sha256", "")
            )
            == teacher_sha,
            "manifest_identity": all(
                str(record.get(field, "")) == manifest_sha
                for field in (
                    "original_manifest_identity",
                    "source_manifest_sha",
                    "pilot_manifest_sha",
                )
            ),
            "target_time": float(record.get("target_time", float("nan")))
            == target_times[index % len(target_times)],
        }
        failed = [name for name, passed in hard_checks.items() if not passed]
        if failed:
            raise ValueError(
                f"Partial cache identity mismatch at index {index}: {failed}"
            )
        if sample_id in seen:
            raise ValueError(
                f"Partial cache contains duplicate sample_id {sample_id!r}"
            )
        seen.add(sample_id)
        rows.append(_record_manifest_row(record, index, manifest_indices[index], path))
    return rows


def _manifest_dataset_indices(dataset, manifest: dict) -> list[int]:
    """Resolve an ordered validation cohort and fail closed on every mismatch."""

    rows = manifest["records"]
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Validation manifest contains duplicate sample_id values")
    if all(str(row.get("source_file_name", "")).strip() for row in rows):
        by_name = {path.name: index for index, path in enumerate(dataset.data_files)}
        ordered = []
        for expected in rows:
            name = str(expected["source_file_name"])
            if name not in by_name:
                raise ValueError(f"Cohort source file is missing: {name}")
            index = by_name[name]
            header = torch.load(
                dataset.data_files[index], map_location="cpu", weights_only=False
            )
            sample_id = str(header.get("sample_id", header.get("mol_id")))
            if sample_id != str(expected["sample_id"]):
                raise ValueError(f"Frozen source_file_name identity mismatch: {name}")
            actual_hash = x_init_sha256(header["x_init"], header["atomic_numbers"])
            if actual_hash != str(header.get("x_init_hash", "")) or actual_hash != str(
                expected["x_init_hash"]
            ):
                raise ValueError(
                    f"x_init_hash mismatch for cohort sample {sample_id!r}"
                )
            ordered.append(index)
        return ordered
    found: dict[str, int] = {}
    for index, path in enumerate(dataset.data_files):
        header = torch.load(path, map_location="cpu", weights_only=False)
        sample_id = str(header.get("sample_id", header.get("mol_id")))
        expected = by_id.get(sample_id)
        if expected is None:
            continue
        if sample_id in found:
            raise ValueError(
                f"Validation cache contains duplicate sample_id {sample_id!r}"
            )
        actual_hash = x_init_sha256(header["x_init"], header["atomic_numbers"])
        persisted_hash = str(header.get("x_init_hash", ""))
        expected_hash = str(expected["x_init_hash"])
        if actual_hash != persisted_hash or actual_hash != expected_hash:
            raise ValueError(
                f"x_init_hash mismatch for validation sample {sample_id!r}"
            )
        actual_mol_id = str(header.get("source_mol_id", header.get("mol_id")))
        if actual_mol_id != str(expected["mol_id"]):
            raise ValueError(f"mol_id mismatch for validation sample {sample_id!r}")
        actual_rotatable = int(torch.as_tensor(header["rotatable_bond_index"]).size(1))
        if actual_rotatable != int(expected["num_rotatable_bonds"]):
            raise ValueError(
                f"num_rotatable_bonds mismatch for validation sample {sample_id!r}"
            )
        found[sample_id] = index
        if len(found) == len(rows):
            break
    missing = [sample_id for sample_id in by_id if sample_id not in found]
    if missing:
        raise ValueError(
            f"Validation cache is missing manifest samples: {missing[:20]}"
        )
    return [found[str(row["sample_id"])] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", required=True, type=Path)
    parser.add_argument("--teacher_config", required=True, type=Path)
    parser.add_argument("--best_configs", type=Path)
    parser.add_argument("--source_cache", required=True, type=Path)
    cohort = parser.add_mutually_exclusive_group(required=True)
    cohort.add_argument("--validation_manifest", type=Path)
    cohort.add_argument("--pilot_manifest", type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--refinement_steps", type=int, default=10)
    parser.add_argument("--update_scale", type=float, default=0.5)
    parser.add_argument("--max_displacement", type=float, default=0.1)
    parser.add_argument("--max_coordinate_norm", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_times", default="0,0.125,0.25")
    parser.add_argument("--ridge", type=float, default=1.0e-5)
    parser.add_argument("--rank_tol", type=float, default=1.0e-6)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("max_records must be positive")
    checkpoint, config, selection = resolve_cartesian_teacher_selection(
        best_configs=args.best_configs,
        checkpoint=args.teacher_checkpoint,
        config=args.teacher_config,
    )
    manifest_path = args.validation_manifest or args.pilot_manifest
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_eval_manifest(manifest_path)
    if manifest.get("formal_large_split") != args.split:
        raise ValueError("Cohort manifest split does not match --split")
    if args.split == "val" and args.validation_manifest is None:
        raise ValueError("Validation cache requires --validation_manifest")
    if args.split == "train" and args.pilot_manifest is None:
        raise ValueError("Train cache requires --pilot_manifest")
    manifest_raw_sha = file_sha256(manifest_path)
    manifest_sha = canonical_sha256(manifest)
    selected_manifest_sha = selection.get("validation_manifest_sha256")
    if (
        args.split == "val"
        and selected_manifest_sha
        and manifest_sha != selected_manifest_sha
    ):
        raise ValueError(
            "Explicit validation manifest does not match validation selection SHA256"
        )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    candidate_identity = cartesian_sampling_identity(
        checkpoint,
        config,
        refinement_steps=args.refinement_steps,
        update_scale=args.update_scale,
        max_displacement=args.max_displacement,
        max_coordinate_norm=args.max_coordinate_norm,
        random_seed=args.seed,
        cohort_manifest_sha256=manifest_sha,
        cohort_manifest_raw_sha256=manifest_raw_sha,
        split=args.split,
        code_commit=_commit(),
        environment=environment,
        selection_identity=selection,
        stage2_target_identity={
            "q_target_mode": "damped_global4d_residual",
            "target_times": [float(value) for value in args.target_times.split(",")],
            "ridge": float(args.ridge),
            "rank_tol": float(args.rank_tol),
            "jacobian_schema": "global-coupled-4d-v1",
            "cache_schema": "serial-global4d-residual-v2",
        },
    )
    output = args.output_dir.expanduser().resolve()
    split_dir = output / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output / f"{args.split}_cache_identity.json"
    identity, generation_code_commits = _resume_identity(
        candidate_identity, identity_path
    )
    target_times = [float(value) for value in args.target_times.split(",")]
    if not target_times or any(not 0.0 <= value <= 1.0 for value in target_times):
        raise ValueError("target_times must contain values in [0, 1]")
    dataset = FlexBondOptimizerDataset(args.source_cache, args.split, validate=True)
    manifest_indices = _manifest_dataset_indices(dataset, manifest)
    limit = (
        len(manifest_indices)
        if args.max_records is None
        else min(len(manifest_indices), args.max_records)
    )
    records = _audit_partial_cache(
        split_dir,
        manifest["records"],
        manifest_indices,
        limit=limit,
        split=args.split,
        identity=identity,
        manifest_sha=manifest_sha,
        target_times=target_times,
    )
    resume_index = len(records)
    teacher_model = (
        load_frozen_cartesian_teacher(checkpoint, device=args.device)
        if resume_index < limit
        else None
    )
    for manifest_index in range(resume_index, limit):
        dataset_index = manifest_indices[manifest_index]
        destination = split_dir / f"{manifest_index:08d}.pt"
        source = dataset[dataset_index]
        x_cart, diagnostics = rollout_frozen_cartesian(
            teacher_model,
            source,
            refinement_steps=args.refinement_steps,
            update_scale=args.update_scale,
            max_displacement=args.max_displacement,
            max_coordinate_norm=args.max_coordinate_norm,
            device=args.device,
        )
        record = build_stage2_training_record(
            source,
            x_cart,
            teacher_sampling_identity=identity,
            original_manifest_identity=manifest_sha,
            split=args.split,
            pilot_manifest_identity=manifest_sha,
        )
        target_time = target_times[manifest_index % len(target_times)]
        record.update(
            materialize_stage2_targets(
                record,
                target_time=target_time,
                ridge=args.ridge,
                rank_tol=args.rank_tol,
            )
        )
        validate_stage2_training_record(record, require_targets=True)
        record["cartesian_rollout_diagnostics"] = diagnostics
        if destination.exists():
            raise ValueError(f"Resume audit did not account for {destination}")
        atomic_torch_save(record, destination)
        records.append(
            _record_manifest_row(record, manifest_index, dataset_index, destination)
        )
        print(f"[{manifest_index + 1}/{limit}] {records[-1]['sample_id']}", flush=True)
    cache_manifest = {
        "stage2_cache_schema_version": "serial-global4d-residual-v2",
        "split": args.split,
        "record_count": len(records),
        "source_record_count": len(dataset),
        "manifest_record_count": len(manifest_indices),
        "complete": len(records) == len(manifest_indices),
        "teacher_sampling_identity_sha256": identity["identity_sha256"],
        "cohort_manifest_sha256": manifest_sha,
        "cohort_manifest_raw_sha256": manifest_raw_sha,
        "generation_code_commits": generation_code_commits,
        "resumed_from_record_count": resume_index,
        "records": records,
    }
    cache_manifest["cache_manifest_sha256"] = canonical_sha256(cache_manifest)
    atomic_json_save(cache_manifest, output / f"{args.split}_manifest.json")
    if cache_manifest["complete"]:
        completed = {
            "status": "COMPLETED",
            "split": args.split,
            "record_count": len(records),
            "teacher_sampling_identity_sha256": identity["identity_sha256"],
            "cohort_manifest_sha256": manifest_sha,
            "cache_manifest_sha256": cache_manifest["cache_manifest_sha256"],
            "generation_code_commits": generation_code_commits,
        }
        completed["cache_identity_sha256"] = canonical_sha256(completed)
        atomic_json_save(completed, output / "COMPLETED.json")


if __name__ == "__main__":
    main()
