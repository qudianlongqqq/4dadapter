#!/usr/bin/env python3
"""Run frozen D1/V5-B/V7 inference on the formal-large validation split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.bac_evaluation import evaluate_bac_candidate  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_v5_constraint_hybrid import (  # noqa: E402
    MCVRNeuralJacobianHybrid,
)
from etflow.ecir.mvr_v7_formal import (  # noqa: E402
    build_v7_formal_model,
    file_sha256,
    load_v7_formal_config,
)
from scripts.report_ecir_mvr_v7_formal_validation import (  # noqa: E402
    METHODS,
    build_report,
)
from scripts.run_ecir_mvr_v2_bac_pilots import _seed  # noqa: E402
from scripts.run_ecir_mvr_v7_10k_validation import (  # noqa: E402
    _build_items,
    _merge_components,
    _merge_solver,
)


V7_CONFIG_SHA256 = "5737ce5aa3bad729a6748a3fb9f0eea515bd96765df15e99bba6bd70297b8b4b"
V5_CONFIG_SHA256 = "d1e70583f77d98e95194fe7ee06eac797da4cc268ef875d54a267295eef92a41"
SOURCE_MANIFEST_SHA256 = "e7d29f971124f51bd385ec987372ab85181b152250ec0789407a867ff81e3c1a"
TARGET_MANIFEST_SHA256 = "4b4ef42c9905c3bbe2dbe911c57827ce594583c66a52f94d7c4d9b5ca70de4c7"
SOURCE_IDENTITY_SHA256 = "3d86eec9ebd82ae96860330ded0fad35938be74111929ed29b9487f8b7e39a0a"
TARGET_IDENTITY_SHA256 = "4d2d45950c92894066e347a966c6d5b877afcb5fe0abe6cdb7c06e70a3148e62"
VALIDITY_IDENTITY_SHA256 = "66dd6ab6cf290057d8dea725e0042ba5a5fcc69ad0ec9e0db8971910df377cd3"
EXPECTED_MOLECULES = 5_000
EXPECTED_RECORDS = 10_000
FROZEN_SEEDS = {
    42: {
        "checkpoint_sha256": "721b4384f3a64eef48ead2fc2b4ea35bf83802b84952e8e3f3aa6c5172e33a2f",
        "training_config_sha256": "69b795a0751453d2f0a2c24acd163dc56e6829f9186f10c697af23cd5755dd73",
    },
    43: {
        "checkpoint_sha256": "c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca",
        "training_config_sha256": "fd1f5b6780c781d8e7681b31fd93b1459f6b30ebf0e6bf4a564ecab5c16e41db",
    },
}
ISOLATION = {
    "test_records_read": 0,
    "test_assets_opened": False,
    "frozen_holdout_records_opened": 0,
    "formal_test_run": False,
    "training_performed": False,
    "target_rematerialization": False,
    "validation_only": True,
}
OUTPUT_FILES = (
    "launch.json",
    "progress.json",
    "formal_validation_per_record.csv",
    "formal_validation_per_molecule.csv",
    "method_summary.json",
    "angle_solver_summary.json",
    "component_summary.json",
    "formal_validation_report.md",
    "run_metadata.json",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _reject_forbidden_path(path: Path, label: str) -> None:
    forbidden = {"test", "formal_test", "formal-test", "holdout", "frozen_holdout"}
    lowered = {part.lower() for part in path.parts}
    if lowered & forbidden:
        raise RuntimeError(f"{label} names test or frozen-holdout data: {path}")


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def _validate_methods(methods: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(methods)
    if selected != METHODS:
        raise RuntimeError(f"formal method set must be exactly {', '.join(METHODS)}")
    return selected


def _validate_seed_contract(
    seed: int,
    checkpoint_sha256: str,
    training_config_sha256: str,
) -> None:
    if seed not in FROZEN_SEEDS:
        raise RuntimeError("formal validation seed must be 42 or 43")
    frozen = FROZEN_SEEDS[seed]
    if checkpoint_sha256 != frozen["checkpoint_sha256"]:
        raise RuntimeError("requested checkpoint SHA differs from frozen seed plan")
    if training_config_sha256 != frozen["training_config_sha256"]:
        raise RuntimeError("training config SHA differs from frozen seed plan")


def _validate_cohort_frames(
    sources: pd.DataFrame, targets: pd.DataFrame
) -> tuple[list[str], str]:
    required_source = {"sample_id", "molecule_id", "split", "test_record"}
    required_target = {"sample_id", "molecule_id", "split", "test_records_read"}
    if required_source.difference(sources.columns):
        raise RuntimeError("formal validation source manifest is missing required fields")
    if required_target.difference(targets.columns):
        raise RuntimeError("formal validation target manifest is missing required fields")
    if len(sources) != EXPECTED_RECORDS or sources.molecule_id.nunique() != EXPECTED_MOLECULES:
        raise RuntimeError("formal validation source cohort size changed")
    if len(targets) != EXPECTED_RECORDS or targets.molecule_id.nunique() != EXPECTED_MOLECULES:
        raise RuntimeError("formal validation target cohort size changed")
    if sources.sample_id.duplicated().any() or targets.sample_id.duplicated().any():
        raise RuntimeError("formal validation contains duplicate sample_id")
    if set(sources["split"].astype(str)) != {"val"} or set(
        targets["split"].astype(str)
    ) != {"val"}:
        raise RuntimeError("formal validation accepts only the val split")
    if bool(sources.test_record.astype(bool).any()) or bool(
        targets.test_records_read.astype(bool).any()
    ):
        raise RuntimeError("formal validation manifest contains test records")
    source_pairs = sources.set_index("sample_id")["molecule_id"].astype(str).sort_index()
    target_pairs = targets.set_index("sample_id")["molecule_id"].astype(str).sort_index()
    if not source_pairs.equals(target_pairs):
        raise RuntimeError("formal validation source/target samples are missing or mismatched")
    counts = source_pairs.value_counts()
    if set(counts.to_numpy()) != {2}:
        raise RuntimeError("formal validation requires exactly two records per molecule")
    molecule_ids = sorted(map(str, sources.molecule_id.unique()))
    identity = _canonical_sha(
        [[sample_id, molecule_id] for sample_id, molecule_id in source_pairs.items()]
    )
    return molecule_ids, identity


def _validate_metadata(
    source_metadata_path: Path,
    target_metadata_path: Path,
    validity_path: Path,
    wrapper: Mapping[str, Any],
) -> None:
    source = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    target = json.loads(target_metadata_path.read_text(encoding="utf-8"))
    validity = json.loads(validity_path.read_text(encoding="utf-8"))
    if source.get("formal_source_identity_sha256") != SOURCE_IDENTITY_SHA256:
        raise RuntimeError("formal validation source identity mismatch")
    if source.get("splits", {}).get("val", {}).get("manifest_sha256") != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("formal validation source metadata manifest mismatch")
    if int(source.get("test_records_read", -1)) != 0:
        raise RuntimeError("formal source metadata reports test access")
    if target.get("formal_target_identity_sha256") != TARGET_IDENTITY_SHA256:
        raise RuntimeError("formal validation target identity mismatch")
    if target.get("splits", {}).get("val", {}).get("target_manifest_sha256") != TARGET_MANIFEST_SHA256:
        raise RuntimeError("formal validation target metadata manifest mismatch")
    if int(target.get("test_records_read", -1)) != 0:
        raise RuntimeError("formal target metadata reports test access")
    if validity.get("identity_sha256") != VALIDITY_IDENTITY_SHA256:
        raise RuntimeError("formal validity statistics identity mismatch")
    expected = wrapper["formal_identities"]
    if expected != {
        "formal_source_identity_sha256": SOURCE_IDENTITY_SHA256,
        "formal_target_identity_sha256": TARGET_IDENTITY_SHA256,
        "validity_statistics_identity_sha256": VALIDITY_IDENTITY_SHA256,
    }:
        raise RuntimeError("V7 wrapper formal identities changed")


def _verify_binding(binding_dir: Path, seed: int, checkpoint_sha: str) -> None:
    required = {
        "checkpoint_identity.json",
        "config.resolved.yaml",
        "PROVENANCE.json",
        "run_metadata.json",
        "SHA256SUMS.txt",
    }
    if not binding_dir.is_dir() or not required.issubset(
        {path.name for path in binding_dir.iterdir()}
    ):
        raise RuntimeError("completed V7 formal binding is missing")
    for line in (binding_dir / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        expected, name = line.split(maxsplit=1)
        if file_sha256(binding_dir / name.strip()) != expected:
            raise RuntimeError(f"V7 binding checksum mismatch: {name.strip()}")
    metadata = json.loads((binding_dir / "run_metadata.json").read_text(encoding="utf-8"))
    expected = {
        "status": "V7_FORMAL_PRIOR_BOUND",
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha,
        "wrapper_config_sha256": V7_CONFIG_SHA256,
        "test_records_read": 0,
        "test_assets_opened": False,
        "formal_test_run": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"V7 binding field mismatch: {key}")


def _load_models(
    checkpoint: Mapping[str, Any],
    wrapper: Mapping[str, Any],
    v5_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    v7 = build_v7_formal_model(checkpoint, wrapper, device=device)
    d1_holder = build_v7_formal_model(checkpoint, wrapper, device=device)
    d1 = d1_holder.prior
    v5_holder = build_v7_formal_model(checkpoint, wrapper, device=device)
    settings = dict(v5_config["prototype_b"])
    jacobian = settings.pop("jacobian")
    v5 = MCVRNeuralJacobianHybrid(
        v5_holder.prior, jacobian_config=jacobian, **settings
    ).to(device)
    models = {"D1": d1, "V5-B": v5, "V7": v7}
    for model in models.values():
        model.eval()
    return models


def _combine_method_evaluations(
    evaluations: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_methods(tuple(evaluations))
    record_frames: list[pd.DataFrame] = []
    molecule_frames: list[pd.DataFrame] = []
    source_records: pd.DataFrame | None = None
    source_molecules: pd.DataFrame | None = None
    for method in METHODS:
        records = evaluations[method]["records"]
        molecules = evaluations[method]["molecules"]
        current_source = records.loc[records.method == "upstream"].sort_values("sample_id")
        current_source_molecules = molecules.loc[
            molecules.method == "upstream"
        ].sort_values(["group", "molecule_id"])
        if source_records is None:
            source_records = current_source.copy()
            source_records["method"] = "Source"
            source_molecules = current_source_molecules.copy()
            source_molecules["method"] = "Source"
            record_frames.append(source_records)
            molecule_frames.append(source_molecules)
        else:
            left = source_records.copy()
            left["method"] = "upstream"
            pd.testing.assert_frame_equal(
                left.reset_index(drop=True),
                current_source.reset_index(drop=True),
                check_exact=True,
            )
            left_molecules = source_molecules.copy()
            left_molecules["method"] = "upstream"
            pd.testing.assert_frame_equal(
                left_molecules.reset_index(drop=True),
                current_source_molecules.reset_index(drop=True),
                check_exact=True,
            )
        candidate_records = records.loc[records.method == "v2_bac_accepted"].copy()
        candidate_records["method"] = method
        candidate_molecules = molecules.loc[
            molecules.method == "v2_bac_accepted"
        ].copy()
        candidate_molecules["method"] = method
        record_frames.append(candidate_records)
        molecule_frames.append(candidate_molecules)
    return (
        pd.concat(record_frames, ignore_index=True),
        pd.concat(molecule_frames, ignore_index=True),
    )


def _chunk_identity(frame: pd.DataFrame, method: str = "Source") -> str:
    selected = frame.loc[frame.method == method, ["sample_id", "molecule_id"]]
    selected = selected.sort_values("sample_id")
    return _canonical_sha(selected.astype(str).values.tolist())


def _completed_chunk(
    chunk_dir: Path,
    *,
    chunk_index: int,
    expected_sample_identity: str,
) -> dict[str, Any] | None:
    summary_path = chunk_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETED" or int(summary.get("chunk", -1)) != chunk_index:
        return None
    if tuple(summary.get("methods", [])) != METHODS:
        raise RuntimeError(f"completed chunk {chunk_index} method set changed")
    if summary.get("sample_identity_sha256") != expected_sample_identity:
        raise RuntimeError(f"completed chunk {chunk_index} sample identity changed")
    for name, details in summary.get("files", {}).items():
        path = chunk_dir / name
        if not path.is_file() or file_sha256(path) != details["sha256"]:
            raise RuntimeError(f"completed chunk {chunk_index} output integrity failed: {name}")
        if len(pd.read_csv(path)) != int(details["rows"]):
            raise RuntimeError(f"completed chunk {chunk_index} row count failed: {name}")
    if set(summary.get("files", {})) != {"records.csv", "molecules.csv"}:
        raise RuntimeError(f"completed chunk {chunk_index} output set changed")
    return summary


def _validate_existing_output(output_dir: Path, resume: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise RuntimeError(f"refusing to overwrite formal validation output: {output_dir}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen V7 formal-large validation evaluator"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--binding-dir", type=Path, required=True)
    parser.add_argument(
        "--v7-config", type=Path, default=Path("configs/ecir_mvr_v7_formal_large.yaml")
    )
    parser.add_argument("--v5-config", type=Path, required=True)
    parser.add_argument("--validation-sources", type=Path, required=True)
    parser.add_argument("--validation-targets", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--target-metadata", type=Path, required=True)
    parser.add_argument(
        "--validity-statistics",
        type=Path,
        default=Path("data/ecir_mvr/validity_reference_stats.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--molecules-per-chunk", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    return parser.parse_args(argv)


def _resolve_paths(args: argparse.Namespace) -> None:
    for name in (
        "checkpoint",
        "training_config",
        "binding_dir",
        "v7_config",
        "v5_config",
        "validation_sources",
        "validation_targets",
        "source_cache_root",
        "target_cache_root",
        "source_metadata",
        "target_metadata",
        "validity_statistics",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    _validate_methods(args.methods)
    if args.molecules_per_chunk < 1 or args.batch_size < 1:
        raise RuntimeError("chunk and batch sizes must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    for name in (
        "validation_sources",
        "validation_targets",
        "source_cache_root",
        "target_cache_root",
    ):
        _reject_forbidden_path(getattr(args, name), name)
    checkpoint_sha = _require_sha(
        args.checkpoint, FROZEN_SEEDS.get(args.seed, {}).get("checkpoint_sha256", ""), "checkpoint"
    )
    training_sha = _require_sha(
        args.training_config,
        FROZEN_SEEDS.get(args.seed, {}).get("training_config_sha256", ""),
        "training config",
    )
    _validate_seed_contract(args.seed, checkpoint_sha, training_sha)
    _require_sha(args.v7_config, V7_CONFIG_SHA256, "V7 config")
    _require_sha(args.v5_config, V5_CONFIG_SHA256, "V5-B config")
    _require_sha(args.validation_sources, SOURCE_MANIFEST_SHA256, "validation sources")
    _require_sha(args.validation_targets, TARGET_MANIFEST_SHA256, "validation targets")
    wrapper = load_v7_formal_config(args.v7_config)
    training = yaml.safe_load(args.training_config.read_text(encoding="utf-8"))
    if int(training.get("seed", -1)) != args.seed:
        raise RuntimeError("training config seed mismatch")
    _validate_metadata(
        args.source_metadata,
        args.target_metadata,
        args.validity_statistics,
        wrapper,
    )
    sources = pd.read_parquet(args.validation_sources)
    targets = pd.read_parquet(args.validation_targets)
    molecule_ids, cohort_identity = _validate_cohort_frames(sources, targets)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint.get("config", {}).get("seed", -1)) != args.seed:
        raise RuntimeError("checkpoint seed mismatch")
    _verify_binding(args.binding_dir, args.seed, checkpoint_sha)
    device = torch.device(args.device)
    v5_config = yaml.safe_load(args.v5_config.read_text(encoding="utf-8"))
    models = _load_models(checkpoint, wrapper, v5_config, device)
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "training_config_sha256": training_sha,
        "wrapper": wrapper,
        "v5_config": v5_config,
        "sources": sources,
        "targets": targets,
        "molecule_ids": molecule_ids,
        "cohort_identity_sha256": cohort_identity,
    }


def _write_checksums(output_dir: Path) -> None:
    lines = [
        f"{file_sha256(output_dir / name)}  {name}\n"
        for name in OUTPUT_FILES
        if (output_dir / name).is_file()
    ]
    (output_dir / "SHA256SUMS.txt").write_text("".join(lines), encoding="ascii")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _resolve_paths(args)
    frozen = _preflight(args)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "V7_FORMAL_VALIDATION_PREFLIGHT_PASSED",
                    "seed": args.seed,
                    "molecules": EXPECTED_MOLECULES,
                    "records": EXPECTED_RECORDS,
                    **ISOLATION,
                },
                sort_keys=True,
            )
        )
        return 0

    _validate_existing_output(args.output_dir, args.resume)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    existing_launch = args.output_dir / "launch.json"
    device = torch.device(args.device)
    launch = {
        "schema_version": "mcvr-v7-formal-validation-launch-v1",
        "status": "RUNNING",
        "seed": args.seed,
        "pid": os.getpid(),
        "started_at": _utcnow(),
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "conda_environment": Path(sys.prefix).name,
        "platform": platform.platform(),
        "device": args.device,
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "output_dir": str(args.output_dir),
        "stdout_log": str(args.output_dir / "stdout.log"),
        "stderr_log": str(args.output_dir / "stderr.log"),
        "methods": list(METHODS),
        "molecules_per_chunk": args.molecules_per_chunk,
        "batch_size": args.batch_size,
        "resume": bool(args.resume),
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "training_config_sha256": frozen["training_config_sha256"],
        "v7_config_sha256": V7_CONFIG_SHA256,
        "v5_config_sha256": V5_CONFIG_SHA256,
        "validation_sources_sha256": SOURCE_MANIFEST_SHA256,
        "validation_targets_sha256": TARGET_MANIFEST_SHA256,
        "cohort_identity_sha256": frozen["cohort_identity_sha256"],
        **ISOLATION,
    }
    if existing_launch.exists() and args.resume:
        previous = json.loads(existing_launch.read_text(encoding="utf-8"))
        for key in (
            "seed",
            "methods",
            "checkpoint_sha256",
            "training_config_sha256",
            "v7_config_sha256",
            "v5_config_sha256",
            "validation_sources_sha256",
            "validation_targets_sha256",
            "cohort_identity_sha256",
        ):
            if previous.get(key) != launch[key]:
                raise RuntimeError(f"resume launch identity mismatch: {key}")
    _atomic_json(existing_launch, launch)

    molecule_ids = frozen["molecule_ids"]
    sources = frozen["sources"]
    targets_by_sample = frozen["targets"].set_index("sample_id", drop=False)
    total_chunks = math.ceil(len(molecule_ids) / args.molecules_per_chunk)
    progress = {
        "schema_version": "mcvr-v7-formal-validation-progress-v1",
        "status": "RUNNING",
        "seed": args.seed,
        "pid": os.getpid(),
        "total_chunks": total_chunks,
        "completed_chunks": [],
        "current_chunk": None,
        "records_completed": 0,
        "molecules_completed": 0,
        "updated_at": _utcnow(),
        **ISOLATION,
    }
    _atomic_json(args.output_dir / "progress.json", progress)

    models = _load_models(
        frozen["checkpoint"], frozen["wrapper"], frozen["v5_config"], device
    )
    inference = dict(frozen["wrapper"]["inference"])
    inference["batch_size"] = args.batch_size
    validity = ChemicalValidity(args.validity_statistics)
    _seed(args.seed)
    solver_summaries: list[dict[str, Any]] = []
    component_summaries: list[dict[str, Any]] = []
    chunk_summaries: list[dict[str, Any]] = []
    started = time.monotonic()

    for chunk_index, start in enumerate(
        range(0, len(molecule_ids), args.molecules_per_chunk), start=1
    ):
        selected = molecule_ids[start : start + args.molecules_per_chunk]
        selected_set = set(selected)
        chunk_sources = sources.loc[
            sources.molecule_id.astype(str).isin(selected_set)
        ].sort_values(["molecule_id", "sample_id"])
        chunk_targets = targets_by_sample.loc[chunk_sources.sample_id].reset_index(drop=True)
        expected_identity = _canonical_sha(
            chunk_sources[["sample_id", "molecule_id"]]
            .sort_values("sample_id")
            .astype(str)
            .values.tolist()
        )
        chunk_dir = chunks_dir / f"chunk_{chunk_index:04d}"
        if args.resume:
            completed = _completed_chunk(
                chunk_dir,
                chunk_index=chunk_index,
                expected_sample_identity=expected_identity,
            )
            if completed is not None:
                chunk_summaries.append(completed)
                if completed.get("angle_solver"):
                    solver_summaries.append(completed["angle_solver"])
                if completed.get("components"):
                    component_summaries.append(completed["components"])
                progress["completed_chunks"].append(chunk_index)
                progress["records_completed"] += int(completed["records"])
                progress["molecules_completed"] += int(completed["molecules"])
                progress["updated_at"] = _utcnow()
                _atomic_json(args.output_dir / "progress.json", progress)
                continue

        chunk_dir.mkdir(parents=True, exist_ok=True)
        progress["current_chunk"] = chunk_index
        progress["updated_at"] = _utcnow()
        _atomic_json(args.output_dir / "progress.json", progress)
        items = _build_items(
            chunk_sources,
            chunk_targets,
            validity,
            source_cache_root=args.source_cache_root,
            target_cache_root=args.target_cache_root,
        )
        evaluations: dict[str, Mapping[str, Any]] = {}
        method_seconds: dict[str, float] = {}
        solver: dict[str, Any] | None = None
        components: dict[str, Any] | None = None
        for method in METHODS:
            model = models[method]
            if method == "V7":
                model.reset_statistics()
            elif method == "V5-B":
                model.reset_solver_statistics()
            method_started = time.monotonic()
            evaluations[method] = evaluate_bac_candidate(
                model,
                items,
                validity,
                device=device,
                inference=inference,
                source_identity_sha256=SOURCE_IDENTITY_SHA256,
                bootstrap_draws=1,
            )
            method_seconds[method] = time.monotonic() - method_started
            if method == "V7":
                solver = model.angle_solver_summary()
                components = model.component_summary()
        records, molecules = _combine_method_evaluations(evaluations)
        if _chunk_identity(records) != expected_identity:
            raise RuntimeError(f"chunk {chunk_index} paired sample identity mismatch")
        record_path = chunk_dir / "records.csv"
        molecule_path = chunk_dir / "molecules.csv"
        _atomic_csv(record_path, records)
        _atomic_csv(molecule_path, molecules)
        summary = {
            "schema_version": "mcvr-v7-formal-validation-chunk-v1",
            "status": "COMPLETED",
            "chunk": chunk_index,
            "total_chunks": total_chunks,
            "methods": list(METHODS),
            "molecules": len(selected),
            "records": len(items),
            "sample_identity_sha256": expected_identity,
            "method_seconds": method_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "angle_solver": solver,
            "components": components,
            "files": {
                "records.csv": {
                    "rows": len(records),
                    "sha256": file_sha256(record_path),
                },
                "molecules.csv": {
                    "rows": len(molecules),
                    "sha256": file_sha256(molecule_path),
                },
            },
            **ISOLATION,
        }
        _atomic_json(chunk_dir / "summary.json", summary)
        chunk_summaries.append(summary)
        if solver:
            solver_summaries.append(solver)
        if components:
            component_summaries.append(components)
        progress["completed_chunks"].append(chunk_index)
        progress["current_chunk"] = None
        progress["records_completed"] += len(items)
        progress["molecules_completed"] += len(selected)
        progress["updated_at"] = _utcnow()
        _atomic_json(args.output_dir / "progress.json", progress)
        print(json.dumps(summary, sort_keys=True), flush=True)
        del items, evaluations, records, molecules
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record_paths = [
        chunks_dir / f"chunk_{index:04d}" / "records.csv"
        for index in range(1, total_chunks + 1)
    ]
    molecule_paths = [
        chunks_dir / f"chunk_{index:04d}" / "molecules.csv"
        for index in range(1, total_chunks + 1)
    ]
    records = pd.concat([pd.read_csv(path) for path in record_paths], ignore_index=True)
    molecules = pd.concat([pd.read_csv(path) for path in molecule_paths], ignore_index=True)
    for method in ("Source", *METHODS):
        selected = records.loc[records.method == method]
        if len(selected) != EXPECTED_RECORDS or selected.sample_id.nunique() != EXPECTED_RECORDS:
            raise RuntimeError(f"final formal validation record identity failed: {method}")
    _atomic_csv(args.output_dir / "formal_validation_per_record.csv", records)
    _atomic_csv(args.output_dir / "formal_validation_per_molecule.csv", molecules)
    angle_solver = _merge_solver(solver_summaries)
    components = _merge_components(component_summaries)
    _atomic_json(args.output_dir / "angle_solver_summary.json", angle_solver)
    _atomic_json(args.output_dir / "component_summary.json", components)
    report = build_report(args.output_dir, seed=args.seed, bootstrap_draws=10_000)
    runtime = time.monotonic() - started
    metadata = {
        "schema_version": "mcvr-v7-formal-validation-run-v1",
        "status": "COMPLETED",
        "seed": args.seed,
        "methods": list(METHODS),
        "molecules": EXPECTED_MOLECULES,
        "records": EXPECTED_RECORDS,
        "chunks": total_chunks,
        "evaluation_seconds": runtime,
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "training_config_sha256": frozen["training_config_sha256"],
        "v7_config_sha256": V7_CONFIG_SHA256,
        "v5_config_sha256": V5_CONFIG_SHA256,
        "validation_sources_sha256": SOURCE_MANIFEST_SHA256,
        "validation_targets_sha256": TARGET_MANIFEST_SHA256,
        "cohort_identity_sha256": frozen["cohort_identity_sha256"],
        "same_sample_identity": bool(report["same_sample_identity"]),
        "same_source_metrics": bool(report["same_source_metrics"]),
        "angle_solver": angle_solver,
        "components": components,
        **ISOLATION,
    }
    _atomic_json(args.output_dir / "run_metadata.json", metadata)
    progress.update(
        {
            "status": "COMPLETED",
            "current_chunk": None,
            "completed_chunks": list(range(1, total_chunks + 1)),
            "records_completed": EXPECTED_RECORDS,
            "molecules_completed": EXPECTED_MOLECULES,
            "updated_at": _utcnow(),
        }
    )
    _atomic_json(args.output_dir / "progress.json", progress)
    launch["status"] = "COMPLETED"
    launch["completed_at"] = _utcnow()
    _atomic_json(existing_launch, launch)
    _write_checksums(args.output_dir)
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
