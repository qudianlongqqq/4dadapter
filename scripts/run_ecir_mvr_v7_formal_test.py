#!/usr/bin/env python3
"""Run the authorized one-time Seed43 MCVR V7 formal test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.commons.record_identity import source_record_identity  # noqa: E402
from etflow.data.flexbond_cache_schema import validate_inference_record  # noqa: E402
from etflow.data.flexbond_eval_manifest import (  # noqa: E402
    load_eval_manifest,
    manifest_content_sha256,
)
from etflow.ecir.bac_evaluation import (  # noqa: E402
    attach_canonical_constraints,
    infer_bac,
)
from etflow.ecir.bac_safety import BACSafetyConfig  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.formal_target_assets import tensor_sha256  # noqa: E402
from etflow.ecir.mvr_v7_formal import (  # noqa: E402
    file_sha256,
    load_v7_formal_config,
)
from etflow.ecir.run_a_evaluation import (  # noqa: E402
    graph_data,
    method_rows,
    nearest_rmsd,
    summarize_groups,
)
from scripts.evaluate_ecir_mvr_formal_test import (  # noqa: E402
    _inference_record,
    _metric_references,
)
from scripts.report_ecir_mvr_v7_formal_test import build_report  # noqa: E402
from scripts.run_ecir_mvr_v2_bac_pilots import _seed  # noqa: E402
from scripts.run_ecir_mvr_v7_10k_validation import (  # noqa: E402
    _merge_components,
    _merge_solver,
)
from scripts.run_ecir_mvr_v7_formal_validation import (  # noqa: E402
    _load_models,
    _verify_binding,
)


SCHEMA_VERSION = "mcvr-v7-formal-test-run-v1"
PLAN_PATH = Path("reports/ecir_mvr/MCVR_V7_FORMAL_TEST_PLAN.json")
PLAN_SHA256 = "1e4436e7c5e1a3393c1ead6b322755c590ea4eac14fb8d5f3c6fd5820698def4"
CHECKPOINT_PATH = Path(
    "artifacts/ecir_mvr/formal_large/d1_b_seed43/best_noninferior_validity.ckpt"
)
CHECKPOINT_SHA256 = "c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca"
TRAINING_CONFIG_PATH = Path("reports/ecir_mvr/D1B_FORMAL_WINDOWS_SEED43.yaml")
TRAINING_CONFIG_SHA256 = "fd1f5b6780c781d8e7681b31fd93b1459f6b30ebf0e6bf4a564ecab5c16e41db"
V5_CONFIG_PATH = Path(
    "diagnostics/ecir_mvr/v5_constraint_hybrid/runs/"
    "v5_b_pilot_seed43018/config.resolved.yaml"
)
V5_CONFIG_SHA256 = "d1e70583f77d98e95194fe7ee06eac797da4cc268ef875d54a267295eef92a41"
V7_CONFIG_PATH = Path("configs/ecir_mvr_v7_formal_large.yaml")
V7_CONFIG_SHA256 = "5737ce5aa3bad729a6748a3fb9f0eea515bd96765df15e99bba6bd70297b8b4b"
BINDING_PATH = Path("artifacts/ecir_mvr/formal_large/v7_seed43")
BINDING_SHA256 = "6799a4b6f0f0f954704feeb2adf8c79432b687d4ab293b5c6524616a4cdf79b4"
MANIFEST_PATH = Path("manifests/formal_large_test.json")
MANIFEST_SHA256 = "7c872843bf4d05202c5d05015d106d5b95eac76c22a20621c541f8405ddd2522"
MANIFEST_CONTENT_SHA256 = "e0c5ff7f6f047b24a818cf13f1e4a9dbe8e79b59ca3ddf2ac4241d331a8579f1"
SOURCE_IDENTITY_SHA256 = "2d60b331b26b01629dac609d565ddbca7bfc6f807578f2523596bc8a31094ef3"
REFERENCE_IDENTITY_SHA256 = "c58c2804f5df2a97b72305da092043d227597b155d03e22fb1ec084d9fde918b"
VALIDITY_PATH = Path("data/ecir_mvr/validity_reference_stats.json")
VALIDITY_SHA256 = "ae5afaa8d3fce1b5418295309bf2c3197997180298e1781b4efc5c265258852e"
SEMANTICS_COMMIT = "065124f6eee0fa13e6e7b56ecca8fd7022aa9dc6"
FROZEN_SOURCE_FILES = {
    "etflow/ecir/bac_evaluation.py": (
        "f0df04815c0dc5a594eebc54a179fb2de88e76744ffc56790e91fe42ebc9c013"
    ),
    "etflow/ecir/bac_safety.py": (
        "01b660432b6ecf94eed7cab255a39777936dc60e90cb66cd0004e18ad87b2a9b"
    ),
    "etflow/ecir/run_a_evaluation.py": (
        "b890459edc7244047a0d2c7547681523315f1ccc95778f753625aba05670576d"
    ),
}
METHODS = ("D1", "V5-B", "V7")
ALL_METHODS = ("Source", *METHODS)
EXPECTED_RECORDS = 23_882
EXPECTED_MOLECULES = 100
BOOTSTRAP_DRAWS = 10_000
EVALUATOR_SEMANTICS = "formal_d1b_weighted_bac_v1"
TRAJECTORY_SEMANTICS = "formal_d1b"
SAFETY_OBJECTIVE_MODE = "weighted_thresholded_validity"
DEFAULT_TEST_CACHE = Path(
    "E:/3dconformergenerationcode/dataset/flexbond_cache_formal_large"
)
PREDICTION_FIELDS = {
    "schema_version",
    "sample_ids",
    "coordinates",
    "metadata",
}
OUTPUT_FILES = (
    "launch.json",
    "progress.json",
    "environment_manifest.json",
    "formal_test_per_record.csv",
    "formal_test_per_molecule.csv",
    "angle_solver_summary.json",
    "component_summary.json",
    "summary.json",
    "summary.md",
    "run_metadata.json",
    "test_identity.json",
)
_CACHE_NAME = re.compile(
    r"^test__(?P<molecule>[0-9a-f]+)__(?P<token>[0-9a-f]+)__"
    r"(?P<generation>gen[0-9]+)\.pt$"
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
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


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authorized one-time Seed43 MCVR V7 formal test"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize-frozen-test", action="store_true")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v7_formal_test/seed43"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--molecules-per-chunk", type=int, default=1)
    parser.add_argument("--test-cache-root", type=Path, default=DEFAULT_TEST_CACHE)
    return parser


def _dry_run(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorize_frozen_test:
        raise RuntimeError("--dry-run and --authorize-frozen-test are mutually exclusive")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "V7_FORMAL_TEST_DRY_RUN_READY",
        "dry_run": True,
        "formal_test_authorized": False,
        "formal_test_started": False,
        "test_records_read": 0,
        "test_assets_opened": False,
        "output_files_created": False,
        "seed": args.seed,
        "methods": list(ALL_METHODS),
        "plan": str(PLAN_PATH),
        "plan_sha256": PLAN_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "evaluator_semantics_version": EVALUATOR_SEMANTICS,
        "trajectory_semantics": TRAJECTORY_SEMANTICS,
        "safety_objective_mode": SAFETY_OBJECTIVE_MODE,
        "reference_policy": "post_prediction_metrics_only",
        "minimal_validity_target_test_required": False,
        "checkpoint_or_config_selected_from_test": False,
        "cohort_selection_from_test_metrics": False,
        "parameter_tuning_from_test": False,
        "training_performed": False,
        "authorization_required": "--authorize-frozen-test",
    }


def _load_plan() -> dict[str, Any]:
    _require_sha(ROOT / PLAN_PATH, PLAN_SHA256, "V7 formal-test plan")
    plan = json.loads((ROOT / PLAN_PATH).read_text(encoding="utf-8"))
    expected = {
        "schema_version": "mcvr-v7-formal-test-plan-v1",
        "status": "MCVR_V7_FORMAL_TEST_PLAN_LOCKED",
        "seed": 43,
        "methods": list(ALL_METHODS),
        "checkpoint_or_config_selected_from_test": False,
        "cohort_selection_from_test_metrics": False,
        "parameter_tuning_from_test": False,
        "method_selection_from_test": False,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise RuntimeError(f"frozen V7 formal-test plan mismatch: {key}")
    if plan["evaluator"] != {
        "semantics_version": EVALUATOR_SEMANTICS,
        "semantics_git_commit": SEMANTICS_COMMIT,
        "trajectory_semantics": TRAJECTORY_SEMANTICS,
        "safety_objective_mode": SAFETY_OBJECTIVE_MODE,
        "bac_evaluation_sha256": FROZEN_SOURCE_FILES[
            "etflow/ecir/bac_evaluation.py"
        ],
        "bac_safety_sha256": FROZEN_SOURCE_FILES["etflow/ecir/bac_safety.py"],
        "run_a_evaluation_sha256": FROZEN_SOURCE_FILES[
            "etflow/ecir/run_a_evaluation.py"
        ],
    }:
        raise RuntimeError("frozen evaluator semantics changed")
    test = plan["test"]
    if (
        int(test["records"]) != EXPECTED_RECORDS
        or int(test["molecules"]) != EXPECTED_MOLECULES
        or test["source_identity_sha256"] != SOURCE_IDENTITY_SHA256
        or test["reference_identity_sha256"] != REFERENCE_IDENTITY_SHA256
        or bool(test["minimal_validity_target_test_required"])
    ):
        raise RuntimeError("frozen formal-test cohort changed")
    if int(plan["metrics"]["bootstrap_draws"]) != BOOTSTRAP_DRAWS:
        raise RuntimeError("frozen bootstrap setting changed")
    return plan


def _static_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != 43:
        raise RuntimeError("this one-time formal test is frozen to Seed43")
    if args.batch_size != 64 or args.molecules_per_chunk != 1:
        raise RuntimeError("frozen formal-test chunk or batch setting changed")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    plan = _load_plan()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", SEMANTICS_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise RuntimeError("evaluator semantics commit is not an ancestor of HEAD")
    for relative, expected in FROZEN_SOURCE_FILES.items():
        _require_sha(ROOT / relative, expected, relative)
    _require_sha(ROOT / CHECKPOINT_PATH, CHECKPOINT_SHA256, "Seed43 checkpoint")
    _require_sha(
        ROOT / TRAINING_CONFIG_PATH,
        TRAINING_CONFIG_SHA256,
        "Seed43 training config",
    )
    _require_sha(ROOT / V5_CONFIG_PATH, V5_CONFIG_SHA256, "V5-B config")
    _require_sha(ROOT / V7_CONFIG_PATH, V7_CONFIG_SHA256, "V7 config")
    _require_sha(ROOT / VALIDITY_PATH, VALIDITY_SHA256, "validity statistics")
    _require_sha(
        ROOT / BINDING_PATH / "SHA256SUMS.txt",
        BINDING_SHA256,
        "Seed43 V7 binding",
    )
    _verify_binding(ROOT / BINDING_PATH, 43, CHECKPOINT_SHA256)
    training = yaml.safe_load(
        (ROOT / TRAINING_CONFIG_PATH).read_text(encoding="utf-8")
    )
    if int(training.get("seed", -1)) != 43:
        raise RuntimeError("Seed43 training config identity changed")
    wrapper = load_v7_formal_config(ROOT / V7_CONFIG_PATH)
    v5_config = yaml.safe_load((ROOT / V5_CONFIG_PATH).read_text(encoding="utf-8"))
    checkpoint = torch.load(
        ROOT / CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("schema_version")
        != "ecir-mvr-medium-rescue-formal-large-d1b-checkpoint-v1"
        or int(checkpoint.get("step", -1)) != 25_000
        or int(checkpoint.get("config", {}).get("seed", -1)) != 43
    ):
        raise RuntimeError("Seed43 checkpoint provenance changed")
    device = torch.device(args.device)
    models = _load_models(checkpoint, wrapper, v5_config, device)
    return {
        "plan": plan,
        "checkpoint": checkpoint,
        "wrapper": wrapper,
        "v5_config": v5_config,
        "models": models,
        "device": device,
        "evaluator_git_head": _git("rev-parse", "HEAD"),
        "evaluator_worktree_dirty": bool(_git("status", "--short")),
    }


def _cache_sample_id(path: Path) -> str:
    match = _CACHE_NAME.fullmatch(path.name)
    if match is None:
        raise RuntimeError(f"unexpected frozen test cache filename: {path.name}")
    return f"test::{match.group('molecule')}__{match.group('generation')}"


def _open_manifest_and_index(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = ROOT / MANIFEST_PATH
    _require_sha(manifest_path, MANIFEST_SHA256, "formal-test manifest")
    manifest = load_eval_manifest(manifest_path)
    content_sha = manifest_content_sha256(manifest)
    if content_sha != MANIFEST_CONTENT_SHA256:
        raise RuntimeError("formal-test manifest content identity changed")
    if manifest.get("formal_large_split") != "test":
        raise RuntimeError("frozen manifest is not the formal test split")
    rows = list(manifest["records"])
    sample_ids = [str(row["sample_id"]) for row in rows]
    molecule_ids = [str(row["mol_id"]) for row in rows]
    if (
        len(rows) != EXPECTED_RECORDS
        or len(set(sample_ids)) != EXPECTED_RECORDS
        or len(set(molecule_ids)) != EXPECTED_MOLECULES
    ):
        raise RuntimeError("formal-test manifest cohort identity changed")
    cache_root = args.test_cache_root.expanduser().resolve()
    split_root = cache_root / "test" if (cache_root / "test").is_dir() else cache_root
    paths = sorted(split_root.glob("*.pt"))
    index: dict[str, Path] = {}
    for path in paths:
        sample_id = _cache_sample_id(path)
        if sample_id in index:
            raise RuntimeError(f"duplicate frozen test cache sample: {sample_id}")
        index[sample_id] = path
    missing = sorted(set(sample_ids).difference(index))
    unexpected = sorted(set(index).difference(sample_ids))
    if missing or unexpected or len(index) != EXPECTED_RECORDS:
        raise RuntimeError(
            "formal-test manifest/cache mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return {
        "manifest": manifest,
        "rows": rows,
        "index": index,
        "cache_root": cache_root,
        "split_root": split_root,
        "manifest_content_sha256": content_sha,
        "ordered_sample_ids_sha256": _canonical_sha(sample_ids),
    }


def _validate_source_identity(
    raw: Mapping[str, Any],
    checked: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = str(manifest_row["sample_id"])
    actual = {
        "mol_id": source_record_identity(raw),
        "sample_id": str(raw.get("sample_id", raw["mol_id"])),
        "x_init_hash": str(checked["x_init_hash"]),
        "num_rotatable_bonds": int(checked["rotatable_bond_index"].size(1)),
    }
    expected = {
        "mol_id": str(manifest_row["mol_id"]),
        "sample_id": sample_id,
        "x_init_hash": str(manifest_row["x_init_hash"]),
        "num_rotatable_bonds": int(manifest_row["num_rotatable_bonds"]),
    }
    if actual != expected:
        raise RuntimeError(f"formal-test source identity mismatch: {sample_id}")
    return {
        **expected,
        "atomic_numbers_sha256": tensor_sha256(checked["atomic_numbers"]),
        "topology_signature": str(raw.get("topology_signature", "")),
    }


def _source_item(
    raw: Mapping[str, Any],
    record: Mapping[str, Any],
    checked: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    validity: ChemicalValidity,
) -> dict[str, Any]:
    if any(
        key.startswith("x_ref") or key.startswith("selected_ref")
        for key in record
    ):
        raise RuntimeError("reference field entered formal-test inference record")
    coordinates = torch.as_tensor(checked["x_init"], dtype=torch.float32)
    input_validity = validity.evaluate(
        coordinates, record, baseline_coordinates=coordinates
    )
    rotatable = int(checked["rotatable_bond_index"].size(1))
    has_ring = bool(torch.as_tensor(record.get("bond_is_in_ring", [])).any())
    clean_names = (
        "bond_outlier_rate",
        "angle_outlier_rate",
        "ring_bond_outlier_rate",
        "ring_planarity_outlier_rate",
        "clash_penetration",
        "severe_clash_rate",
        "stereocenter_degenerate_rate",
    )
    clean = all(float(input_validity[name]) <= 0.0 for name in clean_names)
    active = torch.tensor(
        [
            float(input_validity["bond_outlier_rate"] > 0),
            float(input_validity["angle_outlier_rate"] > 0),
            float(
                input_validity["ring_bond_outlier_rate"] > 0
                or input_validity["ring_planarity_outlier_rate"] > 0
            ),
            float(
                input_validity["clash_penetration"] > 0
                or input_validity["severe_clash_rate"] > 0
            ),
            float(input_validity["torsion_prior_outlier_score"] > 4.0),
            float(clean),
        ]
    )
    metadata = dict(raw.get("metadata") or {})
    row = SimpleNamespace(
        molecule_id=str(manifest_row["mol_id"]),
        sample_id=str(manifest_row["sample_id"]),
        generator_name=str(raw.get("generator_name", "ETFlow_formal_upstream")),
        source_severity=str(raw.get("source_severity", "normal")),
        update_scale=float(metadata.get("update_scale", 0.0) or 0.0),
        NFE=int(metadata.get("NFE", metadata.get("nfe", 0)) or 0),
        seed=int(raw.get("sample_seed", metadata.get("seed", 0)) or 0),
    )
    groups = [
        "all",
        (
            "rotatable_le_2"
            if rotatable <= 2
            else "rotatable_3_5"
            if rotatable <= 5
            else "rotatable_ge_6"
        ),
        "ring" if has_ring else "non_ring",
    ]
    if clean:
        groups.append("clean_valid")
    return {
        "row": row,
        "record": record,
        "input": coordinates,
        "input_validity": input_validity,
        "data": graph_data(record, coordinates, row, active_mode_mask=active),
        "groups": groups,
        "rotatable": rotatable,
        "has_ring": has_ring,
        "clean": clean,
        "atomic_numbers": torch.as_tensor(checked["atomic_numbers"], dtype=torch.long),
        "edge_index": torch.as_tensor(checked["edge_index"], dtype=torch.long),
    }


def _load_source_item(
    path: Path,
    manifest_row: Mapping[str, Any],
    validity: ChemicalValidity,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(raw, Mapping):
        raise TypeError(f"formal-test source payload is not a mapping: {path}")
    record = _inference_record(raw)
    checked = validate_inference_record(record)
    identity = _validate_source_identity(raw, checked, manifest_row)
    return _source_item(raw, record, checked, manifest_row, validity), identity


def _load_metric_item(
    path: Path,
    manifest_row: Mapping[str, Any],
    validity: ChemicalValidity,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise TypeError(f"formal-test metric payload is not a mapping: {path}")
    record = _inference_record(raw)
    checked = validate_inference_record(record)
    _validate_source_identity(raw, checked, manifest_row)
    references = _metric_references(raw, checked)
    item = _source_item(raw, record, checked, manifest_row, validity)
    item["references"] = references
    item["input_rmsd"] = nearest_rmsd(item["input"], references)
    return item, {
        "sample_id": str(manifest_row["sample_id"]),
        "references_sha256": tensor_sha256(references),
    }


def _chunk_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        molecule = str(row["mol_id"])
        if molecule not in grouped:
            grouped[molecule] = []
            order.append(molecule)
        grouped[molecule].append(row)
    return [grouped[molecule] for molecule in order]


def _chunk_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    values = sorted(
        [[str(row["sample_id"]), str(row["mol_id"])] for row in rows]
    )
    return _canonical_sha(values)


def _completed_prediction(
    chunk_dir: Path, chunk_index: int, identity: str
) -> dict[str, Any] | None:
    summary_path = chunk_dir / "summary.json"
    prediction_path = chunk_dir / "predictions.pt"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "PREDICTIONS_COMPLETE"
        or int(summary.get("chunk", -1)) != chunk_index
        or summary.get("sample_identity_sha256") != identity
        or tuple(summary.get("methods", [])) != METHODS
    ):
        return None
    if (
        not prediction_path.is_file()
        or file_sha256(prediction_path) != summary.get("prediction_sha256")
        or set(summary.get("prediction_payload_fields", [])) != PREDICTION_FIELDS
    ):
        raise RuntimeError(f"prediction chunk {chunk_index} integrity failed")
    return summary


def _completed_metrics(
    chunk_dir: Path, chunk_index: int, identity: str
) -> dict[str, Any] | None:
    summary_path = chunk_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "METRICS_COMPLETE"
        or int(summary.get("chunk", -1)) != chunk_index
        or summary.get("sample_identity_sha256") != identity
        or tuple(summary.get("methods", [])) != ALL_METHODS
    ):
        return None
    for name in ("records.csv", "molecules.csv"):
        path = chunk_dir / name
        details = summary.get("files", {}).get(name, {})
        if (
            not path.is_file()
            or file_sha256(path) != details.get("sha256")
            or len(pd.read_csv(path)) != int(details.get("rows", -1))
        ):
            raise RuntimeError(f"metric chunk {chunk_index} integrity failed: {name}")
    return summary


def _prepare_output(
    args: argparse.Namespace, static: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = args.output_dir.expanduser().resolve()
    run_metadata = output / "run_metadata.json"
    if run_metadata.is_file():
        payload = json.loads(run_metadata.read_text(encoding="utf-8"))
        if payload.get("status") == "COMPLETED":
            raise RuntimeError(f"refusing to overwrite completed formal test: {output}")
    existing = [
        path for path in output.iterdir()
        if path.name not in {"stdout.log", "stderr.log"}
    ] if output.is_dir() else []
    if existing and not args.resume:
        raise RuntimeError(f"incomplete formal-test output requires --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = output
    launch = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "started_at": _utcnow(),
        "pid": os.getpid(),
        "formal_test_authorized": True,
        "seed": 43,
        "methods": list(ALL_METHODS),
        "plan_sha256": PLAN_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "training_config_sha256": TRAINING_CONFIG_SHA256,
        "v5_config_sha256": V5_CONFIG_SHA256,
        "v7_config_sha256": V7_CONFIG_SHA256,
        "binding_sha256": BINDING_SHA256,
        "evaluator_semantics_version": EVALUATOR_SEMANTICS,
        "evaluator_semantics_git_commit": SEMANTICS_COMMIT,
        "evaluator_git_head": static["evaluator_git_head"],
        "runner_sha256": file_sha256(Path(__file__)),
        "reporter_sha256": file_sha256(
            ROOT / "scripts/report_ecir_mvr_v7_formal_test.py"
        ),
        "trajectory_semantics": TRAJECTORY_SEMANTICS,
        "safety_objective_mode": SAFETY_OBJECTIVE_MODE,
        "manifest_sha256": MANIFEST_SHA256,
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "source_identity_sha256": SOURCE_IDENTITY_SHA256,
        "reference_identity_sha256": REFERENCE_IDENTITY_SHA256,
        "records": EXPECTED_RECORDS,
        "molecules": EXPECTED_MOLECULES,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "batch_size": args.batch_size,
        "molecules_per_chunk": args.molecules_per_chunk,
        "device": args.device,
        "output_dir": str(output),
        "resume": bool(args.resume),
        "reference_policy": "post_prediction_metrics_only",
        "test_records_read": 0,
        "test_assets_opened": False,
        "training_performed": False,
        "checkpoint_or_config_selected_from_test": False,
        "cohort_selection_from_test_metrics": False,
        "parameter_tuning_from_test": False,
    }
    launch_path = output / "launch.json"
    if launch_path.is_file() and args.resume:
        previous = json.loads(launch_path.read_text(encoding="utf-8"))
        for key in (
            "seed",
            "methods",
            "plan_sha256",
            "checkpoint_sha256",
            "training_config_sha256",
            "v5_config_sha256",
            "v7_config_sha256",
            "binding_sha256",
            "evaluator_semantics_version",
            "trajectory_semantics",
            "safety_objective_mode",
            "manifest_content_sha256",
            "source_identity_sha256",
            "reference_identity_sha256",
            "batch_size",
            "molecules_per_chunk",
        ):
            if previous.get(key) != launch.get(key):
                raise RuntimeError(f"formal-test resume identity mismatch: {key}")
        launch["started_at"] = previous["started_at"]
    _atomic_json(launch_path, launch)
    progress_path = output / "progress.json"
    if progress_path.is_file() and args.resume:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {
            "schema_version": SCHEMA_VERSION,
            "status": "RUNNING",
            "phase": "STATIC_PREFLIGHT_COMPLETE",
            "total_chunks": EXPECTED_MOLECULES,
            "completed_prediction_chunks": [],
            "completed_metric_chunks": [],
            "current_chunk": None,
            "predictions_complete": False,
            "references_opened": False,
            "formal_test_authorized": True,
            "test_records_read": 0,
            "test_record_open_events": 0,
            "test_assets_opened": False,
            "training_performed": False,
            "checkpoint_or_config_selected_from_test": False,
            "cohort_selection_from_test_metrics": False,
            "parameter_tuning_from_test": False,
            "updated_at": _utcnow(),
        }
    _atomic_json(progress_path, progress)
    return launch, progress


def _write_environment(output: Path, args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    _atomic_json(
        output / "environment_manifest.json",
        {
            "schema_version": "mcvr-v7-formal-test-environment-v1",
            "python": sys.version,
            "python_executable": sys.executable,
            "conda_environment": Path(sys.prefix).name,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "torch_geometric": importlib.metadata.version("torch-geometric"),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": pd.__version__,
            "rdkit": importlib.metadata.version("rdkit"),
            "platform": platform.platform(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "formal_test_authorized": True,
            "training_performed": False,
        },
    )


def _write_checksums(output: Path) -> None:
    lines = [
        f"{file_sha256(output / name)}  {name}\n"
        for name in OUTPUT_FILES
        if (output / name).is_file()
    ]
    (output / "SHA256SUMS.txt").write_text("".join(lines), encoding="ascii")


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.authorize_frozen_test:
        raise RuntimeError(
            "formal test access requires explicit --authorize-frozen-test"
        )
    static = _static_preflight(args)
    launch, progress = _prepare_output(args, static)
    output = args.output_dir
    _write_environment(output, args)
    frozen = _open_manifest_and_index(args)
    progress.update(
        {
            "phase": "PREDICTION",
            "test_assets_opened": True,
            "updated_at": _utcnow(),
        }
    )
    launch["test_assets_opened"] = True
    _atomic_json(output / "progress.json", progress)
    _atomic_json(output / "launch.json", launch)

    chunk_rows = _chunk_groups(frozen["rows"])
    if len(chunk_rows) != EXPECTED_MOLECULES:
        raise RuntimeError("formal-test molecule chunk count changed")
    predictions_root = output / "predictions"
    metrics_root = output / "metrics"
    predictions_root.mkdir(exist_ok=True)
    metrics_root.mkdir(exist_ok=True)
    validity = ChemicalValidity(ROOT / VALIDITY_PATH)
    models = static["models"]
    inference = dict(static["wrapper"]["inference"])
    inference["batch_size"] = args.batch_size
    safety = dict(inference["safety"])
    safety["objective_mode"] = SAFETY_OBJECTIVE_MODE
    safety_config = BACSafetyConfig(**safety)
    source_rows: list[dict[str, Any]] = []
    solver_summaries: list[dict[str, Any]] = []
    component_summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    prediction_started = time.monotonic()
    _seed(43)

    for chunk_index, rows in enumerate(chunk_rows, start=1):
        identity = _chunk_identity(rows)
        chunk_dir = predictions_root / f"chunk_{chunk_index:04d}"
        completed = (
            _completed_prediction(chunk_dir, chunk_index, identity)
            if args.resume
            else None
        )
        if completed is not None:
            source_rows.extend(completed["source_identity_rows"])
            if completed.get("angle_solver"):
                solver_summaries.append(completed["angle_solver"])
            if completed.get("components"):
                component_summaries.append(completed["components"])
            continue
        progress.update(
            {
                "phase": "PREDICTION",
                "current_chunk": chunk_index,
                "updated_at": _utcnow(),
            }
        )
        _atomic_json(output / "progress.json", progress)
        items: list[dict[str, Any]] = []
        chunk_source_rows: list[dict[str, Any]] = []
        for row in rows:
            sample_id = str(row["sample_id"])
            item, source_identity = _load_source_item(
                frozen["index"][sample_id], row, validity
            )
            items.append(item)
            chunk_source_rows.append(source_identity)
        attach_canonical_constraints(
            items, validity, source_identity_sha256=SOURCE_IDENTITY_SHA256
        )
        coordinates: dict[str, list[torch.Tensor]] = {}
        metadata: dict[str, list[dict[str, Any]]] = {}
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
            accepted, details = infer_bac(
                model,
                items,
                validity,
                device=static["device"],
                steps=int(inference["teacher_steps"]),
                step_size=float(inference["step_size"]),
                batch_size=args.batch_size,
                safety_config=safety_config,
                trajectory_semantics=TRAJECTORY_SEMANTICS,
            )
            coordinates[method] = [value.detach().cpu() for value in accepted]
            metadata[method] = details
            method_seconds[method] = time.monotonic() - method_started
            if method == "V7":
                solver = model.angle_solver_summary()
                components = model.component_summary()
        prediction = {
            "schema_version": "mcvr-v7-formal-test-predictions-v1",
            "sample_ids": [str(row["sample_id"]) for row in rows],
            "coordinates": coordinates,
            "metadata": metadata,
        }
        prediction_path = chunk_dir / "predictions.pt"
        _atomic_torch(prediction_path, prediction)
        summary = {
            "schema_version": "mcvr-v7-formal-test-prediction-chunk-v1",
            "status": "PREDICTIONS_COMPLETE",
            "chunk": chunk_index,
            "total_chunks": len(chunk_rows),
            "methods": list(METHODS),
            "records": len(rows),
            "molecules": 1,
            "sample_identity_sha256": identity,
            "source_identity_rows": chunk_source_rows,
            "prediction_sha256": file_sha256(prediction_path),
            "prediction_payload_fields": sorted(PREDICTION_FIELDS),
            "prediction_contains_references": False,
            "method_seconds": method_seconds,
            "angle_solver": solver,
            "components": components,
            "formal_test_authorized": True,
            "test_records_read": len(rows),
            "test_assets_opened": True,
            "references_opened": False,
            "training_performed": False,
        }
        _atomic_json(chunk_dir / "summary.json", summary)
        source_rows.extend(chunk_source_rows)
        if solver:
            solver_summaries.append(solver)
        if components:
            component_summaries.append(components)
        completed_predictions = sorted(
            set(progress.get("completed_prediction_chunks", [])) | {chunk_index}
        )
        progress.update(
            {
                "completed_prediction_chunks": completed_predictions,
                "current_chunk": None,
                "test_records_read": sum(
                    len(chunk_rows[index - 1]) for index in completed_predictions
                ),
                "test_record_open_events": sum(
                    len(chunk_rows[index - 1]) for index in completed_predictions
                ),
                "updated_at": _utcnow(),
            }
        )
        _atomic_json(output / "progress.json", progress)
        print(json.dumps(summary, sort_keys=True), flush=True)
        del items, coordinates, metadata, prediction
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if _canonical_sha(source_rows) != SOURCE_IDENTITY_SHA256:
        raise RuntimeError("formal-test source content identity changed")
    prediction_seconds = time.monotonic() - prediction_started
    progress.update(
        {
            "phase": "REFERENCE_METRICS",
            "predictions_complete": True,
            "current_chunk": None,
            "test_records_read": EXPECTED_RECORDS,
            "references_opened": True,
            "updated_at": _utcnow(),
        }
    )
    launch.update(
        {
            "test_records_read": EXPECTED_RECORDS,
            "references_opened_after_predictions": True,
        }
    )
    _atomic_json(output / "progress.json", progress)
    _atomic_json(output / "launch.json", launch)

    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reference_rows: list[dict[str, Any]] = []
    metric_started = time.monotonic()
    for chunk_index, rows in enumerate(chunk_rows, start=1):
        identity = _chunk_identity(rows)
        chunk_dir = metrics_root / f"chunk_{chunk_index:04d}"
        completed = (
            _completed_metrics(chunk_dir, chunk_index, identity)
            if args.resume
            else None
        )
        if completed is not None:
            reference_rows.extend(completed["reference_identity_rows"])
            continue
        progress.update(
            {
                "phase": "REFERENCE_METRICS",
                "current_chunk": chunk_index,
                "updated_at": _utcnow(),
            }
        )
        _atomic_json(output / "progress.json", progress)
        items = []
        chunk_reference_rows = []
        for row in rows:
            sample_id = str(row["sample_id"])
            item, reference_identity = _load_metric_item(
                frozen["index"][sample_id], row, validity
            )
            items.append(item)
            chunk_reference_rows.append(reference_identity)
        prediction_path = (
            predictions_root / f"chunk_{chunk_index:04d}" / "predictions.pt"
        )
        prediction = torch.load(
            prediction_path, map_location="cpu", weights_only=False
        )
        if set(prediction) != PREDICTION_FIELDS:
            raise RuntimeError(f"prediction payload fields changed: chunk {chunk_index}")
        expected_ids = [str(row["sample_id"]) for row in rows]
        if prediction["sample_ids"] != expected_ids:
            raise RuntimeError(f"prediction sample order changed: chunk {chunk_index}")
        method_coordinates: dict[str, Sequence[torch.Tensor]] = {
            "Source": [item["input"] for item in items],
            **prediction["coordinates"],
        }
        method_metadata = {
            "Source": [{"accepted": True} for _ in items],
            **prediction["metadata"],
        }
        records = method_rows(
            items,
            method_coordinates,
            validity,
            method_metadata=method_metadata,
        )
        _, molecules = summarize_groups(records, items, method_coordinates)
        for method in ALL_METHODS:
            selected = records.loc[records.method == method]
            if len(selected) != len(rows) or selected.sample_id.nunique() != len(rows):
                raise RuntimeError(
                    f"formal-test metric pairing failed: {method} chunk {chunk_index}"
                )
        record_path = chunk_dir / "records.csv"
        molecule_path = chunk_dir / "molecules.csv"
        _atomic_csv(record_path, records)
        _atomic_csv(molecule_path, molecules)
        summary = {
            "schema_version": "mcvr-v7-formal-test-metric-chunk-v1",
            "status": "METRICS_COMPLETE",
            "chunk": chunk_index,
            "total_chunks": len(chunk_rows),
            "methods": list(ALL_METHODS),
            "records": len(rows),
            "molecules": 1,
            "sample_identity_sha256": identity,
            "reference_identity_rows": chunk_reference_rows,
            "references_opened_after_all_predictions": True,
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
            "formal_test_authorized": True,
            "test_records_read": len(rows),
            "test_assets_opened": True,
            "training_performed": False,
        }
        _atomic_json(chunk_dir / "summary.json", summary)
        reference_rows.extend(chunk_reference_rows)
        completed_metrics = sorted(
            set(progress.get("completed_metric_chunks", [])) | {chunk_index}
        )
        progress.update(
            {
                "completed_metric_chunks": completed_metrics,
                "current_chunk": None,
                "test_record_open_events": EXPECTED_RECORDS
                + sum(len(chunk_rows[index - 1]) for index in completed_metrics),
                "updated_at": _utcnow(),
            }
        )
        _atomic_json(output / "progress.json", progress)
        print(json.dumps(summary, sort_keys=True), flush=True)
        del items, prediction, method_coordinates, method_metadata, records, molecules
        gc.collect()

    if _canonical_sha(reference_rows) != REFERENCE_IDENTITY_SHA256:
        raise RuntimeError("formal-test reference content identity changed")
    record_paths = [
        metrics_root / f"chunk_{index:04d}" / "records.csv"
        for index in range(1, len(chunk_rows) + 1)
    ]
    molecule_paths = [
        metrics_root / f"chunk_{index:04d}" / "molecules.csv"
        for index in range(1, len(chunk_rows) + 1)
    ]
    records = pd.concat([pd.read_csv(path) for path in record_paths], ignore_index=True)
    molecules = pd.concat(
        [pd.read_csv(path) for path in molecule_paths], ignore_index=True
    )
    for method in ALL_METHODS:
        selected = records.loc[records.method == method]
        if (
            len(selected) != EXPECTED_RECORDS
            or selected.sample_id.nunique() != EXPECTED_RECORDS
        ):
            raise RuntimeError(f"final formal-test record identity failed: {method}")
        molecule_selected = molecules.loc[
            (molecules.method == method) & (molecules.group == "all")
        ]
        if (
            len(molecule_selected) != EXPECTED_MOLECULES
            or molecule_selected.molecule_id.nunique() != EXPECTED_MOLECULES
        ):
            raise RuntimeError(f"final formal-test molecule identity failed: {method}")
    _atomic_csv(output / "formal_test_per_record.csv", records)
    _atomic_csv(output / "formal_test_per_molecule.csv", molecules)
    angle_solver = _merge_solver(solver_summaries)
    components = _merge_components(component_summaries)
    _atomic_json(output / "angle_solver_summary.json", angle_solver)
    _atomic_json(output / "component_summary.json", components)
    report = build_report(
        output,
        seed=43,
        bootstrap_draws=BOOTSTRAP_DRAWS,
        expected_records=EXPECTED_RECORDS,
        expected_molecules=EXPECTED_MOLECULES,
    )
    metric_seconds = time.monotonic() - metric_started
    runtime = time.monotonic() - started
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "formal_test_authorized": True,
        "one_time_evaluation": True,
        "seed": 43,
        "methods": list(ALL_METHODS),
        "records": EXPECTED_RECORDS,
        "molecules": EXPECTED_MOLECULES,
        "chunks": len(chunk_rows),
        "prediction_seconds": prediction_seconds,
        "reference_metric_seconds": metric_seconds,
        "evaluation_seconds": runtime,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "training_config_sha256": TRAINING_CONFIG_SHA256,
        "v5_config_sha256": V5_CONFIG_SHA256,
        "v7_config_sha256": V7_CONFIG_SHA256,
        "binding_sha256": BINDING_SHA256,
        "plan_sha256": PLAN_SHA256,
        "evaluator_semantics_version": EVALUATOR_SEMANTICS,
        "evaluator_semantics_git_commit": SEMANTICS_COMMIT,
        "evaluator_git_head": static["evaluator_git_head"],
        "evaluator_worktree_dirty": static["evaluator_worktree_dirty"],
        "runner_sha256": file_sha256(Path(__file__)),
        "reporter_sha256": file_sha256(
            ROOT / "scripts/report_ecir_mvr_v7_formal_test.py"
        ),
        "trajectory_semantics": TRAJECTORY_SEMANTICS,
        "safety_objective_mode": SAFETY_OBJECTIVE_MODE,
        "manifest_sha256": MANIFEST_SHA256,
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "source_identity_sha256": SOURCE_IDENTITY_SHA256,
        "reference_identity_sha256": REFERENCE_IDENTITY_SHA256,
        "ordered_sample_ids_sha256": frozen["ordered_sample_ids_sha256"],
        "reference_policy": "post_prediction_metrics_only",
        "predictions_complete_before_reference_access": True,
        "prediction_payload_contains_references": False,
        "test_records_read": EXPECTED_RECORDS,
        "test_record_open_events": EXPECTED_RECORDS * 2,
        "test_assets_opened": True,
        "minimal_validity_target_test_used": False,
        "training_performed": False,
        "checkpoint_or_config_selected_from_test": False,
        "cohort_selection_from_test_metrics": False,
        "parameter_tuning_from_test": False,
        "method_selection_from_test": False,
        "angle_solver": angle_solver,
        "components": components,
        "same_sample_identity": bool(report["same_sample_identity"]),
    }
    _atomic_json(output / "run_metadata.json", metadata)
    _atomic_json(
        output / "test_identity.json",
        {
            "schema_version": "mcvr-v7-formal-test-identity-v1",
            "manifest": str((ROOT / MANIFEST_PATH).resolve()),
            "manifest_sha256": MANIFEST_SHA256,
            "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
            "cache_root": str(frozen["cache_root"]),
            "source_identity_sha256": SOURCE_IDENTITY_SHA256,
            "reference_identity_sha256": REFERENCE_IDENTITY_SHA256,
            "ordered_sample_ids_sha256": frozen["ordered_sample_ids_sha256"],
            "records": EXPECTED_RECORDS,
            "molecules": EXPECTED_MOLECULES,
            "test_records_read": EXPECTED_RECORDS,
            "test_assets_opened": True,
        },
    )
    progress.update(
        {
            "status": "COMPLETED",
            "phase": "COMPLETED",
            "current_chunk": None,
            "completed_prediction_chunks": list(range(1, len(chunk_rows) + 1)),
            "completed_metric_chunks": list(range(1, len(chunk_rows) + 1)),
            "predictions_complete": True,
            "references_opened": True,
            "test_records_read": EXPECTED_RECORDS,
            "test_record_open_events": EXPECTED_RECORDS * 2,
            "updated_at": _utcnow(),
        }
    )
    _atomic_json(output / "progress.json", progress)
    launch.update(
        {
            "status": "COMPLETED",
            "completed_at": _utcnow(),
            "test_records_read": EXPECTED_RECORDS,
            "test_assets_opened": True,
            "references_opened_after_predictions": True,
        }
    )
    _atomic_json(output / "launch.json", launch)
    _write_checksums(output)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(_dry_run(args), indent=2, sort_keys=True))
        return 0
    result = _evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
