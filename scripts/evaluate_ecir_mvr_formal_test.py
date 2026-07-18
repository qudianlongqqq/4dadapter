#!/usr/bin/env python
"""Frozen, one-time D1-B evaluation on the independent formal-large test split.

The real evaluation path is deliberately locked behind a two-seed test plan.
``--dry-run`` never opens, stats, hashes, or enumerates a test asset.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import pandas as pd
import torch
import yaml

from etflow.commons.record_identity import source_record_identity
from etflow.data.flexbond_cache_schema import (
    INFERENCE_FORBIDDEN_FIELDS,
    validate_inference_record,
)
from etflow.data.flexbond_eval_manifest import load_eval_manifest, manifest_content_sha256
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.geometry import bond_lengths, unique_bonds
from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir.run_a_evaluation import (
    graph_data,
    infer_mvr,
    method_rows,
    nearest_rmsd,
    paired_bootstrap,
    summarize_groups,
)
from etflow.ecir.formal_target_assets import tensor_sha256


SCHEMA_VERSION = "d1b-formal-frozen-test-evaluator-v1"
PLAN_SCHEMA_VERSION = "d1b-formal-dual-seed-test-plan-v1"
LOCKED_PLAN_STATUS = "D1B_FORMAL_DUAL_SEED_TEST_PLAN_LOCKED"
DRY_RUN_STATUS = "D1B_FORMAL_TEST_DRY_RUN"
COMPLETE_STATUS = "D1B_FORMAL_TEST_COMPLETE"

SEED42_CHECKPOINT = Path(
    "logs_ecir_mvr/formal_large/d1_b_seed42/checkpoints/"
    "best_noninferior_validity.ckpt"
)
SEED42_CHECKPOINT_SHA256 = (
    "721b4384f3a64eef48ead2fc2b4ea35bf83802b84952e8e3f3aa6c5172e33a2f"
)
TRAINING_COMMIT = "a42adace708df60f82980fc51b10cecc5dcde86f"
SELECTED_STEP = 25_000
FROZEN_INFERENCE = {
    "teacher_steps": 4,
    "step_size": 0.25,
    "t_min": 0.0,
    "t_max": 1.0,
    "acceptance_mode": "best_of_trajectory",
}
BROKEN_BOND_LENGTH_ANGSTROM = 2.5


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen D1-B formal-large test evaluator"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize-frozen-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=SEED42_CHECKPOINT)
    parser.add_argument(
        "--checkpoint-sha256", default=SEED42_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("reports/ecir_mvr/D1B_FORMAL_RECOMMENDED_CONFIG.yaml"),
    )
    parser.add_argument(
        "--frozen-test-plan",
        type=Path,
        default=Path("reports/ecir_mvr/D1B_FORMAL_DUAL_SEED_TEST_PLAN.json"),
    )
    parser.add_argument(
        "--test-manifest", type=Path, default=Path("manifests/formal_large_test.json")
    )
    parser.add_argument(
        "--test-cache-root", type=Path, default=Path("data/flexbond_cache_formal_large")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/formal_test/seed42"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    return parser


def _dry_run(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorize_frozen_test:
        raise ValueError("--dry-run and --authorize-frozen-test are mutually exclusive")
    if args.seed != 42:
        raise ValueError("seed43 cannot be planned until its checkpoint is frozen")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": DRY_RUN_STATUS,
        "dry_run": True,
        "test_records_read": 0,
        "test_assets_opened": False,
        "output_files_created": False,
        "formal_test_started": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "selected_step": SELECTED_STEP,
        "training_git_commit": TRAINING_COMMIT,
        "config": str(args.config),
        "frozen_test_plan": str(args.frozen_test_plan),
        "test_manifest": str(args.test_manifest),
        "test_cache_root": str(args.test_cache_root),
        "output_dir": str(args.output_dir),
        "inference": dict(FROZEN_INFERENCE),
        "model_entrypoint": "etflow.ecir.mvr_model.MCVRModel",
        "reused_inference": "etflow.ecir.run_a_evaluation.infer_mvr",
        "reference_policy": "metrics_only",
        "minimal_validity_target_test_required": False,
        "checkpoint_selection_from_test": False,
        "legacy_evaluator_audit": {
            "run_formal_large_final_test.sh": "INCOMPATIBLE_CARTESIAN_GLOBAL4D",
            "sample_formal_large_cartesian.py": "INCOMPATIBLE_FLEXBOND_OPTIMIZER_MODEL",
            "evaluate_ecir_mvr_star": "VALIDATION_ONLY_COMPONENTS_REUSED",
        },
        "authorization_required": {
            "flag": "--authorize-frozen-test",
            "plan_schema_version": PLAN_SCHEMA_VERSION,
            "plan_status": LOCKED_PLAN_STATUS,
            "seed42_and_seed43_checkpoints_frozen": True,
            "test_manifest_sha256_frozen": True,
            "test_source_identity_sha256_frozen": True,
            "test_reference_identity_sha256_frozen": True,
        },
    }


def _load_locked_plan(args: argparse.Namespace) -> dict[str, Any]:
    if not args.authorize_frozen_test:
        raise RuntimeError(
            "formal test access requires explicit --authorize-frozen-test"
        )
    plan = json.loads(args.frozen_test_plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise RuntimeError("frozen dual-seed test plan schema mismatch")
    if plan.get("status") != LOCKED_PLAN_STATUS:
        raise RuntimeError("dual-seed test plan is not locked")
    seeds = plan.get("checkpoints")
    if not isinstance(seeds, list) or {int(row.get("seed", -1)) for row in seeds} != {
        42,
        43,
    }:
        raise RuntimeError("both seed42 and seed43 checkpoints must be frozen")
    selected = next((row for row in seeds if int(row.get("seed", -1)) == args.seed), None)
    if selected is None:
        raise RuntimeError(f"seed {args.seed} is absent from the locked test plan")
    if str(selected["checkpoint_sha256"]) != args.checkpoint_sha256:
        raise RuntimeError("requested checkpoint differs from the locked test plan")
    test = plan.get("test")
    required = {
        "manifest_sha256",
        "manifest_content_sha256",
        "source_identity_sha256",
        "reference_identity_sha256",
    }
    if not isinstance(test, Mapping) or required.difference(test):
        raise RuntimeError("locked test plan is missing test identities")
    # Paths are audit context only. The evaluator supports relocating frozen
    # assets across Windows/Linux and proves identity from the hashes below.
    if bool(plan.get("checkpoint_or_config_selected_from_test", True)):
        raise RuntimeError("locked test plan permits test-driven selection")
    return plan


def _validate_inference_config(config: Mapping[str, Any]) -> None:
    inference = config.get("inference")
    if not isinstance(inference, Mapping):
        raise RuntimeError("checkpoint configuration has no inference block")
    for key, expected in FROZEN_INFERENCE.items():
        if inference.get(key) != expected:
            raise RuntimeError(f"frozen inference setting changed: {key}")
    if int(config.get("training", {}).get("teacher_steps", -1)) != 4:
        raise RuntimeError("training teacher_steps differs from frozen inference")


def _load_checkpoint_and_model(
    args: argparse.Namespace, plan: Mapping[str, Any], device: torch.device
) -> tuple[dict[str, Any], dict[str, Any], MCVRModel, dict[str, Any]]:
    if _file_sha256(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("D1-B checkpoint SHA256 mismatch")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_type") != "MCVRModel":
        raise RuntimeError("checkpoint is not an MCVRModel")
    if payload.get("schema_version") != (
        "ecir-mvr-medium-rescue-formal-large-d1b-checkpoint-v1"
    ):
        raise RuntimeError("checkpoint is not a formal-large D1-B checkpoint")
    if int(payload.get("step", -1)) != SELECTED_STEP:
        raise RuntimeError("checkpoint is not the frozen selected step")
    config_sha = _file_sha256(args.config)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    resolved = payload.get("config")
    if not isinstance(resolved, Mapping):
        raise RuntimeError("checkpoint has no resolved configuration")
    provenance = resolved.get("resolved")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("checkpoint has no resolved provenance")
    if provenance.get("config_sha256") != config_sha:
        raise RuntimeError("checkpoint/config SHA256 mismatch")
    if provenance.get("git_commit") != TRAINING_COMMIT:
        raise RuntimeError("checkpoint training git commit mismatch")
    if int(config.get("seed", -1)) != args.seed:
        raise RuntimeError("configuration seed mismatch")
    for section in ("model", "loss", "inference", "frozen_identities"):
        if config.get(section) != resolved.get(section):
            raise RuntimeError(f"checkpoint/config {section} mismatch")
    _validate_inference_config(config)
    frozen = config.get("frozen_identities")
    if not isinstance(frozen, Mapping) or payload.get("frozen_identities") != frozen:
        raise RuntimeError("checkpoint formal asset identities mismatch")
    source_metadata_path = Path(config["data"]["source_metadata"])
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    source_identity = source_metadata.get("formal_source_identity_sha256")
    if source_identity != frozen.get("formal_source_identity_sha256"):
        raise RuntimeError("formal training source identity changed")
    planned = next(
        row for row in plan["checkpoints"] if int(row.get("seed", -1)) == args.seed
    )
    if planned.get("config_sha256") and planned.get("config_sha256") != config_sha:
        raise RuntimeError("configuration differs from the locked test plan")
    model = MCVRModel(**config["model"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    identity = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_step": int(payload["step"]),
        "config": str(args.config.resolve()),
        "config_sha256": config_sha,
        "training_git_commit": TRAINING_COMMIT,
        "evaluator_git_commit": _git("rev-parse", "HEAD"),
        "evaluator_worktree_dirty": bool(_git("status", "--short")),
        "formal_source_identity_sha256": source_identity,
        "frozen_identities": dict(frozen),
    }
    return payload, config, model, identity


def _inference_record(raw_record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in raw_record.items()
        if key not in INFERENCE_FORBIDDEN_FIELDS
        and not key.startswith("x_ref")
        and not key.startswith("selected_ref")
        and key not in {"x_target", "target_metadata", "rmsd_before", "rmsd_after"}
    }


def _metric_references(
    raw_record: Mapping[str, Any], checked: Mapping[str, Any]
) -> torch.Tensor:
    references = torch.as_tensor(raw_record["x_ref_candidates"], dtype=torch.float32)
    if references.ndim == 2:
        references = references.unsqueeze(0)
    atom_count = int(checked["atomic_numbers"].numel())
    if references.ndim != 3 or tuple(references.shape[1:]) != (atom_count, 3):
        raise ValueError("test reference candidates have an invalid shape")
    if not bool(torch.isfinite(references).all()):
        raise ValueError("test reference candidates contain NaN or Inf")
    reference_numbers = torch.as_tensor(
        raw_record["x_ref_atomic_numbers"], dtype=torch.long
    ).view(-1)
    if not torch.equal(reference_numbers, checked["atomic_numbers"]):
        raise ValueError("test reference atomic-number order differs from source")
    if raw_record.get("x_ref_topology_signature") != raw_record.get(
        "topology_signature"
    ):
        raise ValueError("test reference topology differs from source")
    return references


def _validate_manifest_identity(
    raw_sha256: str,
    content_sha256: str,
    test_plan: Mapping[str, Any],
) -> bool:
    if content_sha256 != test_plan["manifest_content_sha256"]:
        raise RuntimeError("frozen test manifest content SHA256 mismatch")
    return raw_sha256 == test_plan["manifest_sha256"]


def _test_item(
    raw_record: Mapping[str, Any],
    record: Mapping[str, Any],
    checked: Mapping[str, Any],
    references: torch.Tensor,
    manifest_row: Mapping[str, Any],
    validity,
) -> dict[str, Any]:
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
    metadata = dict(raw_record.get("metadata") or {})
    row = SimpleNamespace(
        molecule_id=str(manifest_row["mol_id"]),
        sample_id=str(manifest_row["sample_id"]),
        generator_name=str(raw_record.get("generator_name", "ETFlow_formal_upstream")),
        source_severity=str(raw_record.get("source_severity", "normal")),
        update_scale=float(metadata.get("update_scale", 0.0) or 0.0),
        NFE=int(metadata.get("NFE", metadata.get("nfe", 0)) or 0),
        seed=int(raw_record.get("sample_seed", metadata.get("seed", 0)) or 0),
    )
    groups = [
        "all",
        "rotatable_le_2"
        if rotatable <= 2
        else "rotatable_3_5"
        if rotatable <= 5
        else "rotatable_ge_6",
        "ring" if has_ring else "non_ring",
    ]
    if clean:
        groups.append("clean_valid")
    return {
        "row": row,
        "record": record,
        "input": coordinates,
        "references": references,
        "input_validity": input_validity,
        "input_rmsd": nearest_rmsd(coordinates, references),
        "data": graph_data(record, coordinates, row, active_mode_mask=active),
        "groups": groups,
        "rotatable": rotatable,
        "has_ring": has_ring,
        "clean": clean,
        "atomic_numbers": torch.as_tensor(checked["atomic_numbers"], dtype=torch.long),
        "edge_index": torch.as_tensor(checked["edge_index"], dtype=torch.long),
    }


def _load_test_items(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    validity: ChemicalValidity,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    test_plan = plan["test"]
    raw_manifest_sha = _file_sha256(args.test_manifest)
    manifest = load_eval_manifest(args.test_manifest)
    content_sha = manifest_content_sha256(manifest)
    raw_manifest_sha_match = _validate_manifest_identity(
        raw_manifest_sha, content_sha, test_plan
    )
    if manifest.get("formal_large_split") not in {None, "test"}:
        raise RuntimeError("frozen manifest is not the test split")

    root = args.test_cache_root.expanduser()
    if (root / "test").is_dir():
        root = root / "test"
    paths = sorted(root.glob("*.pt"))
    if not paths:
        raise RuntimeError(f"frozen test cache has no PT records: {root}")
    loaded: dict[
        str, tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any], torch.Tensor]
    ] = {}
    failures: list[dict[str, Any]] = []
    for path in paths:
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(raw, Mapping):
                raise TypeError("test source payload is not a mapping")
            record = _inference_record(raw)
            checked = validate_inference_record(record)
            references = _metric_references(raw, checked)
            sample_id = str(raw.get("sample_id", raw["mol_id"]))
            if sample_id in loaded:
                raise ValueError(f"duplicate test sample_id: {sample_id}")
            loaded[sample_id] = (raw, record, checked, references)
        except Exception as error:  # retain every source-load failure
            failures.append(
                {
                    "phase": "source_load",
                    "path": str(path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    rows = manifest["records"]
    expected_ids = [str(row["sample_id"]) for row in rows]
    missing = [sample_id for sample_id in expected_ids if sample_id not in loaded]
    unexpected = sorted(set(loaded).difference(expected_ids))
    if missing or unexpected or failures:
        raise RuntimeError(
            "frozen test cache/manifest mismatch: "
            f"missing={missing[:20]} unexpected={unexpected[:20]} "
            f"load_failures={len(failures)}"
        )

    source_rows = []
    reference_rows = []
    items = []
    for row in rows:
        sample_id = str(row["sample_id"])
        raw, record, checked, references = loaded[sample_id]
        rotatable_count = int(checked["rotatable_bond_index"].size(1))
        actual = {
            "mol_id": source_record_identity(raw),
            "sample_id": sample_id,
            "x_init_hash": str(checked["x_init_hash"]),
            "num_rotatable_bonds": rotatable_count,
        }
        expected = {
            "mol_id": str(row["mol_id"]),
            "sample_id": sample_id,
            "x_init_hash": str(row["x_init_hash"]),
            "num_rotatable_bonds": int(row["num_rotatable_bonds"]),
        }
        if actual != expected:
            raise RuntimeError(f"test manifest mismatch for {sample_id}")
        source_rows.append(
            {
                **expected,
                "atomic_numbers_sha256": tensor_sha256(checked["atomic_numbers"]),
                "topology_signature": str(raw.get("topology_signature", "")),
            }
        )
        reference_rows.append(
            {"sample_id": sample_id, "references_sha256": tensor_sha256(references)}
        )
        items.append(
            _test_item(raw, record, checked, references, row, validity)
        )
    source_identity = _canonical_sha256(source_rows)
    reference_identity = _canonical_sha256(reference_rows)
    if source_identity != test_plan["source_identity_sha256"]:
        raise RuntimeError("frozen test source identity mismatch")
    if reference_identity != test_plan["reference_identity_sha256"]:
        raise RuntimeError("frozen test reference identity mismatch")
    identity = {
        "manifest": str(args.test_manifest.resolve()),
        "manifest_sha256": raw_manifest_sha,
        "planned_manifest_sha256": test_plan["manifest_sha256"],
        "manifest_file_sha256_match": raw_manifest_sha_match,
        "manifest_content_sha256": content_sha,
        "test_cache_root": str(args.test_cache_root.resolve()),
        "source_identity_sha256": source_identity,
        "reference_identity_sha256": reference_identity,
        "ordered_sample_ids_sha256": _canonical_sha256(expected_ids),
        "test_records_planned": len(rows),
        "test_records_read": len(loaded),
        "test_molecules": len({str(row["mol_id"]) for row in rows}),
    }
    return items, identity, failures


def _run_inference(
    model: MCVRModel,
    items: Sequence[dict[str, Any]],
    validity: ChemicalValidity,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[dict[str, Any]]]:
    successful_items: list[dict[str, Any]] = []
    accepted: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for start in range(0, len(items), batch_size):
        chunk = list(items[start : start + batch_size])
        try:
            _, chunk_accepted, chunk_metadata = infer_mvr(
                model,
                chunk,
                validity,
                device=device,
                steps=4,
                step_size=0.25,
                batch_size=batch_size,
                acceptance_mode="best_of_trajectory",
            )
            successful_items.extend(chunk)
            accepted.extend(chunk_accepted)
            metadata.extend(chunk_metadata)
        except Exception:
            for item in chunk:
                try:
                    _, one_accepted, one_metadata = infer_mvr(
                        model,
                        [item],
                        validity,
                        device=device,
                        steps=4,
                        step_size=0.25,
                        batch_size=1,
                        acceptance_mode="best_of_trajectory",
                    )
                    successful_items.append(item)
                    accepted.extend(one_accepted)
                    metadata.extend(one_metadata)
                except Exception as error:
                    failures.append(
                        {
                            "phase": "inference",
                            "sample_id": str(item["row"].sample_id),
                            "molecule_id": str(item["row"].molecule_id),
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
    return successful_items, accepted, metadata, failures


def _new_broken_bonds(item: Mapping[str, Any], candidate: torch.Tensor) -> int:
    bonds = unique_bonds(item["edge_index"])
    baseline = bond_lengths(item["input"], bonds)
    refined = bond_lengths(candidate, bonds)
    return int(
        (
            (baseline <= BROKEN_BOND_LENGTH_ANGSTROM)
            & (refined > BROKEN_BOND_LENGTH_ANGSTROM)
        ).sum()
    )


def _enrich_rows(
    frame: pd.DataFrame,
    items: Sequence[dict[str, Any]],
    methods: Mapping[str, Sequence[torch.Tensor]],
) -> pd.DataFrame:
    item_by_id = {str(item["row"].sample_id): item for item in items}
    coordinate_by_method = {
        method: {
            str(item["row"].sample_id): coordinate
            for item, coordinate in zip(items, coordinates, strict=True)
        }
        for method, coordinates in methods.items()
    }
    enriched = frame.copy()
    enriched["atom_identity_failure"] = False
    enriched["bond_identity_failure"] = False
    enriched["chirality_identity_failure"] = enriched["chirality_error"] > 0.0
    enriched["new_clash"] = enriched["delta_severe_clash_rate"] > 0.0
    enriched["new_broken_bond_count"] = [
        _new_broken_bonds(
            item_by_id[str(row.sample_id)],
            coordinate_by_method[str(row.method)][str(row.sample_id)],
        )
        for row in enriched.itertuples(index=False)
    ]
    enriched["status"] = "success"
    return enriched


def _failure_rate_ci(failures: int, total: int, draws: int, seed: int) -> dict[str, float]:
    if total == 0:
        return {"mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    values = np.concatenate([np.ones(failures), np.zeros(total - failures)])
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [rng.choice(values, size=total, replace=True).mean() for _ in range(draws)]
    )
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def _extra_paired_bootstrap(
    molecule_frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    metrics = (
        "mean_displacement",
        "molecule_rms_displacement",
        "max_displacement",
        "coordinate_unchanged",
        "accepted",
        "chirality_identity_failure",
        "atom_identity_failure",
        "bond_identity_failure",
        "new_clash",
        "new_broken_bond_count",
        "high_flex_torsion_change",
    )
    all_frame = molecule_frame[molecule_frame.group == "all"]
    result = {}
    for offset, metric in enumerate(metrics):
        if metric not in all_frame:
            continue
        pivot = all_frame.pivot(
            index="molecule_id", columns="method", values=metric
        ).dropna()
        if candidate not in pivot or baseline not in pivot:
            continue
        delta = pivot[candidate].to_numpy() - pivot[baseline].to_numpy()
        if not delta.size:
            continue
        rng = np.random.default_rng(seed + offset)
        means = np.asarray(
            [rng.choice(delta, size=delta.size, replace=True).mean() for _ in range(draws)]
        )
        result[metric] = {
            "mean": float(delta.mean()),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
        }
    return result


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    refined = summary["metrics"]["d1b_refined"]
    baseline = summary["metrics"]["source_baseline"]
    lines = [
        "# D1-B Formal-Large Frozen Test",
        "",
        f"Status: `{summary['status']}`",
        f"Test records read: `{summary['test_records_read']}`",
        f"Failures: `{summary['failure_count']}` ({summary['failure_rate']:.6f})",
        "",
        "| Metric | Source baseline | D1-B refined |",
        "|---|---:|---:|",
    ]
    for metric in (
        "aligned_RMSD",
        "MAT_P",
        "MAT_R",
        "COV_P",
        "COV_R",
        "bond_outlier_rate",
        "bond_outlier_magnitude",
        "angle_outlier_rate",
        "ring_bond_outlier_rate",
        "total_thresholded_validity_score",
        "molecule_rms_displacement",
        "accepted_fraction",
    ):
        lines.append(
            f"| {metric} | {baseline.get(metric, math.nan):.8g} | "
            f"{refined.get(metric, math.nan):.8g} |"
        )
    lines.extend(
        [
            "",
            "References were used only for final RMSD/MAT/COV metrics. The test "
            "results were not used to select a checkpoint, threshold, step, or configuration.",
            "",
        ]
    )
    return "\n".join(lines)


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed not in {42, 43}:
        raise ValueError("only frozen seed42 or seed43 evaluation is supported")
    if args.batch_size < 1 or args.bootstrap_draws < 1:
        raise ValueError("batch size and bootstrap draws must be positive")
    plan = _load_locked_plan(args)
    device = torch.device(args.device)
    payload, config, model, checkpoint_identity = _load_checkpoint_and_model(
        args, plan, device
    )
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    expected_validity = config["frozen_identities"][
        "validity_statistics_identity_sha256"
    ]
    if validity.statistics.get("identity_sha256") != expected_validity:
        raise RuntimeError("validity statistics identity changed")

    # This is the first operation allowed to touch test assets.
    items, test_identity, source_failures = _load_test_items(args, plan, validity)
    successful, accepted, metadata, inference_failures = _run_inference(
        model, items, validity, device=device, batch_size=args.batch_size
    )
    failures = [*source_failures, *inference_failures]
    methods = {
        "source_baseline": [item["input"] for item in successful],
        "d1b_refined": accepted,
    }
    record_rows = method_rows(
        successful,
        methods,
        validity,
        method_metadata={"d1b_refined": metadata},
    )
    record_rows = _enrich_rows(record_rows, successful, methods)
    group_summary, molecule_rows = summarize_groups(record_rows, successful, methods)
    bootstrap = paired_bootstrap(
        molecule_rows,
        candidate="d1b_refined",
        baseline="source_baseline",
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    bootstrap.update(
        _extra_paired_bootstrap(
            molecule_rows,
            candidate="d1b_refined",
            baseline="source_baseline",
            draws=args.bootstrap_draws,
            seed=args.seed,
        )
    )
    all_rows = group_summary[group_summary.group == "all"].set_index("method")
    metrics = {
        method: {
            key: float(value)
            for key, value in all_rows.loc[method].to_dict().items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
        for method in ("source_baseline", "d1b_refined")
    }
    refined_records = record_rows[record_rows.method == "d1b_refined"]
    clean_records = refined_records[refined_records.clean_valid]
    high_flex = group_summary[
        (group_summary.group == "rotatable_ge_6")
        & (group_summary.method == "d1b_refined")
    ]
    failure_count = len(failures)
    total = int(test_identity["test_records_planned"])
    failure_rows = pd.DataFrame(
        [
            {
                "method": "d1b_refined",
                "molecule_id": failure.get("molecule_id", ""),
                "sample_id": failure.get("sample_id", ""),
                "status": "failed",
                "error_type": failure.get("error_type", ""),
                "error": failure.get("error", ""),
            }
            for failure in failures
        ]
    )
    output_record_rows = (
        pd.concat([record_rows, failure_rows], ignore_index=True, sort=False)
        if not failure_rows.empty
        else record_rows
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": COMPLETE_STATUS,
        "seed": args.seed,
        "selected_step": int(payload["step"]),
        "test_records_read": int(test_identity["test_records_read"]),
        "test_records_evaluated": len(successful),
        "failure_count": failure_count,
        "failure_rate": failure_count / total if total else math.nan,
        "failure_rate_bootstrap_95_ci": _failure_rate_ci(
            failure_count, total, args.bootstrap_draws, args.seed
        ),
        "metrics": metrics,
        "bootstrap_delta_refined_minus_source_95_ci": bootstrap,
        "clean_identity_fraction": (
            float(clean_records.coordinate_unchanged.mean())
            if not clean_records.empty
            else math.nan
        ),
        "atom_identity_failures": int(refined_records.atom_identity_failure.sum()),
        "bond_identity_failures": int(refined_records.bond_identity_failure.sum()),
        "chirality_identity_failures": int(
            refined_records.chirality_identity_failure.sum()
        ),
        "new_clash_records": int(refined_records.new_clash.sum()),
        "new_broken_bonds": int(refined_records.new_broken_bond_count.sum()),
        "acceptance_fraction": (
            float(refined_records.accepted.mean()) if not refined_records.empty else math.nan
        ),
        "high_flex_subset": _jsonable(
            high_flex.iloc[0].to_dict() if not high_flex.empty else {"records": 0}
        ),
        "inference": dict(FROZEN_INFERENCE),
        "metric_thresholds": {
            "new_broken_bond_length_angstrom": BROKEN_BOND_LENGTH_ANGSTROM,
            "cov_rmsd_angstrom": 1.25,
        },
        "reference_policy": "metrics_only",
        "minimal_validity_target_test_used": False,
        "checkpoint_or_config_selected_from_test": False,
        "checkpoint_identity": checkpoint_identity,
        "test_identity": test_identity,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_csv(output_record_rows, args.output_dir / "per_record.csv")
    _atomic_csv(molecule_rows, args.output_dir / "per_molecule.csv")
    _atomic_json(summary, args.output_dir / "summary.json")
    (args.output_dir / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            **checkpoint_identity,
            **test_identity,
            "inference": dict(FROZEN_INFERENCE),
        },
        args.output_dir / "test_identity.json",
    )
    _atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "failure_count": failure_count,
            "failures": failures,
        },
        args.output_dir / "failures.json",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(_dry_run(args), indent=2))
        return 0
    result = _evaluate(args)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
