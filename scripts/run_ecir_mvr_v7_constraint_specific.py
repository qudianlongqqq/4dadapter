#!/usr/bin/env python3
"""Run the frozen inference-only V7 development candidate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.bac_evaluation import evaluate_bac_candidate  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_v2_bac import MCVRBACModel  # noqa: E402
from etflow.ecir.mvr_v7_constraint_specific import (  # noqa: E402
    MCVRConstraintSpecificHybrid,
)
from scripts.run_ecir_mvr_v2_bac_pilots import (  # noqa: E402
    _canonical_sha,
    _seed,
    _sha,
    _write_json,
)
from scripts.run_ecir_mvr_v5_constraint_hybrid import (  # noqa: E402
    DEVELOPMENT_IDENTITY,
    D1_DEVELOPMENT_SHA256,
    _development_items,
    _write_evaluation,
)


SEED = 43018


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument(
        "--development-manifests",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_recovery/manifests"),
    )
    parser.add_argument(
        "--d1-checkpoint",
        type=Path,
        default=Path(
            "diagnostics/ecir_mvr/v2_bac_recovery/runs/"
            "d1_pilot_1000step_seed43018/checkpoint_final.ckpt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v7_constraint_specific/runs"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--validation-record-limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.smoke and args.validation_record_limit != 128:
        raise ValueError("V7 smoke requires --validation-record-limit 128")
    if not args.smoke and args.validation_record_limit is not None:
        raise ValueError("V7 full development run must use all 1024 records")
    if args.validation_batch_size < 1:
        raise ValueError("validation batch size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested V7 CUDA device is unavailable")


def _config(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "mcvr-v7-constraint-specific-experiment-v1",
        "experiment_id": (
            f"v7_constraint_specific_{'smoke' if args.smoke else 'pilot'}_seed{SEED}"
        ),
        "seed": SEED,
        "prior_model": checkpoint["config"]["model"],
        "v7": {
            "integration_step_size": 0.25,
            "angle_max_graph_rms": 0.01,
            "angle_max_atom": 0.02,
            "clash_max_graph_rms": 0.01,
            "clash_max_atom": 0.02,
            "clash_cutoff": 2.0,
            "clash_allowed_contact": 1.0,
            "clash_exclude_topology_distance": 2,
            "max_clash_edges_per_graph": 128,
            "jacobian_config": {
                "damping_lambda": 1.0e-3,
                "rank_tol": 1.0e-6,
                "max_condition_number": 1.0e8,
                "near_linear_sine_threshold": 1.0e-3,
                "near_linear_weight": 0.1,
            },
        },
        "inference": {
            "teacher_steps": 4,
            "step_size": 0.25,
            "batch_size": int(args.validation_batch_size),
            "safety": {
                "max_atom_displacement": 0.12,
                "max_molecule_rms_displacement": 0.06,
                "epsilon_bond": 0.0,
                "epsilon_angle": 0.0,
                "epsilon_clash": 0.0,
                "epsilon_ring": 0.0,
                "minimum_bac_gain": 1.0e-8,
                "backtracking_scales": [1.0, 0.5, 0.25],
                "enable_backtracking": True,
            },
        },
        "data": {
            "development_manifest_identity_sha256": DEVELOPMENT_IDENTITY,
            "development_record_limit": args.validation_record_limit,
        },
        "d1_checkpoint_sha256": D1_DEVELOPMENT_SHA256,
        "training_performed": False,
        "learned_fusion": False,
        "hidden_or_layer_change": False,
        "target_rematerialization": False,
        "formal_large_run": False,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }


def _build_model(
    checkpoint: dict[str, Any], config: dict[str, Any], device: torch.device
) -> MCVRConstraintSpecificHybrid:
    prior = MCVRBACModel(**config["prior_model"])
    incompatible = prior.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("V7 frozen D1 prior strict-load failed")
    settings = dict(config["v7"])
    jacobian = settings.pop("jacobian_config")
    model = MCVRConstraintSpecificHybrid(
        prior, jacobian_config=jacobian, **settings
    ).to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("V7 unexpectedly contains trainable parameters")
    return model


def main() -> None:
    args = parse_args()
    _validate_args(args)
    for name in (
        "formal_root",
        "source_cache_root",
        "development_manifests",
        "d1_checkpoint",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    manifest_path = args.development_manifests / "recovery_development_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["identity_sha256"] != DEVELOPMENT_IDENTITY:
        raise RuntimeError("V7 development manifest identity mismatch")
    if (
        int(manifest["test_records_read"]) != 0
        or bool(manifest["test_assets_opened"])
        or int(manifest["frozen_holdout_records_opened"]) != 0
    ):
        raise RuntimeError("V7 development manifest violates isolation")
    if _sha(args.d1_checkpoint) != D1_DEVELOPMENT_SHA256:
        raise RuntimeError("V7 frozen D1 checkpoint SHA mismatch")
    checkpoint = torch.load(args.d1_checkpoint, map_location="cpu", weights_only=False)
    config = _config(args, checkpoint)
    run_dir = args.output_dir / config["experiment_id"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite V7 run: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    command = " ".join(sys.argv)
    _write_json(
        run_dir / "launch.json",
        {
            "pid": os.getpid(),
            "command": command,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": _canonical_sha(config),
            "test_records_read": 0,
            "test_assets_opened": False,
            "frozen_holdout_records_opened": 0,
        },
    )
    validity = ChemicalValidity("data/ecir_mvr/validity_reference_stats.json")
    source_metadata = json.loads(
        (args.formal_root / "real_sources" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    items = _development_items(args, run_dir, validity)
    expected_records = 128 if args.smoke else 1024
    if len(items) != expected_records:
        raise RuntimeError(f"V7 development record count changed: {len(items)}")
    _seed(SEED)
    device = torch.device(args.device)
    model = _build_model(checkpoint, config, device)
    model.eval()
    model.reset_statistics()
    started = time.monotonic()
    evaluation = evaluate_bac_candidate(
        model,
        items,
        validity,
        device=device,
        inference=config["inference"],
        source_identity_sha256=source_metadata["formal_source_identity_sha256"],
        bootstrap_draws=500,
    )
    evaluation_seconds = time.monotonic() - started
    active = _write_evaluation(run_dir, evaluation)
    angle_solver = model.angle_solver_summary()
    components = model.component_summary()
    _write_json(run_dir / "angle_solver_summary.json", angle_solver)
    _write_json(run_dir / "component_summary.json", components)
    with (run_dir / "angle_solver_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in model.angle_solver_trace():
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (run_dir / "component_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in model.component_trace():
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "schema_version": "mcvr-v7-constraint-specific-run-v1",
        "status": "COMPLETED",
        "smoke": args.smoke,
        "records": len(items),
        "training_performed": False,
        "training_elapsed_seconds": 0.0,
        "evaluation_seconds": evaluation_seconds,
        "checkpoint": str(args.d1_checkpoint),
        "checkpoint_sha256": D1_DEVELOPMENT_SHA256,
        "checkpoint_strict_load": True,
        "metrics": evaluation["metrics"],
        "active_subsets": active,
        "angle_solver": angle_solver,
        "components": components,
        "command": command,
        "formal_large_run": False,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
