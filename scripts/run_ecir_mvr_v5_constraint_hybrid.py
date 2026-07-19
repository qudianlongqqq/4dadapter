#!/usr/bin/env python3
"""Run frozen Windows development experiments for MCVR V5 prototypes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.bac_evaluation import evaluate_bac_candidate, summary_json  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_dataset import MCVRMixedDataset  # noqa: E402
from etflow.ecir.mvr_v2_bac import MCVRBACModel  # noqa: E402
from etflow.ecir.mvr_v5_constraint_hybrid import (  # noqa: E402
    MCVRConstraintMultiHeadModel,
    MCVRNeuralJacobianHybrid,
)
from etflow.ecir.mvr_v5_constraint_loss import MCVRConstraintMultiHeadLoss  # noqa: E402
from etflow.ecir.run_a_evaluation import build_items  # noqa: E402
from scripts.run_ecir_mvr_v2_bac_pilots import (  # noqa: E402
    _atomic_torch_save,
    _canonical_sha,
    _gradient_norms,
    _learning_rate,
    _seed,
    _sha,
    _write_json,
)
from scripts.run_ecir_mvr_v2_bac_recovery import _active_subset_metrics  # noqa: E402


DEVELOPMENT_IDENTITY = "3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51"
D1_DEVELOPMENT_SHA256 = "9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426"
SEED = 43018


def _common_config(args: argparse.Namespace, base: dict[str, Any]) -> dict[str, Any]:
    smoke = bool(args.smoke)
    prototype = str(args.prototype)
    identifier = f"v5_{prototype.lower()}_{'smoke' if smoke else 'pilot'}"
    if prototype == "A":
        identifier += f"_{args.steps}step_seed{SEED}"
    else:
        identifier += f"_seed{SEED}"
    model = {
        **base["model"],
        "bac_constraint_scale": 0.05,
        "clash_cutoff": 2.0,
        "clash_allowed_contact": 1.0,
        "clash_exclude_topology_distance": 2,
        "max_clash_edges_per_graph": 128,
        "bac_active_constraint_normalization": True,
    }
    return {
        "schema_version": "mcvr-v5-constraint-hybrid-experiment-v1",
        "experiment_id": identifier,
        "prototype": prototype,
        "seed": SEED,
        "model": model,
        "base_loss": base["loss"],
        "bac_loss": {
            "bond_residual": 1.0,
            "angle_residual": 1.0,
            "clash_penetration": 1.0,
            "zero_error_noop": 0.25,
            "preservation": 0.25,
            "no_new_violation": 0.5,
            "confidence": 0.05,
            "gate": 0.05,
        },
        "head_loss": {
            "bond_specialization": 0.5,
            "angle_specialization": 0.5,
            "clash_specialization": 0.5,
            "cross_preservation": 0.25,
            "fusion_assignment": 0.1,
        },
        "prototype_a": {
            "component_max_graph_rms": 0.02,
        },
        "prototype_b": {
            "correction_lambda": 1.0,
            "integration_step_size": 0.25,
            "max_correction_graph_rms": 0.01,
            "max_correction_atom": 0.02,
            "jacobian": {
                "damping_lambda": 1.0e-3,
                "rank_tol": 1.0e-6,
                "max_condition_number": 1.0e8,
                "near_linear_sine_threshold": 1.0e-3,
                "near_linear_weight": 0.1,
            },
        },
        "training": {
            "optimizer_steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "sample_exposures": int(args.steps) * int(args.batch_size),
            "learning_rate": 2.0e-4,
            "final_learning_rate": 2.0e-5,
            "warmup_steps": min(100, max(1, int(args.steps) // 10)) if args.steps else 0,
            "weight_decay": 1.0e-6,
            "gradient_clip_norm": 1.0,
            "num_workers": int(args.num_workers),
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
            "training_assets_used": prototype == "A",
        },
        "hidden_or_layer_change": False,
        "target_rematerialization": False,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype", choices=("A", "B"), required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument(
        "--training-assets",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_overnight/pilot_targets"),
    )
    parser.add_argument(
        "--development-manifests",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_recovery/manifests"),
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path("artifacts/ecir_mvr/formal_large/d1_b_seed43/best_noninferior_validity.ckpt"),
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
        "--base-config",
        type=Path,
        default=Path("configs/ecir_mvr_formal_large_d1b_base.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v5_constraint_hybrid/runs"),
    )
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--validation-record-limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.prototype == "A":
        expected = 200 if args.smoke else 1000
        if args.steps != expected:
            raise ValueError(f"Prototype A requires exactly {expected} steps")
    elif args.steps != 0:
        raise ValueError("Prototype B is frozen and requires --steps 0")
    if args.smoke and args.validation_record_limit != 128:
        raise ValueError("V5 smoke requires --validation-record-limit 128")
    if not args.smoke and args.validation_record_limit is not None:
        raise ValueError("V5 pilot must use all 1024 development records")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested V5 CUDA device is unavailable")


def _development_items(
    args: argparse.Namespace,
    run_dir: Path,
    validity: ChemicalValidity,
) -> list[dict[str, Any]]:
    source_manifest = args.development_manifests / "development_sources.parquet"
    target_manifest = args.development_manifests / "development_targets.parquet"
    if args.validation_record_limit is not None:
        sources = pd.read_parquet(source_manifest).head(args.validation_record_limit)
        targets = pd.read_parquet(target_manifest)
        targets = targets[targets.sample_id.isin(sources.sample_id)]
        runtime = run_dir / "runtime_manifests"
        runtime.mkdir()
        source_manifest = runtime / "development_sources.parquet"
        target_manifest = runtime / "development_targets.parquet"
        sources.to_parquet(source_manifest, index=False)
        targets.to_parquet(target_manifest, index=False)
    return build_items(
        source_manifest,
        target_manifest,
        validity,
        source_cache_root=args.source_cache_root,
        target_cache_root=args.formal_root / "minimal_targets",
    )


def _write_evaluation(run_dir: Path, evaluation: dict[str, Any]) -> dict[str, Any]:
    evaluation["records"].to_csv(run_dir / "development_per_record.csv", index=False)
    evaluation["molecules"].to_csv(run_dir / "development_per_molecule.csv", index=False)
    evaluation["summary"].to_csv(run_dir / "development_summary.csv", index=False)
    _write_json(run_dir / "development_summary.json", summary_json(evaluation))
    active = _active_subset_metrics(evaluation["records"])
    _write_json(
        run_dir / "active_subset_metrics.json",
        {
            "schema_version": "mcvr-v5-active-subsets-v1",
            "subsets": active,
            "test_records_read": 0,
            "test_assets_opened": False,
            "frozen_holdout_records_opened": 0,
        },
    )
    return active


def _train_prototype_a(
    args: argparse.Namespace,
    config: dict[str, Any],
    validity: ChemicalValidity,
    source_identity: str,
    run_dir: Path,
    device: torch.device,
) -> tuple[MCVRConstraintMultiHeadModel, dict[str, Any]]:
    dataset = MCVRMixedDataset(
        args.training_assets / "sources_train.parquet",
        args.training_assets / "targets_train.parquet",
        validity,
        length=args.steps * args.batch_size,
        seed=SEED,
        formal_adapter_lru_size=0,
        precompute_training_topology=True,
        source_cache_root=args.source_cache_root,
        canonical_constraints=True,
        constraint_source_identity_sha256=source_identity,
    )
    loader_kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != 25000:
        raise RuntimeError("Prototype A initialization is not frozen step 25000")
    model = MCVRConstraintMultiHeadModel(**config["model"], **config["prototype_a"]).to(device)
    missing, unexpected = model.load_d1b_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Prototype A D1 initialization failed: {missing}, {unexpected}")
    loss_fn = MCVRConstraintMultiHeadLoss(
        config["base_loss"], config["bac_loss"], config["head_loss"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    rows = []
    started = time.monotonic()
    for step, batch in enumerate(loader, start=1):
        if step > args.steps:
            break
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        losses = loss_fn(model, batch)
        if not bool(torch.isfinite(losses["loss"])):
            raise RuntimeError(f"nonfinite Prototype A loss at step {step}")
        losses["loss"].backward()
        gradients = _gradient_norms(model)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"nonfinite Prototype A gradient at step {step}")
        learning_rate = _learning_rate(
            step,
            args.steps,
            config["training"]["warmup_steps"],
            config["training"]["learning_rate"],
            config["training"]["final_learning_rate"],
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(losses["loss"].detach()),
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm),
                **gradients,
                **{
                    name: float(value.detach())
                    for name, value in losses.items()
                    if name != "loss" and value.numel() == 1
                },
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(run_dir / "training_metrics.csv", index=False)
            with (run_dir / "training.log").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    checkpoint_path = run_dir / "checkpoint_final.ckpt"
    _atomic_torch_save(
        {
            "schema_version": "mcvr-v5-prototype-a-checkpoint-v1",
            "step": args.steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "test_records_read": 0,
        },
        checkpoint_path,
    )
    roundtrip = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    roundtrip_model = MCVRConstraintMultiHeadModel(**config["model"], **config["prototype_a"])
    incompatible = roundtrip_model.load_state_dict(roundtrip["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Prototype A checkpoint strict roundtrip failed")
    return model, {
        "training_elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "checkpoint_roundtrip": True,
    }


def _build_prototype_b(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[MCVRNeuralJacobianHybrid, dict[str, Any]]:
    if _sha(args.d1_checkpoint) != D1_DEVELOPMENT_SHA256:
        raise RuntimeError("Prototype B frozen D1 checkpoint SHA mismatch")
    checkpoint = torch.load(args.d1_checkpoint, map_location="cpu", weights_only=False)
    prior = MCVRBACModel(**checkpoint["config"]["model"]).to(device)
    incompatible = prior.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Prototype B D1 prior strict-load failed")
    settings = dict(config["prototype_b"])
    jacobian = settings.pop("jacobian")
    model = MCVRNeuralJacobianHybrid(prior, jacobian_config=jacobian, **settings).to(device)
    return model, {
        "training_elapsed_seconds": 0.0,
        "checkpoint": str(args.d1_checkpoint),
        "checkpoint_sha256": D1_DEVELOPMENT_SHA256,
        "checkpoint_roundtrip": True,
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)
    for name in (
        "formal_root",
        "source_cache_root",
        "training_assets",
        "development_manifests",
        "initial_checkpoint",
        "d1_checkpoint",
        "base_config",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    manifest = json.loads(
        (args.development_manifests / "recovery_development_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest["identity_sha256"] != DEVELOPMENT_IDENTITY:
        raise RuntimeError("V5 development manifest identity mismatch")
    if (
        int(manifest["test_records_read"]) != 0
        or bool(manifest["test_assets_opened"])
        or int(manifest["frozen_holdout_records_opened"]) != 0
    ):
        raise RuntimeError("V5 development manifest violates isolation")
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config = _common_config(args, base)
    run_dir = args.output_dir / config["experiment_id"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite V5 run: {run_dir}")
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
            "prototype": args.prototype,
            "config_sha256": _canonical_sha(config),
            "test_records_read": 0,
            "test_assets_opened": False,
            "frozen_holdout_records_opened": 0,
        },
    )
    validity = ChemicalValidity("data/ecir_mvr/validity_reference_stats.json")
    source_metadata = json.loads(
        (args.formal_root / "real_sources" / "metadata.json").read_text(encoding="utf-8")
    )
    items = _development_items(args, run_dir, validity)
    expected_records = 128 if args.smoke else 1024
    if len(items) != expected_records:
        raise RuntimeError(f"V5 development record count changed: {len(items)}")
    _seed(SEED)
    device = torch.device(args.device)
    if args.prototype == "A":
        model, run = _train_prototype_a(
            args,
            config,
            validity,
            source_metadata["formal_source_identity_sha256"],
            run_dir,
            device,
        )
    else:
        model, run = _build_prototype_b(args, config, device)
    evaluation_started = time.monotonic()
    evaluation = evaluate_bac_candidate(
        model,
        items,
        validity,
        device=device,
        inference=config["inference"],
        source_identity_sha256=source_metadata["formal_source_identity_sha256"],
        bootstrap_draws=500,
    )
    evaluation_seconds = time.monotonic() - evaluation_started
    active = _write_evaluation(run_dir, evaluation)
    solver = None
    if isinstance(model, MCVRNeuralJacobianHybrid):
        solver = model.solver_summary()
        _write_json(run_dir / "solver_summary.json", solver)
        with (run_dir / "solver_trace.jsonl").open("w", encoding="utf-8") as handle:
            for row in model.solver_trace():
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "schema_version": "mcvr-v5-constraint-hybrid-run-v1",
        "status": "COMPLETED",
        "prototype": args.prototype,
        "smoke": args.smoke,
        "completed_steps": args.steps,
        "records": len(items),
        "metrics": evaluation["metrics"],
        "active_subsets": active,
        "evaluation_seconds": evaluation_seconds,
        "solver": solver,
        **run,
        "command": command,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
