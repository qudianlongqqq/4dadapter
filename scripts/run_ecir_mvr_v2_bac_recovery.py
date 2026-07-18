#!/usr/bin/env python3
"""Run one preregistered Phase-1 Cartesian BAC recovery candidate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.bac_evaluation import (  # noqa: E402
    evaluate_bac_candidate,
    summary_json,
)
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_dataset import MCVRMixedDataset  # noqa: E402
from etflow.ecir.mvr_loss import MCVRLoss  # noqa: E402
from etflow.ecir.mvr_v2_bac import (  # noqa: E402
    MCVRBACModel,
    V2_A_BOND_ONLY,
    V2_D_BOND_ANGLE_CLASH,
)
from etflow.ecir.mvr_v2_bac_loss import MCVRBACLoss  # noqa: E402
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


CANDIDATES = ("A0", "D0", "D1")
RECOVERY_MANIFEST_IDENTITY = (
    "3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51"
)


def _candidate_config(
    candidate: str, args: argparse.Namespace, base: dict[str, Any]
) -> dict[str, Any]:
    mode = V2_A_BOND_ONLY if candidate == "A0" else V2_D_BOND_ANGLE_CLASH
    recovered = candidate == "D1"
    return {
        "schema_version": "mcvr-v2-bac-recovery-candidate-v1",
        "experiment_id": (
            f"{candidate.lower()}_{'smoke' if args.smoke else 'pilot'}_"
            f"{args.steps}step_seed{args.seed}"
        ),
        "candidate": candidate,
        "mode": mode,
        "seed": int(args.seed),
        "model": {
            **base["model"],
            "bac_mode": mode,
            "bac_constraint_scale": 0.05,
            "clash_cutoff": 2.0,
            "clash_allowed_contact": 1.0,
            "clash_exclude_topology_distance": 2,
            "max_clash_edges_per_graph": 128,
            "bac_active_constraint_normalization": recovered,
        },
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
        "bac_loss_behavior": {
            "proposal_mode": "full_inference_field" if recovered else "branches_only",
            "proposal_time": 1.0 if recovered else 0.0,
            "proposal_step_size": 0.25 if recovered else 1.0,
        },
        "training": {
            "optimizer_steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "sample_exposures": int(args.steps) * int(args.batch_size),
            "learning_rate": 2.0e-4,
            "final_learning_rate": 2.0e-5,
            "warmup_steps": min(100, max(1, int(args.steps) // 10)),
            "weight_decay": 1.0e-6,
            "gradient_clip_norm": 1.0,
            "num_workers": int(args.num_workers),
            "formal_adapter_lru_size": 0,
            "precompute_training_topology": True,
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
                "enable_backtracking": recovered,
            },
        },
        "data": {
            "training_source_manifest": str(args.training_assets / "sources_train.parquet"),
            "training_target_manifest": str(args.training_assets / "targets_train.parquet"),
            "development_manifest_identity_sha256": RECOVERY_MANIFEST_IDENTITY,
            "development_record_limit": args.validation_record_limit,
        },
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }


def _active_subset_metrics(records: pd.DataFrame, draws: int = 500) -> dict[str, Any]:
    source = records[records.method == "upstream"].set_index("sample_id")
    candidate = records[records.method == "v2_bac_accepted"].set_index("sample_id")
    common = source.index.intersection(candidate.index)
    source = source.loc[common]
    candidate = candidate.loc[common]
    definitions = {
        "all": pd.Series(True, index=common),
        "bond_active": source.bond_outlier_rate > 0,
        "angle_active": source.angle_outlier_rate > 0,
        "clash_active": (source.clash_penetration > 0) | (source.severe_clash_rate > 0),
    }
    metric_columns = {
        "bond_delta": "bond_outlier_rate",
        "angle_delta": "angle_outlier_rate",
        "clash_delta": "clash_penetration",
        "weighted_bac_delta": "total_thresholded_validity_score",
        "ring_delta": "ring_bond_outlier_rate",
        "chirality_delta": "chirality_error",
        "rmsd_delta": "aligned_RMSD",
    }
    result = {}
    rng = np.random.default_rng(43017)
    for name, mask in definitions.items():
        selected_ids = common[mask.to_numpy()]
        if not len(selected_ids):
            result[name] = {"records": 0, "molecules": 0, "status": "NO_DATA_SUPPORT"}
            continue
        left = source.loc[selected_ids]
        right = candidate.loc[selected_ids]
        deltas = pd.DataFrame(
            {
                output: right[column].to_numpy() - left[column].to_numpy()
                for output, column in metric_columns.items()
            },
            index=selected_ids,
        )
        deltas["molecule_id"] = left.molecule_id.to_numpy()
        molecule = deltas.groupby("molecule_id").mean(numeric_only=True)
        metrics = {}
        for column in metric_columns:
            values = molecule[column].to_numpy(dtype=np.float64)
            sampled = np.asarray(
                [rng.choice(values, size=len(values), replace=True).mean() for _ in range(draws)]
            )
            metrics[column] = {
                "mean": float(values.mean()),
                "ci95_low": float(np.quantile(sampled, 0.025)),
                "ci95_high": float(np.quantile(sampled, 0.975)),
            }
        accepted = right.accepted.to_numpy(dtype=np.float64)
        result[name] = {
            "records": len(selected_ids),
            "molecules": int(molecule.shape[0]),
            "metrics": metrics,
            "acceptance_fraction": float(accepted.mean()),
            "mean_displacement": float(right.molecule_rms_displacement.mean()),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
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
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/ecir_mvr/formal_large/d1_b_seed43/"
            "best_noninferior_validity.ckpt"
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
        default=Path("diagnostics/ecir_mvr/v2_bac_recovery/runs"),
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--validation-record-limit", type=int)
    parser.add_argument("--seed", type=int, default=43018)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke and not 1 <= args.steps <= 200:
        raise ValueError("recovery smoke is capped at 200 steps")
    if not args.smoke and not 1 <= args.steps <= 1000:
        raise ValueError("recovery pilot is capped at 1000 steps")
    if args.smoke and args.candidate != "D1":
        raise ValueError("the single preregistered smoke is reserved for D1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested recovery CUDA device is unavailable")
    for name in (
        "formal_root",
        "source_cache_root",
        "training_assets",
        "development_manifests",
        "checkpoint",
        "base_config",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    manifest = json.loads(
        (args.development_manifests / "recovery_development_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest["identity_sha256"] != RECOVERY_MANIFEST_IDENTITY:
        raise RuntimeError("recovery development manifest identity mismatch")
    if (
        int(manifest["test_records_read"]) != 0
        or bool(manifest["test_assets_opened"])
        or int(manifest["frozen_holdout_records_opened"]) != 0
    ):
        raise RuntimeError("recovery development manifest violates isolation")

    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config = _candidate_config(args.candidate, args, base)
    run_dir = args.output_dir / config["experiment_id"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite recovery run: {run_dir}")
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
            "candidate": args.candidate,
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
    source_manifest = args.development_manifests / "development_sources.parquet"
    target_manifest = args.development_manifests / "development_targets.parquet"
    if args.validation_record_limit is not None:
        source_rows = pd.read_parquet(source_manifest).head(args.validation_record_limit)
        target_rows = pd.read_parquet(target_manifest)
        target_rows = target_rows[target_rows.sample_id.isin(source_rows.sample_id)]
        runtime = run_dir / "runtime_manifests"
        runtime.mkdir()
        source_manifest = runtime / "development_sources.parquet"
        target_manifest = runtime / "development_targets.parquet"
        source_rows.to_parquet(source_manifest, index=False)
        target_rows.to_parquet(target_manifest, index=False)
    validation_items = build_items(
        source_manifest,
        target_manifest,
        validity,
        source_cache_root=args.source_cache_root,
        target_cache_root=args.formal_root / "minimal_targets",
    )

    _seed(args.seed)
    dataset = MCVRMixedDataset(
        args.training_assets / "sources_train.parquet",
        args.training_assets / "targets_train.parquet",
        validity,
        length=args.steps * args.batch_size,
        seed=args.seed,
        formal_adapter_lru_size=0,
        precompute_training_topology=True,
        source_cache_root=args.source_cache_root,
        canonical_constraints=True,
        constraint_source_identity_sha256=source_metadata[
            "formal_source_identity_sha256"
        ],
    )
    loader_kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != 25000:
        raise RuntimeError("recovery initialization is not a frozen step-25000 checkpoint")
    device = torch.device(args.device)
    model = MCVRBACModel(**config["model"]).to(device)
    missing, unexpected = model.load_d1b_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    if missing or unexpected:
        raise RuntimeError(f"D1-B strict initialization failed: {missing}, {unexpected}")
    loss_fn: torch.nn.Module
    if args.candidate == "A0":
        loss_fn = MCVRLoss(config["base_loss"])
    else:
        loss_fn = MCVRBACLoss(
            config["base_loss"],
            config["bac_loss"],
            **config["bac_loss_behavior"],
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    rows = []
    log_path = run_dir / "training.log"
    started = time.monotonic()
    for step, batch in enumerate(loader, start=1):
        if step > args.steps:
            break
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        losses = loss_fn(model, batch)
        if not bool(torch.isfinite(losses["loss"])):
            raise RuntimeError(f"nonfinite loss at step {step}")
        losses["loss"].backward()
        gradients = _gradient_norms(model)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"nonfinite gradient at step {step}")
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
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    checkpoint_path = run_dir / "checkpoint_final.ckpt"
    _atomic_torch_save(
        {
            "schema_version": "mcvr-v2-bac-recovery-checkpoint-v1",
            "step": args.steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "test_records_read": 0,
        },
        checkpoint_path,
    )
    roundtrip = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    roundtrip_model = MCVRBACModel(**config["model"])
    incompatible = roundtrip_model.load_state_dict(
        roundtrip["model_state_dict"], strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("recovery checkpoint strict roundtrip failed")

    evaluation = evaluate_bac_candidate(
        model,
        validation_items,
        validity,
        device=device,
        inference=config["inference"],
        source_identity_sha256=source_metadata["formal_source_identity_sha256"],
        bootstrap_draws=500,
    )
    evaluation["records"].to_csv(run_dir / "development_per_record.csv", index=False)
    evaluation["molecules"].to_csv(run_dir / "development_per_molecule.csv", index=False)
    evaluation["summary"].to_csv(run_dir / "development_summary.csv", index=False)
    _write_json(run_dir / "development_summary.json", summary_json(evaluation))
    active_subsets = _active_subset_metrics(evaluation["records"])
    _write_json(
        run_dir / "active_subset_metrics.json",
        {
            "schema_version": "mcvr-v2-bac-recovery-active-subsets-v1",
            "subsets": active_subsets,
            "test_records_read": 0,
            "test_assets_opened": False,
            "frozen_holdout_records_opened": 0,
        },
    )
    metadata = {
        "schema_version": "mcvr-v2-bac-recovery-run-v1",
        "status": "COMPLETED",
        "candidate": args.candidate,
        "smoke": args.smoke,
        "completed_steps": args.steps,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "checkpoint_roundtrip": True,
        "metrics": evaluation["metrics"],
        "active_subsets": active_subsets,
        "pid": os.getpid(),
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
