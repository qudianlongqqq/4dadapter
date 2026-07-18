#!/usr/bin/env python3
"""Run preregistered sequential MCVR V2-BAC train/validation pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
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
    V2_B_BOND_ANGLE,
    V2_C_BOND_CLASH,
    V2_D_BOND_ANGLE_CLASH,
)
from etflow.ecir.mvr_v2_bac_loss import MCVRBACLoss  # noqa: E402
from etflow.ecir.run_a_evaluation import build_items  # noqa: E402


MODES = (
    V2_A_BOND_ONLY,
    V2_B_BOND_ANGLE,
    V2_C_BOND_CLASH,
    V2_D_BOND_ANGLE_CLASH,
)
HYPOTHESES = {
    V2_A_BOND_ONLY: "D1-B initialization and training remain numerically compatible",
    V2_B_BOND_ANGLE: "explicit angle triplets improve angle validity without bond loss",
    V2_C_BOND_CLASH: "sparse spatial edges reduce clashes without bond loss",
    V2_D_BOND_ANGLE_CLASH: "single fused BAC correction improves all active modes safely",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8"
    ).strip()


def _worktree_code_identity() -> str:
    diff = subprocess.check_output(["git", "diff", "--binary"])
    paths = (
        "etflow/ecir/bac_constraints.py",
        "etflow/ecir/bac_target.py",
        "etflow/ecir/bac_safety.py",
        "etflow/ecir/bac_evaluation.py",
        "etflow/ecir/mvr_v2_bac.py",
        "etflow/ecir/mvr_v2_bac_loss.py",
        "etflow/ecir/mvr_model.py",
        "etflow/ecir/mvr_dataset.py",
    )
    payload = {
        "head": _git("rev-parse", "HEAD"),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "files": {
            path: _sha(ROOT / path) for path in paths if (ROOT / path).is_file()
        },
    }
    return _canonical_sha(payload)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _learning_rate(step: int, total: int, warmup: int, peak: float, final: float) -> float:
    if step <= warmup:
        return peak * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * progress))


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    groups = {
        "shared_backbone": ("backbone.", "error_encoder."),
        "bond": ("bond_explicit_head.",),
        "angle": ("angle_constraint_",),
        "clash": ("clash_constraint_",),
        "fusion": ("constraint_fusion.", "constraint_type_embedding."),
    }
    result = {}
    named = list(model.named_parameters())
    for group, prefixes in groups.items():
        total = 0.0
        for name, parameter in named:
            if parameter.grad is None or not name.startswith(prefixes):
                continue
            total += float(parameter.grad.detach().square().sum())
        result[f"gradient_norm_{group}"] = math.sqrt(total)
    return result


def _mode_bac_weights(mode: str) -> dict[str, float]:
    return {
        "bond_residual": 1.0,
        "angle_residual": 1.0 if mode in {V2_B_BOND_ANGLE, V2_D_BOND_ANGLE_CLASH} else 0.0,
        "clash_penetration": 1.0 if mode in {V2_C_BOND_CLASH, V2_D_BOND_ANGLE_CLASH} else 0.0,
        "zero_error_noop": 0.25,
        "preservation": 0.25,
        "no_new_violation": 0.5,
        "confidence": 0.05,
        "gate": 0.05,
    }


def _resolved_config(args: argparse.Namespace, mode: str, base: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "mcvr-v2-bac-pilot-config-v1",
        "experiment_id": f"{mode.lower()}_{args.steps}step_seed{args.seed}",
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
        },
        "base_loss": base["loss"],
        "bac_loss": _mode_bac_weights(mode),
        "training": {
            "optimizer": "AdamW",
            "optimizer_steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "effective_batch_size": int(args.batch_size),
            "sample_exposures": int(args.steps) * int(args.batch_size),
            "learning_rate": 2.0e-4,
            "final_learning_rate": 2.0e-5,
            "warmup_steps": min(200, max(1, int(args.steps) // 10)),
            "weight_decay": 1.0e-6,
            "gradient_clip_norm": 1.0,
            "num_workers": int(args.num_workers),
            "precompute_training_topology": True,
            "formal_adapter_lru_size": 0,
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
            },
        },
        "data": {
            "pilot_source_manifest": str(args.pilot_assets / "sources_train.parquet"),
            "pilot_target_manifest": str(args.pilot_assets / "targets_train.parquet"),
            "formal_source_identity_sha256": args.formal_source_identity,
            "validation_cohort_identity_sha256": args.validation_cohort_identity,
        },
        "test_records_read": 0,
        "test_assets_opened": False,
        "validation_only": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--pilot-assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/ecir_mvr_formal_large_d1b_base.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_overnight"),
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--validation-record-limit", type=int)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-hours", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.formal_root = args.formal_root.expanduser().resolve()
    args.source_cache_root = args.source_cache_root.expanduser().resolve()
    args.pilot_assets = args.pilot_assets.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if len(args.modes) > 6:
        raise ValueError("at most six sequential GPU runs are permitted")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested pilot")
    device = torch.device(args.device)
    source_metadata = json.loads(
        (args.formal_root / "real_sources" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    cohorts = json.loads(
        (args.output_dir / "validation_cohorts.json").read_text(encoding="utf-8")
    )
    args.formal_source_identity = source_metadata["formal_source_identity_sha256"]
    args.validation_cohort_identity = cohorts["identity_sha256"]
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != 25000:
        raise RuntimeError("D1-B initialization checkpoint is not step 25000")
    validity = ChemicalValidity("data/ecir_mvr/validity_reference_stats.json")
    tune_molecules = set(cohorts["validation_tune"]["molecule_ids"])
    val_sources = pd.read_parquet(args.formal_root / "real_sources" / "val.parquet")
    val_targets = pd.read_parquet(args.formal_root / "minimal_targets" / "val.parquet")
    val_sources = val_sources[val_sources.molecule_id.astype(str).isin(tune_molecules)]
    if args.validation_record_limit is not None:
        val_sources = val_sources.sort_values(["molecule_id", "sample_id"]).head(
            int(args.validation_record_limit)
        )
    val_targets = val_targets[val_targets.sample_id.isin(val_sources.sample_id)]
    runtime_dir = args.output_dir / "runtime_manifests"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    val_source_path = runtime_dir / "validation_tune_sources.parquet"
    val_target_path = runtime_dir / "validation_tune_targets.parquet"
    val_sources.to_parquet(val_source_path, index=False)
    val_targets.to_parquet(val_target_path, index=False)
    validation_items = build_items(
        val_source_path,
        val_target_path,
        validity,
        source_cache_root=args.source_cache_root,
        target_cache_root=args.formal_root / "minimal_targets",
    )
    start_all = time.monotonic()
    inventory_rows = []
    for mode in args.modes:
        if time.monotonic() - start_all > float(args.maximum_hours) * 3600:
            raise RuntimeError("overnight wall-clock budget exhausted")
        config = _resolved_config(args, mode, base)
        experiment_id = config["experiment_id"]
        run_dir = args.output_dir / "runs" / experiment_id
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"refusing to overwrite run directory: {run_dir}")
        run_dir.mkdir(parents=True)
        config_sha = _canonical_sha(config)
        _append_jsonl(
            args.output_dir / "decision_log.jsonl",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment_id": experiment_id,
                "parent_experiment": "D1-B seed43 step25000",
                "hypothesis": HYPOTHESES[mode],
                "code_identity": _worktree_code_identity(),
                "worktree_dirty": bool(_git("status", "--short")),
                "config_identity": config_sha,
                "changed_parameters": {"bac_mode": mode},
                "why_changed": HYPOTHESES[mode],
                "expected_outcome": "BAC improvement under hard noninferiority",
                "stop_criteria": [
                    "nonfinite",
                    "identity_or_chirality_change",
                    "ring_regression",
                    "repeated_oom",
                ],
                "data_identities": {
                    "formal_source": args.formal_source_identity,
                    "validation_cohort": args.validation_cohort_identity,
                },
                "test_records_read": 0,
                "test_assets_opened": False,
                "validation_only": True,
            },
        )
        (run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        _seed(int(args.seed))
        dataset = MCVRMixedDataset(
            args.pilot_assets / "sources_train.parquet",
            args.pilot_assets / "targets_train.parquet",
            validity,
            length=int(args.steps) * int(args.batch_size),
            seed=int(args.seed),
            formal_adapter_lru_size=0,
            precompute_training_topology=True,
            source_cache_root=args.source_cache_root,
            canonical_constraints=True,
            constraint_source_identity_sha256=args.formal_source_identity,
        )
        loader_kwargs = {
            "num_workers": int(args.num_workers),
            "pin_memory": True,
        }
        if int(args.num_workers) > 0:
            loader_kwargs.update(
                {"persistent_workers": True, "prefetch_factor": 2}
            )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            **loader_kwargs,
        )
        model = MCVRBACModel(**config["model"]).to(device)
        missing, unexpected = model.load_d1b_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        if missing or unexpected:
            raise RuntimeError(f"D1-B initialization mismatch: {missing}, {unexpected}")
        loss_fn: torch.nn.Module
        if mode == V2_A_BOND_ONLY:
            loss_fn = MCVRLoss(config["base_loss"])
        else:
            loss_fn = MCVRBACLoss(config["base_loss"], config["bac_loss"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2.0e-4,
            weight_decay=float(config["training"]["weight_decay"]),
        )
        metric_path = run_dir / "training_metrics.csv"
        log_path = run_dir / "training.log"
        rows = []
        iterator = iter(loader)
        status = "RUNNING"
        started = time.monotonic()
        for step in range(1, int(args.steps) + 1):
            batch = next(iterator).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_fn(model, batch)
            if not bool(torch.isfinite(losses["loss"])):
                status = "FAILED_NONFINITE"
                break
            losses["loss"].backward()
            gradients = _gradient_norms(model)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                status = "FAILED_NONFINITE_GRADIENT"
                break
            lr = _learning_rate(
                step,
                int(args.steps),
                int(config["training"]["warmup_steps"]),
                2.0e-4,
                2.0e-5,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == int(args.steps):
                row = {
                    "step": step,
                    "loss": float(losses["loss"].detach()),
                    "learning_rate": lr,
                    "gradient_norm": float(gradient_norm),
                    **gradients,
                    **{
                        name: float(value.detach())
                        for name, value in losses.items()
                        if name != "loss" and value.numel() == 1
                    },
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(metric_path, index=False)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        else:
            status = "TRAINING_COMPLETE"
        if status != "TRAINING_COMPLETE":
            _write_json(
                run_dir / "run_metadata.json",
                {
                    "status": status,
                    "completed_steps": rows[-1]["step"] if rows else 0,
                    "test_records_read": 0,
                    "test_assets_opened": False,
                    "validation_only": True,
                },
            )
            raise RuntimeError(f"{experiment_id} stopped: {status}")
        checkpoint_path = run_dir / "checkpoint_final.ckpt"
        _atomic_torch_save(
            {
                "schema_version": "mcvr-v2-bac-pilot-checkpoint-v1",
                "step": int(args.steps),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "test_records_read": 0,
            },
            checkpoint_path,
        )
        roundtrip = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        roundtrip_model = MCVRBACModel(**config["model"])
        incompatible = roundtrip_model.load_state_dict(
            roundtrip["model_state_dict"], strict=True
        )
        checkpoint_roundtrip = (
            not incompatible.missing_keys
            and not incompatible.unexpected_keys
            and int(roundtrip["step"]) == int(args.steps)
        )
        if not checkpoint_roundtrip:
            raise RuntimeError("checkpoint strict roundtrip failed")
        evaluation = evaluate_bac_candidate(
            model,
            validation_items,
            validity,
            device=device,
            inference=config["inference"],
            source_identity_sha256=args.formal_source_identity,
            bootstrap_draws=500,
        )
        evaluation["records"].to_csv(run_dir / "validation_per_record.csv", index=False)
        evaluation["molecules"].to_csv(
            run_dir / "validation_per_molecule.csv", index=False
        )
        evaluation["summary"].to_csv(run_dir / "validation_summary.csv", index=False)
        _write_json(run_dir / "validation_summary.json", summary_json(evaluation))
        metrics = evaluation["metrics"]
        qualified = (
            metrics["failure_rate"] == 0.0
            and metrics["ring_delta"] <= 1.0e-12
            and metrics["rmsd_delta"] <= 0.015
            and metrics["mat_p_delta"] <= 0.015
            and metrics["mat_r_delta"] <= 0.015
            and metrics["cov_p_delta"] >= -0.005
            and metrics["cov_r_delta"] >= -0.005
        )
        metadata = {
            "status": "COMPLETED",
            "experiment_id": experiment_id,
            "mode": mode,
            "completed_steps": int(args.steps),
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha(checkpoint_path),
            "checkpoint_roundtrip": checkpoint_roundtrip,
            "config_sha256": config_sha,
            "qualified": qualified,
            "metrics": metrics,
            "test_records_read": 0,
            "test_assets_opened": False,
            "validation_only": True,
        }
        _write_json(run_dir / "run_metadata.json", metadata)
        inventory_rows.append(metadata)
        pd.DataFrame(inventory_rows).to_csv(
            args.output_dir / "experiment_inventory.csv", index=False
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "MCVR_V2_BAC_PILOTS_COMPLETE",
                "runs": len(inventory_rows),
                "test_records_read": 0,
                "test_assets_opened": False,
                "validation_only": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
