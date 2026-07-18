#!/usr/bin/env python
"""Train the one authorized MCVR Stage 2b Run A rigid-only pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

from etflow.commons.global_coupled_4d_sampling import atomic_json_save, atomic_torch_save
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.mvr_dataset import MCVRMixedDataset
from etflow.ecir.mvr_loss import MCVRLoss
from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir.run_a_evaluation import (
    build_clean_control_items,
    build_items,
    evaluate_run_a_only,
)


LOSS_NAMES = (
    "flow_loss", "validity_mode_loss", "identity_loss", "anchor_loss", "sparse_loss",
    "torsion_anchor_loss", "error_loss", "uncertainty_loss", "trust_loss", "total_loss",
    "bond_residual_loss", "bond_direction_loss", "bond_sparse_loss",
    "bond_confidence_loss", "bond_uncertainty_loss", "bond_consistency_loss",
)
METRIC_FIELDS = (
    "step", "split", *LOSS_NAMES, "rigid_gate_mean", "global_safety_gate_mean",
    "uncertainty_mean", "velocity_norm_mean", "molecule_displacement_mean",
    "max_atom_displacement_mean", "identity_subset_displacement",
    "high_flex_torsion_change", "records_per_second",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _loss_value(values, name: str):
    return values["loss" if name == "total_loss" else name]


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _assert_identity(config: dict, audit_path: Path) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    stage_d = config["experiment_name"] in {
        "ecir_mvr_stage_d_d1_a_aux_only_seed42_5k",
        "ecir_mvr_stage_d_d1_b_explicit_bond_seed42_5k",
    }
    medium = stage_d or config["experiment_name"] in {
        "ecir_mvr_medium_5k_500_run_a_seed42_20k",
        "ecir_mvr_medium_5k_500_run_a_seed42_20k_rescue_v2",
        "ecir_mvr_medium_5k_500_run_a_seed42_20k_rescue_v3",
    }
    if audit["status"] != "PASS" or audit["test_records_read"] != 0:
        raise RuntimeError("Run A data audit is not a test-free PASS")
    if audit["identities"] != config["frozen_identities"]:
        raise RuntimeError("Run A frozen identities changed after audit")
    if medium:
        source = json.loads(Path(config["data"]["source_metadata"]).read_text(encoding="utf-8"))
        target = json.loads(Path(config["data"]["target_metadata"]).read_text(encoding="utf-8"))
        if source["medium_real_source_identity_sha256"] != config["frozen_identities"]["medium_real_source_identity_sha256"]:
            raise RuntimeError("medium real-source identity changed after preflight")
        if target["decision"] != "PASS" or target["medium_target_identity_sha256"] != config["frozen_identities"]["medium_target_identity_sha256"]:
            raise RuntimeError("medium target identity changed after preflight")
        state = json.loads(Path("reports/ecir_mvr/progressive_state.json").read_text(encoding="utf-8"))
        permitted = (
            state.get("stage_d_oracle_decision") == "PASS"
            if stage_d else (
                bool(state.get("20k_permitted"))
                or bool(state.get("medium_rescue_v2_permitted"))
                or bool(state.get("medium_rescue_v3_permitted"))
            )
        )
        if not permitted or state["100k_permitted"]:
            raise RuntimeError("medium permission boundary changed")
        return audit
    stage = json.loads(Path("diagnostics/ecir_mvr/stage_c/decision.json").read_text())
    if stage["20k_permitted"] or stage["100k_permitted"]:
        raise RuntimeError("long-run permissions unexpectedly enabled")
    return audit


def _dataset(
    config: dict, split: str, validity, *, runtime_statistics=None
) -> MCVRMixedDataset:
    data = config["data"]
    runtime = data.get("runtime_optimizations", {})
    return MCVRMixedDataset(
        data[f"{split}_sources"], data[f"{split}_targets"], validity,
        length=int(data[f"{split}_epoch_size"]), ratios=data["mixture"],
        synthetic_ratios=data["synthetic_mixture"],
        seed=int(config["seed"]) + (0 if split == "train" else 100_000),
        out_of_domain_extreme_ratio=float(data["out_of_domain_extreme_fraction"]),
        formal_adapter_lru_size=int(runtime.get("formal_adapter_lru_size", 0)),
        precompute_training_topology=bool(
            runtime.get("precompute_training_topology", False)
        ),
        runtime_statistics=runtime_statistics,
    )


@torch.inference_mode()
def _diagnostics(model, batch, step_size: float = 0.25) -> dict[str, float]:
    model.eval()
    graphs = int(batch.num_graphs)
    output = model(batch, batch.x_input, batch.x_input.new_full((graphs,), 0.5))
    atom_batch = batch.batch
    displacement = float(step_size) * output["v_final"]
    norms = torch.linalg.vector_norm(displacement, dim=-1)
    energy = displacement.new_zeros(graphs)
    energy.index_add_(0, atom_batch, displacement.square().sum(-1))
    counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1).to(displacement.dtype)
    graph_rms = torch.sqrt(energy / counts + 1e-12)
    clean = batch.active_mode_mask.reshape(graphs, 6)[:, 5] > 0
    identity = norms[clean[atom_batch]].mean() if bool(clean.any()) else norms.new_zeros(())
    model.train()
    return {
        "rigid_gate_mean": float(output["rigid_gate"].mean()),
        "global_safety_gate_mean": float(output["global_safety_gate"].mean()),
        "uncertainty_mean": float(output["uncertainty"].mean()),
        "velocity_norm_mean": float(torch.linalg.vector_norm(output["v_final"], dim=-1).mean()),
        "molecule_displacement_mean": float(graph_rms.mean()),
        "max_atom_displacement_mean": float(norms.max()),
        "identity_subset_displacement": float(identity),
        "high_flex_torsion_change": 0.0,
    }


@torch.inference_mode()
def _validate_losses(model, loss_fn, loader, device) -> dict[str, float]:
    model.eval()
    rows = []
    diagnostic_rows = []
    for batch in loader:
        batch = batch.to(device)
        values = loss_fn(model, batch)
        rows.append({name: float(_loss_value(values, name)) for name in LOSS_NAMES})
        diagnostic_rows.append(_diagnostics(model, batch))
    model.train()
    result = {name: float(np.mean([row[name] for row in rows])) for name in LOSS_NAMES}
    result.update({
        name: float(np.mean([row[name] for row in diagnostic_rows]))
        for name in diagnostic_rows[0]
    })
    return result


def _checkpoint_payload(model, optimizer, step, resolved, validation=None):
    return {
        "schema_version": "ecir-mvr-run-a-v1",
        "model_type": "MCVRModel",
        "run_mode": "rigid_only",
        "step": int(step),
        "config": resolved,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_audit", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume_checkpoint", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    medium = config["experiment_name"] == "ecir_mvr_medium_5k_500_run_a_seed42_20k"
    if config["experiment_name"] not in {
        "ecir_mvr_stage2b_run_a_rigid_only_seed42_5k",
        "ecir_mvr_medium_5k_500_run_a_seed42_20k",
    }:
        raise ValueError("only the frozen Run A experiment is authorized")
    if not config["run_a_mode"]["torsion_gate_fixed_zero"] or config["model"]["torsion_scale"] != 0.0:
        raise ValueError("Run A must have zero torsion gate and scale")
    requested_steps = int(args.steps or config["training"]["optimizer_steps"])
    if medium:
        if requested_steps != 20000 or args.steps is not None:
            raise ValueError("frozen medium Run A requires exactly 20000 optimizer steps")
        if args.resume_checkpoint is not None or config.get("initialize_from_checkpoint") is not None or config.get("resume_checkpoint") is not None:
            raise ValueError("medium seed42 must train from scratch")
    else:
        if requested_steps > 5000:
            raise ValueError("Run A may not exceed 5000 optimizer steps")
        if requested_steps != 5000 and args.steps is None:
            raise ValueError("frozen Run A requires exactly 5000 optimizer steps")
    audit = _assert_identity(config, args.data_audit)
    seed = int(config["seed"])
    _seed(seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Run A requires the audited CUDA environment")

    output = Path(config["output_dir"])
    checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    config_sha = _sha(args.config)
    git_commit = _git("rev-parse", "HEAD")
    resume_payload = None
    start_step = 0
    if args.resume_checkpoint is not None:
        resume_payload = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if resume_payload.get("model_type") != "MCVRModel" or resume_payload.get("run_mode") != "rigid_only":
            raise RuntimeError("resume checkpoint is not rigid-only MCVR Run A")
        start_step = int(resume_payload["step"])
        if start_step <= 0 or start_step >= requested_steps:
            raise RuntimeError("resume checkpoint step must be between 1 and requested_steps-1")
        frozen_sha = resume_payload["config"]["resolved"]["config_sha256"]
        if frozen_sha != config_sha:
            raise RuntimeError("resume checkpoint config identity mismatch")
    resolved = {
        **config,
        "resolved": {
            "config_sha256": config_sha, "git_commit": git_commit,
            "device": str(device), "gpu": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
        },
    }
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    existing_metadata_path = output / "run_metadata.json"
    existing_metadata = (
        json.loads(existing_metadata_path.read_text(encoding="utf-8"))
        if resume_payload is not None and existing_metadata_path.is_file() else {}
    )
    prior_elapsed = float(existing_metadata.get("elapsed_seconds", 0.0))
    run_metadata = {
        **existing_metadata,
        "status": "RUNNING", "experiment_name": config["experiment_name"],
        "seed": seed, "optimizer_steps": requested_steps, "teacher_steps": 4,
        "config_sha256": config_sha, "git_commit": git_commit,
        "data_audit_identity": audit["identity_sha256"],
        "frozen_identities": config["frozen_identities"],
        "host": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda), "gpu": torch.cuda.get_device_name(0),
        "test_records_read": 0, "run_b_started": False, "run_c_started": False,
        "20k_started": medium, "100k_started": False,
        "started_at_unix": existing_metadata.get("started_at_unix", time.time()),
        "resumed_from_step": start_step if resume_payload is not None else None,
        "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
    }
    atomic_json_save(run_metadata, output / "run_metadata.json")
    log_path = output / "training.log"
    log_handle = log_path.open(
        "a" if resume_payload is not None else "w", encoding="utf-8", buffering=1
    )

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")

    log(json.dumps({
        "split_identity": audit["identities"],
        "train": audit["train"], "val": audit["val"],
        "training_plan": audit.get("training_plan", {
            "optimizer_steps": requested_steps,
            "train_epoch_size": config["data"]["train_epoch_size"],
            "val_epoch_size": config["data"]["val_epoch_size"],
            "mixture": config["data"]["mixture"],
        }), "config_sha256": config_sha,
        "git_commit": git_commit, "gpu": run_metadata["gpu"],
        "cuda": run_metadata["cuda"], "torch": run_metadata["torch"], "seed": seed,
    }, sort_keys=True))

    validity = ChemicalValidity(config["data"]["validity_statistics"])
    train_data = _dataset(config, "train", validity)
    val_data = _dataset(config, "val", validity)
    training = config["training"]
    loader_kwargs = {
        "num_workers": int(training["num_workers"]), "pin_memory": True,
    }
    train_loader = DataLoader(
        train_data, batch_size=int(training["batch_size"]), shuffle=False, **loader_kwargs
    )
    val_loader = DataLoader(
        val_data, batch_size=int(training["val_batch_size"]), shuffle=False, **loader_kwargs
    )
    val_items = build_items(
        config["data"]["val_sources"], config["data"]["val_targets"], validity
    )
    clean_control_items = build_clean_control_items(val_items, validity, limit=20)
    if len(clean_control_items) < 10:
        raise RuntimeError("fewer than 10 clean validation-reference identity controls")
    model = MCVRModel(**config["model"]).to(device)
    loss_fn = MCVRLoss(config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        log(f"RESUME checkpoint={args.resume_checkpoint} step={start_step}")
    metrics_path = output / "metrics.csv"
    append_metrics = resume_payload is not None and metrics_path.is_file()
    metrics_handle = metrics_path.open(
        "a" if append_metrics else "w", newline="", encoding="utf-8"
    )
    writer = csv.DictWriter(metrics_handle, fieldnames=METRIC_FIELDS)
    if not append_metrics:
        writer.writeheader()
    diagnostic_dir = Path(config["diagnostics_dir"])
    comparison_path = diagnostic_dir / "checkpoint_comparison.csv"
    comparison_rows = (
        pd.read_csv(comparison_path).to_dict("records")
        if resume_payload is not None and comparison_path.is_file() else []
    )
    train_window = []
    batches_per_epoch = len(train_loader)
    epoch, batch_offset = divmod(start_step, batches_per_epoch)
    train_data.set_epoch(epoch)
    iterator = iter(train_loader)
    for _ in range(batch_offset):
        next(iterator)
    seen = 0
    started = time.perf_counter()
    stop_reason = None
    resumed_validation = resume_payload.get("validation") if resume_payload else None
    best = None
    if resumed_validation and resumed_validation.get("accuracy_noninferior"):
        best = {
            "step": start_step,
            "key": (
                round(float(resumed_validation["validity_delta"]), 6),
                float(resumed_validation["mean_displacement"]),
                -float(resumed_validation.get("identity_fraction", 0.0)),
                float(resumed_validation.get("acceptance_fraction", 1.0)),
            ),
            "validation": resumed_validation,
        }
    validation_history = [resumed_validation] if resumed_validation else []
    diagnostic_history = []
    model.train()
    step = start_step
    for step in range(start_step + 1, requested_steps + 1):
        if step % 50 == 1:
            _assert_identity(config, args.data_audit)
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            train_data.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = loss_fn(model, batch)
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            stop_reason = "nan_or_inf_loss"
            log(f"EARLY_STOP {stop_reason} step={step}")
            break
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(grad_norm)) or float(grad_norm) > 100.0:
            stop_reason = "gradient_explosion"
            log(f"EARLY_STOP {stop_reason} step={step} grad_norm={float(grad_norm)}")
            break
        optimizer.step()
        seen += int(batch.num_graphs)
        train_window.append({
            name: float(_loss_value(losses, name).detach()) for name in LOSS_NAMES
        })
        if step == 1 or step % int(training["log_interval"]) == 0:
            diag = _diagnostics(model, batch)
            row = {
                "step": step, "split": "train",
                **{name: float(np.mean([value[name] for value in train_window])) for name in LOSS_NAMES},
                **diag, "records_per_second": seen / max(time.perf_counter() - started, 1e-9),
            }
            writer.writerow(row); metrics_handle.flush(); train_window.clear()
            diagnostic_history.append(row)
            log(
                f"step={step} loss={row['total_loss']:.6f} flow={row['flow_loss']:.6f} "
                f"rigid_gate={row['rigid_gate_mean']:.4f} safety={row['global_safety_gate_mean']:.4f} "
                f"velocity={row['velocity_norm_mean']:.6f}"
            )
            if row["velocity_norm_mean"] > 0.2:
                stop_reason = "velocity_norm_abnormal_growth"
                break
            if len(diagnostic_history) >= 5:
                recent_diagnostics = diagnostic_history[-5:]
                if all(value["rigid_gate_mean"] < 0.01 for value in recent_diagnostics):
                    stop_reason = "rigid_gate_collapsed_near_zero"
                elif all(value["rigid_gate_mean"] > 0.99 for value in recent_diagnostics):
                    stop_reason = "rigid_gate_collapsed_near_one"
                elif all(value["global_safety_gate_mean"] < 0.05 for value in recent_diagnostics):
                    stop_reason = "global_safety_gate_collapsed"
                elif (
                    all(
                        recent_diagnostics[index]["velocity_norm_mean"]
                        < recent_diagnostics[index + 1]["velocity_norm_mean"]
                        for index in range(4)
                    )
                    and recent_diagnostics[-1]["velocity_norm_mean"]
                    > 2.0 * recent_diagnostics[0]["velocity_norm_mean"]
                ):
                    stop_reason = "velocity_norm_sustained_growth"
                elif (
                    all(
                        recent_diagnostics[index]["molecule_displacement_mean"]
                        < recent_diagnostics[index + 1]["molecule_displacement_mean"]
                        for index in range(4)
                    )
                    and recent_diagnostics[-1]["molecule_displacement_mean"]
                    > 2.0 * recent_diagnostics[0]["molecule_displacement_mean"]
                ):
                    stop_reason = "molecule_displacement_sustained_growth"
                if stop_reason:
                    log(f"EARLY_STOP {stop_reason} step={step}")
                    break

        if step % int(training["validation_interval"]) == 0 or step == requested_steps:
            _assert_identity(config, args.data_audit)
            val_losses = _validate_losses(model, loss_fn, val_loader, device)
            writer.writerow({
                "step": step, "split": "val", **val_losses, "records_per_second": "",
            }); metrics_handle.flush()
            full = evaluate_run_a_only(
                model, val_items, validity, device=device,
                inference=config["inference"], margins=config["noninferiority"],
                bootstrap_draws=500, clean_control_items=clean_control_items,
            )
            all_summary = full["summary"]
            all_summary.to_csv(output / f"validation_step{step:06d}.csv", index=False)
            validation = {
                "step": step, "losses": val_losses,
                "accuracy_gate": full["accuracy_gate"],
                "accuracy_noninferior": full["accuracy_noninferior"],
                "validity_delta": full["validity_delta"],
                "mean_displacement": full["mean_displacement"],
                "acceptance_fraction": full["acceptance_fraction"],
                "validity_worsened_fraction": full["validity_worsened_fraction"],
                "chirality_delta": full["chirality_delta"],
                "severe_clash_delta": full["severe_clash_delta"],
                "high_flex_torsion_change": full["high_flex_torsion_change"],
                "identity_fraction": full["identity_fraction"],
                "acceptance_fraction": full["acceptance_fraction"],
                "torsion_gate_max": full["torsion_gate_max"],
                "torsion_contribution_max": full["torsion_contribution_max"],
                "bootstrap": full["bootstrap"],
            }
            summary_index = full["summary"].set_index(["group", "method"])
            for group, prefix in (("rotatable_ge_6", "high_flex"), ("unseen_update_scale_0.35", "unseen")):
                candidate_row = summary_index.loc[(group, "run_a_accepted")]
                upstream_row = summary_index.loc[(group, "upstream")]
                validation[f"{prefix}_validity_delta"] = float(
                    candidate_row.total_thresholded_validity_score - upstream_row.total_thresholded_validity_score
                )
                validation[f"{prefix}_rmsd_delta"] = float(candidate_row.aligned_RMSD - upstream_row.aligned_RMSD)
            validation_history.append(validation)
            payload = _checkpoint_payload(model, optimizer, step, resolved, validation)
            checkpoint_path = checkpoints / f"step{step:06d}.ckpt"
            atomic_torch_save(payload, checkpoint_path)
            comparison_rows.append({
                "step": step, "checkpoint": str(checkpoint_path.resolve()),
                "accuracy_noninferior": full["accuracy_noninferior"],
                "validity_delta": full["validity_delta"],
                "mean_displacement": full["mean_displacement"],
                "identity_fraction": full["identity_fraction"],
                "rmsd_delta": full["bootstrap"]["aligned_RMSD"]["mean"],
                "rmsd_ci_high": full["bootstrap"]["aligned_RMSD"]["ci95_high"],
                "mat_p_delta": full["bootstrap"]["MAT_P"]["mean"],
                "mat_r_delta": full["bootstrap"]["MAT_R"]["mean"],
            })
            log(
                f"validation step={step} loss={val_losses['total_loss']:.6f} "
                f"accuracy_noninferior={full['accuracy_noninferior']} "
                f"validity_delta={full['validity_delta']:.6f} "
                f"rmsd_delta={full['bootstrap']['aligned_RMSD']['mean']:.6f}"
            )
            if full["torsion_gate_max"] != 0.0 or full["torsion_contribution_max"] != 0.0:
                stop_reason = "torsion_branch_nonzero"
            candidate_key = (
                round(full["validity_delta"], 6), full["mean_displacement"],
                -float(full["identity_fraction"] if math.isfinite(full["identity_fraction"]) else 0.0),
                validation["high_flex_validity_delta"] if medium else full["acceptance_fraction"],
                validation["unseen_validity_delta"] if medium else full["acceptance_fraction"],
            )
            if full["accuracy_noninferior"] and (best is None or candidate_key < best["key"]):
                best = {"step": step, "key": candidate_key, "validation": validation}
                atomic_torch_save(payload, checkpoints / "best_noninferior_validity.ckpt")
            if len(validation_history) >= 2:
                recent = validation_history[-2:]
                margins = config["noninferiority"]
                simultaneous = all(
                    value["bootstrap"]["aligned_RMSD"]["mean"] > float(margins["rmsd_mean_delta_max"])
                    and value["bootstrap"]["MAT_P"]["mean"] > float(margins["mat_p_mean_delta_max"])
                    and value["bootstrap"]["MAT_R"]["mean"] > float(margins["mat_r_mean_delta_max"])
                    for value in recent
                )
                if simultaneous:
                    stop_reason = "two_validations_accuracy_noninferiority_failed"
                finite_identity = [
                    value["identity_fraction"] for value in recent
                    if math.isfinite(value["identity_fraction"])
                ]
                if len(finite_identity) == 2 and all(value < 0.90 for value in finite_identity):
                    stop_reason = "clean_identity_degraded"
                if any(value["chirality_delta"] > 1e-9 for value in recent):
                    stop_reason = "chirality_worsened"
                if any(value["severe_clash_delta"] > 1e-9 for value in recent):
                    stop_reason = "severe_clash_increased"
                flex = [
                    value["high_flex_torsion_change"] for value in recent
                    if math.isfinite(value["high_flex_torsion_change"])
                ]
                if len(flex) == 2 and flex[-1] > max(0.05, 1.25 * flex[0]):
                    stop_reason = "high_flex_torsion_change_increased"
                if (
                    recent[-1]["validity_delta"] >= recent[-2]["validity_delta"] - 1e-6
                    and recent[-1]["mean_displacement"] > recent[-2]["mean_displacement"] + 1e-4
                ):
                    stop_reason = "validity_plateau_with_displacement_growth"
                if (
                    recent[-1]["losses"]["total_loss"] > recent[-2]["losses"]["total_loss"]
                    and len(diagnostic_history) >= 2
                    and diagnostic_history[-1]["total_loss"] < diagnostic_history[-2]["total_loss"]
                ):
                    stop_reason = "train_loss_down_validation_worse"
            if medium and (
                validation["unseen_validity_delta"] >= 0.0
                or validation["unseen_rmsd_delta"] > float(config["noninferiority"]["rmsd_mean_delta_max"])
            ):
                stop_reason = "unseen_condition_failed"
            if stop_reason:
                log(f"EARLY_STOP {stop_reason} step={step}")
                break

    final_step = step
    final_payload = _checkpoint_payload(
        model, optimizer, final_step, resolved,
        validation_history[-1] if validation_history else None,
    )
    atomic_torch_save(final_payload, checkpoints / "last.ckpt")
    comparison = pd.DataFrame(comparison_rows)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(diagnostic_dir / "checkpoint_comparison.csv", index=False)
    metrics_handle.close()
    metrics = pd.read_csv(metrics_path)
    loss_summary = metrics.groupby("split", as_index=False)[list(LOSS_NAMES)].agg(["first", "last", "min"])
    loss_summary.columns = ["_".join(part for part in column if part) for column in loss_summary.columns]
    loss_summary.to_csv(diagnostic_dir / "loss_summary.csv", index=False)
    elapsed = time.perf_counter() - started
    run_metadata.update({
        "status": "COMPLETED" if stop_reason is None else "EARLY_STOPPED",
        "completed_steps": final_step,
        "elapsed_seconds": prior_elapsed + elapsed,
        "latest_segment_elapsed_seconds": elapsed,
        "stop_reason": stop_reason, "best_noninferior_step": best["step"] if best else None,
        "20k_completed": bool(medium and stop_reason is None and final_step == 20000),
        "completed_at_unix": time.time(),
    })
    atomic_json_save(run_metadata, output / "run_metadata.json")
    log(f"finished steps={final_step} elapsed_seconds={elapsed:.3f} stop_reason={stop_reason}")
    log_handle.close()
    print(json.dumps(run_metadata, indent=2))


if __name__ == "__main__":
    main()
