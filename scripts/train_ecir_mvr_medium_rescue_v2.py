#!/usr/bin/env python
"""Train the frozen Medium Seed42 Rescue V2 with unattended safety controls."""

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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
from etflow.commons.run_timing import RunTiming, iso_now, write_heartbeat
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.mvr_loss import MCVRLoss
from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir.run_a_evaluation import build_clean_control_items, build_items, evaluate_run_a_only
from scripts.train_ecir_mvr_run_a import LOSS_NAMES, _assert_identity, _dataset, _loss_value, _seed, _validate_losses


METRIC_FIELDS = (
    "step", "split", *LOSS_NAMES, "rigid_gate_mean", "global_safety_gate_mean",
    "uncertainty_mean", "velocity_norm_mean", "velocity_graph_rms",
    "velocity_atom_max", "raw_trust_clipping_fraction", "molecule_displacement_mean",
    "max_atom_displacement_mean", "identity_subset_displacement",
    "high_flex_torsion_change", "records_per_second",
)
GPU_FIELDS = (
    "step", "timestamp", "torch_allocated_mib", "torch_reserved_mib",
    "torch_peak_allocated_mib", "torch_peak_reserved_mib", "card_memory_used_mib",
    "card_memory_free_mib", "gpu_utilization", "temperature_c", "power_w",
    "shared_memory_mib",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _gpu_telemetry(step: int) -> dict[str, Any]:
    values: dict[str, Any] = {
        "step": int(step), "timestamp": iso_now(),
        "torch_allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "torch_reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "card_memory_used_mib": math.nan, "card_memory_free_mib": math.nan,
        "gpu_utilization": math.nan, "temperature_c": math.nan,
        "power_w": math.nan, "shared_memory_mib": math.nan,
    }
    try:
        query = "memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw"
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", "0"],
            text=True, timeout=5,
        ).strip().splitlines()[0]
        parsed = [float(value.strip()) for value in output.split(",")]
        values.update(dict(zip(
            ("card_memory_used_mib", "card_memory_free_mib", "gpu_utilization", "temperature_c", "power_w"),
            parsed,
        )))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return values


@torch.inference_mode()
def _diagnostics(model: MCVRModel, batch, step_size: float = 0.25) -> dict[str, float]:
    model.eval()
    graphs = int(batch.num_graphs)
    output = model(batch, batch.x_input, batch.x_input.new_full((graphs,), 0.5))
    atom_batch = batch.batch
    final = output["v_final"]
    raw = output["v_raw"]
    atom_velocity = torch.linalg.vector_norm(final, dim=-1)
    displacement = float(step_size) * final
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)
    energy = final.new_zeros(graphs)
    energy.index_add_(0, atom_batch, final.square().sum(-1))
    counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1).to(final.dtype)
    graph_rms = torch.sqrt(energy / counts + 1e-12)
    raw_energy = raw.new_zeros(graphs)
    raw_energy.index_add_(0, atom_batch, raw.square().sum(-1))
    raw_graph_rms = torch.sqrt(raw_energy / counts + 1e-12)
    raw_atom_norm = torch.linalg.vector_norm(raw, dim=-1)
    clipped_atoms = raw_atom_norm >= model.max_velocity_atom_norm
    clipped_graphs = raw_graph_rms >= model.max_velocity_graph_rms
    clipped = clipped_atoms | clipped_graphs[atom_batch]
    clean = batch.active_mode_mask.reshape(graphs, 6)[:, 5] > 0
    identity = displacement_norm[clean[atom_batch]].mean() if bool(clean.any()) else displacement_norm.new_zeros(())
    model.train()
    return {
        "rigid_gate_mean": float(output["rigid_gate"].mean()),
        "global_safety_gate_mean": float(output["global_safety_gate"].mean()),
        "uncertainty_mean": float(output["uncertainty"].mean()),
        "velocity_norm_mean": float(atom_velocity.mean()),
        "velocity_graph_rms": float(graph_rms.max()),
        "velocity_atom_max": float(atom_velocity.max()),
        "raw_trust_clipping_fraction": float(clipped.float().mean()),
        "molecule_displacement_mean": float((float(step_size) * graph_rms).mean()),
        "max_atom_displacement_mean": float(displacement_norm.max()),
        "identity_subset_displacement": float(identity),
        "high_flex_torsion_change": 0.0,
    }


def _checkpoint_payload(
    model, optimizer, step: int, resolved: dict, validation: dict | None,
    *, epoch: int, batch_offset: int, active_seconds: float,
    interval_rows: list[dict[str, Any]], frozen_identities: dict,
) -> dict[str, Any]:
    return {
        "schema_version": "ecir-mvr-medium-rescue-v2-checkpoint-v1",
        "model_type": "MCVRModel", "run_mode": "rigid_only_rescue_v2",
        "step": int(step), "global_step": int(step), "config": resolved,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": None,
        "scaler_state_dict": None, "validation": validation,
        "rng_states": _capture_rng(),
        "sampler_state": {"epoch": int(epoch), "batch_offset": int(batch_offset)},
        "timing_accumulator": {
            "active_optimizer_seconds": float(active_seconds),
            "interval_rows": interval_rows,
        },
        "frozen_identities": frozen_identities,
    }


def _loader_settings(config: dict) -> dict[str, Any]:
    training = config["training"]
    settings = {
        "num_workers": int(training["num_workers"]),
        "pin_memory": bool(training.get("pin_memory", True)),
    }
    benchmark_path = Path(config["dataloader_benchmark_result"])
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if benchmark["status"] != "PASS" or benchmark["sample_order_preserved"] is not True:
        raise RuntimeError("DataLoader benchmark is not a sample-order-preserving PASS")
    selected = benchmark["selected"]
    settings.update({
        "num_workers": int(selected["num_workers"]),
        "pin_memory": bool(selected["pin_memory"]),
    })
    if settings["num_workers"] > 0:
        settings.update({
            "persistent_workers": bool(selected["persistent_workers"]),
            "prefetch_factor": int(selected["prefetch_factor"]),
        })
    return settings


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_audit", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume_checkpoint", type=Path)
    parser.add_argument("--controller_resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["experiment_name"] != "ecir_mvr_medium_5k_500_run_a_seed42_20k_rescue_v2":
        raise ValueError("only frozen Medium Seed42 Rescue V2 is authorized")
    training = config["training"]
    if not (
        training["batch_size"] == training["effective_batch_size"] == 8
        and training["gradient_accumulation_steps"] == 1
        and training["optimizer_steps"] == 20000
        and float(training["learning_rate"]) == 0.0002
    ):
        raise ValueError("Rescue V2 scientific training budget changed")
    if config.get("initialize_from_checkpoint") is not None or config.get("resume_checkpoint") is not None:
        raise ValueError("configured training must start from step 0")
    if args.resume_checkpoint is not None and not args.controller_resume:
        raise ValueError("resume is restricted to the overnight controller")

    audit = _assert_identity(config, args.data_audit)
    _seed(int(config["seed"]))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Rescue V2 requires the audited CUDA environment")
    torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(training.get("allow_tf32", True))

    output = Path(config["output_dir"])
    checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True); checkpoints.mkdir(parents=True, exist_ok=True)
    timing = RunTiming(output)
    timing.mark("training_process_start", resume=bool(args.resume_checkpoint))
    (output / "training.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    started_at = iso_now()
    process_started = time.monotonic()
    heartbeat_template = {
        "status": "RUNNING", "pid": os.getpid(), "current_step": 0,
        "target_step": 20000, "started_at": started_at, "elapsed_seconds": 0.0,
        "active_training_seconds": 0.0, "last_validation_step": 0,
        "last_checkpoint": None, "latest_total_loss": None,
        "velocity_graph_rms": 0.0, "velocity_atom_max": 0.0,
        "cuda_allocated_mib": 0.0, "cuda_reserved_mib": 0.0,
        "gpu_utilization": math.nan, "estimated_finish_time": None,
        "latest_warning": None, "latest_error": None,
    }
    write_heartbeat(output, **heartbeat_template)

    log_handle = (output / "training.log").open(
        "a" if args.resume_checkpoint else "w", encoding="utf-8", buffering=1
    )
    def log(message: str) -> None:
        line = f"[{iso_now()}] {message}"
        print(line, flush=True); log_handle.write(line + "\n")

    try:
        config_sha = _sha(args.config)
        git_commit = _git("rev-parse", "HEAD")
        loader_settings = _loader_settings(config)
        resolved = {
            **config,
            "training": {**training, **loader_settings},
            "resolved": {
                "config_sha256": config_sha, "git_commit": git_commit,
                "device": str(device), "gpu": torch.cuda.get_device_name(0),
                "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
                "dataloader_settings": loader_settings,
            },
        }
        (output / "config.resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        validity = ChemicalValidity(config["data"]["validity_statistics"])
        train_data = _dataset(config, "train", validity)
        val_data = _dataset(config, "val", validity)
        train_loader = DataLoader(
            train_data, batch_size=8, shuffle=False, **loader_settings
        )
        val_settings = dict(loader_settings)
        val_loader = DataLoader(
            val_data, batch_size=int(training["val_batch_size"]), shuffle=False, **val_settings
        )
        val_items = build_items(config["data"]["val_sources"], config["data"]["val_targets"], validity)
        clean_control_items = build_clean_control_items(val_items, validity, limit=20)
        if len(clean_control_items) < 10:
            raise RuntimeError("fewer than 10 clean validation-reference controls")
        model = MCVRModel(**config["model"]).to(device)
        loss_fn = MCVRLoss(config["loss"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        resume_payload = None
        start_step = 0
        active_seconds = 0.0
        interval_rows: list[dict[str, Any]] = []
        prior_training_seconds = 0.0
        validation_history: list[dict[str, Any]] = []
        if args.resume_checkpoint:
            resume_payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
            if resume_payload.get("schema_version") != "ecir-mvr-medium-rescue-v2-checkpoint-v1":
                raise RuntimeError("resume checkpoint schema mismatch")
            if resume_payload["config"]["resolved"]["config_sha256"] != config_sha:
                raise RuntimeError("resume config identity mismatch")
            if resume_payload["frozen_identities"] != config["frozen_identities"]:
                raise RuntimeError("resume frozen identity mismatch")
            start_step = int(resume_payload["step"])
            if not 0 < start_step < 20000 or start_step % 1000:
                raise RuntimeError("resume checkpoint is not a complete 1000-step recovery point")
            model.load_state_dict(resume_payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            active_seconds = float(resume_payload["timing_accumulator"]["active_optimizer_seconds"])
            interval_rows = list(resume_payload["timing_accumulator"]["interval_rows"])
            prior_training_seconds = float(interval_rows[-1]["cumulative_training_seconds"]) if interval_rows else 0.0
            if resume_payload.get("validation"):
                validation_history.append(resume_payload["validation"])
            log(f"CONTROLLER_RESUME step={start_step} checkpoint={args.resume_checkpoint}")

        metadata_path = output / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() and args.resume_checkpoint else {}
        metadata.update({
            "status": "RUNNING", "experiment_name": config["experiment_name"],
            "seed": 42, "optimizer_steps": 20000, "batch_size": 8,
            "effective_batch_size": 8, "learning_rate": 0.0002,
            "config_sha256": config_sha, "git_commit": git_commit,
            "data_audit_identity": audit["identity_sha256"],
            "frozen_identities": config["frozen_identities"],
            "host": platform.node(), "platform": platform.platform(),
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda), "gpu": torch.cuda.get_device_name(0),
            "test_records_read": 0, "20k_started": True, "100k_started": False,
            "started_at": metadata.get("started_at", started_at),
            "resumed": bool(args.resume_checkpoint), "resumed_from_step": start_step or None,
            "dataloader_settings": loader_settings,
        })
        atomic_json_save(metadata, metadata_path)

        metrics_path = output / "metrics.csv"
        metrics_handle = metrics_path.open("a" if args.resume_checkpoint else "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(metrics_handle, fieldnames=METRIC_FIELDS)
        if not args.resume_checkpoint:
            writer.writeheader()
        gpu_path = output / "gpu_metrics.csv"
        gpu_rows: list[dict[str, Any]] = (
            pd.read_csv(gpu_path).to_dict("records") if args.resume_checkpoint and gpu_path.is_file() else []
        )
        comparison_rows: list[dict[str, Any]] = []
        comparison_path = Path(config["diagnostics_dir"]) / "checkpoint_comparison.csv"
        if args.resume_checkpoint and comparison_path.is_file():
            comparison_rows = pd.read_csv(comparison_path).to_dict("records")

        batches_per_epoch = len(train_loader)
        epoch, batch_offset = divmod(start_step, batches_per_epoch)
        if resume_payload:
            saved_sampler = resume_payload["sampler_state"]
            if (epoch, batch_offset) != (int(saved_sampler["epoch"]), int(saved_sampler["batch_offset"])):
                raise RuntimeError("resume sampler position does not match global step")
        train_data.set_epoch(epoch)
        iterator = iter(train_loader)
        for _ in range(batch_offset):
            next(iterator)
        if resume_payload:
            _restore_rng(resume_payload["rng_states"])
        train_window: list[dict[str, float]] = []
        diagnostic_history: list[dict[str, float]] = []
        clipping_history: list[float] = []
        best: dict[str, Any] | None = None
        checkpoint_steps = set(range(1000, 20001, 1000))
        validation_steps = set(int(value) for value in training["checkpoint_validation_steps"])
        last_heartbeat = time.monotonic()
        interval_started = time.monotonic()
        interval_active_start = active_seconds
        interval_gpu_start = len(gpu_rows)
        interval_validation_start = timing.event_seconds("validation_start", "validation_end")
        seen = epoch * len(train_data) + min(batch_offset * 8, len(train_data))
        interval_seen_start = seen
        stop_reason = None
        warning = None
        latest_loss = None
        last_validation_step = validation_history[-1]["step"] if validation_history else 0
        last_checkpoint = str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None
        torch.cuda.reset_peak_memory_stats()
        timing.mark("active_optimizer_start", start_step=start_step)
        model.train()
        step = start_step
        completed_step = start_step
        for step in range(start_step + 1, 20001):
            if step % 50 == 1:
                _assert_identity(config, args.data_audit)
            load_started = time.monotonic()
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1; batch_offset = 0
                train_data.set_epoch(epoch); iterator = iter(train_loader); batch = next(iterator)
            batch_offset += 1
            batch = batch.to(device, non_blocking=bool(loader_settings["pin_memory"]))
            optimizer_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            losses = loss_fn(model, batch)
            if not all(bool(torch.isfinite(value)) for value in losses.values()):
                stop_reason = "nan_or_inf_loss"; break
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            if not bool(torch.isfinite(grad_norm)):
                stop_reason = "gradient_nonfinite"; break
            optimizer.step()
            torch.cuda.synchronize()
            active_seconds += time.monotonic() - optimizer_started
            completed_step = step
            seen += int(batch.num_graphs)
            train_window.append({name: float(_loss_value(losses, name).detach()) for name in LOSS_NAMES})

            latest_loss = float(losses["loss"].detach())
            latest_diag = diagnostic_history[-1] if diagnostic_history else {
                "velocity_graph_rms": 0.0, "velocity_atom_max": 0.0,
            }
            if step == 1 or step % int(training["log_interval"]) == 0:
                diag = _diagnostics(model, batch)
                row = {
                    "step": step, "split": "train",
                    **{name: float(np.mean([value[name] for value in train_window])) for name in LOSS_NAMES},
                    **diag, "records_per_second": seen / max(time.monotonic() - process_started, 1e-9),
                }
                writer.writerow(row); metrics_handle.flush(); train_window.clear()
                diagnostic_history.append(row); latest_diag = row
                clipping_history.append(row["raw_trust_clipping_fraction"])
                clipping_history = clipping_history[-int(config["safety"]["trust_clipping_windows"]):]
                recent_velocity = diagnostic_history[-5:]
                if len(recent_velocity) == 5 and all(
                    recent_velocity[index]["velocity_norm_mean"] < recent_velocity[index + 1]["velocity_norm_mean"]
                    for index in range(4)
                ) and recent_velocity[-1]["velocity_norm_mean"] > 2.0 * recent_velocity[0]["velocity_norm_mean"]:
                    warning = "INFO velocity_norm_sustained_growth_below_hard_limits"
                    log(f"{warning} step={step} velocity={row['velocity_norm_mean']:.6f}")
                if row["velocity_graph_rms"] >= float(config["safety"]["max_velocity_graph_rms"]):
                    stop_reason = "velocity_graph_rms_hard_limit"
                elif row["velocity_atom_max"] >= float(config["safety"]["max_velocity_atom_norm"]):
                    stop_reason = "velocity_atom_max_hard_limit"
                elif len(clipping_history) == int(config["safety"]["trust_clipping_windows"]) and all(
                    value >= float(config["safety"]["trust_clipping_fraction_hard"]) for value in clipping_history
                ):
                    stop_reason = "sustained_large_area_trust_clipping"
                log(
                    f"step={step} loss={row['total_loss']:.6f} velocity_mean={row['velocity_norm_mean']:.6f} "
                    f"velocity_graph_rms={row['velocity_graph_rms']:.6f} velocity_atom_max={row['velocity_atom_max']:.6f} "
                    f"trust_clipping={row['raw_trust_clipping_fraction']:.4f}"
                )
                telemetry = _gpu_telemetry(step); gpu_rows.append(telemetry)
                _write_csv(gpu_path, gpu_rows, GPU_FIELDS)
            if stop_reason:
                log(f"SAFETY_STOP {stop_reason} step={step}"); break

            validation = None
            if step in validation_steps:
                _assert_identity(config, args.data_audit)
                timing.mark("validation_start", step=step)
                validation_started = time.monotonic()
                val_losses = _validate_losses(model, loss_fn, val_loader, device)
                full = evaluate_run_a_only(
                    model, val_items, validity, device=device,
                    inference=config["inference"], margins=config["noninferiority"],
                    bootstrap_draws=500, clean_control_items=clean_control_items,
                )
                full["summary"].to_csv(output / f"validation_step{step:06d}.csv", index=False)
                validation = {
                    "step": step, "losses": val_losses, "accuracy_gate": full["accuracy_gate"],
                    "accuracy_noninferior": full["accuracy_noninferior"],
                    "validity_delta": full["validity_delta"],
                    "mean_displacement": full["mean_displacement"],
                    "acceptance_fraction": full["acceptance_fraction"],
                    "validity_worsened_fraction": full["validity_worsened_fraction"],
                    "chirality_delta": full["chirality_delta"],
                    "severe_clash_delta": full["severe_clash_delta"],
                    "high_flex_torsion_change": full["high_flex_torsion_change"],
                    "identity_fraction": full["identity_fraction"],
                    "torsion_gate_max": full["torsion_gate_max"],
                    "torsion_contribution_max": full["torsion_contribution_max"],
                    "bootstrap": full["bootstrap"],
                }
                summary_index = full["summary"].set_index(["group", "method"])
                for group, prefix in (("rotatable_ge_6", "high_flex"), ("unseen_update_scale_0.35", "unseen")):
                    candidate = summary_index.loc[(group, "run_a_accepted")]
                    upstream = summary_index.loc[(group, "upstream")]
                    validation[f"{prefix}_validity_delta"] = float(candidate.total_thresholded_validity_score - upstream.total_thresholded_validity_score)
                    validation[f"{prefix}_rmsd_delta"] = float(candidate.aligned_RMSD - upstream.aligned_RMSD)
                validation_history.append(validation); last_validation_step = step
                writer.writerow({"step": step, "split": "val", **val_losses, "records_per_second": ""}); metrics_handle.flush()
                timing.mark("validation_end", step=step, seconds=time.monotonic() - validation_started)
                log(
                    f"validation step={step} validity_delta={full['validity_delta']:.6f} "
                    f"rmsd_delta={full['bootstrap']['aligned_RMSD']['mean']:.6f} "
                    f"identity={full['identity_fraction']:.6f}"
                )
                if full["torsion_gate_max"] != 0.0 or full["torsion_contribution_max"] != 0.0:
                    stop_reason = "torsion_branch_nonzero"
                elif math.isfinite(full["identity_fraction"]) and full["identity_fraction"] < 0.90:
                    stop_reason = "clean_identity_below_90pct"
                elif full["chirality_delta"] > 1e-9:
                    stop_reason = "chirality_worsened"
                elif full["severe_clash_delta"] > 1e-9:
                    stop_reason = "severe_clash_increased"
                if len(validation_history) >= 3:
                    transition_flags = []
                    for previous, current in zip(validation_history[-3:-1], validation_history[-2:]):
                        accuracy_worse = all(
                            current["bootstrap"][metric]["mean"] > 0.0
                            for metric in ("aligned_RMSD", "MAT_P", "MAT_R")
                        )
                        transition_flags.append(
                            current["validity_delta"] > 0.0
                            and current["mean_displacement"] > previous["mean_displacement"] + 1e-4
                            and accuracy_worse
                        )
                    if all(transition_flags):
                        stop_reason = "two_validations_joint_validity_displacement_accuracy_worsening"

            if step in checkpoint_steps:
                interval_end = time.monotonic()
                interval_gpu = gpu_rows[interval_gpu_start:]
                util = np.asarray([row["gpu_utilization"] for row in interval_gpu], dtype=float)
                util = util[np.isfinite(util)]
                interval_seconds = interval_end - interval_started
                step_start = max(start_step, step - 1000)
                interval_steps = step - step_start
                wall_rate = interval_steps / max(interval_seconds, 1e-9)
                remaining = 20000 - step
                eta_seconds = remaining / max(wall_rate, 1e-9)
                interval_rows.append({
                    "step_start": step_start, "step_end": step,
                    "interval_seconds": interval_seconds,
                    "cumulative_training_seconds": prior_training_seconds + interval_end - process_started,
                    "active_optimizer_seconds": active_seconds - interval_active_start,
                    "validation_seconds": timing.event_seconds("validation_start", "validation_end") - interval_validation_start,
                    "steps_per_second": interval_steps / max(active_seconds - interval_active_start, 1e-9),
                    "examples_per_second": (seen - interval_seen_start) / max(active_seconds - interval_active_start, 1e-9),
                    "ETA_seconds": eta_seconds,
                    "ETA_finish_time": (datetime.now().astimezone() + timedelta(seconds=eta_seconds)).isoformat(timespec="seconds"),
                    "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                    "gpu_utilization_mean": float(util.mean()) if util.size else math.nan,
                    "gpu_utilization_p95": float(np.quantile(util, 0.95)) if util.size else math.nan,
                })
                timing.write_intervals(interval_rows)
                timing.mark("checkpoint_save_start", step=step)
                checkpoint_started = time.monotonic()
                payload = _checkpoint_payload(
                    model, optimizer, step, resolved, validation,
                    epoch=epoch, batch_offset=batch_offset, active_seconds=active_seconds,
                    interval_rows=interval_rows, frozen_identities=config["frozen_identities"],
                )
                checkpoint_path = checkpoints / f"step{step:06d}.ckpt"
                atomic_torch_save(payload, checkpoint_path); last_checkpoint = str(checkpoint_path.resolve())
                if validation is not None:
                    candidate_key = (
                        round(float(validation["validity_delta"]), 6),
                        float(validation["mean_displacement"]),
                        -float(validation["identity_fraction"] if math.isfinite(validation["identity_fraction"]) else 0.0),
                        float(validation["high_flex_validity_delta"]), float(validation["unseen_validity_delta"]),
                    )
                    if validation["accuracy_noninferior"] and (best is None or candidate_key < best["key"]):
                        best = {"step": step, "key": candidate_key, "validation": validation}
                        atomic_torch_save(payload, checkpoints / "best_noninferior_validity.ckpt")
                    comparison_rows.append({
                        "step": step, "checkpoint": str(checkpoint_path.resolve()),
                        "checkpoint_sha256": _sha(checkpoint_path),
                        "accuracy_noninferior": validation["accuracy_noninferior"],
                        "validity_delta": validation["validity_delta"],
                        "mean_displacement": validation["mean_displacement"],
                        "identity_fraction": validation["identity_fraction"],
                        "rmsd_delta": validation["bootstrap"]["aligned_RMSD"]["mean"],
                        "mat_p_delta": validation["bootstrap"]["MAT_P"]["mean"],
                        "mat_r_delta": validation["bootstrap"]["MAT_R"]["mean"],
                        "high_flex_validity_delta": validation["high_flex_validity_delta"],
                        "unseen_validity_delta": validation["unseen_validity_delta"],
                    })
                    comparison_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
                timing.mark("checkpoint_save_end", step=step, seconds=time.monotonic() - checkpoint_started)

                interval_started = time.monotonic(); interval_active_start = active_seconds
                interval_seen_start = seen; interval_gpu_start = len(gpu_rows)
                interval_validation_start = timing.event_seconds("validation_start", "validation_end")
            now = time.monotonic()
            if now - last_heartbeat >= 60.0 or step in checkpoint_steps:
                telemetry = gpu_rows[-1] if gpu_rows else _gpu_telemetry(step)
                elapsed = now - process_started
                rate = (step - start_step) / max(elapsed, 1e-9)
                eta = (20000 - step) / max(rate, 1e-9)
                write_heartbeat(
                    output, status="RUNNING", pid=os.getpid(), current_step=step,
                    target_step=20000, started_at=heartbeat_template["started_at"],
                    elapsed_seconds=elapsed, active_training_seconds=active_seconds,
                    last_validation_step=last_validation_step, last_checkpoint=last_checkpoint,
                    latest_total_loss=latest_loss,
                    velocity_graph_rms=latest_diag["velocity_graph_rms"],
                    velocity_atom_max=latest_diag["velocity_atom_max"],
                    cuda_allocated_mib=telemetry["torch_allocated_mib"],
                    cuda_reserved_mib=telemetry["torch_reserved_mib"],
                    gpu_utilization=telemetry["gpu_utilization"],
                    estimated_finish_time=(datetime.now().astimezone() + timedelta(seconds=eta)).isoformat(timespec="seconds"),
                    latest_warning=warning, latest_error=None,
                )
                last_heartbeat = now
            if stop_reason:
                log(f"SAFETY_STOP {stop_reason} step={step}"); break

        final_step = completed_step
        timing.mark("active_optimizer_end", final_step=final_step)
        final_payload = _checkpoint_payload(
            model, optimizer, final_step, resolved,
            validation_history[-1] if validation_history else None,
            epoch=epoch, batch_offset=batch_offset, active_seconds=active_seconds,
            interval_rows=interval_rows, frozen_identities=config["frozen_identities"],
        )
        timing.mark("checkpoint_save_start", step=final_step, kind="last")
        atomic_torch_save(final_payload, checkpoints / "last.ckpt")
        timing.mark("checkpoint_save_end", step=final_step, kind="last")
        status = "COMPLETED" if final_step == 20000 and stop_reason is None else "SAFETY_STOPPED"
        timing.mark("training_process_end", status=status, final_step=final_step)
        all_gpu = np.asarray([row["gpu_utilization"] for row in gpu_rows], dtype=float)
        all_gpu = all_gpu[np.isfinite(all_gpu)]
        card_used = np.asarray([row["card_memory_used_mib"] for row in gpu_rows], dtype=float)
        card_used = card_used[np.isfinite(card_used)]
        timing_state = timing.finalize(
            completed_optimizer_steps=final_step, batch_size=8,
            active_optimizer_seconds=active_seconds, interval_rows=interval_rows,
            extra={
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "peak_card_memory_used_mib": float(card_used.max()) if card_used.size else math.nan,
                "gpu_utilization_mean": float(all_gpu.mean()) if all_gpu.size else math.nan,
                "gpu_utilization_p95": float(np.quantile(all_gpu, 0.95)) if all_gpu.size else math.nan,
                "examples_seen": int(seen),
                "mean_examples_per_second": seen / active_seconds if active_seconds > 0 else 0.0,
            },
        )
        metadata.update({
            "status": status, "completed_steps": final_step, "stop_reason": stop_reason,
            "20k_completed": status == "COMPLETED", "completed_at": iso_now(),
            "active_optimizer_seconds": active_seconds,
            "best_noninferior_step": best["step"] if best else None,
            "peak_cuda_allocated_mib": timing_state["peak_cuda_allocated_mib"],
            "peak_cuda_reserved_mib": timing_state["peak_cuda_reserved_mib"],
        })
        atomic_json_save(metadata, metadata_path)
        write_heartbeat(
            output, status=status, current_step=final_step, target_step=20000,
            elapsed_seconds=time.monotonic() - process_started,
            active_training_seconds=active_seconds, last_validation_step=last_validation_step,
            last_checkpoint=str((checkpoints / "last.ckpt").resolve()),
            latest_total_loss=latest_loss, latest_warning=warning,
            latest_error=stop_reason if status == "SAFETY_STOPPED" else None,
        )
        atomic_json_save({"resume_allowed": False, "reason": status}, output / "resume_control.json")
        metrics_handle.close(); log(f"finished status={status} step={final_step} stop_reason={stop_reason}")
        log_handle.close()
        print(json.dumps(metadata, indent=2))
    except BaseException as error:
        message = f"{type(error).__name__}: {error}"
        oom = isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
        recoverable = isinstance(error, OSError) and not oom
        status = "FAILED"
        write_heartbeat(output, status=status, latest_error=message)
        atomic_json_save(
            {"resume_allowed": bool(recoverable), "reason": "temporary_io_error" if recoverable else "nonrecoverable_error", "error": message},
            output / "resume_control.json",
        )
        try:
            timing.mark("training_process_end", status=status, error=message)
        finally:
            log(f"FAILED {message}"); log_handle.close()
        raise


if __name__ == "__main__":
    main()
