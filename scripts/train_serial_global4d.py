#!/usr/bin/env python
"""Train Serial Global4D Phase A coefficients or Phase B benefit gate."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from etflow.commons.global_coupled_4d_sampling import (
    atomic_json_save,
    atomic_torch_save,
)
from etflow.serial_global4d.cache import SerialGlobal4DResidualDataset
from etflow.serial_global4d.model import SerialGlobal4DResidualRefiner


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model(config: dict[str, Any]) -> SerialGlobal4DResidualRefiner:
    return SerialGlobal4DResidualRefiner(**config["model"])


def _load_checkpoint(
    path: Path,
    model: SerialGlobal4DResidualRefiner,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(
        payload["model_state_dict"], strict=True
    )
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload.get("step", 0)), payload


def _graph_metrics(batch, output: dict[str, torch.Tensor]) -> dict[str, float]:
    atom_batch = output["atom_batch"]
    graphs = int(batch.num_graphs)
    target = batch.r_J_star.to(output["v_internal"])
    residual = batch.u_stage2.to(output["v_internal"])
    prediction = output["v_internal"].detach()
    target = target.detach()
    residual = residual.detach()
    dot_target = prediction.new_zeros(graphs)
    pred_energy = prediction.new_zeros(graphs)
    target_energy = prediction.new_zeros(graphs)
    residual_energy = prediction.new_zeros(graphs)
    corrected_energy = prediction.new_zeros(graphs)
    internal_error_energy = prediction.new_zeros(graphs)
    dot_target.index_add_(0, atom_batch, (prediction * target).sum(-1))
    pred_energy.index_add_(0, atom_batch, prediction.square().sum(-1))
    target_energy.index_add_(0, atom_batch, target.square().sum(-1))
    residual_energy.index_add_(0, atom_batch, residual.square().sum(-1))
    corrected_energy.index_add_(0, atom_batch, (residual - prediction).square().sum(-1))
    internal_error_energy.index_add_(
        0, atom_batch, (target - prediction).square().sum(-1)
    )
    coordinate_counts = (
        3.0 * torch.bincount(atom_batch, minlength=graphs).to(prediction.dtype)
    ).clamp_min(1.0)
    cosine = dot_target / (pred_energy * target_energy).sqrt().clamp_min(1.0e-12)
    gain = residual_energy - corrected_energy
    norm_ratio = pred_energy.sqrt() / target_energy.sqrt().clamp_min(1.0e-12)
    return {
        "internal_cosine": float(cosine.mean()),
        "predicted_target_norm_ratio": float(norm_ratio.mean()),
        "raw_gain_mean": float(gain.mean()),
        "positive_gain_fraction": float((gain > 0).float().mean()),
        "negative_gain_fraction": float((gain < 0).float().mean()),
        "q_norm": (
            float(torch.linalg.vector_norm(output["q_pred"].detach(), dim=-1).mean())
            if output["q_pred"].numel()
            else 0.0
        ),
        "internal_mse": float((internal_error_energy / coordinate_counts).mean()),
        "zero_predictor_internal_mse": float(
            (target_energy / coordinate_counts).mean()
        ),
    }


@torch.no_grad()
def _validate(model, loader, phase: str, config: dict, device: str) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    records = 0
    for batch in loader:
        batch = batch.to(device)
        output = (
            model.phase_a_loss(
                batch,
                coefficient_weight=config["loss"]["coefficient_weight"],
                internal_weight=config["loss"]["internal_weight"],
            )
            if phase == "a"
            else model.phase_b_loss(batch)
        )
        metrics = _graph_metrics(batch, output)
        q_loss = float(output.get("q_loss", output["loss"].new_zeros(())))
        internal_loss = metrics["internal_mse"]
        gate_loss = float(output.get("gate_loss", output["loss"].new_zeros(())))
        validation_loss = (
            float(config["loss"]["coefficient_weight"]) * q_loss
            + float(config["loss"]["internal_weight"]) * internal_loss
            if phase == "a"
            else gate_loss
        )
        metrics.update(
            {
                "loss": validation_loss,
                "q_loss": q_loss,
                "internal_loss": internal_loss,
                "gate_loss": gate_loss,
                "gate_mean": float(output["gate"].mean()),
            }
        )
        if phase == "b":
            metrics["gate_target_mean"] = float(output["gate_target"].mean())
            metrics["gate_calibration_mae"] = float(
                (output["gate"] - output["gate_target"]).abs().mean()
            )
        weight = int(batch.num_graphs)
        records += weight
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * weight
    model.train()
    return {key: value / records for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--phase", choices=("a", "b"), default="a")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    phase = args.phase
    seed = int(config["training"]["seed"])
    _seed(seed)
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = SerialGlobal4DResidualDataset(
        config["data"]["train_cache_root"], "train", require_completed=True
    )
    val_dataset = SerialGlobal4DResidualDataset(
        config["data"]["val_cache_root"], "val", require_completed=True
    )
    batch_size = int(config["training"]["batch_size"])
    loader_kwargs = {
        "num_workers": int(config["data"].get("num_workers", 0)),
        "persistent_workers": bool(config["data"].get("num_workers", 0)),
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("val_batch_size", batch_size)),
        shuffle=False,
        num_workers=0,
    )
    model = _model(config).to(args.device)
    if phase == "a":
        model.gate_head.requires_grad_(False)
    else:
        if args.checkpoint is None:
            raise ValueError("Phase B requires a Phase A --checkpoint")
        _load_checkpoint(args.checkpoint, model)
        model.freeze_for_phase_b()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 1.0e-6)),
    )
    start_step = 0
    if args.resume:
        if args.checkpoint is None:
            raise ValueError("--resume requires --checkpoint")
        start_step, _ = _load_checkpoint(args.checkpoint, model, optimizer)
    max_steps = int(args.max_steps or config["training"][f"phase_{phase}_max_steps"])
    val_interval = int(config["training"].get("validation_interval", 250))
    checkpoint_interval = int(config["training"].get("checkpoint_interval", 250))
    log_interval = int(config["training"].get("log_interval", 10))
    history = []
    iterator = iter(train_loader)
    model.train()
    torch.cuda.reset_peak_memory_stats() if args.device.startswith("cuda") else None
    wall_started = time.perf_counter()
    for step in range(start_step + 1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = batch.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        output = (
            model.phase_a_loss(
                batch,
                coefficient_weight=config["loss"]["coefficient_weight"],
                internal_weight=config["loss"]["internal_weight"],
            )
            if phase == "a"
            else model.phase_b_loss(batch)
        )
        if not bool(torch.isfinite(output["loss"])):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        output["loss"].backward()
        finite_grad = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        if not finite_grad:
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(config["training"].get("grad_clip", 1.0))
        )
        optimizer.step()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        if step == 1 or step % log_interval == 0 or step == max_steps:
            metrics = _graph_metrics(batch, output)
            row = {
                "step": step,
                "split": "train",
                "phase": phase,
                "loss": float(output["loss"].detach()),
                "q_loss": float(
                    output.get("q_loss", output["loss"].new_zeros(())).detach()
                ),
                "internal_loss": float(
                    output.get("internal_loss", output["loss"].new_zeros(())).detach()
                ),
                "gate_loss": float(
                    output.get("gate_loss", output["loss"].new_zeros(())).detach()
                ),
                "gate_mean": float(output["gate"].detach().mean()),
                "grad_norm": float(grad_norm.detach()),
                "step_seconds": time.perf_counter() - step_started,
                "records_per_second": int(batch.num_graphs)
                / max(time.perf_counter() - step_started, 1.0e-12),
                **metrics,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % val_interval == 0 or step == max_steps:
            metrics = _validate(model, val_loader, phase, config, args.device)
            row = {"step": step, "split": "val", "phase": phase, **metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % checkpoint_interval == 0 or step == max_steps:
            atomic_torch_save(
                {
                    "model_type": "serial_global4d_residual_refiner",
                    "phase": phase,
                    "step": step,
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                run_dir / f"step{step:06d}.ckpt",
            )
    fields = sorted({key for row in history for key in row})
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "status": "COMPLETED",
        "phase": phase,
        "steps": max_steps,
        "batch_size": batch_size,
        "lr": float(config["training"]["lr"]),
        "wall_seconds": time.perf_counter() - wall_started,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda")
            else None
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None
        ),
        "last_train": next(row for row in reversed(history) if row["split"] == "train"),
        "last_val": next(row for row in reversed(history) if row["split"] == "val"),
        "checkpoint": str(run_dir / f"step{max_steps:06d}.ckpt"),
    }
    atomic_json_save(summary, run_dir / "summary.json")


if __name__ == "__main__":
    main()
