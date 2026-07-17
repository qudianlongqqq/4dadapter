#!/usr/bin/env python
"""Fit Stage G on contiguous tensors with explicit CUDA residency and large batches."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np
import pandas as pd
import torch
import yaml

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.ecir.bounded_residual_confidence import (
    BoundedResidualSignSafeCalibrator,
    calibrator_identity_payload,
    checkpoint_selection_priority,
    dataframe_stage_g_tensors,
    feature_view,
    move_tensor_bundle,
    pin_tensor_bundle,
    select_stage_g_checkpoint,
    selection_metrics,
    stage_g_loss,
    tensor_bundle_nbytes,
    validate_stage_g_frame,
    validate_stage_g_manifest,
    verify_stage_f_identity,
)


def _model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config["calibrator"]
    return {
        name: value[name]
        for name in (
            "hidden_dim",
            "num_layers",
            "bond_type_embedding_dim",
            "element_pair_embedding_dim",
            "element_pair_buckets",
            "time_embedding_dim",
            "min_multiplier",
            "max_multiplier",
            "epsilon",
            "dropout",
        )
    }


def resolve_dataset_residency(
    requested: str,
    *,
    device: torch.device,
    tensors: Mapping[str, torch.Tensor],
    cuda_headroom_fraction: float = 0.25,
) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("dataset residency must be auto, cpu, or cuda")
    if requested == "cuda" and device.type != "cuda":
        raise ValueError("dataset-residency=cuda requires --device cuda")
    if requested == "cpu" or device.type != "cuda":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    free_bytes, _ = torch.cuda.mem_get_info(device)
    required = tensor_bundle_nbytes(tensors)
    return "cuda" if required <= free_bytes * (1.0 - float(cuda_headroom_fraction)) else "cpu"


def _take_batch(
    bundle: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    storage = next(iter(bundle.values())).device
    local_indices = indices.to(storage, non_blocking=storage.type == "cuda")
    selected = {name: value.index_select(0, local_indices) for name, value in bundle.items()}
    if storage != device:
        selected = move_tensor_bundle(selected, device, non_blocking=True)
    return selected


@torch.inference_mode()
def evaluate_holdout(
    calibrator: BoundedResidualSignSafeCalibrator,
    frame: pd.DataFrame,
    bundle: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    threshold: float,
) -> dict[str, Any]:
    confidences, multipliers = [], []
    for start in range(0, len(frame), int(batch_size)):
        stop = min(start + int(batch_size), len(frame))
        indices = torch.arange(start, stop, device=next(iter(bundle.values())).device)
        batch = _take_batch(bundle, indices, device=device)
        confidence, multiplier, _ = calibrator.forward_components(feature_view(batch))
        confidences.append(confidence.detach().cpu())
        multipliers.append(multiplier.detach().cpu())
    confidence_array = torch.cat(confidences).numpy()
    multiplier_array = torch.cat(multipliers).numpy()
    return selection_metrics(
        frame, confidence_array, multiplier_array, threshold=float(threshold)
    )


def _cuda_profile_row(
    *,
    device: torch.device,
    step: int,
    batch_size: int,
    rows_processed: int,
    step_seconds: float,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "batch_size": int(batch_size),
        "rows_processed": int(rows_processed),
        "step_seconds": float(step_seconds),
        "rows_per_second": float(batch_size / max(step_seconds, 1.0e-12)),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device_name": torch.cuda.get_device_name(device),
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    pd.read_csv(temporary)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ecir_mvr_stage_g_bounded_residual.yaml")
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dataset-residency", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--profile-cuda-memory", action="store_true", default=None)
    parser.add_argument("--profile-every-steps", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    verify_stage_f_identity(config)
    source = args.input_dir or Path(config["output_dir"])
    output = args.output_dir or source
    manifest = json.loads((source / "calibration_manifest.json").read_text(encoding="utf-8"))
    validate_stage_g_manifest(manifest)
    frame = pd.read_parquet(source / "calibration_dataset.parquet")
    validate_stage_g_frame(frame, manifest)
    training = config["training"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage G requested CUDA but CUDA is unavailable")
    batch_size = int(args.batch_size or training["batch_size"])
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    num_workers = int(
        args.num_workers if args.num_workers is not None else training["num_workers"]
    )
    if num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    pin_memory = bool(
        args.pin_memory if args.pin_memory is not None else training["pin_memory"]
    )
    profile = bool(
        args.profile_cuda_memory
        if args.profile_cuda_memory is not None
        else training["profile_cuda_memory"]
    )
    profile_every = int(args.profile_every_steps or training["profile_every_steps"])
    if profile_every < 1:
        raise ValueError("profile-every-steps must be positive")
    seed = int(args.seed if args.seed is not None else config["seed"])
    if seed != int(manifest["seed"]):
        raise RuntimeError("Stage G training seed differs from calibration manifest")
    fit = frame[frame.split.eq("fit")].reset_index(drop=True)
    check = frame[frame.split.eq("internal_check")].reset_index(drop=True)
    fit_bundle = dataframe_stage_g_tensors(fit)
    check_bundle = dataframe_stage_g_tensors(check)
    residency_request = args.dataset_residency or training["dataset_residency"]
    residency = resolve_dataset_residency(
        residency_request,
        device=device,
        tensors={**{f"fit_{k}": v for k, v in fit_bundle.items()}, **{f"check_{k}": v for k, v in check_bundle.items()}},
    )
    if residency == "cuda":
        fit_bundle = move_tensor_bundle(fit_bundle, device)
        check_bundle = move_tensor_bundle(check_bundle, device)
        pin_memory = False
    elif pin_memory and device.type == "cuda":
        fit_bundle = pin_tensor_bundle(fit_bundle)
        check_bundle = pin_tensor_bundle(check_bundle)
    if residency == "cpu" and num_workers > 0:
        torch.set_num_threads(max(1, num_workers))
    model_config = _model_config(config)
    calibrator = BoundedResidualSignSafeCalibrator(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        calibrator.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    max_steps = int(args.max_steps or training["optimizer_steps"])
    checkpoint_steps = sorted(
        {int(step) for step in training["checkpoint_steps"] if int(step) <= max_steps}
        | {max_steps}
    )
    monitor_interval = int(training["monitor_interval"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    storage_device = next(iter(fit_bundle.values())).device
    generator = torch.Generator(device=storage_device.type).manual_seed(seed)
    if profile and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for step in range(1, max_steps + 1):
            started = time.perf_counter()
            indices = torch.randint(
                0, len(fit), (batch_size,), generator=generator, device=storage_device
            )
            batch = _take_batch(fit_bundle, indices, device=device)
            features = feature_view(batch)
            confidence, multiplier, _ = calibrator.forward_components(features)
            false_positive = batch["zero_target"] | batch["already_valid_unsafe"]
            loss, parts = stage_g_loss(
                confidence,
                multiplier,
                optimal_scale=batch["optimal_scale"],
                scale_weight=batch["scale_weight"],
                wrong_sign=batch["wrong_sign"],
                false_positive=false_positive,
                beneficial=batch["beneficial"],
                molecule_ids=batch["molecule_code"],
                lambda_wrong_sign=training["lambda_wrong_sign"],
                lambda_false_positive=training["lambda_false_positive"],
                lambda_overactivation=training["lambda_overactivation"],
                lambda_rank=training["lambda_rank"],
                lambda_beneficial_recall=training["lambda_beneficial_recall"],
                lambda_multiplier_identity=training["lambda_multiplier_identity"],
                beneficial_confidence_floor=training["beneficial_confidence_floor"],
                smooth_l1_beta=training["smooth_l1_beta"],
                rank_margin=training["rank_margin"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            should_profile = profile and device.type == "cuda" and (
                step == 1 or step % profile_every == 0 or step == max_steps
            )
            if should_profile:
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                profile_rows.append(
                    _cuda_profile_row(
                        device=device,
                        step=step,
                        batch_size=batch_size,
                        rows_processed=step * batch_size,
                        step_seconds=elapsed,
                    )
                )
            evaluate = (
                step == 1
                or step % monitor_interval == 0
                or step in checkpoint_steps
                or step == max_steps
            )
            if evaluate:
                metrics = evaluate_holdout(
                    calibrator,
                    check,
                    check_bundle,
                    device=device,
                    batch_size=batch_size,
                    threshold=float(training["reporting_activation_threshold"]),
                )
                row = {
                    "step": step,
                    "checkpoint": step in checkpoint_steps,
                    "batch_size": batch_size,
                    "dataset_residency": residency,
                    "loss": float(loss.detach()),
                    **{name: float(value.detach()) for name, value in parts.items()},
                    **metrics,
                }
                history.append(row)
                if step in checkpoint_steps:
                    checkpoint_path = checkpoint_dir / f"step{step:06d}.ckpt"
                    torch.save(
                        {
                            "step": step,
                            "calibrator_state_dict": calibrator.state_dict(),
                            "model_config": model_config,
                            "selection_metrics": metrics,
                        },
                        checkpoint_path,
                    )
                    selection_rows.append({**row, "checkpoint_path": checkpoint_path.as_posix()})
    except torch.cuda.OutOfMemoryError:
        print(
            json.dumps(
                {
                    "status": "CUDA_OOM",
                    "batch_size": batch_size,
                    "dataset_residency": residency,
                }
            )
        )
        raise
    _atomic_csv(pd.DataFrame(history), output / "training_history.csv")
    _atomic_csv(pd.DataFrame(selection_rows), output / "checkpoint_selection.csv")
    if profile_rows:
        _atomic_csv(pd.DataFrame(profile_rows), output / "cuda_memory_profile.csv")
    selected = select_stage_g_checkpoint(selection_rows)
    if selected is None:
        result = {
            "schema_version": "ecir-mvr-stage-g-fit-v1",
            "decision": "STAGE_G_COLLAPSED",
            "all_checkpoints_collapsed": True,
            "selected_step": None,
            "batch_size": batch_size,
            "dataset_residency": residency,
            "device": str(device),
            "validation_used": False,
            "test_records_read": 0,
            "next_command": None,
            "next_commands": [],
        }
        atomic_json_save(result, output / "fit_result.json")
        print(json.dumps(result, indent=2))
        return
    selected_path = Path(str(selected["checkpoint_path"]))
    checkpoint = torch.load(selected_path, map_location=device, weights_only=False)
    calibrator.load_state_dict(checkpoint["calibrator_state_dict"], strict=True)
    best_path = checkpoint_dir / "best_internal_check.ckpt"
    torch.save(
        {
            "step": int(selected["step"]),
            "calibrator_state_dict": calibrator.state_dict(),
            "model_config": model_config,
            "selection_metrics": {
                key: selected[key]
                for key in selected
                if key not in {"checkpoint_path", "dataset_residency"}
            },
        },
        best_path,
    )
    selection_payload = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in selected.items()
        if key not in {"checkpoint_path", "dataset_residency"}
    }
    payload = calibrator_identity_payload(
        calibrator,
        model_config=model_config,
        checkpoint_sha256=config["checkpoint"]["sha256"],
        training_molecule_identity_sha256=manifest["training_molecule_identity_sha256"],
        manifest_identity_sha256=manifest["stage_g_manifest_identity_sha256"],
        selected_step=int(selected["step"]),
        selection=selection_payload,
        smoke=bool(manifest.get("smoke")),
    )
    payload.update(
        {
            "selected_checkpoint": best_path.as_posix(),
            "decision": "STAGE_G_CALIBRATOR_SELECTED",
            "batch_size": batch_size,
            "dataset_residency": residency,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "profile_cuda_memory": profile,
        }
    )
    atomic_json_save(payload, output / "calibrator.json")
    atomic_json_save(
        {
            "schema_version": "ecir-mvr-stage-g-fit-v1",
            "decision": "STAGE_G_CALIBRATOR_SELECTED",
            "all_checkpoints_collapsed": False,
            "selected_step": int(selected["step"]),
            "selection_priority": training["selection_priority"],
            "batch_size": batch_size,
            "dataset_residency": residency,
            "device": str(device),
            "validation_used": False,
            "test_records_read": 0,
        },
        output / "fit_result.json",
    )
    print(
        json.dumps(
            {
                "status": "SMOKE_COMPLETE" if manifest.get("smoke") else "COMPLETE",
                "optimizer_steps": max_steps,
                "selected_step": int(selected["step"]),
                "batch_size": batch_size,
                "dataset_residency": residency,
                "device": str(device),
                "selection_priority": checkpoint_selection_priority(selected),
                "validation_used": False,
                "test_records_read": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
