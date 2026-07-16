#!/usr/bin/env python
"""Train ECIR-Flow with independently logged loss terms."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from etflow.commons.global_coupled_4d_sampling import atomic_json_save, atomic_torch_save
from etflow.ecir.dataset import ECIRMixedDataset
from etflow.ecir.model import ECIRFlowSystem


LOSS_COLUMNS = (
    "loss",
    "flow_matching_loss",
    "internal_mode_loss",
    "error_prediction_loss",
    "identity_loss",
    "trust_loss",
    "gate_mean",
)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _dataset(config: dict, split: str) -> ECIRMixedDataset:
    data = config["data"]
    stage = config.get("stage", {})
    atlas_path = None
    if data.get("atlas_dir"):
        candidate = Path(data["atlas_dir"]) / f"{split}.parquet"
        if candidate.is_file():
            atlas_path = candidate
    return ECIRMixedDataset(
        data["cache_dir"],
        split,
        atlas_path=atlas_path,
        target_cache_dir=data["target_cache_dir"],
        real_error_ratio=float(data["real_error_ratio"]),
        synthetic_error_ratio=float(data["synthetic_error_ratio"]),
        clean_identity_ratio=float(data["clean_identity_ratio"]),
        seed=int(config.get("seed", 42)) + (0 if split == "train" else 100_000),
        max_records=stage.get(f"max_{split}_records"),
        max_molecules=stage.get(f"max_{split}_molecules"),
        allow_online_target_building=False,
    )


@torch.inference_mode()
def _validate(model, loader, device, loss_weights) -> dict[str, float]:
    model.eval()
    rows = []
    for batch in loader:
        batch = batch.to(device)
        result = model.loss(batch, **loss_weights)
        rows.append({key: float(value) for key, value in result.items()})
    model.train()
    return {key: sum(row[key] for row in rows) / len(rows) for key in LOSS_COLUMNS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--target_cache_dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.cache_dir is not None:
        config["data"]["cache_dir"] = str(args.cache_dir)
    if args.target_cache_dir is not None:
        config["data"]["target_cache_dir"] = str(args.target_cache_dir)
    seed = int(config.get("seed", 42))
    _seed(seed)
    training = config["training"]
    device = torch.device(args.device or training.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    steps = int(args.steps or training["steps"])
    output = args.output_dir or Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    train_data = _dataset(config, "train")
    val_data = _dataset(config, "val")
    loader_kwargs = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    model = ECIRFlowSystem(**dict(config.get("model") or {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training.get("weight_decay", 1.0e-6)),
    )
    start_step = 0
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload["step"])
    loss_weights = {
        key: float(value)
        for key, value in config["loss"].items()
        if key in {"lambda_mode", "lambda_error", "lambda_identity", "lambda_trust"}
    }
    resolved = {**config, "resolved_device": str(device), "resolved_steps": steps}
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    history_path = output / "history.csv"
    history_file = history_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(history_file, fieldnames=("step", "split", *LOSS_COLUMNS, "records_per_second"))
    if history_path.stat().st_size == 0:
        writer.writeheader()
    interval = int(training.get("validation_interval", 250))
    checkpoint_interval = int(training.get("checkpoint_interval", interval))
    iterator = iter(train_loader)
    seen = 0
    started = time.perf_counter()
    last_validation: dict[str, float] | None = None
    model.train()
    for step in range(start_step + 1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            train_data.set_epoch(train_data.epoch + 1)
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(batch, **loss_weights)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("grad_clip", 1.0)))
        optimizer.step()
        graphs = int(batch.num_graphs)
        seen += graphs
        if step == 1 or step % int(training.get("log_interval", 25)) == 0:
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            writer.writerow(
                {
                    "step": step,
                    "split": "train",
                    **{key: float(losses[key].detach()) for key in LOSS_COLUMNS},
                    "records_per_second": seen / elapsed,
                }
            )
            history_file.flush()
        if step % interval == 0 or step == steps:
            last_validation = _validate(model, val_loader, device, loss_weights)
            writer.writerow(
                {"step": step, "split": "val", **last_validation, "records_per_second": ""}
            )
            history_file.flush()
        if step % checkpoint_interval == 0 or step == steps:
            atomic_torch_save(
                {
                    "schema_version": "1.0",
                    "model_type": "ECIRFlowSystem",
                    "step": step,
                    "config": resolved,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                output / f"step{step:06d}.ckpt",
            )
    history_file.close()
    summary = {
        "status": "COMPLETED",
        "steps": steps,
        "device": str(device),
        "train_records": len(train_data),
        "val_records": len(val_data),
        "mix_ratios": dict(zip(("real_error", "synthetic_error", "clean_identity"), train_data.ratios.tolist())),
        "elapsed_seconds": time.perf_counter() - started,
        "last_validation": last_validation,
        "checkpoint": str((output / f"step{steps:06d}.ckpt").resolve()),
    }
    atomic_json_save(summary, output / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
