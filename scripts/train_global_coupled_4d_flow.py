#!/usr/bin/env python
"""Train the Global Coupled 4D refiner with resumable milestone checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import re
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from etflow.commons.provenance import write_run_provenance
from etflow.commons.run_state import update_run_state
from etflow.data.flexbond_datamodule import FlexBondOptimizerDataModule
from etflow.models.global_coupled_4d_flow import GlobalCoupled4DFlowLightningModule
from etflow.models.global4d_checkpoint import warm_start_global4d


class MilestoneCheckpoint(Callback):
    def __init__(self, directory: Path, milestones: list[int]):
        self.directory = directory
        self.milestones = set(int(value) for value in milestones if int(value) > 0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step in self.milestones:
            path = self.directory / f"step{step}.ckpt"
            if not path.is_file() or path.stat().st_size == 0:
                trainer.save_checkpoint(path)


def _hash_config(config: dict) -> str:
    serialized = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _parse_steps(value: str | None, final: int) -> list[int]:
    if value:
        result = sorted({int(item) for item in value.split(",") if item.strip()})
    else:
        defaults = (2000, 5000, 10000, 20000, 50000, 75000, final)
        result = sorted({step for step in defaults if step <= final})
    if final not in result:
        result.append(final)
    return result


def main(
    default_config: str = "configs/global_coupled_4d_local025_matched.yaml",
    required_fusion_mode: str | None = None,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--cache_dir")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--max_molecules", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint_steps")
    parser.add_argument("--val_check_interval", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--accumulate_grad_batches", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument(
        "--pin_memory", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--persistent_workers", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--prefetch_factor", type=int)
    parser.add_argument(
        "--fusion_mode",
        choices=("strict_orthogonal", "additive", "gated_additive"),
    )
    parser.add_argument("--internal_beta", type=float)
    parser.add_argument("--warm_start_checkpoint")
    parser.add_argument("--initialize_missing_gate", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default="auto")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.cache_dir:
        config["data"]["cache_dir"] = args.cache_dir
    if args.max_steps is not None:
        config["trainer"]["max_steps"] = args.max_steps
    if args.max_molecules is not None:
        config["data"]["max_molecules"] = args.max_molecules
    if args.seed is not None:
        config["seed"] = args.seed
    if args.val_check_interval is not None:
        config["trainer"]["val_check_interval"] = args.val_check_interval
    data_overrides = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
    }
    for key, value in data_overrides.items():
        if value is not None:
            config["data"][key] = value
    if args.accumulate_grad_batches is not None:
        config["trainer"]["accumulate_grad_batches"] = args.accumulate_grad_batches
    if args.fusion_mode is not None:
        config["model"]["fusion_mode"] = args.fusion_mode
    if args.internal_beta is not None:
        config["model"]["internal_beta"] = args.internal_beta
    resolved_fusion = str(config["model"].get("fusion_mode", "strict_orthogonal"))
    if required_fusion_mode is not None and resolved_fusion != required_fusion_mode:
        parser.error(
            f"this entry point requires fusion_mode={required_fusion_mode}, "
            f"got {resolved_fusion}"
        )
    if args.initialize_missing_gate and not args.warm_start_checkpoint:
        parser.error("--initialize_missing_gate requires --warm_start_checkpoint")
    if int(config["data"]["batch_size"]) < 1:
        raise ValueError("batch_size must be positive")
    if int(config["trainer"].get("accumulate_grad_batches", 1)) < 1:
        raise ValueError("accumulate_grad_batches must be positive")
    data = FlexBondOptimizerDataModule(**config["data"])
    # Persist actual DataLoader semantics, including worker-free normalization.
    config["data"].update(data.resolved_loader_config())
    accumulate = int(config["trainer"].get("accumulate_grad_batches", 1))
    training_runtime = {
        "batch_size": int(config["data"]["batch_size"]),
        "accumulate_grad_batches": accumulate,
        "effective_batch_size": int(config["data"]["batch_size"]) * accumulate,
        "num_workers": int(config["data"]["num_workers"]),
        "pin_memory": bool(config["data"]["pin_memory"]),
        "persistent_workers": bool(config["data"]["persistent_workers"]),
        "prefetch_factor": config["data"]["prefetch_factor"],
    }
    max_steps = int(config["trainer"]["max_steps"])
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resolved = args.output_dir / "config.resolved.yaml"
    config_hash = _hash_config(config)
    resume = None
    if args.resume_from_checkpoint == "auto":
        candidate = checkpoint_dir / "last.ckpt"
        if candidate.is_file() and candidate.stat().st_size:
            resume = str(candidate)
        else:
            candidates = [path for path in checkpoint_dir.glob("*.ckpt") if path.stat().st_size]
            def checkpoint_rank(path):
                match = re.search(r"step[=_-]?(\d+)", path.name)
                return int(match.group(1)) if match else 0
            if candidates:
                resume = str(max(candidates, key=checkpoint_rank))
    elif args.resume_from_checkpoint.lower() not in ("none", ""):
        resume = args.resume_from_checkpoint
    if resolved.exists():
        old = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        old_hash = _hash_config(old)
        # Extending max_steps is the only normal resume-time config difference.
        old_for_compare, new_for_compare = dict(old), dict(config)
        old_for_compare["trainer"] = dict(old.get("trainer", {}))
        new_for_compare["trainer"] = dict(config.get("trainer", {}))
        old_for_compare["trainer"].pop("max_steps", None)
        new_for_compare["trainer"].pop("max_steps", None)
        if _hash_config(old_for_compare) != _hash_config(new_for_compare):
            raise RuntimeError(f"resume config mismatch: old={old_hash}, new={config_hash}")
    update_run_state(args.output_dir, "started", stage="training", mode=args.mode,
                     max_steps=max_steps, resumed=bool(resume), config_hash=config_hash,
                     fusion_mode=str(config["model"].get("fusion_mode", "strict_orthogonal")),
                     internal_beta=float(config["model"].get("internal_beta", 1.0)),
                     **training_runtime)
    try:
        resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        provenance = write_run_provenance(
            args.output_dir / "run_provenance.json",
            config_path=args.config,
            checkpoint_path=resume,
            cache_path=config["data"]["cache_dir"],
        )
        provenance.update({"mode": args.mode, "config_hash": config_hash, "resume": resume})
        (args.output_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        pl.seed_everything(int(config.get("seed", 42)), workers=True)
        model_args = {
            **config["model"], **config["loss"], **config["optimizer"], **config["time_sampling"]
        }
        model_args.pop("scheduler", None)
        model_args.update(
            data_loader_config=data.resolved_loader_config(),
            training_runtime_config=training_runtime,
        )
        warm_start_report = None
        if resume is None and args.warm_start_checkpoint:
            warm_config = {
                **config,
                "model": {
                    **config["model"],
                    "data_loader_config": data.resolved_loader_config(),
                    "training_runtime_config": training_runtime,
                },
            }
            model, warm_start_report = warm_start_global4d(
                args.warm_start_checkpoint,
                warm_config,
                initialize_missing_gate=args.initialize_missing_gate,
            )
            (args.output_dir / "warm_start_report.json").write_text(
                json.dumps(warm_start_report, indent=2), encoding="utf-8"
            )
        else:
            model = GlobalCoupled4DFlowLightningModule(**model_args)
        last_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir, filename="val-best-{step}", monitor="val/final_loss",
            mode="min", save_top_k=3, save_last=True, auto_insert_metric_name=False,
        )
        milestone = MilestoneCheckpoint(checkpoint_dir, _parse_steps(args.checkpoint_steps, max_steps))
        logger = CSVLogger(args.output_dir, name="csv")
        trainer = pl.Trainer(
            **config["trainer"], default_root_dir=args.output_dir, logger=logger,
            callbacks=[last_checkpoint, milestone],
        )
        training_started = time.perf_counter()
        trainer.fit(model, datamodule=data, ckpt_path=resume)
        wall_clock_seconds = time.perf_counter() - training_started
        metrics = Path(logger.log_dir) / "metrics.csv"
        if metrics.is_file():
            shutil.copy2(metrics, args.output_dir / "metrics.csv")
        final_milestone = checkpoint_dir / f"step{max_steps}.ckpt"
        if not final_milestone.is_file() and int(trainer.global_step) >= max_steps:
            trainer.save_checkpoint(final_milestone)
        expected = [args.output_dir / "metrics.csv", checkpoint_dir / "last.ckpt", final_milestone]
        if not all(path.is_file() and path.stat().st_size for path in expected):
            raise RuntimeError(f"training ended without complete outputs: {expected}")
        checkpoint_bytes = final_milestone.stat().st_size
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        update_run_state(
            args.output_dir, "completed", stage="training", mode=args.mode,
            global_step=int(trainer.global_step), checkpoint=str(final_milestone),
            parameter_count=parameter_count, checkpoint_bytes=checkpoint_bytes,
            wall_clock_seconds=wall_clock_seconds,
            iterations_per_second=int(trainer.global_step) / max(wall_clock_seconds, 1.0e-9),
            gpu_hours=wall_clock_seconds * int(config["trainer"].get("devices", 1)) / 3600.0,
            **training_runtime,
            fusion_mode=str(config["model"].get("fusion_mode", "strict_orthogonal")),
            internal_beta=float(config["model"].get("internal_beta", 1.0)),
            warm_start_report=warm_start_report,
        )
    except KeyboardInterrupt:
        update_run_state(args.output_dir, "stopped", stage="training", mode=args.mode,
                         reason="KeyboardInterrupt")
        raise
    except Exception as exc:
        update_run_state(args.output_dir, "failed", stage="training", mode=args.mode, error=repr(exc))
        raise


if __name__ == "__main__":
    main()
