#!/usr/bin/env python
"""Train the light EGNN FlexBond secondary optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from etflow.data.flexbond_datamodule import FlexBondOptimizerDataModule
from etflow.commons.provenance import write_run_provenance
from etflow.commons.run_state import update_run_state
from etflow.models.flexbond_optimizer import (
    OPTIMIZER_MODES,
    FlexBondOptimizerLightningModule,
)


class MilestoneCheckpoint(Callback):
    def __init__(self, directory: Path, milestones: list[int]):
        self.directory = directory
        self.milestones = set(milestones)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step in self.milestones:
            path = self.directory / f"step{step}.ckpt"
            if not path.is_file():
                trainer.save_checkpoint(path)


def _config_hash(config: dict) -> str:
    return hashlib.sha256(
        yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _checkpoint_rank(path: Path) -> int:
    match = re.search(r"step[=_-]?(\d+)", path.name)
    return int(match.group(1)) if match else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flexbond_optimizer_egnn.yaml")
    parser.add_argument("--mode", choices=OPTIMIZER_MODES)
    parser.add_argument("--cache_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--max_molecules", type=int)
    parser.add_argument("--t_min", type=float)
    parser.add_argument("--t_max", type=float)
    parser.add_argument("--resume_from_checkpoint", default="auto")
    parser.add_argument("--checkpoint_steps")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.mode:
        config["model"]["mode"] = args.mode
    if args.cache_dir:
        config["data"]["cache_dir"] = args.cache_dir
    if args.max_steps is not None:
        config["trainer"]["max_steps"] = args.max_steps
    if args.max_molecules is not None:
        config["data"]["max_molecules"] = args.max_molecules
    time_sampling = config.setdefault(
        "time_sampling", {"t_min": 0.0, "t_max": 1.0}
    )
    if args.t_min is not None:
        time_sampling["t_min"] = args.t_min
    if args.t_max is not None:
        time_sampling["t_max"] = args.t_max
    t_min = float(time_sampling.get("t_min", 0.0))
    t_max = float(time_sampling.get("t_max", 1.0))
    if not 0.0 <= t_min <= t_max <= 1.0:
        raise ValueError("Require 0 <= t_min <= t_max <= 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "config.resolved.yaml"
    config_hash = _config_hash(config)
    if resolved_path.is_file():
        previous = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        if _config_hash(previous) != config_hash:
            raise RuntimeError("Refusing to resume Cartesian training with a changed config")
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    resume = None
    if args.resume_from_checkpoint == "auto":
        last = checkpoint_dir / "last.ckpt"
        if last.is_file() and last.stat().st_size:
            resume = str(last)
        else:
            candidates = [path for path in checkpoint_dir.glob("*.ckpt") if path.stat().st_size]
            if candidates:
                resume = str(max(candidates, key=_checkpoint_rank))
    elif args.resume_from_checkpoint.lower() not in ("", "none"):
        resume = args.resume_from_checkpoint
    provenance = write_run_provenance(
        output_dir / "run_provenance.json",
        config_path=args.config,
        checkpoint_path=resume,
        cache_path=config["data"]["cache_dir"],
    )
    provenance.update({"config_hash": config_hash, "resume": resume})
    (output_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    pl.seed_everything(int(config.get("seed", 42)), workers=True)
    datamodule = FlexBondOptimizerDataModule(**config["data"])
    model = FlexBondOptimizerLightningModule(
        **config["model"], t_min=t_min, t_max=t_max
    )
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="flexbond-{step}",
        monitor="val/final_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    max_steps = int(config["trainer"]["max_steps"])
    milestones = [
        int(value)
        for value in (args.checkpoint_steps or str(max_steps)).split(",")
    ]
    if max_steps not in milestones:
        milestones.append(max_steps)
    milestone = MilestoneCheckpoint(checkpoint_dir, milestones)
    logger = CSVLogger(output_dir, name="csv")
    trainer = pl.Trainer(
        **config["trainer"],
        default_root_dir=output_dir,
        logger=logger,
        callbacks=[checkpoint, milestone],
    )
    update_run_state(
        output_dir,
        "started",
        stage="training",
        method="cartesian",
        max_steps=max_steps,
        resumed=bool(resume),
        config_hash=config_hash,
    )
    try:
        started = time.perf_counter()
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume)
        wall_clock = time.perf_counter() - started
        metrics_path = Path(logger.log_dir) / "metrics.csv"
        if metrics_path.is_file():
            shutil.copy2(metrics_path, output_dir / "metrics.csv")
        final = checkpoint_dir / f"step{max_steps}.ckpt"
        if not final.is_file() and int(trainer.global_step) >= max_steps:
            trainer.save_checkpoint(final)
        expected = (output_dir / "metrics.csv", checkpoint_dir / "last.ckpt", final)
        if not all(path.is_file() and path.stat().st_size for path in expected):
            raise RuntimeError(f"Cartesian training ended without outputs: {expected}")
        update_run_state(
            output_dir,
            "completed",
            stage="training",
            method="cartesian",
            global_step=int(trainer.global_step),
            checkpoint=str(final),
            wall_clock_seconds=wall_clock,
        )
    except Exception as exc:
        update_run_state(
            output_dir, "failed", stage="training", method="cartesian", error=repr(exc)
        )
        raise


if __name__ == "__main__":
    main()
