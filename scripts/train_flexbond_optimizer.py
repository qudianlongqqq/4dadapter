#!/usr/bin/env python
"""Train the light EGNN FlexBond secondary optimizer."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from etflow.data.flexbond_datamodule import FlexBondOptimizerDataModule
from etflow.commons.provenance import write_run_provenance
from etflow.models.flexbond_optimizer import (
    OPTIMIZER_MODES,
    FlexBondOptimizerLightningModule,
)


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
    parser.add_argument("--resume_from_checkpoint")
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
    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    write_run_provenance(
        output_dir / "run_provenance.json",
        config_path=args.config,
        checkpoint_path=args.resume_from_checkpoint,
        cache_path=config["data"]["cache_dir"],
    )
    pl.seed_everything(int(config.get("seed", 42)), workers=True)
    datamodule = FlexBondOptimizerDataModule(**config["data"])
    model = FlexBondOptimizerLightningModule(
        **config["model"], t_min=t_min, t_max=t_max
    )
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="flexbond-{step}",
        monitor="val/final_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    logger = CSVLogger(output_dir, name="csv")
    trainer = pl.Trainer(
        **config["trainer"],
        default_root_dir=output_dir,
        logger=logger,
        callbacks=[checkpoint],
    )
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume_from_checkpoint,
    )
    metrics_path = Path(logger.log_dir) / "metrics.csv"
    if metrics_path.is_file():
        shutil.copy2(metrics_path, output_dir / "metrics.csv")


if __name__ == "__main__":
    main()
