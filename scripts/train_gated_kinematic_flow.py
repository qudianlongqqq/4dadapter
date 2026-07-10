#!/usr/bin/env python
"""Train Gated Molecular Kinematic Flow on the labeled refinement cache."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from etflow.commons.provenance import write_run_provenance
from etflow.commons.run_state import update_run_state
from etflow.data.flexbond_datamodule import FlexBondOptimizerDataModule
from etflow.models.gated_kinematic_flow import GatedKinematicFlowLightningModule


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/gated_kinematic_local025_small.yaml")
    parser.add_argument("--cache_dir"); parser.add_argument("--output_dir",required=True,type=Path)
    parser.add_argument("--max_steps",type=int); parser.add_argument("--max_molecules",type=int)
    parser.add_argument("--seed",type=int); parser.add_argument("--resume_from_checkpoint")
    args=parser.parse_args()
    with open(args.config,encoding="utf-8") as handle: config=yaml.safe_load(handle)
    if args.cache_dir: config["data"]["cache_dir"]=args.cache_dir
    if args.max_steps: config["trainer"]["max_steps"]=args.max_steps
    if args.max_molecules: config["data"]["max_molecules"]=args.max_molecules
    if args.seed is not None: config["seed"]=args.seed
    args.output_dir.mkdir(parents=True,exist_ok=True)
    resolved=args.output_dir/"config.resolved.yaml"
    if resolved.exists() and not args.resume_from_checkpoint:
        raise FileExistsError("Output directory already contains a run; use a new name or explicit resume.")
    update_run_state(args.output_dir,"started",stage="training",expected_outputs=["metrics.csv","checkpoints/last.ckpt"])
    try:
        with resolved.open("w",encoding="utf-8") as handle: yaml.safe_dump(config,handle,sort_keys=False)
        provenance=write_run_provenance(args.output_dir/"run_provenance.json",config_path=args.config,
            checkpoint_path=args.resume_from_checkpoint,cache_path=config["data"]["cache_dir"])
        provenance["gate_target_method"]=config["model"].get("gate_target_method","sigmoid_threshold")
        with (args.output_dir/"run_provenance.json").open("w",encoding="utf-8") as handle:
            json.dump(provenance,handle,indent=2)
        pl.seed_everything(int(config.get("seed",42)),workers=True)
        data=FlexBondOptimizerDataModule(**config["data"])
        model_args={**config["model"],**config["loss"],**config["optimizer"],**config["time_sampling"]}
        model=GatedKinematicFlowLightningModule(**model_args)
        checkpoint=ModelCheckpoint(dirpath=args.output_dir/"checkpoints",filename="gated-kinematic-{step}",
            monitor="val/final_loss",mode="min",save_top_k=3,save_last=True)
        logger=CSVLogger(args.output_dir,name="csv")
        trainer=pl.Trainer(**config["trainer"],default_root_dir=args.output_dir,logger=logger,callbacks=[checkpoint])
        trainer.fit(model,datamodule=data,ckpt_path=args.resume_from_checkpoint)
        metrics=Path(logger.log_dir)/"metrics.csv"
        if metrics.is_file(): shutil.copy2(metrics,args.output_dir/"metrics.csv")
        expected=[args.output_dir/"metrics.csv",args.output_dir/"checkpoints"/"last.ckpt"]
        if not all(path.is_file() for path in expected): raise RuntimeError("Training ended without all expected outputs")
        update_run_state(args.output_dir,"completed",stage="training",checkpoint=str(checkpoint.last_model_path))
    except KeyboardInterrupt:
        (args.output_dir/"STOPPED_REASON.txt").write_text("KeyboardInterrupt\n",encoding="utf-8")
        update_run_state(args.output_dir,"stopped",stage="training",reason="KeyboardInterrupt"); raise
    except Exception as exc:
        update_run_state(args.output_dir,"failed",stage="training",error=repr(exc)); raise


if __name__=="__main__": main()
