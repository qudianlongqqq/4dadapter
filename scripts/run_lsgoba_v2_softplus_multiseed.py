#!/usr/bin/env python3
"""Preregistered seed331/353 replication using the frozen Softplus-v2 recipe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from scripts import run_lsgoba_v2_softplus_seed307 as training

PLAN_PATH = ROOT / "configs/ecir_mvr_lsgoba_v2_softplus_multiseed.json"
BASE_CONFIG_PATH = ROOT / "configs/ecir_mvr_lsgoba_v2_softplus_seed307.json"
STUDY_REPORT = ROOT / "reports/ecir_mvr/lsgoba_v2_softplus_multiseed"
STUDY_ARTIFACT = ROOT / "artifacts/ecir_mvr/lsgoba_v2_softplus_multiseed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan() -> dict[str, Any]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def resolved_config(seed: int) -> dict[str, Any]:
    protocol = plan()
    if sha256(BASE_CONFIG_PATH) != protocol["base_config_sha256"]:
        raise RuntimeError("frozen seed307 base config SHA changed")
    if seed not in protocol["replicate_seeds"]:
        raise ValueError(f"seed {seed} is not preregistered")
    cfg = copy.deepcopy(json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8")))
    cfg["schema_version"] = "lsgoba-v2-softplus-replicate-resolved-v1"
    cfg["study_name"] = f"LSGO-BA v2 positivity-only Softplus seed{seed} replication"
    cfg["seed"] = seed
    cfg["lineage_base_commit"] = protocol["lineage_base_commit"]
    cfg["training"]["optimizer_steps"] = protocol["train_steps"]
    cfg["training"]["scheduler_horizon"] = protocol["scheduler_horizon"]
    cfg["training"]["checkpoint_steps"] = protocol["checkpoint_steps"]
    cfg["validation"]["primary_checkpoint_step"] = protocol["primary_checkpoint_step"]
    cfg["validation"]["checkpoint_rule"] = "preregistered step17500 replicate; no external-metric selection"
    cfg["validation"]["selection_rule"] = "fixed step17500 before seed331/353 training"
    cfg["replication"] = {
        "reference_seed": protocol["reference_seed"],
        "run_order": protocol["run_order"],
        "only_random_seed_changed": True,
        "scheduler_horizon_preserved_from_seed307": True,
        "seed307_retrained": False,
    }
    return cfg


def paths(seed: int) -> dict[str, Path]:
    report = STUDY_REPORT / f"seed{seed}"
    artifact = STUDY_ARTIFACT / f"seed{seed}"
    return {
        "report": report,
        "artifact": artifact,
        "status": report / "STATUS.json",
        "best": artifact / "checkpoints/final_step17500.ckpt",
        "last": artifact / "checkpoints/last_recovery.ckpt",
        "initial": artifact / f"checkpoints/initial_random_seed{seed}.ckpt",
    }


def configure(seed: int) -> dict[str, Any]:
    cfg = resolved_config(seed)
    p = paths(seed)
    source_payload = Path(cfg["source_payload"]["path"])
    training.CONFIG_PATH = PLAN_PATH
    training.REPORT = p["report"]
    training.ARTIFACT = p["artifact"]
    training.STATUS = p["status"]
    training.SOURCE_PAYLOAD = source_payload
    training.BEST = p["best"]
    training.LAST = p["last"]
    training.INITIAL = p["initial"]
    training.config = lambda: copy.deepcopy(cfg)
    training.base.CONFIG_PATH = PLAN_PATH
    training.base.REPORT = p["report"]
    training.base.ARTIFACT = p["artifact"]
    training.base.STATUS = p["status"]
    training.base.SOURCE_PAYLOAD = source_payload
    training.base.BEST = p["best"]
    training.base.LAST = p["last"]
    training.base.config = training.config
    training.base.status = training.status
    training.base.make_model = training.make_model
    training.base.loss_terms = training.loss_terms
    return cfg


def preflight(seed: int) -> None:
    configure(seed)
    training.architecture_report()
    training.write_config_artifact()
    training.preflight()


def train(seed: int) -> None:
    configure(seed)
    training.architecture_report()
    training.write_config_artifact()
    training.train()


def verify(seed: int) -> None:
    cfg = configure(seed)
    p = paths(seed)
    status = json.loads(p["status"].read_text(encoding="utf-8"))
    expected_checkpoints = [p["artifact"] / f"checkpoints/step{step:05d}.ckpt" for step in (12500, 15000, 17500)]
    if status.get("status") != "PASS" or status.get("completed_steps") != 17500:
        raise RuntimeError("training status is not complete at the preregistered step")
    if any(not checkpoint.is_file() for checkpoint in expected_checkpoints):
        raise RuntimeError("one or more preregistered checkpoints are missing")
    payload = torch.load(expected_checkpoints[-1], map_location="cpu", weights_only=False)
    if payload.get("seed") != seed or payload.get("global_step") != 17500:
        raise RuntimeError("final checkpoint seed/step identity changed")
    if payload.get("initialization") != "FROM_SCRATCH_FULL_TRAINABLE_MODEL" or payload.get("warm_start") is not False:
        raise RuntimeError("replicate is not a from-scratch full-model run")
    scheduler_state = payload["scheduler_state"]
    if scheduler_state.get("T_max") != 22500 or scheduler_state.get("last_epoch") != 17500:
        raise RuntimeError("scheduler trajectory identity changed")
    if payload.get("bond_mean_parameterization") != "softplus(z_B)":
        raise RuntimeError("Bond parameterization changed")
    if not all(torch.isfinite(value).all() for value in payload["model_state"].values() if torch.is_tensor(value)):
        raise RuntimeError("final checkpoint contains nonfinite model values")
    log = pd.read_csv(p["report"] / "05_TRAIN_LOG.csv")
    required = ["L_prior", "L_post", "L_move", "L_total", "bond_mu_mean", "bond_mu_max", "tau_mean"]
    if int(log.step.max()) != 17500 or not np.isfinite(log[required].to_numpy()).all():
        raise RuntimeError("training log is incomplete or nonfinite")
    validation = pd.read_csv(p["report"] / "VALIDATION_CHECKPOINTS.csv")
    if validation.step.astype(int).tolist() != [12500, 15000, 17500]:
        raise RuntimeError("validation checkpoint schedule changed")
    report = {
        "schema_version": "lsgoba-v2-softplus-replicate-integrity-v1",
        "status": "PASS",
        "seed": seed,
        "from_scratch_full_model": True,
        "warm_start": False,
        "train_steps": 17500,
        "scheduler": cfg["training"]["scheduler"],
        "scheduler_T_max": scheduler_state["T_max"],
        "scheduler_last_epoch": scheduler_state["last_epoch"],
        "checkpoint_paths": [str(path) for path in expected_checkpoints],
        "checkpoint_sha256": [sha256(path) for path in expected_checkpoints],
        "bond_parameterization": "SOFTPLUS_POSITIVITY_ONLY",
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
        "xtb_started": False,
        "orca_started": False,
        "docking_started": False,
    }
    training.base.atomic_json(p["report"] / "INTEGRITY.json", report)
    print(json.dumps(report, indent=2), flush=True)


def show_status(seed: int) -> None:
    status_path = paths(seed)["status"]
    print(status_path.read_text(encoding="utf-8") if status_path.is_file() else json.dumps({"status": "NOT_STARTED", "seed": seed}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "train", "verify", "status"))
    parser.add_argument("--seed", type=int, required=True, choices=(331, 353))
    args = parser.parse_args()
    if args.phase == "status":
        show_status(args.seed)
        return 0
    try:
        globals()[args.phase](args.seed)
    except Exception as error:
        configure(args.seed)
        training.status("FAILED", state="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
