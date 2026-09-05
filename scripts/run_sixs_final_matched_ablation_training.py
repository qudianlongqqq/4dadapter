#!/usr/bin/env python
"""GPU-only matched retraining for the two frozen final SIXS ablations.

Only Reliability or Adaptive-BA is replaced.  The model is first initialized
exactly as the final unrestricted model so the active parameters retain the
same seed-matched initial state; the disabled module is replaced afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
import scripts.run_sixs_j1r1_full_joint_unrestricted_movement as u


class UnitReliability(nn.Module):
    def forward(self, bond_features, angle_features, *unused):
        return bond_features.new_ones(bond_features.shape[0]), angle_features.new_ones(angle_features.shape[0])


class EqualBA(nn.Module):
    def forward(self, graph_embedding):
        return graph_embedding.new_full((graph_embedding.shape[0], 2), 0.5)


def configure(args: argparse.Namespace) -> None:
    u.CONFIG_PATH = args.config
    u.EXPERIMENT_ID = f"SIXS_FINAL_UNRESTRICTED_{args.variant.upper()}_SEED{args.seed}_STEP17500"
    u.RUN_ROOT = args.report_dir.parent
    u.REPORT = args.report_dir
    u.ARTIFACT = args.artifact_dir
    u.STATUS = args.report_dir / "FINAL_STATUS.json"
    u.FINAL = args.report_dir / "FINAL_CHECKPOINT.pt"
    u.RECOVERY = args.artifact_dir / "RECOVERY_CHECKPOINT.pt"
    u.TRAIN_LOG = args.report_dir / "TRAIN_LOG.csv"

    original_build = u.build_model

    def build_model(device: torch.device):
        model = original_build(device)
        if args.variant == "reliability_off":
            model.reliability = UnitReliability().to(device)
        else:
            model.adaptive_ba = EqualBA().to(device)
        return model

    def optimizer_for(model):
        training = u.cfg()["training"]
        groups = model.parameter_groups()
        rates = {
            "backbone": training["backbone_learning_rate"],
            "mu": training["head_learning_rate"],
            "j1_sigma": training["head_learning_rate"],
            "reliability": training["head_learning_rate"],
            "adaptive_ba": training["head_learning_rate"],
            "magnitude": training["head_learning_rate"],
        }
        active = [{"params": values, "lr": rates[name], "name": name} for name, values in groups.items() if values]
        optimizer = torch.optim.AdamW(active, weight_decay=training["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["scheduler_horizon"])
        return optimizer, scheduler

    def guards():
        return {
            "EXPERIMENT_ID": u.EXPERIMENT_ID, "SEED": args.seed,
            "FINAL_PRIMARY_METHOD": "J1_R1_FULL_JOINT_ADAPTIVE_BA_UNRESTRICTED_MOVEMENT",
            "MATCHED_ABLATION": args.variant.upper(), "TRAINING_STEPS": 17500,
            "BETA_NLL_BETA": 0.5, "MOVEMENT_REGULARIZER": "NONE",
            "TAU_PARAMETERIZATION": "SOFTPLUS_RAW_NO_FINITE_UPPER_BOUND",
            "TAU_MAX": "NONE", "ATOM_CAP": "NONE", "ROLLBACK_USED": "NO",
            "RELIABILITY": "UNIT_WEIGHT" if args.variant == "reliability_off" else "LEARNED_R1",
            "ADAPTIVE_BA": "EQUAL_0.5_0.5" if args.variant == "equal_ba" else "LEARNED",
            "TRAINING_DEVICE": "cuda", "CPU_TRAINING_FALLBACK": "NO",
            "CHECKPOINT_SELECTED_BY_DEV_OUTCOME": "NO", "DEV_READ": "NO",
            "FORMAL_READ": "NO", "LARGE_HOLDOUT_READ": "NO",
        }

    u.build_model = build_model
    u.base.build_model = build_model
    u.base.optimizer_for = optimizer_for
    u.guards = guards
    u.patch_base()
    u.base.guards = guards


def preflight(args: argparse.Namespace) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("STOP_BEFORE_TRAINING: CUDA unavailable; CPU fallback forbidden")
    config = u.cfg()
    if int(config["seed"]) != args.seed or int(config["training"]["optimizer_steps"]) != 17500:
        raise RuntimeError("frozen seed or endpoint mismatch")
    if float(config["model"]["beta_nll_beta"]) != 0.5:
        raise RuntimeError("frozen beta mismatch")
    device = torch.device("cuda:0")
    prepared, source_payload = u.base.frozen.load_inputs()
    sources = u.base.frozen.source_index(source_payload["train"])
    generator = u.base.seed_all(args.seed)
    graphs, bg, source, reference, _ = u.base.frozen.sample_batch(prepared["train"], sources, generator, 64, device)
    model = u.build_model(device)
    optimizer, _ = u.base.optimizer_for(model)
    prediction, action, losses, _, _ = u.batch_losses(model, graphs, bg, source, reference)
    losses["total"].backward()
    groups = model.parameter_groups()
    checks = {
        "MODEL_ON_GPU": all(p.device.type == "cuda" for p in model.parameters()),
        "BATCH_ON_GPU": all(v.device.type == "cuda" for v in (source, reference, bg.atom_categorical, bg.edge_index)),
        "FORWARD_ON_GPU": losses["total"].device.type == "cuda" and prediction["bond_mu"].device.type == "cuda",
        "BACKWARD_ON_GPU": all(p.grad is None or p.grad.device.type == "cuda" for p in model.parameters()),
        "OPTIMIZER_ON_GPU": all(p.device.type == "cuda" for g in optimizer.param_groups for p in g["params"]),
        "FINITE": bool(torch.isfinite(losses["total"])) and bool(torch.isfinite(action.proposal).all()),
        "DISABLED_GROUP_EMPTY": len(groups["reliability" if args.variant == "reliability_off" else "adaptive_ba"]) == 0,
    }
    payload = {
        "schema_version": "sixs-final-matched-ablation-gpu-preflight-v1", "status": "PASS" if all(checks.values()) else "FAIL",
        "variant": args.variant, "seed": args.seed, "CUDA_AVAILABLE": True,
        "CUDA_DEVICE_NAME": torch.cuda.get_device_name(0), "PYTORCH_CUDA_VERSION": torch.version.cuda,
        **{key: "PASS" if value else "FAIL" for key, value in checks.items()},
    }
    u.base.atomic_json(args.report_dir / "GPU_PREFLIGHT.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"matched ablation preflight failed: {checks}")
    del model, optimizer, prepared, source_payload, graphs, bg, source, reference
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("reliability_off", "equal_ba"), required=True)
    parser.add_argument("--seed", type=int, choices=(307, 331, 353), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    for key in ("config", "report_dir", "artifact_dir"):
        setattr(args, key, getattr(args, key).resolve())
    configure(args)
    if u.FINAL.is_file():
        saved = torch.load(u.FINAL, map_location="cpu", weights_only=False)
        if int(saved.get("step", -1)) != 17500 or saved.get("MATCHED_ABLATION") != args.variant.upper():
            raise RuntimeError("existing final checkpoint has incompatible identity")
        return 0
    preflight(args)
    u.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
