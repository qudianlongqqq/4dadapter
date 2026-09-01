#!/usr/bin/env python3
"""GPU-only seed307 Full Joint unrestricted-movement capability branch."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
import scripts.run_sixs_j1r1_full_joint_adaptive_ba_movement as base
from etflow.ecir.j1r1_full_joint_unrestricted import (
    INITIAL_RAW_PARAMETER,
    INITIAL_TAU_ANGSTROM,
    UnrestrictedFullJointModel,
    unrestricted_full_joint_action,
)
from etflow.ecir.learned_geometry import geometry_values
from etflow.ecir.musigma_reliability import action_loss, belief_loss


CONFIG_PATH = Path(os.environ.get("SIXS_MULTISEED_CONFIG", ROOT / "configs/sixs_j1r1_full_joint_unrestricted_movement.json"))
EXPERIMENT_ID = os.environ.get("SIXS_MULTISEED_EXPERIMENT_ID", "SIXS_J1_R1_FULL_JOINT_UNRESTRICTED_MOVEMENT_SEED307")
RUN_ROOT = Path(os.environ.get("SIXS_MULTISEED_RUN_ROOT", ROOT / "reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307"))
REPORT = Path(os.environ.get("SIXS_MULTISEED_REPORT", RUN_ROOT / "02_UNRESTRICTED_MOVEMENT"))
ARTIFACT = Path(os.environ.get("SIXS_MULTISEED_ARTIFACT", ROOT / "artifacts/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT"))
STATUS = REPORT / "FINAL_STATUS.json"
FINAL = REPORT / "FINAL_CHECKPOINT.pt"
RECOVERY = ARTIFACT / "RECOVERY_CHECKPOINT.pt"
TRAIN_LOG = REPORT / "UNRESTRICTED_TRAIN_LOG.csv"
EVAL = ARTIFACT / "dev_evaluation"
COORDINATES_READY = EVAL / "COORDINATES_READY.json"
PAYLOAD = EVAL / "EVALUATION_PAYLOAD.pt"
SDF = EVAL / "PROPOSAL.sdf"
PER_RECORD = EVAL / "PER_RECORD.parquet"
PB = EVAL / "POSEBUSTERS.parquet"
V3D = EVAL / "VALIDITY3D.parquet"


def cfg() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def guards() -> dict[str, Any]:
    seed = int(cfg()["seed"])
    return {
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "SEED": seed,
        "CURRENT_FINAL_MODEL": "J1_R1_FULL_JOINT_ADAPTIVE_BA_MOVEMENT_STEP17500",
        "CAPABILITY_BRANCH_ONLY": "YES",
        "FULL_JOINT_TRAINING": "YES",
        "MOVEMENT_REGULARIZER": "NONE",
        "TAU_PARAMETERIZATION": "SOFTPLUS_RAW_NO_FINITE_UPPER_BOUND",
        "TAU_MAX": "NONE",
        "ATOM_CAP": "NONE",
        "ROLLBACK_USED": "NO",
        "REFERENCE_USED_AT_INFERENCE": "NO",
        "SECOND_ORDER_REQUIRED": "NO",
        "TRAINING_DEVICE": "cuda",
        "CPU_TRAINING_FALLBACK": "NO",
        "FORMAL_READ": "NO",
        "LARGE_HOLDOUT_READ": "NO",
        "SEED331_STARTED": "YES" if seed == 331 else "NO",
        "SEED353_STARTED": "YES" if seed == 353 else "NO",
        "CHECKPOINT_SELECTED_BY_DEV_OUTCOME": "NO",
        "HYPERPARAMETER_SWEEP": "NO",
    }


def status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    previous = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    if state != "FAIL":
        for key in ("ERROR", "ERROR_TYPE", "TRACEBACK"):
            previous.pop(key, None)
    base.atomic_json(STATUS, {
        **previous,
        "schema_version": "sixs-unrestricted-movement-status-v1",
        "PIPELINE_STATUS": state,
        "CURRENT_STAGE": stage,
        "WORKER_PID": os.getpid(),
        "UPDATED_AT": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        **guards(),
        **extra,
    })


def patch_base() -> None:
    base.CONFIG_PATH = CONFIG_PATH
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.RUN_NAME = "sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT"
    base.REPORT = REPORT
    base.ARTIFACT = ARTIFACT
    base.STATUS = STATUS
    base.FINAL = FINAL
    base.RECOVERY = RECOVERY
    base.TRAIN_LOG = TRAIN_LOG
    base.cfg = cfg
    base.guards = guards


def cuda_snapshot() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    name = torch.cuda.get_device_name(0) if count else None
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace",
        )
        smi = query.stdout.strip() if query.returncode == 0 else f"ERROR[{query.returncode}]: {query.stderr.strip()}"
    except Exception as error:
        smi = f"{type(error).__name__}: {error}"
    return {
        "CUDA_AVAILABLE": available,
        "CUDA_DEVICE_COUNT": count,
        "CUDA_DEVICE_NAME": name,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<UNSET>"),
        "PYTORCH_CUDA_VERSION": torch.version.cuda,
        "MODEL_DEVICE": "cuda" if available else "unavailable",
        "TRAINING_DEVICE": "cuda" if available else "unavailable",
        "NVIDIA_SMI": smi,
    }


def execution_device() -> torch.device:
    snapshot = cuda_snapshot()
    for key in ("CUDA_AVAILABLE", "CUDA_DEVICE_COUNT", "CUDA_DEVICE_NAME", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_VERSION"):
        print(f"{key} = {snapshot[key]}", flush=True)
    if not snapshot["CUDA_AVAILABLE"] or snapshot["CUDA_DEVICE_COUNT"] < 1:
        status("STOP_BEFORE_TRAINING", "FAIL", GPU_TRAINING_READY="NO", UNRESTRICTED_TRAINING_STARTED="NO", **snapshot)
        raise RuntimeError("GPU_TRAINING_REQUIRED_BUT_CUDA_UNAVAILABLE")
    print("MODEL_DEVICE = cuda", flush=True)
    print("TRAINING_DEVICE = cuda", flush=True)
    return torch.device("cuda:0")


def build_model(device: torch.device) -> UnrestrictedFullJointModel:
    config = cfg()
    base.seed_all(config["seed"])
    model = UnrestrictedFullJointModel(config["model"]["hidden_dim"], config["model"]["layers"])
    base.load_standard_base_geometry(model)
    state = json.loads((ROOT / config["data"]["magnitude_state_preflight"]).read_text(encoding="utf-8"))
    model.set_state_normalization(torch.tensor(state["state_mean"]), torch.tensor(state["state_std"]))
    return model.to(device)


def batch_losses(model, graphs, bg, source, reference):
    config = cfg()
    prediction = model.belief(bg, detach_sigma_features=False)
    ref_bond, ref_angle = geometry_values(reference, bg)
    belief, belief_parts = belief_loss(
        "J1", prediction,
        ref_bond.to(prediction["bond_mu"]), ref_angle.to(prediction["angle_mu"]),
        graphs, config["model"]["beta_nll_beta"],
    )
    action = unrestricted_full_joint_action(model, source, graphs, prediction)
    post = action_loss(action, reference, graphs, prediction)
    zero = post.new_zeros(())
    return prediction, action, {"total": belief + post, "belief": belief, "post": post, "move": zero, **belief_parts}, ref_bond, ref_angle


def parameter_devices(model, optimizer, source, reference, bg) -> dict[str, Any]:
    model_cuda = all(parameter.device.type == "cuda" for parameter in model.parameters())
    optimizer_cuda = all(
        parameter.device.type == "cuda"
        for group in optimizer.param_groups for parameter in group["params"]
    )
    batch_cuda = all(tensor.device.type == "cuda" for tensor in (source, reference, bg.atom_categorical, bg.edge_index))
    return {
        "MODEL_ON_GPU": "PASS" if model_cuda else "FAIL",
        "BATCH_ON_GPU": "PASS" if batch_cuda else "FAIL",
        "OPTIMIZER_PARAMETERS_ON_GPU": "PASS" if optimizer_cuda else "FAIL",
    }


def preflight() -> None:
    gate_path = REPORT / "GPU_PREFLIGHT.json"
    if gate_path.is_file() and (REPORT / "GRADIENT_PATH_AUDIT.json").is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("GPU_TRAINING_READY") == "YES":
            return
    REPORT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    status("GPU_PREFLIGHT", "RUNNING", UNRESTRICTED_TRAINING_STARTED="NO")
    input_hashes = base.verify_inputs()
    config = cfg()
    device = execution_device()
    prepared, source_payload = base.frozen.load_inputs()
    sources = base.frozen.source_index(source_payload["train"])
    generator = base.seed_all(config["seed"])
    graphs, bg, source, reference, _ = base.frozen.sample_batch(prepared["train"], sources, generator, 64, device)
    model = build_model(device)
    optimizer, _ = base.optimizer_for(model)
    devices = parameter_devices(model, optimizer, source, reference, bg)
    if any(value != "PASS" for value in devices.values()):
        status("STOP_BEFORE_TRAINING", "FAIL", GPU_TRAINING_READY="NO", **devices)
        raise RuntimeError("GPU_DEVICE_PREFLIGHT_FAILED")

    prediction, action, losses, _, _ = batch_losses(model, graphs, bg, source, reference)
    groups = model.parameter_groups()
    flat = [parameter for parameters in groups.values() for parameter in parameters]
    routes = {}
    for name in ("belief", "post"):
        gradients = torch.autograd.grad(losses[name], flat, retain_graph=True, allow_unused=True)
        offset, norms = 0, {}
        for group_name, parameters in groups.items():
            local = gradients[offset : offset + len(parameters)]
            offset += len(parameters)
            norms[group_name] = base.parameter_norm(local)
        routes[name] = norms
    losses["total"].backward()
    total_gradients = {name: base.parameter_norm([p.grad for p in params]) for name, params in groups.items()}
    exact_initial_tau = float((action.tau.detach() - INITIAL_TAU_ANGSTROM).abs().max()) <= 1e-7
    required = (
        routes["belief"]["backbone"] > 0 and routes["belief"]["mu"] > 0 and routes["belief"]["j1_sigma"] > 0
        and routes["post"]["backbone"] > 0 and routes["post"]["reliability"] > 0
        and routes["post"]["adaptive_ba"] > 0 and routes["post"]["magnitude"] > 0
        and routes["post"]["mu"] == 0 and routes["post"]["j1_sigma"] == 0
        and all(value > 0 and math.isfinite(value) for value in total_gradients.values())
        and bool(torch.isfinite(losses["total"])) and bool(torch.isfinite(action.proposal).all())
        and exact_initial_tau
    )
    snapshot = cuda_snapshot()
    gpu_gate = {
        "schema_version": "sixs-unrestricted-gpu-preflight-v1",
        "GPU_TRAINING_READY": "YES" if required else "NO",
        **snapshot, **devices,
        "INITIAL_TAU_TARGET": INITIAL_TAU_ANGSTROM,
        "INITIAL_LOGIT_OR_RAW_PARAMETER": INITIAL_RAW_PARAMETER,
        "INITIAL_TAU_EXACT": exact_initial_tau,
        "FORWARD_ON_GPU": "PASS" if losses["total"].device.type == "cuda" else "FAIL",
        "BACKWARD_ON_GPU": "PASS" if all(p.grad is None or p.grad.device.type == "cuda" for p in model.parameters()) else "FAIL",
        **guards(),
    }
    base.atomic_json(gate_path, gpu_gate)
    base.atomic_json(REPORT / "GRADIENT_PATH_AUDIT.json", {
        "status": "PASS" if required else "FAIL",
        "loss_routes": routes,
        "total_gradient_norms": total_gradients,
        "MOVEMENT_REGULARIZER": "NONE",
        "MAGNITUDE_SUPERVISION": "L_POST_ONLY",
        "SECOND_ORDER_REQUIRED": "NO",
        **guards(),
    })
    base.atomic_json(REPORT / "PARAMETER_GROUP_AUDIT.json", {
        "status": "PASS",
        "groups": {name: base.parameter_count(params) for name, params in groups.items()},
        "groups_disjoint": len(flat) == len({id(p) for p in flat}),
        **guards(),
    })
    identity = f"""# Unrestricted Full Joint model identity

This is an independent seed307 capability branch. The frozen current final model is unchanged. The branch starts from the same Softplus-v2 backbone and mu initialization and trains all J1/R1/Adaptive-BA/magnitude modules for exactly 17,500 steps.

```text
MOVEMENT_REGULARIZER = NONE
TAU = softplus(raw)
INITIAL_TAU_TARGET = {INITIAL_TAU_ANGSTROM}
INITIAL_LOGIT_OR_RAW_PARAMETER = {INITIAL_RAW_PARAMETER}
TAU_FINITE_UPPER_BOUND = NONE
PER_ATOM_CAP = NONE
ROLLBACK_USED = NO
TRAINING_DEVICE = cuda
CPU_TRAINING_FALLBACK = NO
```
"""
    base.atomic_text(REPORT / "UNRESTRICTED_MODEL_IDENTITY.md", identity)
    base.atomic_json(REPORT / "IMPLEMENTATION_CONFIG.json", {
        **config,
        "config_sha256": base.sha256(CONFIG_PATH),
        "module_sha256": base.sha256(ROOT / "etflow/ecir/j1r1_full_joint_unrestricted.py"),
        "runner_sha256": base.sha256(Path(__file__)),
        "input_hashes": input_hashes,
        **guards(),
    })
    if not required:
        status("STOP_BEFORE_TRAINING", "FAIL", GPU_TRAINING_READY="NO")
        raise RuntimeError("UNRESTRICTED_PREFLIGHT_FAILED")
    status("PREFLIGHT_COMPLETE", "PASS", GPU_TRAINING_READY="YES", UNRESTRICTED_TRAINING_STARTED="NO", **devices)
    del model, prepared, source_payload, graphs, bg, source, reference
    torch.cuda.empty_cache()


def training_row(step, losses, prediction, action, ref_bond, ref_angle, groups, seconds):
    row = base.training_row(step, losses, prediction, action, ref_bond, ref_angle, groups, seconds)
    row["MOVEMENT_LOSS"] = 0.0
    row["MOVEMENT_REGULARIZER"] = "NONE"
    return row


def training_gpu_verification(step, model, optimizer, source, reference, bg, prediction, losses) -> None:
    devices = parameter_devices(model, optimizer, source, reference, bg)
    result = {
        "schema_version": "sixs-unrestricted-gpu-training-verification-v1",
        "step": step,
        **devices,
        "FORWARD_ON_GPU": "PASS" if prediction["bond_mu"].device.type == "cuda" and losses["total"].device.type == "cuda" else "FAIL",
        "BACKWARD_ON_GPU": "PASS" if all(p.grad is None or p.grad.device.type == "cuda" for p in model.parameters()) else "FAIL",
        **cuda_snapshot(), **guards(),
    }
    result["status"] = "PASS" if all(result[key] == "PASS" for key in (
        "MODEL_ON_GPU", "BATCH_ON_GPU", "OPTIMIZER_PARAMETERS_ON_GPU", "FORWARD_ON_GPU", "BACKWARD_ON_GPU"
    )) else "FAIL"
    base.atomic_json(REPORT / "GPU_TRAINING_VERIFICATION.json", result)
    for key in ("MODEL_ON_GPU", "BATCH_ON_GPU", "FORWARD_ON_GPU", "BACKWARD_ON_GPU"):
        print(f"{key} = {result[key]}", flush=True)
    print(f"NVIDIA_SMI = {result['NVIDIA_SMI']}", flush=True)
    if result["status"] != "PASS":
        raise RuntimeError("GPU_TRAINING_VERIFICATION_FAILED")


def train() -> None:
    if FINAL.is_file():
        return
    config, training = cfg(), cfg()["training"]
    device = execution_device()
    torch.cuda.reset_peak_memory_stats(device)
    prepared, source_payload = base.frozen.load_inputs()
    sources = base.frozen.source_index(source_payload["train"])
    generator = base.seed_all(config["seed"])
    model = build_model(device)
    optimizer, scheduler = base.optimizer_for(model)
    groups = model.parameter_groups()
    start, logs = 0, []
    if RECOVERY.is_file():
        saved = torch.load(RECOVERY, map_location="cpu", weights_only=False)
        if saved.get("config_sha256") != base.sha256(CONFIG_PATH):
            raise RuntimeError("recovery config identity mismatch")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        generator.set_state(saved["generator_state"])
        random.setstate(saved["python_rng_state"]); np.random.set_state(saved["numpy_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"]); torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        start = int(saved["step"])
        if TRAIN_LOG.is_file():
            logs = pd.read_csv(TRAIN_LOG).to_dict("records")
    status("UNRESTRICTED_GPU_TRAINING", "RUNNING", CURRENT_STEP=start, TOTAL_STEPS=17500, GPU_TRAINING_READY="YES", UNRESTRICTED_TRAINING_STARTED="YES")
    started = time.time()
    for step in range(start + 1, training["optimizer_steps"] + 1):
        tick = time.time(); model.train()
        graphs, bg, source, reference, _ = base.frozen.sample_batch(
            prepared["train"], sources, generator, training["batch_molecules"], device
        )
        optimizer.zero_grad(set_to_none=True)
        prediction, action, losses, ref_bond, ref_angle = batch_losses(model, graphs, bg, source, reference)
        if not bool(torch.isfinite(losses["total"])) or not bool(torch.isfinite(action.proposal).all()):
            raise RuntimeError(f"nonfinite unrestricted forward at step {step}")
        losses["total"].backward()
        if step == 25:
            training_gpu_verification(step, model, optimizer, source, reference, bg, prediction, losses)
        if step == 1 or step % training["log_interval"] == 0:
            row = training_row(step, losses, prediction, action, ref_bond, ref_angle, groups, time.time() - tick)
            logs.append(row)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
        if not bool(torch.isfinite(gradient)):
            raise RuntimeError(f"nonfinite unrestricted gradient at step {step}")
        optimizer.step(); scheduler.step()
        if step == 1 or step % training["log_interval"] == 0:
            current = logs[-1]
            status("UNRESTRICTED_GPU_TRAINING", "RUNNING", CURRENT_STEP=step, TOTAL_STEPS=17500,
                   TOTAL_LOSS=current["TOTAL_LOSS"], POST_LOSS=current["POST_LOSS"], TAU_MEAN=current["TAU_MEAN"],
                   ELAPSED_SECONDS=time.time() - started, GPU_TRAINING_READY="YES", UNRESTRICTED_TRAINING_STARTED="YES")
            print(json.dumps({"stage": "UNRESTRICTED_GPU_TRAINING", **current}), flush=True)
        if step % training["recovery_interval"] == 0:
            base.atomic_torch(RECOVERY, {
                "schema_version": "sixs-unrestricted-recovery-v1", "step": step,
                "config_sha256": base.sha256(CONFIG_PATH), "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "generator_state": generator.get_state(), "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(), **guards(),
            })
            base.atomic_frame(TRAIN_LOG, pd.DataFrame(logs))
    base.atomic_torch(FINAL, {
        "schema_version": "sixs-unrestricted-final-v1", "experiment_id": EXPERIMENT_ID,
        "seed": int(config["seed"]), "step": training["optimizer_steps"], "config_sha256": base.sha256(CONFIG_PATH),
        "model_state": model.state_dict(), "optimizer_group_names": list(groups), **guards(),
    })
    base.atomic_frame(TRAIN_LOG, pd.DataFrame(logs))
    base.atomic_text(REPORT / "FINAL_CHECKPOINT_SHA256.txt", base.sha256(FINAL))
    tail = pd.DataFrame(logs).tail(20)
    base.atomic_json(REPORT / "TRAIN_SUMMARY.json", {
        "status": "PASS", "steps": 17500, "checkpoint_sha256": base.sha256(FINAL),
        "last_log": logs[-1], "late_mean": tail.mean(numeric_only=True).to_dict(),
        "elapsed_seconds": time.time() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "nonfinite_count": 0, **guards(),
    })
    status("FINAL_CHECKPOINT_FROZEN", "PASS", CURRENT_STEP=17500, TOTAL_STEPS=17500,
           FINAL_CHECKPOINT_SHA256=base.sha256(FINAL), GPU_TRAINING_READY="YES", UNRESTRICTED_TRAINING_STARTED="YES")


def evaluate_coordinates() -> None:
    if COORDINATES_READY.is_file() and PER_RECORD.is_file() and SDF.is_file():
        return
    config, device = cfg(), execution_device()
    status("UNRESTRICTED_DEV_COORDINATE_GENERATION", "RUNNING", EVALUATION_RECORDS=0, EVALUATION_TOTAL=5000)
    prepared, source_payload = base.frozen.load_inputs()
    by_item = {str(item["molecule_id"]): item for item in prepared["val"]}
    by_sample = {str(row["sample_id"]): row for row in source_payload["val"]}
    manifest = json.loads((ROOT / config["data"]["dev_manifest"]).read_text(encoding="utf-8"))
    ids = [str(sample) for row in manifest["rows"] for sample in row["sample_ids"]]
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise RuntimeError("DEV identity changed")
    final = torch.load(FINAL, map_location="cpu", weights_only=False)
    model = build_model(device); model.load_state_dict(final["model_state"], strict=True); model.eval()
    val_manifest = pd.read_parquet(config["data"]["val_manifest"])
    manifest_by_sample = {str(row.sample_id): row for row in val_manifest.itertuples(index=False)}
    SDF.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(SDF) + f".tmp.{os.getpid()}"); writer = Chem.SDWriter(str(temporary))
    primitive_rows, record_rows = [], []
    try:
        for start in range(0, len(ids), 64):
            batch_ids = ids[start : start + 64]
            source_rows = [by_sample[sample] for sample in batch_ids]
            items = [by_item[str(row["molecule_id"])] for row in source_rows]
            graphs = [item["graph"] for item in items]
            bg = base.collate_graphs(graphs).to(device)
            source = torch.cat([torch.as_tensor(row["source"], dtype=torch.float64) for row in source_rows]).to(device)
            reference = torch.cat([torch.as_tensor(item["references"][0], dtype=torch.float64) for item in items]).to(device)
            with torch.no_grad():
                prediction = model.belief(bg, detach_sigma_features=False)
                action = unrestricted_full_joint_action(model, source, graphs, prediction)
            if not bool(torch.isfinite(action.proposal).all()) or not bool(torch.isfinite(action.tau).all()):
                raise RuntimeError(f"NONFINITE_UNRESTRICTED_COORDINATES_AT_BATCH_{start}")
            ref_bond, ref_angle = geometry_values(reference, bg)
            src_bond, src_angle = geometry_values(source, bg)
            prop_bond, prop_angle = geometry_values(action.proposal, bg)
            bo = ao = xo = 0
            for local, (sample_id, source_row, graph) in enumerate(zip(batch_ids, source_rows, graphs, strict=True)):
                nb, na, nat = int(graph.bonds.size(1)), int(graph.angles.size(0)), int(graph.atom_categorical.size(0))
                bb, aa, xx = slice(bo, bo+nb), slice(ao, ao+na), slice(xo, xo+nat)
                bo += nb; ao += na; xo += nat
                bmu = prediction["bond_mu"][bb].detach().cpu().double().numpy(); amu = prediction["angle_mu"][aa].detach().cpu().double().numpy()
                bs = prediction["bond_sigma"][bb].detach().cpu().double().numpy(); ass = prediction["angle_sigma"][aa].detach().cpu().double().numpy()
                bstat = bg.bond_fixed[bb,1].detach().cpu().double().numpy(); astat = bg.angle_fixed[aa,1].detach().cpu().double().numpy()
                rb = action.bond_reliability[bb].detach().cpu().double().numpy(); ra = action.angle_reliability[aa].detach().cpu().double().numpy()
                yb, ya = ref_bond[bb].detach().cpu().numpy(), ref_angle[aa].detach().cpu().numpy()
                sb, sa = src_bond[bb].detach().cpu().numpy(), src_angle[aa].detach().cpu().numpy()
                pb, pa = prop_bond[bb].detach().cpu().numpy(), prop_angle[aa].detach().cpu().numpy()
                primitive_rows.append({
                    "record_id": sample_id, "molecule_id": str(source_row["molecule_id"]),
                    "bond_reference": yb, "bond_mu": bmu, "bond_sigma": bs, "bond_sigma_stat": bstat,
                    "bond_reliability": rb, "bond_source_abs_defect": np.abs(sb-bmu), "bond_source_abs_standardized_defect": np.abs((sb-bmu)/bs),
                    "angle_reference": ya, "angle_mu": amu, "angle_sigma": ass, "angle_sigma_stat": astat,
                    "angle_reliability": ra, "angle_source_abs_defect": np.abs(sa-amu), "angle_source_abs_standardized_defect": np.abs((sa-amu)/ass),
                })
                common_source = .5*(np.mean(((sb-yb)/bstat)**2)+np.mean(((sa-ya)/astat)**2))
                common_post = .5*(np.mean(((pb-yb)/bstat)**2)+np.mean(((pa-ya)/astat)**2))
                delta = action.proposal[xx] - source[xx]
                atom_displacement = torch.linalg.vector_norm(delta, dim=-1).detach().cpu().double().numpy()
                source_rmsd = float(delta.square().sum(-1).mean().sqrt())
                record_rows.append({
                    "record_id": sample_id, "molecule_id": str(source_row["molecule_id"]), "arm": EXPERIMENT_ID,
                    "internal_post": common_post, "common_source_objective": common_source,
                    "direction_improvement": common_source-common_post, "bond_raw_mae": float(np.mean(np.abs(pb-yb))),
                    "angle_raw_mae": float(np.mean(np.abs(pa-ya))), "bond_mu_mae": float(np.mean(np.abs(bmu-yb))),
                    "angle_mu_mae": float(np.mean(np.abs(amu-ya))), "source_rmsd": source_rmsd,
                    "proposal_movement": source_rmsd, "tau": float(action.tau[local]),
                    "w_B": float(action.family_weights[local,0]), "w_A": float(action.family_weights[local,1]),
                    "atom_displacements": atom_displacement, "max_atom_displacement": float(atom_displacement.max()),
                    "atom_cap_active": False, "rollback": False,
                })
                metadata = manifest_by_sample[sample_id]
                cache = Path(config["data"]["val_cache"]) / Path(str(metadata.source_path)).name
                raw = cache.read_bytes()
                if hashlib.sha256(raw).hexdigest() != str(metadata.source_file_sha256):
                    raise RuntimeError("DEV source cache hash changed")
                record = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
                base.frozen.write_molecule(writer, record, action.proposal[xx], sample_id, EXPERIMENT_ID)
            status("UNRESTRICTED_DEV_COORDINATE_GENERATION", "RUNNING", EVALUATION_RECORDS=min(start+64,5000), EVALUATION_TOTAL=5000)
    finally:
        writer.close()
    os.replace(temporary, SDF)
    base.atomic_frame(PER_RECORD, pd.DataFrame(record_rows))
    base.atomic_torch(PAYLOAD, {"record_ids": ids, "record_rows": record_rows, "primitive_rows": primitive_rows, **guards()})
    base.atomic_json(COORDINATES_READY, {
        "status": "PASS", "records": 5000, "record_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "final_checkpoint_sha256": base.sha256(FINAL), "sdf_sha256": base.sha256(SDF), "payload_sha256": base.sha256(PAYLOAD), **guards(),
    })
    status("UNRESTRICTED_COORDINATES_READY", "PASS", EVALUATION_RECORDS=5000, EVALUATION_TOTAL=5000)


def evaluate_validity() -> None:
    evaluate_coordinates()
    ids = pd.read_parquet(PER_RECORD).record_id.astype(str).tolist()
    if base.evaluator_artifact_status(PB, ids) != "COMPLETE" or base.evaluator_artifact_status(V3D, ids) != "COMPLETE":
        status("UNRESTRICTED_EXTERNAL_VALIDITY", "RUNNING", EVALUATION_RECORDS=5000, EVALUATION_TOTAL=5000)
        base.frozen.run_external_evaluators(EXPERIMENT_ID, {"sdf": SDF, "per_record": PER_RECORD, "pb": PB, "v3d": V3D}, ids)
    status("UNRESTRICTED_VALIDITY_COMPLETE", "PASS", EVALUATION_RECORDS=5000, EVALUATION_TOTAL=5000)


def pipeline() -> None:
    preflight(); train(); evaluate_validity()


def main() -> int:
    patch_base()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "train", "coordinates", "validity", "pipeline"), default="pipeline")
    args = parser.parse_args()
    try:
        if args.stage == "preflight": preflight()
        elif args.stage == "train": preflight(); train()
        elif args.stage == "coordinates": preflight(); evaluate_coordinates()
        elif args.stage == "validity": preflight(); evaluate_validity()
        else: pipeline()
        return 0
    except Exception as error:
        REPORT.mkdir(parents=True, exist_ok=True)
        trace = traceback.format_exc(); base.atomic_text(REPORT / "PIPELINE_TRACEBACK.txt", trace)
        status("STOPPED_FAIL_CLOSED", "FAIL", ERROR_TYPE=type(error).__name__, ERROR=str(error), TRACEBACK=str(REPORT / "PIPELINE_TRACEBACK.txt"))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
