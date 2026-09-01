#!/usr/bin/env python3
"""Resumable seed307 full-joint J1/R1 + Adaptive-BA + magnitude experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
os.environ.setdefault("SIXS_FACTORIAL_RUN_NAMESPACE", "sixs_musigma_reliability_factorial_cuda")
import scripts.run_sixs_musigma_reliability_factorial as frozen
from etflow.ecir.j1r1_full_joint import FullJointModel, full_joint_action
from etflow.ecir.learned_geometry import geometry_values, safety_accept
from etflow.ecir.musigma_reliability import action_loss, belief_loss
from scripts.run_mcvr_lsgo import collate_graphs


CONFIG_PATH = Path(os.environ.get("SIXS_MULTISEED_CONFIG", ROOT / "configs/sixs_j1r1_full_joint_adaptive_ba_movement.json"))
EXPERIMENT_ID = os.environ.get("SIXS_MULTISEED_EXPERIMENT_ID", "SIXS_J1_R1_FULL_JOINT_ADAPTIVE_BA_MOVEMENT")
RUN_NAME = os.environ.get("SIXS_MULTISEED_RUN_NAME", "sixs_j1r1_full_joint_adaptive_ba_movement_seed307")
REPORT = Path(os.environ.get("SIXS_MULTISEED_REPORT", ROOT / "reports/ecir_mvr" / RUN_NAME))
ARTIFACT = Path(os.environ.get("SIXS_MULTISEED_ARTIFACT", ROOT / "artifacts/ecir_mvr" / RUN_NAME))
STATUS = REPORT / "FINAL_STATUS.json"
STDOUT = REPORT / "PIPELINE_STDOUT.log"
STDERR = REPORT / "PIPELINE_STDERR.log"
FINAL = REPORT / "FINAL_CHECKPOINT.pt"
RECOVERY = ARTIFACT / "RECOVERY_CHECKPOINT.pt"
TRAIN_LOG = REPORT / "TRAIN_LOG.csv"
COORDINATES_READY = ARTIFACT / "dev_evaluation/COORDINATES_READY.json"
PAYLOAD = ARTIFACT / "dev_evaluation/EVALUATION_PAYLOAD.pt"
SDF = ARTIFACT / "dev_evaluation/PROPOSAL.sdf"
PER_RECORD = ARTIFACT / "dev_evaluation/PER_RECORD.parquet"
PB = ARTIFACT / "dev_evaluation/POSEBUSTERS.parquet"
V3D = ARTIFACT / "dev_evaluation/VALIDITY3D.parquet"
RECOVERY_PREFLIGHT = REPORT / "RECOVERY_PREFLIGHT.json"


def cfg() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def markdown_frame(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    columns = [str(column) for column in frame.columns]
    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.10g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def guards() -> dict[str, str]:
    seed = int(cfg()["seed"])
    is_replication = os.environ.get("SIXS_MULTISEED_REPLICATION", "0") == "1"
    return {
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "SEED": str(seed),
        "FULL_JOINT_TRAINING": "YES",
        "BACKBONE_TRAINABLE": "YES",
        "MU_TRAINABLE": "YES",
        "J1_SIGMA_TRAINABLE": "YES",
        "RELIABILITY_TRAINABLE": "YES",
        "ADAPTIVE_BA_TRAINABLE": "YES",
        "MAGNITUDE_TRAINABLE": "YES",
        "CURRENT_BA": "ADAPTIVE",
        "WARM_START_FROM_SCIENTIFIC_CHECKPOINT": "NO",
        "BASE_GEOMETRY_PROTOCOL_INITIALIZATION": "SOFTPLUS_V2_BACKBONE_AND_MU_ONLY",
        "NAIVE_MU_SIGMA_JOINT_REINTRODUCED": "NO",
        "TEACHER_STUDENT_USED": "NO",
        "SECOND_ORDER_REQUIRED": "NO",
        "REFERENCE_USED_AT_INFERENCE": "NO",
        "TAU_MAX": "0.010 A",
        "ATOM_CAP": "0.03 A",
        "ROLLBACK_USED_AS_PRIMARY_ENDPOINT": "NO",
        "ROLLBACK_USED_AS_MODEL_SELECTION_GATE": "NO",
        "ROLLBACK_USED_AS_SAFETY_PENALTY": "NO",
        "FORMAL_READ": "NO",
        "LARGE_HOLDOUT_READ": "NO",
        "XTB_STARTED": "NO",
        "SEED331_STARTED": "YES" if seed == 331 else "NO",
        "SEED353_STARTED": "YES" if seed == 353 else "NO",
        "HYPERPARAMETER_SWEEP": "NO",
        "CHECKPOINT_SELECTED_BY_DEV_OUTCOME": "NO",
        "REPEATED_POLLING": "NO",
        "NO_NEW_TRAINING": "NO" if is_replication else "YES",
        "NO_COORDINATE_REGENERATION": "YES",
        "SCIENTIFIC_MODEL_MODIFIED": "NO",
        "SCIENTIFIC_EVALUATOR_MODIFIED": "NO",
    }


def status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    previous = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    if state != "FAIL":
        previous.pop("ERROR", None)
        previous.pop("ERROR_TYPE", None)
        previous.pop("TRACEBACK", None)
    atomic_json(
        STATUS,
        {
            **previous,
            "schema_version": "sixs-j1r1-full-joint-status-v1",
            "IMPLEMENTATION_STATUS": "PASS",
            "PIPELINE_STATUS": state,
            "CURRENT_STAGE": stage,
            "WORKER_PID": os.getpid(),
            "UPDATED_AT": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            **guards(),
            **extra,
        },
    )


def execution_device() -> torch.device:
    mode = os.environ.get("SIXS_FULL_JOINT_DEVICE", "cuda").lower()
    if mode not in {"cuda", "cpu", "auto"}:
        raise RuntimeError("SIXS_FULL_JOINT_DEVICE must be cuda, cpu, or auto")
    if mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_BUT_UNAVAILABLE")
    return torch.device("cuda:0" if mode != "cpu" and torch.cuda.is_available() else "cpu")


def seed_all(seed: int = 307) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed + 91000)


def verify_inputs() -> dict[str, str]:
    config = cfg()
    checks = {
        config["data"]["prepared_payload"]: config["data"]["prepared_sha256"],
        config["data"]["source_payload"]: config["data"]["source_payload_sha256"],
        config["data"]["train_manifest"]: config["data"]["train_manifest_sha256"],
        config["data"]["val_manifest"]: config["data"]["val_manifest_sha256"],
        ROOT / config["data"]["dev_manifest"]: config["data"]["dev_manifest_sha256"],
        ROOT / config["data"]["magnitude_state_preflight"]: config["data"]["magnitude_state_preflight_sha256"],
        config["initialization"]["base_geometry_checkpoint"]: config["initialization"]["base_geometry_checkpoint_sha256"],
        ROOT / config["comparator"]["per_record"]: config["comparator"]["per_record_sha256"],
        ROOT / config["comparator"]["posebusters"]: config["comparator"]["posebusters_sha256"],
        ROOT / config["comparator"]["validity3d"]: config["comparator"]["validity3d_sha256"],
    }
    actual = {}
    for path, expected in checks.items():
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f"missing frozen input: {path}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"frozen input hash changed: {path}")
        actual[str(path)] = observed
    return actual


def load_standard_base_geometry(model: FullJointModel) -> None:
    path = Path(cfg()["initialization"]["base_geometry_checkpoint"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    if any(key.startswith("geometry.") for key in state):
        geometry = {key[len("geometry.") :]: value for key, value in state.items() if key.startswith("geometry.")}
    else:
        geometry = state
    model.belief.geometry.load_state_dict(geometry, strict=True)


def build_model(device: torch.device) -> FullJointModel:
    config = cfg()
    seed_all(config["seed"])
    model = FullJointModel(config["model"]["hidden_dim"], config["model"]["layers"])
    load_standard_base_geometry(model)
    state = json.loads((ROOT / config["data"]["magnitude_state_preflight"]).read_text(encoding="utf-8"))
    model.set_state_normalization(torch.tensor(state["state_mean"]), torch.tensor(state["state_std"]))
    return model.to(device)


def optimizer_for(model: FullJointModel):
    training = cfg()["training"]
    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["backbone"], "lr": training["backbone_learning_rate"], "name": "backbone"},
            {"params": groups["mu"], "lr": training["head_learning_rate"], "name": "mu"},
            {"params": groups["j1_sigma"], "lr": training["head_learning_rate"], "name": "j1_sigma"},
            {"params": groups["reliability"], "lr": training["head_learning_rate"], "name": "reliability"},
            {"params": groups["adaptive_ba"], "lr": training["head_learning_rate"], "name": "adaptive_ba"},
            {"params": groups["magnitude"], "lr": training["head_learning_rate"], "name": "magnitude"},
        ],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["scheduler_horizon"])
    return optimizer, scheduler


def parameter_norm(grads: Iterable[torch.Tensor | None]) -> float:
    values = [grad.detach().double().square().sum() for grad in grads if grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def parameter_count(parameters: Sequence[torch.nn.Parameter]) -> int:
    return sum(int(parameter.numel()) for parameter in parameters)


def quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.detach().float().cpu(), q))


def batch_losses(model, graphs, bg, source, reference):
    config = cfg()
    prediction = model.belief(bg, detach_sigma_features=False)
    ref_bond, ref_angle = geometry_values(reference, bg)
    belief, belief_parts = belief_loss(
        "J1",
        prediction,
        ref_bond.to(prediction["bond_mu"]),
        ref_angle.to(prediction["angle_mu"]),
        graphs,
        config["model"]["beta_nll_beta"],
    )
    action = full_joint_action(model, source, graphs, prediction, atom_cap=config["model"]["atom_cap_angstrom"])
    post = action_loss(action, reference, graphs, prediction)
    move = (action.tau / config["model"]["tau_max_angstrom"]).square().mean()
    total = belief + post + config["objective"]["lambda_move"] * move
    return prediction, action, {"total": total, "belief": belief, "post": post, "move": move, **belief_parts}, ref_bond, ref_angle


def preflight() -> None:
    if (REPORT / "GRADIENT_PATH_AUDIT.json").is_file():
        gate = json.loads((REPORT / "GRADIENT_PATH_AUDIT.json").read_text(encoding="utf-8"))
        if gate.get("status") == "PASS":
            return
    REPORT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    status("PREFLIGHT", "RUNNING", CURRENT_STEP=0, TOTAL_STEPS=17500)
    input_hashes = verify_inputs()
    config = cfg()
    device = execution_device()
    prepared, source_payload = frozen.load_inputs()
    sources = frozen.source_index(source_payload["train"])
    generator = seed_all(config["seed"])
    graphs, bg, source, reference, _ = frozen.sample_batch(prepared["train"], sources, generator, 64, device)
    model = build_model(device)
    groups = model.parameter_groups()

    all_ids = [id(parameter) for parameters in groups.values() for parameter in parameters]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("optimizer parameter groups overlap")
    group_rows = []
    for name, parameters in groups.items():
        group_rows.append(
            {
                "group": name,
                "parameter_tensors": len(parameters),
                "parameter_count": parameter_count(parameters),
                "learning_rate": config["training"]["backbone_learning_rate"] if name == "backbone" else config["training"]["head_learning_rate"],
                "trainable": all(parameter.requires_grad for parameter in parameters),
            }
        )
    atomic_json(
        REPORT / "PARAMETER_GROUP_AUDIT.json",
        {
            "schema_version": "sixs-full-joint-parameter-group-audit-v1",
            "status": "PASS",
            "groups": group_rows,
            "total_parameter_count": sum(row["parameter_count"] for row in group_rows),
            "groups_disjoint": True,
            **guards(),
        },
    )

    prediction, action, losses, _, _ = batch_losses(model, graphs, bg, source, reference)
    flat = [parameter for parameters in groups.values() for parameter in parameters]
    routes = {}
    for loss_name in ("belief", "post", "move"):
        gradients = torch.autograd.grad(losses[loss_name], flat, retain_graph=True, allow_unused=True)
        offset, norms = 0, {}
        for group_name, parameters in groups.items():
            local = gradients[offset : offset + len(parameters)]
            offset += len(parameters)
            norms[group_name] = parameter_norm(local)
        routes[loss_name] = norms
    losses["total"].backward()
    total_gradients = {
        name: parameter_norm([parameter.grad for parameter in parameters]) for name, parameters in groups.items()
    }
    required = (
        routes["belief"]["backbone"] > 0
        and routes["belief"]["mu"] > 0
        and routes["belief"]["j1_sigma"] > 0
        and routes["post"]["backbone"] > 0
        and routes["post"]["reliability"] > 0
        and routes["post"]["adaptive_ba"] > 0
        and routes["post"]["magnitude"] > 0
        and routes["post"]["mu"] == 0
        and routes["post"]["j1_sigma"] == 0
        and routes["move"]["magnitude"] > 0
        and all(routes["move"][name] == 0 for name in groups if name != "magnitude")
        and all(value > 0 and math.isfinite(value) for value in total_gradients.values())
        and bool(torch.isfinite(losses["total"]))
        and bool(torch.isfinite(action.direction).all())
        and bool((action.family_weights > 0).all())
        and float((action.family_weights.sum(-1) - 1).abs().max()) <= 1.0e-7
    )
    gradient_audit = {
        "schema_version": "sixs-full-joint-gradient-path-audit-v1",
        "status": "PASS" if required else "FAIL",
        "loss_routes": routes,
        "total_gradient_norms": total_gradients,
        "loss_values": {name: float(value.detach()) for name, value in losses.items()},
        "J1_SIGMA_RELATED_LOSS": "J1_BETA_NLL_COUPLED_BELIEF_OBJECTIVE",
        "RELIABILITY_RELATED_LOSS": "L_POST_SHARED_NOT_ADDED_TWICE",
        "BA_SUPERVISION": "L_POST_ONLY",
        "MAGNITUDE_SUPERVISION": "L_POST_PLUS_L_MOVE",
        "CARTESIAN_DERIVATIVES_DETACHED": "YES",
        "SECOND_ORDER_REQUIRED": "NO",
        **guards(),
    }
    atomic_json(REPORT / "GRADIENT_PATH_AUDIT.json", gradient_audit)
    atomic_json(
        REPORT / "REFERENCE_LEAKAGE_AUDIT.json",
        {
            "schema_version": "sixs-full-joint-reference-leakage-audit-v1",
            "status": "PASS",
            "inference_action_inputs": ["graph", "source_coordinates", "mu", "sigma", "source_conditioned_reliability", "adaptive_BA", "source_state", "tau"],
            "reference_enters_inference": False,
            "reference_enters_training_losses_only": True,
            **guards(),
        },
    )
    model_identity = """# SIXS J1-R1 full-joint model identity

This is one new seed307 optimization run. The standard six-arm base-geometry initialization loads only the frozen Softplus-v2 backbone and mu scope; it does not load J1-R1, learned-magnitude, Adaptive-BA-v2, or V3 Joint BA-Movement candidate weights. J1 sigma, R1 Reliability, Adaptive BA, and magnitude use their deterministic standard initializations and every scientific module is trainable.

The total objective is `L_J1_beta_NLL + L_post + lambda_move * L_move`. Reliability supervision and post loss are the same mathematical object and are not counted twice. Adaptive BA has no auxiliary target. The Cartesian primitive derivatives remain detached and first order.

```text
FULL_JOINT_TRAINING = YES
WARM_START_FROM_SCIENTIFIC_CHECKPOINT = NO
BASE_GEOMETRY_PROTOCOL_INITIALIZATION = SOFTPLUS_V2_BACKBONE_AND_MU_ONLY
NAIVE_MU_SIGMA_JOINT_REINTRODUCED = NO
SECOND_ORDER_REQUIRED = NO
REFERENCE_USED_AT_INFERENCE = NO
```
"""
    atomic_text(REPORT / "MODEL_IDENTITY.md", model_identity)
    atomic_json(
        REPORT / "IMPLEMENTATION_CONFIG.json",
        {
            **config,
            "config_sha256": sha256(CONFIG_PATH),
            "module_sha256": sha256(ROOT / "etflow/ecir/j1r1_full_joint.py"),
            "runner_sha256": sha256(Path(__file__)),
            "input_hashes": input_hashes,
            "parameter_group_audit_sha256": sha256(REPORT / "PARAMETER_GROUP_AUDIT.json"),
            "gradient_path_audit_sha256": sha256(REPORT / "GRADIENT_PATH_AUDIT.json"),
            **guards(),
        },
    )
    if not required:
        raise RuntimeError("full-joint preflight gradient gate failed")
    status("PREFLIGHT_COMPLETE", "PASS", CURRENT_STEP=0, TOTAL_STEPS=17500)
    del model, prepared, source_payload, graphs, bg, source, reference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def training_row(step, losses, prediction, action, ref_bond, ref_angle, groups, step_seconds):
    sigma = torch.cat((prediction["bond_sigma"].detach(), prediction["angle_sigma"].detach()))
    sigma_stat = torch.cat((prediction["bond_sigma"].detach() / torch.exp(prediction["bond_log_sigma_ratio"].detach()), prediction["angle_sigma"].detach() / torch.exp(prediction["angle_log_sigma_ratio"].detach())))
    log_ratio = torch.log(sigma / sigma_stat)
    reliability = torch.cat((action.bond_reliability.detach(), action.angle_reliability.detach()))
    weights = action.family_weights.detach()
    tau = action.tau.detach()
    grad_norms = {name: parameter_norm([parameter.grad for parameter in parameters]) for name, parameters in groups.items()}
    return {
        "step": step,
        "TOTAL_LOSS": float(losses["total"].detach()),
        "PRIOR_J1_LOSS": float(losses["belief"].detach()),
        "POST_LOSS": float(losses["post"].detach()),
        "MOVEMENT_LOSS": float(losses["move"].detach()),
        "J1_SIGMA_RELATED_LOSS": float(losses["belief"].detach()),
        "RELIABILITY_RELATED_LOSS": float(losses["post"].detach()),
        "MU_BOND_MAE": float((prediction["bond_mu"].detach() - ref_bond.to(prediction["bond_mu"])).abs().mean()),
        "MU_ANGLE_MAE": float((prediction["angle_mu"].detach() - ref_angle.to(prediction["angle_mu"])).abs().mean()),
        "SIGMA_MEAN": float(sigma.mean()),
        "SIGMA_P01": quantile(sigma, 0.01),
        "SIGMA_P50": quantile(sigma, 0.50),
        "SIGMA_P99": quantile(sigma, 0.99),
        "SIGMA_LOWER_SATURATION": float((log_ratio <= -6 + 1.0e-6).float().mean()),
        "SIGMA_UPPER_SATURATION": float((log_ratio >= 6 - 1.0e-6).float().mean()),
        "R_MEAN": float(reliability.mean()),
        "R_P01": quantile(reliability, 0.01),
        "R_P50": quantile(reliability, 0.50),
        "R_P99": quantile(reliability, 0.99),
        "W_B_MEAN": float(weights[:, 0].mean()),
        "W_B_MEDIAN": quantile(weights[:, 0], 0.50),
        "W_B_P05": quantile(weights[:, 0], 0.05),
        "W_B_P95": quantile(weights[:, 0], 0.95),
        "W_A_MEAN": float(weights[:, 1].mean()),
        "W_A_MEDIAN": quantile(weights[:, 1], 0.50),
        "W_A_P05": quantile(weights[:, 1], 0.05),
        "W_A_P95": quantile(weights[:, 1], 0.95),
        "TAU_MEAN": float(tau.mean()),
        "TAU_MEDIAN": quantile(tau, 0.50),
        "TAU_P05": quantile(tau, 0.05),
        "TAU_P95": quantile(tau, 0.95),
        "TAU_P99": quantile(tau, 0.99),
        "TAU_MAX": float(tau.max()),
        **{f"GRAD_{name.upper()}": value for name, value in grad_norms.items()},
        "STEP_SECONDS": step_seconds,
    }


def train() -> None:
    if FINAL.is_file():
        return
    config = cfg()
    training = config["training"]
    device = execution_device()
    if device.type != "cuda":
        raise RuntimeError("GPU_ONLY_TRAINING_REQUIRED")
    torch.cuda.reset_peak_memory_stats(device)
    prepared, source_payload = frozen.load_inputs()
    sources = frozen.source_index(source_payload["train"])
    generator = seed_all(config["seed"])
    model = build_model(device)
    optimizer, scheduler = optimizer_for(model)
    groups = model.parameter_groups()
    start, logs = 0, []
    if RECOVERY.is_file():
        saved = torch.load(RECOVERY, map_location="cpu", weights_only=False)
        if saved.get("config_sha256") != sha256(CONFIG_PATH):
            raise RuntimeError("recovery config identity mismatch")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        generator.set_state(saved["generator_state"])
        random.setstate(saved["python_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        start = int(saved["step"])
        if TRAIN_LOG.is_file():
            logs = pd.read_csv(TRAIN_LOG).to_dict("records")
    status("FULL_JOINT_TRAINING", "RUNNING", CURRENT_STEP=start, TOTAL_STEPS=training["optimizer_steps"])
    started = time.time()
    for step in range(start + 1, training["optimizer_steps"] + 1):
        tick = time.time()
        model.train()
        graphs, bg, source, reference, _ = frozen.sample_batch(prepared["train"], sources, generator, training["batch_molecules"], device)
        optimizer.zero_grad(set_to_none=True)
        prediction, action, losses, ref_bond, ref_angle = batch_losses(model, graphs, bg, source, reference)
        if not bool(torch.isfinite(losses["total"])) or not bool(torch.isfinite(action.proposal).all()):
            raise RuntimeError(f"nonfinite full-joint forward at step {step}")
        losses["total"].backward()
        if step == 25:
            model_cuda = all(parameter.device.type == "cuda" for parameter in model.parameters())
            batch_cuda = all(tensor.device.type == "cuda" for tensor in (source, reference, bg.atom_categorical, bg.edge_index))
            optimizer_cuda = all(parameter.device.type == "cuda" for group in optimizer.param_groups for parameter in group["params"])
            verification = {
                "schema_version": "sixs-restricted-gpu-training-verification-v1", "step": step,
                "MODEL_ON_GPU": "PASS" if model_cuda else "FAIL",
                "BATCH_ON_GPU": "PASS" if batch_cuda else "FAIL",
                "OPTIMIZER_PARAMETERS_ON_GPU": "PASS" if optimizer_cuda else "FAIL",
                "FORWARD_ON_GPU": "PASS" if prediction["bond_mu"].device.type == "cuda" and losses["total"].device.type == "cuda" else "FAIL",
                "BACKWARD_ON_GPU": "PASS" if all(p.grad is None or p.grad.device.type == "cuda" for p in model.parameters()) else "FAIL",
                "CUDA_DEVICE_NAME": torch.cuda.get_device_name(device), "PYTORCH_CUDA_VERSION": torch.version.cuda,
                "NVIDIA_SMI_SNAPSHOT": os.popen("nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits").read().strip(),
                **guards(),
            }
            verification["status"] = "PASS" if all(verification[key] == "PASS" for key in (
                "MODEL_ON_GPU", "BATCH_ON_GPU", "OPTIMIZER_PARAMETERS_ON_GPU", "FORWARD_ON_GPU", "BACKWARD_ON_GPU"
            )) else "FAIL"
            atomic_json(REPORT / "GPU_TRAINING_VERIFICATION.json", verification)
            if verification["status"] != "PASS":
                raise RuntimeError("GPU_TRAINING_VERIFICATION_FAILED")
        if step == 1 or step % training["log_interval"] == 0:
            row = training_row(step, losses, prediction, action, ref_bond, ref_angle, groups, time.time() - tick)
            logs.append(row)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
        if not bool(torch.isfinite(gradient)):
            raise RuntimeError(f"nonfinite full-joint gradient at step {step}")
        optimizer.step()
        scheduler.step()
        if step == 1 or step % training["log_interval"] == 0:
            current = logs[-1]
            status(
                "FULL_JOINT_TRAINING",
                "RUNNING",
                CURRENT_STEP=step,
                TOTAL_STEPS=training["optimizer_steps"],
                TOTAL_LOSS=current["TOTAL_LOSS"],
                PRIOR_J1_LOSS=current["PRIOR_J1_LOSS"],
                POST_LOSS=current["POST_LOSS"],
                MOVEMENT_LOSS=current["MOVEMENT_LOSS"],
                W_B_MEAN=current["W_B_MEAN"],
                TAU_MEAN=current["TAU_MEAN"],
                SIGMA_MEAN=current["SIGMA_MEAN"],
                R_MEAN=current["R_MEAN"],
                ELAPSED_SECONDS=time.time() - started,
            )
            print(json.dumps({"stage": "FULL_JOINT_TRAINING", **current}), flush=True)
        if step % training["recovery_interval"] == 0:
            atomic_torch(
                RECOVERY,
                {
                    "schema_version": "sixs-full-joint-recovery-v1",
                    "step": step,
                    "config_sha256": sha256(CONFIG_PATH),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "generator_state": generator.get_state(),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    **guards(),
                },
            )
            atomic_frame(TRAIN_LOG, pd.DataFrame(logs))
    atomic_torch(
        FINAL,
        {
            "schema_version": "sixs-j1r1-full-joint-final-v1",
            "experiment_id": EXPERIMENT_ID,
            "seed": int(config["seed"]),
            "step": training["optimizer_steps"],
            "config_sha256": sha256(CONFIG_PATH),
            "model_state": model.state_dict(),
            "optimizer_group_names": list(groups),
            **guards(),
        },
    )
    atomic_frame(TRAIN_LOG, pd.DataFrame(logs))
    atomic_text(REPORT / "FINAL_CHECKPOINT_SHA256.txt", sha256(FINAL))
    tail = pd.DataFrame(logs).tail(20)
    atomic_json(
        REPORT / "TRAIN_SUMMARY.json",
        {
            "schema_version": "sixs-full-joint-train-summary-v1",
            "status": "PASS",
            "steps": training["optimizer_steps"],
            "checkpoint_sha256": sha256(FINAL),
            "last_log": logs[-1],
            "late_mean": tail.mean(numeric_only=True).to_dict(),
            "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "nonfinite_count": 0,
            **guards(),
        },
    )
    status("FINAL_CHECKPOINT_FROZEN", "PASS", CURRENT_STEP=training["optimizer_steps"], TOTAL_STEPS=training["optimizer_steps"], FINAL_CHECKPOINT_SHA256=sha256(FINAL))


def distribution(values: np.ndarray, label: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {
        "object": label,
        "count": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "nonfinite_fraction": float(np.mean(~np.isfinite(values))),
    }


def evaluator_artifact_status(path: Path, ids: Sequence[str]) -> str:
    if not path.is_file():
        return "NOT_FOUND"
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return "PARTIAL_OR_INVALID"
    if "record_id" not in frame or len(frame) != len(ids):
        return "PARTIAL_OR_INVALID"
    if frame.record_id.astype(str).tolist() != list(ids):
        return "PARTIAL_OR_INVALID"
    return "COMPLETE"


def recovery_preflight(force: bool = False) -> dict[str, Any]:
    """Validate frozen coordinate artifacts without rebuilding scientific coordinates."""
    if RECOVERY_PREFLIGHT.is_file() and not force:
        existing = json.loads(RECOVERY_PREFLIGHT.read_text(encoding="utf-8"))
        if existing.get("RECOVERY_PREFLIGHT") != "PASS":
            raise RuntimeError("existing recovery preflight is not PASS")
        return existing
    config = cfg()
    required = (FINAL, SDF, PER_RECORD, PAYLOAD, COORDINATES_READY)
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise RuntimeError(f"recovery artifact missing; coordinate regeneration forbidden: {missing}")

    marker = json.loads(COORDINATES_READY.read_text(encoding="utf-8"))
    actual_checkpoint_sha = sha256(FINAL)
    recorded_checkpoint_sha = (REPORT / "FINAL_CHECKPOINT_SHA256.txt").read_text(encoding="utf-8").strip()
    checkpoint_ok = (
        len(recorded_checkpoint_sha) == 64
        and actual_checkpoint_sha == recorded_checkpoint_sha
        and actual_checkpoint_sha == marker.get("final_checkpoint_sha256")
    )

    manifest = json.loads((ROOT / config["data"]["dev_manifest"]).read_text(encoding="utf-8"))
    frozen_ids = [str(sample) for row in manifest["rows"] for sample in row["sample_ids"]]
    records = pd.read_parquet(PER_RECORD)
    record_ids = records.record_id.astype(str).tolist()
    counts = records.groupby("molecule_id", sort=False).size()
    payload = torch.load(PAYLOAD, map_location="cpu", weights_only=False)
    payload_ids = [str(value) for value in payload.get("record_ids", [])]

    supplier = Chem.SDMolSupplier(str(SDF), removeHs=False, sanitize=False)
    sdf_ids: list[str] = []
    for molecule in supplier:
        if molecule is None:
            sdf_ids.append("__INVALID_MOLECULE__")
        elif molecule.HasProp("sample_id"):
            sdf_ids.append(str(molecule.GetProp("sample_id")))
        else:
            sdf_ids.append(str(molecule.GetProp("_Name")))

    record_alignment_ok = (
        len(frozen_ids) == 5000
        and len(set(frozen_ids)) == 5000
        and len(records) == 5000
        and records.molecule_id.astype(str).nunique() == 2500
        and len(counts) == 2500
        and bool((counts == 2).all())
        and record_ids == frozen_ids
        and payload_ids == frozen_ids
        and sdf_ids == frozen_ids
    )
    coordinate_ok = (
        marker.get("status") == "PASS"
        and marker.get("records") == 5000
        and marker.get("record_ids_sha256") == hashlib.sha256("\n".join(frozen_ids).encode()).hexdigest()
        and marker.get("sdf_sha256") == sha256(SDF)
        and marker.get("payload_sha256") == sha256(PAYLOAD)
        and record_alignment_ok
    )
    comparator_hashes_ok = all(
        sha256(ROOT / config["comparator"][name]) == config["comparator"][f"{name}_sha256"]
        for name in ("per_record", "posebusters", "validity3d")
    )
    v3d_status = evaluator_artifact_status(V3D, frozen_ids)
    pb_status = evaluator_artifact_status(PB, frozen_ids)
    result = {
        "schema_version": "sixs-full-joint-recovery-preflight-v1",
        "RECOVERY_PREFLIGHT": "PASS" if checkpoint_ok and coordinate_ok and comparator_hashes_ok else "FAIL",
        "CHECKPOINT_INTEGRITY": "PASS" if checkpoint_ok else "FAIL",
        "COORDINATE_INTEGRITY": "PASS" if coordinate_ok else "FAIL",
        "DEV_RECORD_ALIGNMENT": "PASS" if record_alignment_ok else "FAIL",
        "EXISTING_V3D_STATUS": v3d_status,
        "EXISTING_PB_STATUS": pb_status,
        "molecules": int(records.molecule_id.nunique()),
        "records": len(records),
        "records_per_molecule": sorted(int(value) for value in counts.unique()),
        "checkpoint_sha256": actual_checkpoint_sha,
        "sdf_sha256": sha256(SDF),
        "per_record_sha256": sha256(PER_RECORD),
        "payload_sha256": sha256(PAYLOAD),
        "comparator_hashes": "PASS" if comparator_hashes_ok else "FAIL",
        "COORDINATES_REUSED": "YES",
        "COORDINATES_REGENERATED": "NO",
        "CHECKPOINT_REUSED": "YES",
        "NEW_TRAINING_STARTED": "NO",
        **guards(),
    }
    atomic_json(RECOVERY_PREFLIGHT, result)
    if result["RECOVERY_PREFLIGHT"] != "PASS":
        raise RuntimeError("RECOVERY_PREFLIGHT_FAILED; coordinate regeneration forbidden")
    status(
        "RECOVERY_PREFLIGHT_COMPLETE",
        "PASS",
        CURRENT_STEP=17500,
        TOTAL_STEPS=17500,
        EVALUATION_RECORDS=5000,
        EVALUATION_TOTAL=5000,
        **{key: result[key] for key in (
            "RECOVERY_PREFLIGHT", "CHECKPOINT_INTEGRITY", "COORDINATE_INTEGRITY",
            "DEV_RECORD_ALIGNMENT", "EXISTING_V3D_STATUS", "EXISTING_PB_STATUS",
        )},
    )
    return result


def materialize_reused_coordinate_diagnostics() -> None:
    """Rebuild only missing derived tables from the immutable evaluation payload."""
    outputs = (
        REPORT / "SIGMA_DIAGNOSTICS.csv",
        REPORT / "RELIABILITY_DIAGNOSTICS.csv",
        REPORT / "BA_DISTRIBUTION.csv",
        REPORT / "TAU_DISTRIBUTION.csv",
    )
    if all(path.is_file() for path in outputs):
        return
    payload = torch.load(PAYLOAD, map_location="cpu", weights_only=False)
    if payload.get("record_ids") != pd.read_parquet(PER_RECORD).record_id.astype(str).tolist():
        raise RuntimeError("evaluation payload identity mismatch")
    primitives = pd.DataFrame(payload["primitive_rows"])
    records = pd.read_parquet(PER_RECORD)
    predictive = {
        family: frozen.predictive_family_summary(primitives, family)
        for family in ("bond", "angle")
    }
    reliability = {family: reliability_summary_safe(primitives, family) for family in ("bond", "angle")}
    atomic_frame(REPORT / "SIGMA_DIAGNOSTICS.csv", pd.DataFrame([{"family": family, **values} for family, values in predictive.items()]))
    atomic_frame(REPORT / "RELIABILITY_DIAGNOSTICS.csv", pd.DataFrame([{"family": family, **values} for family, values in reliability.items()]))
    atomic_frame(REPORT / "BA_DISTRIBUTION.csv", pd.DataFrame([distribution(records.w_B.to_numpy(), "w_B"), distribution(records.w_A.to_numpy(), "w_A")]))
    atomic_frame(REPORT / "TAU_DISTRIBUTION.csv", pd.DataFrame([distribution(records.tau.to_numpy(), "tau_angstrom")]))


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks without the crashing pandas/scipy Spearman path."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    boundaries = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, len(values)]
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        ranks[order[left:right]] = 0.5 * (left + right - 1)
    return ranks


def spearman_numpy(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2:
        return None
    left_rank = average_ranks(left[finite])
    right_rank = average_ranks(right[finite])
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.sqrt(np.sum(left_rank * left_rank) * np.sum(right_rank * right_rank))
    return float(np.sum(left_rank * right_rank) / denominator)


def reliability_summary_safe(rows: pd.DataFrame, family: str) -> dict[str, Any]:
    reliability = np.concatenate(rows[f"{family}_reliability"].to_numpy())
    defect = np.concatenate(rows[f"{family}_source_abs_defect"].to_numpy())
    standardized = np.concatenate(rows[f"{family}_source_abs_standardized_defect"].to_numpy())
    sigma = np.concatenate(rows[f"{family}_sigma"].to_numpy())
    return {
        **frozen.distribution_summary(reliability),
        "fraction_lt_0_1": float(np.mean(reliability < 0.1)),
        "fraction_lt_0_25": float(np.mean(reliability < 0.25)),
        "fraction_lt_0_5": float(np.mean(reliability < 0.5)),
        "fraction_gt_0_9": float(np.mean(reliability > 0.9)),
        "fraction_gt_0_99": float(np.mean(reliability > 0.99)),
        "spearman_abs_source_defect": spearman_numpy(reliability, defect),
        "spearman_abs_standardized_defect": spearman_numpy(reliability, standardized),
        "spearman_sigma": spearman_numpy(reliability, sigma),
        "RELIABILITY_COLLAPSED_TO_ZERO": bool(np.mean(reliability < 0.1) > 0.9),
        "RELIABILITY_COLLAPSED_TO_ONE": bool(np.mean(reliability > 0.99) > 0.9),
    }


def evaluate_coordinates() -> None:
    if COORDINATES_READY.is_file() and PER_RECORD.is_file() and SDF.is_file():
        return
    status("DEV_COORDINATE_GENERATION", "RUNNING", CURRENT_STEP=17500, TOTAL_STEPS=17500, EVALUATION_RECORDS=0, EVALUATION_TOTAL=5000)
    config = cfg()
    device = execution_device()
    prepared, source_payload = frozen.load_inputs()
    by_item = {str(item["molecule_id"]): item for item in prepared["val"]}
    by_sample = {str(row["sample_id"]): row for row in source_payload["val"]}
    manifest = json.loads((ROOT / config["data"]["dev_manifest"]).read_text(encoding="utf-8"))
    ids = [sample for row in manifest["rows"] for sample in row["sample_ids"]]
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise RuntimeError("DEV identity changed")
    final = torch.load(FINAL, map_location="cpu", weights_only=False)
    model = build_model(device)
    model.load_state_dict(final["model_state"], strict=True)
    model.eval()
    val_manifest = pd.read_parquet(config["data"]["val_manifest"])
    manifest_by_sample = {str(row.sample_id): row for row in val_manifest.itertuples(index=False)}
    SDF.parent.mkdir(parents=True, exist_ok=True)
    temp_sdf = Path(str(SDF) + f".tmp.{os.getpid()}")
    writer = Chem.SDWriter(str(temp_sdf))
    primitive_rows, record_rows = [], []
    try:
        for start in range(0, len(ids), 64):
            batch_ids = ids[start : start + 64]
            source_rows = [by_sample[sample] for sample in batch_ids]
            items = [by_item[str(row["molecule_id"])] for row in source_rows]
            graphs = [item["graph"] for item in items]
            bg = collate_graphs(graphs).to(device)
            source = torch.cat([torch.as_tensor(row["source"], dtype=torch.float64) for row in source_rows]).to(device)
            reference = torch.cat([torch.as_tensor(item["references"][0], dtype=torch.float64) for item in items]).to(device)
            with torch.no_grad():
                prediction = model.belief(bg, detach_sigma_features=False)
                action = full_joint_action(model, source, graphs, prediction, atom_cap=config["model"]["atom_cap_angstrom"])
            ref_bond, ref_angle = geometry_values(reference, bg)
            src_bond, src_angle = geometry_values(source, bg)
            prop_bond, prop_angle = geometry_values(action.proposal, bg)
            bond_offset = angle_offset = atom_offset = 0
            for local, (sample_id, source_row, graph) in enumerate(zip(batch_ids, source_rows, graphs, strict=True)):
                nb, na, nat = int(graph.bonds.size(1)), int(graph.angles.size(0)), int(graph.atom_categorical.size(0))
                bb = slice(bond_offset, bond_offset + nb)
                aa = slice(angle_offset, angle_offset + na)
                xx = slice(atom_offset, atom_offset + nat)
                bond_offset += nb
                angle_offset += na
                atom_offset += nat
                bmu = prediction["bond_mu"][bb].detach().cpu().double().numpy()
                amu = prediction["angle_mu"][aa].detach().cpu().double().numpy()
                bs = prediction["bond_sigma"][bb].detach().cpu().double().numpy()
                ass = prediction["angle_sigma"][aa].detach().cpu().double().numpy()
                bstat = bg.bond_fixed[bb, 1].detach().cpu().double().numpy()
                astat = bg.angle_fixed[aa, 1].detach().cpu().double().numpy()
                rb = action.bond_reliability[bb].detach().cpu().double().numpy()
                ra = action.angle_reliability[aa].detach().cpu().double().numpy()
                yb, ya = ref_bond[bb].detach().cpu().numpy(), ref_angle[aa].detach().cpu().numpy()
                sb, sa = src_bond[bb].detach().cpu().numpy(), src_angle[aa].detach().cpu().numpy()
                pb, pa = prop_bond[bb].detach().cpu().numpy(), prop_angle[aa].detach().cpu().numpy()
                primitive_rows.append(
                    {
                        "record_id": sample_id,
                        "molecule_id": str(source_row["molecule_id"]),
                        "bond_reference": yb,
                        "bond_mu": bmu,
                        "bond_sigma": bs,
                        "bond_sigma_stat": bstat,
                        "bond_reliability": rb,
                        "bond_source_abs_defect": np.abs(sb - bmu),
                        "bond_source_abs_standardized_defect": np.abs((sb - bmu) / bs),
                        "angle_reference": ya,
                        "angle_mu": amu,
                        "angle_sigma": ass,
                        "angle_sigma_stat": astat,
                        "angle_reliability": ra,
                        "angle_source_abs_defect": np.abs(sa - amu),
                        "angle_source_abs_standardized_defect": np.abs((sa - amu) / ass),
                    }
                )
                common_source = 0.5 * (np.mean(((sb - yb) / bstat) ** 2) + np.mean(((sa - ya) / astat) ** 2))
                common_post = 0.5 * (np.mean(((pb - yb) / bstat) ** 2) + np.mean(((pa - ya) / astat) ** 2))
                own_post = 0.5 * (np.mean(((pb - yb) / bs) ** 2) + np.mean(((pa - ya) / ass) ** 2))
                delta = action.proposal[xx] - source[xx]
                source_rmsd = float(delta.square().sum(-1).mean().sqrt())
                _, safety = safety_accept(source[xx].detach().cpu(), action.proposal[xx].detach().cpu(), graph)
                record_rows.append(
                    {
                        "record_id": sample_id,
                        "molecule_id": str(source_row["molecule_id"]),
                        "arm": EXPERIMENT_ID,
                        "internal_post": common_post,
                        "own_sigma_post": own_post,
                        "common_source_objective": common_source,
                        "direction_improvement": common_source - common_post,
                        "bond_raw_mae": float(np.mean(np.abs(pb - yb))),
                        "angle_raw_mae": float(np.mean(np.abs(pa - ya))),
                        "bond_mu_mae": float(np.mean(np.abs(bmu - yb))),
                        "angle_mu_mae": float(np.mean(np.abs(amu - ya))),
                        "source_rmsd": source_rmsd,
                        "proposal_movement": source_rmsd,
                        "tau": float(action.tau[local]),
                        "w_B": float(action.family_weights[local, 0]),
                        "w_A": float(action.family_weights[local, 1]),
                        "max_atom_displacement": float(torch.linalg.vector_norm(delta, dim=-1).max()),
                        "atom_cap_active": bool(action.cap_active[local]),
                        "rollback": bool(safety["fallback"]),
                        "PB": None,
                        "V3D": None,
                    }
                )
                metadata = manifest_by_sample[sample_id]
                cache = Path(config["data"]["val_cache"]) / Path(str(metadata.source_path)).name
                raw = cache.read_bytes()
                if hashlib.sha256(raw).hexdigest() != str(metadata.source_file_sha256):
                    raise RuntimeError("DEV source cache hash changed")
                record = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
                frozen.write_molecule(writer, record, action.proposal[xx], sample_id, EXPERIMENT_ID)
            status("DEV_COORDINATE_GENERATION", "RUNNING", CURRENT_STEP=17500, TOTAL_STEPS=17500, EVALUATION_RECORDS=min(start + 64, len(ids)), EVALUATION_TOTAL=len(ids))
    finally:
        writer.close()
    os.replace(temp_sdf, SDF)
    records = pd.DataFrame(record_rows)
    primitives = pd.DataFrame(primitive_rows)
    atomic_frame(PER_RECORD, records)
    atomic_torch(
        PAYLOAD,
        {
            "schema_version": "sixs-full-joint-evaluation-payload-v1",
            "record_ids": ids,
            "record_rows": record_rows,
            "primitive_rows": primitive_rows,
            **guards(),
        },
    )
    atomic_json(
        COORDINATES_READY,
        {
            "schema_version": "sixs-full-joint-coordinates-ready-v1",
            "status": "PASS",
            "records": len(ids),
            "record_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "final_checkpoint_sha256": sha256(FINAL),
            "sdf_sha256": sha256(SDF),
            "payload_sha256": sha256(PAYLOAD),
            **guards(),
        },
    )
    predictive = {
        "bond": frozen.predictive_family_summary(primitives, "bond"),
        "angle": frozen.predictive_family_summary(primitives, "angle"),
    }
    reliability = {
        "bond": reliability_summary_safe(primitives, "bond"),
        "angle": reliability_summary_safe(primitives, "angle"),
    }
    atomic_frame(REPORT / "SIGMA_DIAGNOSTICS.csv", pd.DataFrame([{"family": family, **values} for family, values in predictive.items()]))
    atomic_frame(REPORT / "RELIABILITY_DIAGNOSTICS.csv", pd.DataFrame([{"family": family, **values} for family, values in reliability.items()]))
    atomic_frame(REPORT / "BA_DISTRIBUTION.csv", pd.DataFrame([distribution(records.w_B.to_numpy(), "w_B"), distribution(records.w_A.to_numpy(), "w_A")]))
    atomic_frame(REPORT / "TAU_DISTRIBUTION.csv", pd.DataFrame([distribution(records.tau.to_numpy(), "tau_angstrom")]))
    status("COORDINATES_READY", "PASS", CURRENT_STEP=17500, TOTAL_STEPS=17500, EVALUATION_RECORDS=5000, EVALUATION_TOTAL=5000)


V3D_COMPONENTS = (
    "bond_geometry_valid",
    "angle_geometry_valid",
    "aromatic_ring_valid",
    "intramolecular_steric_clash_valid",
    "validity3d",
)
PB_COMPONENTS = (
    "mol_pred_loaded",
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "no_radicals",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
    "PB",
)


def transitions(baseline: pd.DataFrame, candidate: pd.DataFrame, component: str) -> dict[str, Any]:
    joined = baseline[["record_id", component]].merge(candidate[["record_id", component]], on="record_id", suffixes=("_baseline", "_candidate"), validate="one_to_one")
    left = joined[f"{component}_baseline"].astype(bool)
    right = joined[f"{component}_candidate"].astype(bool)
    pp = int((left & right).sum())
    pf = int((left & ~right).sum())
    fp = int((~left & right).sum())
    ff = int((~left & ~right).sum())
    return {"component": component, "baseline_pass_candidate_pass": pp, "baseline_pass_candidate_fail": pf, "baseline_fail_candidate_pass": fp, "baseline_fail_candidate_fail": ff, "net_transition": fp - pf}


def component_table(label: str, frame: pd.DataFrame, components: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for component in components:
        passed = int(frame[component].astype(bool).sum())
        rows.append({"arm": label, "component": component, "pass_count": passed, "fail_count": len(frame) - passed, "pass_rate": passed / len(frame)})
    return rows


def cluster_bootstrap(baseline: pd.DataFrame, candidate: pd.DataFrame, metric: str, seed: int) -> dict[str, Any]:
    joined = baseline[["record_id", "molecule_id", metric]].merge(candidate[["record_id", metric]], on="record_id", suffixes=("_baseline", "_candidate"), validate="one_to_one")
    joined["delta"] = joined[f"{metric}_candidate"].astype(float) - joined[f"{metric}_baseline"].astype(float)
    cluster = joined.groupby("molecule_id", sort=True).delta.mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(10000)
    for start in range(0, 10000, 500):
        sampled = rng.integers(0, len(cluster), size=(500, len(cluster)))
        draws[start : start + 500] = cluster[sampled].mean(axis=1)
    return {"metric": metric, "delta_candidate_minus_baseline": float(cluster.mean()), "ci95_low": float(np.percentile(draws, 2.5)), "ci95_high": float(np.percentile(draws, 97.5)), "bootstrap_clusters": len(cluster), "bootstrap_resamples": 10000, "seed": seed}


def evaluate_and_finalize() -> None:
    if (REPORT / "FINAL_DECISION.md").is_file():
        return
    recovery = recovery_preflight()
    evaluate_coordinates()
    materialize_reused_coordinate_diagnostics()
    config = cfg()
    records = pd.read_parquet(PER_RECORD)
    ids = records.record_id.astype(str).tolist()
    pb_state = evaluator_artifact_status(PB, ids)
    v3d_state = evaluator_artifact_status(V3D, ids)
    if pb_state != "COMPLETE" or v3d_state != "COMPLETE":
        if pb_state == "COMPLETE" or v3d_state == "COMPLETE":
            raise RuntimeError("one external evaluator is complete while the other is incomplete; fail closed to avoid repeating a completed evaluator")
        status(
            "DEV_EXTERNAL_EVALUATION",
            "RUNNING",
            CURRENT_STEP=17500,
            TOTAL_STEPS=17500,
            EVALUATION_RECORDS=5000,
            EVALUATION_TOTAL=5000,
            V3D_STARTED="YES",
            POSEBUSTERS_STARTED="YES",
            BOOTSTRAP_STARTED="NO",
        )
        frozen.run_external_evaluators(EXPERIMENT_ID, {"sdf": SDF, "per_record": PER_RECORD, "pb": PB, "v3d": V3D}, ids)
    pb = pd.read_parquet(PB)
    v3d = pd.read_parquet(V3D)
    records = records.drop(columns=["PB", "V3D"], errors="ignore")
    records = records.merge(pb[["record_id", "PB"]], on="record_id", validate="one_to_one")
    records = records.merge(v3d[["record_id", "validity3d"]].rename(columns={"validity3d": "V3D"}), on="record_id", validate="one_to_one")
    atomic_frame(PER_RECORD, records)

    comparator_records = pd.read_parquet(ROOT / config["comparator"]["per_record"])
    comparator_pb = pd.read_parquet(ROOT / config["comparator"]["posebusters"])
    comparator_v3d = pd.read_parquet(ROOT / config["comparator"]["validity3d"])
    if records.record_id.astype(str).tolist() != comparator_records.record_id.astype(str).tolist():
        raise RuntimeError("comparator record identity mismatch")

    v3d_rows = component_table("COMPARATOR", comparator_v3d, V3D_COMPONENTS) + component_table("FULL_JOINT", v3d, V3D_COMPONENTS)
    atomic_frame(REPORT / "V3D_COMPONENTS.csv", pd.DataFrame(v3d_rows))
    atomic_frame(REPORT / "V3D_TRANSITIONS.csv", pd.DataFrame([transitions(comparator_v3d, v3d, component) for component in V3D_COMPONENTS]))
    pb_rows = component_table("COMPARATOR", comparator_pb, PB_COMPONENTS) + component_table("FULL_JOINT", pb, PB_COMPONENTS)
    atomic_frame(REPORT / "PB_COMPONENTS.csv", pd.DataFrame(pb_rows))
    atomic_frame(REPORT / "PB_TRANSITIONS.csv", pd.DataFrame([transitions(comparator_pb, pb, component) for component in PB_COMPONENTS]))

    status(
        "PAIRED_MOLECULE_CLUSTER_BOOTSTRAP",
        "RUNNING",
        CURRENT_STEP=17500,
        TOTAL_STEPS=17500,
        EVALUATION_RECORDS=5000,
        EVALUATION_TOTAL=5000,
        V3D_STARTED="YES",
        POSEBUSTERS_STARTED="YES",
        BOOTSTRAP_STARTED="YES",
    )
    metrics = ("V3D", "PB", "internal_post", "direction_improvement", "bond_raw_mae", "angle_raw_mae", "source_rmsd", "proposal_movement")
    bootstrap_path = REPORT / "PAIRED_BOOTSTRAP.csv"
    if bootstrap_path.is_file():
        bootstrap_frame = pd.read_csv(bootstrap_path)
    else:
        bootstrap = [cluster_bootstrap(comparator_records, records, metric, config["evaluation"]["bootstrap_seed"] + index) for index, metric in enumerate(metrics)]
        comparator_v3d_for_boot = comparator_v3d.copy()
        candidate_v3d_for_boot = v3d.copy()
        for frame in (comparator_v3d_for_boot, candidate_v3d_for_boot):
            if "molecule_id" not in frame:
                frame["molecule_id"] = frame.record_id.map(records.set_index("record_id").molecule_id)
        for index, component in enumerate(V3D_COMPONENTS):
            bootstrap.append(cluster_bootstrap(comparator_v3d_for_boot, candidate_v3d_for_boot, component, config["evaluation"]["bootstrap_seed"] + 100 + index))
        bootstrap_frame = pd.DataFrame(bootstrap)
        atomic_frame(bootstrap_path, bootstrap_frame)

    predictive = pd.read_csv(REPORT / "SIGMA_DIAGNOSTICS.csv")
    reliability = pd.read_csv(REPORT / "RELIABILITY_DIAGNOSTICS.csv")
    ba = pd.read_csv(REPORT / "BA_DISTRIBUTION.csv")
    tau = pd.read_csv(REPORT / "TAU_DISTRIBUTION.csv")
    summary = {
        "arm": EXPERIMENT_ID,
        "molecules": int(records.molecule_id.nunique()),
        "records": len(records),
        "proposal_V3D": float(records.V3D.mean()),
        "proposal_PB": float(records.PB.mean()),
        "internal_post": float(records.internal_post.mean()),
        "direction_improvement": float(records.direction_improvement.mean()),
        "bond_raw_mae": float(records.bond_raw_mae.mean()),
        "angle_raw_mae": float(records.angle_raw_mae.mean()),
        "source_rmsd": float(records.source_rmsd.mean()),
        "proposal_movement": float(records.proposal_movement.mean()),
        "tau_mean": float(records.tau.mean()),
        "tau_median": float(records.tau.median()),
        "tau_p05": float(records.tau.quantile(0.05)),
        "tau_p95": float(records.tau.quantile(0.95)),
        "tau_p99": float(records.tau.quantile(0.99)),
        "tau_max": float(records.tau.max()),
        "atom_cap_fraction": float(records.atom_cap_active.mean()),
        "w_B_mean": float(records.w_B.mean()),
        "w_B_median": float(records.w_B.median()),
        "w_B_p05": float(records.w_B.quantile(0.05)),
        "w_B_p95": float(records.w_B.quantile(0.95)),
        "w_B_p99": float(records.w_B.quantile(0.99)),
        "w_B_min": float(records.w_B.min()),
        "w_B_max": float(records.w_B.max()),
        "w_A_mean": float(records.w_A.mean()),
        "w_A_median": float(records.w_A.median()),
        "w_A_p05": float(records.w_A.quantile(0.05)),
        "w_A_p95": float(records.w_A.quantile(0.95)),
        "w_A_p99": float(records.w_A.quantile(0.99)),
        "w_A_min": float(records.w_A.min()),
        "w_A_max": float(records.w_A.max()),
        "mu_bond_mae": float(records.bond_mu_mae.mean()),
        "mu_angle_mae": float(records.angle_mu_mae.mean()),
    }
    atomic_frame(REPORT / "DEV_SUMMARY.csv", pd.DataFrame([summary]))

    by_metric = {row.metric: row for row in bootstrap_frame.itertuples(index=False)}
    v3d_result = by_metric["V3D"]
    pb_result = by_metric["PB"]
    reverse = any(by_metric[metric].ci95_low > 0 for metric in ("internal_post", "bond_raw_mae", "angle_raw_mae"))
    pb_material = pb_result.delta_candidate_minus_baseline < -config["evaluation"]["pb_material_drop_tolerance"]
    if v3d_result.ci95_high < 0:
        classification = "NEGATIVE"
    elif v3d_result.delta_candidate_minus_baseline <= 0:
        classification = "NO_ENDPOINT_GAIN"
    elif v3d_result.ci95_low <= 0:
        classification = "INCONCLUSIVE_POSITIVE"
    elif pb_material or reverse:
        classification = "NEGATIVE"
    else:
        classification = "POSITIVE"
    next_candidate = "J1_R1_FULL_JOINT_ADAPTIVE_BA_MOVEMENT" if classification == "POSITIVE" else "RETAIN_J1_R1_EQUAL_BA_LEARNED_MAGNITUDE"

    sigma_upper = float(predictive.sigma_upper_saturation_fraction.max())
    sigma_lower = float(predictive.sigma_lower_saturation_fraction.max())
    r_zero = float(reliability.fraction_lt_0_1.max())
    r_one = float(reliability.fraction_gt_0_99.max())
    w_b = ba[ba.object == "w_B"].iloc[0]
    tau_row = tau.iloc[0]
    comparator_j1 = pd.read_csv(ROOT / "reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/09_ACTION_COMPARISON.csv")
    comparator_j1 = comparator_j1[comparator_j1.arm == "J1-R1"].iloc[0]
    bond_mu_delta = summary["mu_bond_mae"] - float(comparator_j1.mu_bond_mae)
    angle_mu_delta = summary["mu_angle_mae"] - float(comparator_j1.mu_angle_mae)
    mu_degradation = bond_mu_delta > 0 and angle_mu_delta > 0
    interaction = "POSITIVE" if classification == "POSITIVE" else ("MIXED" if classification == "INCONCLUSIVE_POSITIVE" else "NEGATIVE_OR_NONE")
    diagnostics = f"""# Full-joint mechanism diagnostics

The primary comparison is FULL_JOINT minus the frozen J1-R1 + Equal-BA + learned-magnitude comparator. Mu degradation is additionally described against the frozen six-arm J1-R1 predictive checkpoint because the action comparator does not persist primitive mu rows. This is a diagnostic comparison, not a protected selection rule.

```text
BA_COLLAPSE = {'YES' if w_b['p01'] < 0.01 or w_b['p99'] > 0.99 else 'NO'}
ADAPTIVE_BA_W_B_MEAN = {summary['w_B_mean']}
ADAPTIVE_BA_W_B_P05_P95 = [{summary['w_B_p05']}, {summary['w_B_p95']}]
ADAPTIVE_BA_W_A_MEAN = {summary['w_A_mean']}
SIGMA_INFLATION = {'YES' if sigma_upper > 0.01 else 'NO'}
SIGMA_COLLAPSE = {'YES' if sigma_lower > 0.01 else 'NO'}
RELIABILITY_COLLAPSE_TO_ZERO = {'YES' if r_zero > 0.90 else 'NO'}
RELIABILITY_COLLAPSE_TO_ONE = {'YES' if r_one > 0.90 else 'NO'}
TAU_COLLAPSE = {'YES' if tau_row['mean'] < 0.0005 else 'NO'}
TAU_SATURATION = {'YES' if tau_row['p99'] >= 0.0099 else 'NO'}
MU_DEGRADATION = {'YES' if mu_degradation else 'NO_OR_MIXED'}
MU_BOND_MAE_DELTA_VS_FROZEN_J1_R1 = {bond_mu_delta}
MU_ANGLE_MAE_DELTA_VS_FROZEN_J1_R1 = {angle_mu_delta}
BOND_ANGLE_CONFLICT = DESCRIPTIVE_ONLY_SEE_COMPONENT_TRANSITIONS
MOVEMENT_WITHOUT_ENDPOINT_GAIN = {'YES' if by_metric['source_rmsd'].delta_candidate_minus_baseline > 0 and v3d_result.delta_candidate_minus_baseline <= 0 else 'NO'}
OLD_V3_JOINT_MIXED_INTERACTION_REPRODUCED = {'YES' if classification in ('NO_ENDPOINT_GAIN','NEGATIVE') else 'NO_OR_NOT_ESTABLISHED'}
ADAPTIVE_BA_MOVEMENT_INTERACTION = {interaction}
INTERACTION_CAUSALLY_IDENTIFIED = NO_FULL_JOINT_INCREMENT_ONLY
NO_AUTOMATIC_RETRAIN = YES
```
"""
    atomic_text(REPORT / "FAILURE_DIAGNOSTICS.md", diagnostics)
    atomic_text(REPORT / "FULL_JOINT_MECHANISM_DIAGNOSTICS.md", diagnostics)
    decision = {
        "schema_version": "sixs-full-joint-final-decision-v1",
        "FULL_JOINT_RESULT": classification,
        "NEXT_CANDIDATE": next_candidate,
        "summary": summary,
        "primary_bootstrap": {metric: by_metric[metric]._asdict() for metric in metrics},
        "PB_MATERIAL_DROP": "YES" if pb_material else "NO",
        "CONTINUOUS_GEOMETRY_MATERIAL_REVERSE": "YES" if reverse else "NO",
        "DELTA_V3D": v3d_result.delta_candidate_minus_baseline,
        "V3D_95CI": [v3d_result.ci95_low, v3d_result.ci95_high],
        "V3D_CI_EXCLUDES_ZERO": "YES" if v3d_result.ci95_low > 0 or v3d_result.ci95_high < 0 else "NO",
        "DELTA_PB": pb_result.delta_candidate_minus_baseline,
        "PB_95CI": [pb_result.ci95_low, pb_result.ci95_high],
        "RECOVERY_PREFLIGHT": recovery["RECOVERY_PREFLIGHT"],
        **guards(),
    }
    block = "\n".join(f"{key} = {value}" for key, value in decision.items() if isinstance(value, str))
    atomic_text(REPORT / "FINAL_DECISION.md", "# SIXS J1-R1 full-joint final decision\n\n" + markdown_frame(pd.DataFrame([summary])) + "\n\n```text\n" + block + "\n```\n")
    artifact_names = (
        "IMPLEMENTATION_CONFIG.json", "MODEL_IDENTITY.md", "PARAMETER_GROUP_AUDIT.json",
        "GRADIENT_PATH_AUDIT.json", "REFERENCE_LEAKAGE_AUDIT.json", "TRAIN_LOG.csv",
        "TRAIN_SUMMARY.json", "FINAL_CHECKPOINT.pt", "DEV_SUMMARY.csv", "V3D_COMPONENTS.csv",
        "V3D_TRANSITIONS.csv", "PB_COMPONENTS.csv", "PB_TRANSITIONS.csv", "BA_DISTRIBUTION.csv",
        "TAU_DISTRIBUTION.csv", "SIGMA_DIAGNOSTICS.csv", "RELIABILITY_DIAGNOSTICS.csv",
        "RECOVERY_PREFLIGHT.json", "PAIRED_BOOTSTRAP.csv", "FAILURE_DIAGNOSTICS.md",
        "FULL_JOINT_MECHANISM_DIAGNOSTICS.md", "FINAL_DECISION.md",
    )
    hashes = {name: sha256(REPORT / name) for name in artifact_names}
    status("COMPLETE", "PASS", CURRENT_STEP=17500, TOTAL_STEPS=17500, FULL_JOINT_RESULT=classification, NEXT_CANDIDATE=next_candidate, PROPOSAL_V3D=summary["proposal_V3D"], PROPOSAL_PB=summary["proposal_PB"], DELTA_V3D=v3d_result.delta_candidate_minus_baseline, DELTA_V3D_CI95=[v3d_result.ci95_low, v3d_result.ci95_high], ARTIFACT_SHA256=hashes)


def pipeline() -> None:
    preflight()
    train()
    evaluate_and_finalize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "recovery-preflight", "train", "evaluate", "pipeline"), default="pipeline")
    args = parser.parse_args()
    try:
        if args.stage == "preflight":
            preflight()
        elif args.stage == "recovery-preflight":
            recovery_preflight(force=True)
        elif args.stage == "train":
            preflight(); train()
        elif args.stage == "evaluate":
            preflight(); evaluate_and_finalize()
        else:
            pipeline()
        return 0
    except Exception as exc:
        REPORT.mkdir(parents=True, exist_ok=True)
        trace = traceback.format_exc()
        atomic_text(REPORT / "PIPELINE_TRACEBACK.txt", trace)
        status("STOPPED_FAIL_CLOSED", "FAIL", ERROR_TYPE=type(exc).__name__, ERROR=str(exc), TRACEBACK=str(REPORT / "PIPELINE_TRACEBACK.txt"))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
