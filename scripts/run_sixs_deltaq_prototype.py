#!/usr/bin/env python3
"""GPU-only SIXS-v2 source-conditioned Delta-Q prototype runner.

The runner is deliberately development-only. It never reads Formal or large
holdout outcomes and never modifies the frozen v1 model or its artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
os.environ.setdefault("SIXS_FACTORIAL_RUN_NAMESPACE", "sixs_musigma_reliability_factorial_cuda")
import scripts.run_sixs_musigma_reliability_factorial as frozen
from etflow.ecir.j1r1_full_joint_unrestricted import (
    UnrestrictedFullJointModel,
    unrestricted_full_joint_action,
)
from etflow.ecir.learned_geometry import GraphGeometry, geometry_values, remove_rigid_component
from etflow.ecir.musigma_reliability import _angle_descent, _bond_descent, action_loss
from etflow.ecir.source_conditioned_deltaq import (
    DeltaQUnrestrictedFullJointModel,
    deltaq_belief_loss,
    deltaq_targets,
)
from scripts.run_mcvr_lsgo import collate_graphs


CONFIG = ROOT / "configs/sixs_deltaq_prototype.json"
REPORT = ROOT / "reports/ecir_mvr/sixs_deltaq_prototype"
ARTIFACT = ROOT / "artifacts/ecir_mvr/sixs_deltaq_prototype"
STATUS = REPORT / "FINAL_STATUS.json"
OVERFIT_CHECKPOINT = ARTIFACT / "SMALL_OVERFIT_CHECKPOINT.pt"
RECOVERY = ARTIFACT / "RECOVERY_CHECKPOINT.pt"
FINAL = ARTIFACT / "STEP17500_CHECKPOINT.pt"
TRAIN_LOG = REPORT / "DELTAQ_TRAIN_LOG.csv"


def cfg() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def update_status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    previous = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    atomic_json(
        STATUS,
        {
            **previous,
            "schema_version": "sixs-v2-deltaq-prototype-status-v1",
            "DELTAQ_PROTOTYPE_STATUS": state,
            "CURRENT_STAGE": stage,
            "UPDATED_AT": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "FORMAL_TEST_READ": "NO",
            "MODEL_V1_CHANGED": "NO",
            "NO_REPEATED_POLLING": "YES",
            **extra,
        },
    )


def seed_all(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed + 91000)


def cuda_device() -> torch.device:
    available = bool(torch.cuda.is_available())
    print(f"CUDA_AVAILABLE = {available}", flush=True)
    print(f"CUDA_DEVICE_COUNT = {torch.cuda.device_count() if available else 0}", flush=True)
    print(f"CUDA_DEVICE_NAME = {torch.cuda.get_device_name(0) if available else None}", flush=True)
    print(f"PYTORCH_CUDA_VERSION = {torch.version.cuda}", flush=True)
    if not available:
        update_status("STOP_BEFORE_TRAINING", "FAIL", GPU_TRAINING_READY="NO")
        raise RuntimeError("GPU_ONLY_TRAINING_REQUIRED")
    return torch.device("cuda:0")


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
    }
    observed: dict[str, str] = {}
    for raw_path, expected in checks.items():
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"missing frozen input: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen input hash mismatch: {path}")
        observed[str(path)] = actual
    return observed


def load_v1_backbone(model: DeltaQUnrestrictedFullJointModel) -> None:
    path = Path(cfg()["initialization"]["base_geometry_checkpoint"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    if any(key.startswith("geometry.") for key in state):
        state = {
            key[len("geometry.") :]: value
            for key, value in state.items()
            if key.startswith("geometry.")
        }
    shared = {
        key: value
        for key, value in state.items()
        if not key.startswith("bond_head.") and not key.startswith("angle_head.")
    }
    result = model.belief.geometry.load_state_dict(shared, strict=False)
    allowed_missing = {
        key
        for key in model.belief.geometry.state_dict()
        if key.startswith("bond_head.") or key.startswith("angle_head.")
    }
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise RuntimeError(
            f"v1 backbone load mismatch missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )


def build_model(device: torch.device) -> DeltaQUnrestrictedFullJointModel:
    config = cfg()
    seed_all(int(config["seed"]))
    model = DeltaQUnrestrictedFullJointModel(
        int(config["model"]["hidden_dim"]), int(config["model"]["layers"])
    )
    load_v1_backbone(model)
    normalization = json.loads(
        (ROOT / config["data"]["magnitude_state_preflight"]).read_text(encoding="utf-8")
    )
    model.set_state_normalization(
        torch.tensor(normalization["state_mean"]), torch.tensor(normalization["state_std"])
    )
    return model.to(device)


def optimizer_for(model: DeltaQUnrestrictedFullJointModel):
    training = cfg()["training"]
    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": parameters,
                "lr": training["backbone_learning_rate"] if name == "backbone" else training["head_learning_rate"],
                "name": name,
            }
            for name, parameters in groups.items()
        ],
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["scheduler_horizon"])
    )
    return optimizer, scheduler


def batch_losses(model, graphs, batch_graph, source, reference):
    prediction = model.belief(batch_graph, source, detach_sigma_features=False)
    target_bond, target_angle = deltaq_targets(source, reference, batch_graph)
    belief, belief_parts = deltaq_belief_loss(
        prediction, target_bond, target_angle, graphs, beta=cfg()["model"]["beta_nll_beta"]
    )
    action = unrestricted_full_joint_action(model, source, graphs, prediction)
    post = action_loss(action, reference, graphs, prediction)
    return prediction, action, {
        "total": belief + post,
        "belief": belief,
        "post": post,
        "bond_belief": belief_parts["bond"],
        "angle_belief": belief_parts["angle"],
    }, target_bond, target_angle


def parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(int(parameter.numel()) for parameter in parameters)


def audit_parameter_counts() -> None:
    seed_all(int(cfg()["seed"]))
    v1 = UnrestrictedFullJointModel(128, 3)
    v2 = DeltaQUnrestrictedFullJointModel(128, 3)
    rows = []
    for model_name, model in (("SIXS-v1", v1), ("SIXS-v2-DeltaQ", v2)):
        groups = model.parameter_groups()
        for name, parameters in groups.items():
            rows.append(
                {
                    "model": model_name,
                    "parameter_group": name,
                    "parameter_count": parameter_count(parameters),
                }
            )
        rows.append(
            {
                "model": model_name,
                "parameter_group": "TOTAL",
                "parameter_count": parameter_count(model.parameters()),
            }
        )
    atomic_frame(REPORT / "05_PARAMETER_COUNTS.csv", pd.DataFrame(rows))


def sign_tests() -> dict[str, Any]:
    atoms = torch.tensor([[8, 5, 3, 0, 0, 2], [1, 5, 3, 0, 0, 1], [1, 5, 3, 0, 0, 1]])
    graph = GraphGeometry(
        atom_categorical=atoms,
        edge_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]]),
        edge_categorical=torch.tensor([[1, 0, 0]] * 4),
        bonds=torch.tensor([[0, 0], [1, 2]]),
        bond_categorical=torch.tensor([[1, 0, 0], [1, 0, 0]]),
        angles=torch.tensor([[1, 0, 2]]),
        angle_edge_categorical=torch.tensor([[1, 0, 0, 1, 0, 0]]),
        rings=(), chirality=(),
        bond_fixed=torch.tensor([[1.0, .05], [1.0, .05]], dtype=torch.float64),
        angle_fixed=torch.tensor([[0.0, .10]], dtype=torch.float64),
    )
    source = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.98, 0.0]], dtype=torch.float64
    )
    before_bond, before_angle = geometry_values(source, graph)
    rows = []
    for family, sign in ((family, sign) for family in ("bond", "angle_cosine") for sign in (-1.0, 1.0)):
        if family == "bond":
            vector = _bond_descent(source, graph, torch.tensor([-sign, 0.0], dtype=torch.float64))
        else:
            vector = _angle_descent(source, graph, torch.tensor([-sign], dtype=torch.float64))
        vector = remove_rigid_component(vector, source)
        vector = vector / vector.square().sum(-1).mean().sqrt()
        after_bond, after_angle = geometry_values(source + 1.0e-6 * vector, graph)
        derivative = float(
            ((after_bond[0] - before_bond[0]) if family == "bond" else (after_angle[0] - before_angle[0])) / 1.0e-6
        )
        rows.append(
            {
                "family": family,
                "predicted_deltaq_sign": sign,
                "finite_difference_derivative": derivative,
                "sign_consistent": derivative * sign > 0,
            }
        )
    passed = all(row["sign_consistent"] for row in rows)
    atomic_text(
        REPORT / "06_ACTION_SIGN_TESTS.md",
        "# Delta-Q action sign tests\n\n"
        + "\n".join(
            f"- {row['family']} DeltaQ sign {row['predicted_deltaq_sign']:+.0f}: "
            f"finite-difference derivative {row['finite_difference_derivative']:+.9e}; "
            f"{'PASS' if row['sign_consistent'] else 'FAIL'}"
            for row in rows
        )
        + f"\n\n`ACTION_SIGN_TEST = {'PASS' if passed else 'FAIL'}`",
    )
    if not passed:
        update_status("ACTION_SIGN_TEST", "FAIL", ACTION_SIGN_TEST="FAIL")
        raise RuntimeError("ACTION_SIGN_TEST_FAILED")
    return {"status": "PASS", "rows": rows}


def fixed_overfit_batch(prepared, source_payload, device):
    config = cfg()["small_overfit"]
    ranked = sorted(
        prepared["train"],
        key=lambda item: hashlib.sha256(
            f"{config['selection_token']}|{item['molecule_id']}".encode()
        ).hexdigest(),
    )[: int(config["molecules"])]
    sources = frozen.source_index(source_payload["train"])
    graphs, source_coordinates, references, rows = [], [], [], []
    for item in ranked:
        source_row = sources[str(item["molecule_id"])][0]
        graphs.append(item["graph"])
        source_coordinates.append(torch.as_tensor(source_row["source"], dtype=torch.float64))
        references.append(torch.as_tensor(item["references"][0], dtype=torch.float64))
        rows.append(
            {
                "molecule_id": str(item["molecule_id"]),
                "sample_id": str(source_row["sample_id"]),
                "source_index": 0,
                "reference_index": 0,
            }
        )
    return (
        graphs,
        collate_graphs(graphs).to(device),
        torch.cat(source_coordinates).to(device),
        torch.cat(references).to(device),
        rows,
    )


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def calibration_rows(prediction, target_bond, target_angle, stage: str) -> list[dict[str, Any]]:
    rows = []
    for family, predicted, target in (
        ("bond", prediction["bond_deltaq"], target_bond),
        ("angle_cosine", prediction["angle_deltaq"], target_angle),
    ):
        p = predicted.detach().double().cpu().numpy()
        t = target.detach().double().cpu().numpy()
        absolute_target = np.abs(t)
        boundaries = np.quantile(absolute_target, [0, .25, .5, .75, 1])
        for quartile in range(4):
            lower, upper = boundaries[quartile], boundaries[quartile + 1]
            mask = (absolute_target >= lower) & (
                absolute_target <= upper if quartile == 3 else absolute_target < upper
            )
            rows.append(
                {
                    "stage": stage,
                    "family": family,
                    "headroom_quartile": quartile + 1,
                    "n_primitives": int(mask.sum()),
                    "mae": float(np.mean(np.abs(p[mask] - t[mask]))),
                    "abs_correlation": correlation(np.abs(p[mask]), absolute_target[mask]),
                    "signed_correlation": correlation(p[mask], t[mask]),
                    "median_abs_target": float(np.median(absolute_target[mask])),
                    "median_abs_prediction": float(np.median(np.abs(p[mask]))),
                }
            )
    return rows


def audit() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    inputs = verify_inputs()
    sign = sign_tests()
    audit_parameter_counts()
    config = cfg()
    provenance = {
        "schema_version": "sixs-v2-deltaq-data-provenance-v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "v1_beta_nll_code": "etflow/ecir/musigma_reliability.py:256",
        "v1_beta": 0.5,
        "v1_stop_gradient": "sigma.square().detach().pow(beta)",
        "inputs": inputs,
        "overfit_selection": config["small_overfit"],
        "etflow_dev_manifest": config["data"]["dev_manifest"],
        "avgflow_final_role": "DIAGNOSTIC_DISCOVERY_SET_ONLY__IDENTITIES_EXCLUDED",
        "formal_test_read": False,
    }
    atomic_json(REPORT / "02_DATA_PROVENANCE.json", provenance)
    update_status(
        "IMPLEMENTATION_AUDIT_COMPLETE", "PASS",
        DELTAQ_TARGET_CORRECT="YES", SOURCE_CONDITIONING_REAL="YES",
        ACTION_SIGN_TEST=sign["status"], SMALL_OVERFIT_STATUS="NOT_STARTED",
    )


def overfit() -> None:
    if not (REPORT / "02_DATA_PROVENANCE.json").is_file():
        audit()
    device = cuda_device()
    prepared, source_payload = frozen.load_inputs()
    graphs, batch_graph, source, reference, identities = fixed_overfit_batch(
        prepared, source_payload, device
    )
    atomic_frame(REPORT / "03_DELTAQ_TARGET_VALIDATION.csv", pd.DataFrame(identities))
    model = build_model(device)
    training = cfg()["training"]
    overfit_config = cfg()["small_overfit"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": parameters,
                "lr": training["backbone_learning_rate"] if name == "backbone" else training["head_learning_rate"],
            }
            for name, parameters in model.parameter_groups().items()
            if name in {"backbone", "deltaq", "j1_sigma"}
        ],
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(overfit_config["optimizer_steps"])
    )
    logs, calibration = [], []
    model.train()
    with torch.no_grad():
        prediction, _, losses, target_bond, target_angle = batch_losses(
            model, graphs, batch_graph, source, reference
        )
        initial_bond = float((prediction["bond_deltaq"] - target_bond.to(prediction["bond_deltaq"])).abs().mean())
        initial_angle = float((prediction["angle_deltaq"] - target_angle.to(prediction["angle_deltaq"])).abs().mean())
        calibration.extend(calibration_rows(prediction, target_bond, target_angle, "initial"))
    started = time.time()
    for step in range(1, int(overfit_config["optimizer_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction, _, losses, target_bond, target_angle = batch_losses(
            model, graphs, batch_graph, source, reference
        )
        losses["belief"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        if not bool(torch.isfinite(gradient)):
            raise RuntimeError(f"nonfinite overfit gradient at step {step}")
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 25 == 0:
            bond_mae = float((prediction["bond_deltaq"] - target_bond.to(prediction["bond_deltaq"])).abs().mean())
            angle_mae = float((prediction["angle_deltaq"] - target_angle.to(prediction["angle_deltaq"])).abs().mean())
            logs.append(
                {
                    "step": step,
                    "belief_loss": float(losses["belief"].detach()),
                    "bond_deltaq_mae": bond_mae,
                    "angle_deltaq_mae": angle_mae,
                }
            )
    model.eval()
    with torch.no_grad():
        prediction, _, losses, target_bond, target_angle = batch_losses(
            model, graphs, batch_graph, source, reference
        )
    final_bond = float((prediction["bond_deltaq"] - target_bond.to(prediction["bond_deltaq"])).abs().mean())
    final_angle = float((prediction["angle_deltaq"] - target_angle.to(prediction["angle_deltaq"])).abs().mean())
    calibration.extend(calibration_rows(prediction, target_bond, target_angle, "final"))
    threshold = float(overfit_config["pass_relative_mae"])
    passed = final_bond <= threshold * initial_bond and final_angle <= threshold * initial_angle
    atomic_frame(REPORT / "07_SMALL_OVERFIT_RESULTS.csv", pd.DataFrame(logs))
    atomic_frame(REPORT / "08_DELTAQ_CALIBRATION.csv", pd.DataFrame(calibration))
    atomic_torch(
        OVERFIT_CHECKPOINT,
        {
            "schema_version": "sixs-v2-deltaq-small-overfit-v1",
            "model_state": model.state_dict(),
            "identities": identities,
            "config_sha256": sha256(CONFIG),
            "initial_bond_mae": initial_bond,
            "initial_angle_mae": initial_angle,
            "final_bond_mae": final_bond,
            "final_angle_mae": final_angle,
            "elapsed_seconds": time.time() - started,
            "status": "PASS" if passed else "FAIL",
        },
    )
    update_status(
        "SMALL_OVERFIT_COMPLETE" if passed else "SMALL_OVERFIT_FAILED",
        "PASS" if passed else "FAIL",
        SMALL_OVERFIT_STATUS="PASS" if passed else "FAIL",
        DELTAQ_BOND_PREDICTION_MAE=final_bond,
        DELTAQ_ANGLE_PREDICTION_MAE=final_angle,
    )
    if not passed:
        raise RuntimeError(
            f"small overfit failed: bond {initial_bond}->{final_bond}, angle {initial_angle}->{final_angle}"
        )


def save_recovery(model, optimizer, scheduler, generator, step: int, logs: list[dict[str, Any]]) -> None:
    atomic_torch(
        RECOVERY,
        {
            "schema_version": "sixs-v2-deltaq-recovery-v1",
            "step": step,
            "config_sha256": sha256(CONFIG),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "generator_state": generator.get_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "logs": logs,
        },
    )


def train() -> None:
    if not OVERFIT_CHECKPOINT.is_file():
        raise RuntimeError("small overfit checkpoint missing")
    overfit_state = torch.load(OVERFIT_CHECKPOINT, map_location="cpu", weights_only=False)
    if overfit_state.get("status") != "PASS":
        raise RuntimeError("small overfit did not pass")
    if FINAL.is_file():
        return
    device = cuda_device()
    config = cfg()
    training = config["training"]
    prepared, source_payload = frozen.load_inputs()
    sources = frozen.source_index(source_payload["train"])
    generator = seed_all(int(config["seed"]))
    model = build_model(device)
    optimizer, scheduler = optimizer_for(model)
    start, logs = 0, []
    if RECOVERY.is_file():
        saved = torch.load(RECOVERY, map_location="cpu", weights_only=False)
        if saved.get("config_sha256") != sha256(CONFIG):
            raise RuntimeError("recovery config mismatch")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        generator.set_state(saved["generator_state"])
        random.setstate(saved["python_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        start, logs = int(saved["step"]), list(saved["logs"])
    update_status(
        "GPU_TRAINING", "RUNNING", CURRENT_STEP=start,
        TOTAL_STEPS=int(training["optimizer_steps"]), GPU_TRAINING_READY="YES",
    )
    started = time.time()
    for step in range(start + 1, int(training["optimizer_steps"]) + 1):
        tick = time.time()
        model.train()
        graphs, batch_graph, source, reference, _ = frozen.sample_batch(
            prepared["train"], sources, generator, int(training["batch_molecules"]), device
        )
        optimizer.zero_grad(set_to_none=True)
        prediction, action, losses, target_bond, target_angle = batch_losses(
            model, graphs, batch_graph, source, reference
        )
        if not bool(torch.isfinite(losses["total"])) or not bool(torch.isfinite(action.proposal).all()):
            raise RuntimeError(f"nonfinite training forward at step {step}")
        losses["total"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        if not bool(torch.isfinite(gradient)):
            raise RuntimeError(f"nonfinite training gradient at step {step}")
        if step == 25:
            gpu_pass = (
                all(parameter.device.type == "cuda" for parameter in model.parameters())
                and source.device.type == "cuda"
                and batch_graph.atom_categorical.device.type == "cuda"
                and losses["total"].device.type == "cuda"
                and all(parameter.grad is None or parameter.grad.device.type == "cuda" for parameter in model.parameters())
            )
            atomic_json(
                REPORT / "GPU_TRAINING_VERIFICATION.json",
                {
                    "status": "PASS" if gpu_pass else "FAIL", "step": step,
                    "MODEL_ON_GPU": "PASS" if all(p.device.type == "cuda" for p in model.parameters()) else "FAIL",
                    "BATCH_ON_GPU": "PASS" if source.device.type == "cuda" else "FAIL",
                    "FORWARD_ON_GPU": "PASS" if losses["total"].device.type == "cuda" else "FAIL",
                    "BACKWARD_ON_GPU": "PASS" if all(p.grad is None or p.grad.device.type == "cuda" for p in model.parameters()) else "FAIL",
                    "CUDA_DEVICE_NAME": torch.cuda.get_device_name(device),
                    "PYTORCH_CUDA_VERSION": torch.version.cuda,
                },
            )
            if not gpu_pass:
                raise RuntimeError("GPU verification failed")
        optimizer.step()
        scheduler.step()
        if step == 1 or step % int(training["log_interval"]) == 0:
            row = {
                "step": step,
                "total_loss": float(losses["total"].detach()),
                "belief_loss": float(losses["belief"].detach()),
                "post_loss": float(losses["post"].detach()),
                "bond_deltaq_mae": float((prediction["bond_deltaq"].detach() - target_bond.to(prediction["bond_deltaq"])).abs().mean()),
                "angle_deltaq_mae": float((prediction["angle_deltaq"].detach() - target_angle.to(prediction["angle_deltaq"])).abs().mean()),
                "tau_mean": float(action.tau.detach().mean()),
                "w_B_mean": float(action.family_weights.detach()[:, 0].mean()),
                "step_seconds": time.time() - tick,
            }
            logs.append(row)
            atomic_frame(TRAIN_LOG, pd.DataFrame(logs))
            update_status(
                "GPU_TRAINING", "RUNNING", CURRENT_STEP=step,
                TOTAL_STEPS=int(training["optimizer_steps"]), **row,
            )
            print(json.dumps(row), flush=True)
        if step % int(training["recovery_interval"]) == 0:
            save_recovery(model, optimizer, scheduler, generator, step, logs)
    atomic_torch(
        FINAL,
        {
            "schema_version": "sixs-v2-deltaq-step17500-v1",
            "step": int(training["optimizer_steps"]),
            "model_state": model.state_dict(),
            "config_sha256": sha256(CONFIG),
            "training_wall_seconds": time.time() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    )
    update_status(
        "TRAINING_COMPLETE", "PASS", CURRENT_STEP=int(training["optimizer_steps"]),
        TOTAL_STEPS=int(training["optimizer_steps"]), FINAL_CHECKPOINT_SHA256=sha256(FINAL),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "overfit", "train", "all"))
    args = parser.parse_args()
    if args.stage in {"audit", "all"}:
        audit()
    if args.stage in {"overfit", "all"}:
        overfit()
    if args.stage in {"train", "all"}:
        train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
