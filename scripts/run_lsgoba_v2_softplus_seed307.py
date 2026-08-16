#!/usr/bin/env python3
"""From-scratch LSGO-BA v2 Softplus training implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.learned_geometry import geometry_values, parameter_count, structured_objective
from etflow.ecir.lsgoba_v2_joint_magnitude import (
    JointMagnitudeLSGO,
    SOURCE_STATE_FEATURES,
    batch_source_directions_and_state,
    scaled_proposal,
)
from scripts import run_lsgoba_v2_joint_magnitude_full307 as base


CONFIG_PATH = ROOT / "configs/ecir_mvr_lsgoba_v2_softplus_seed307.json"
REPORT = ROOT / "reports/ecir_mvr/lsgoba_v2_softplus_seed307"
ARTIFACT = ROOT / "artifacts/ecir_mvr/lsgoba_v2_softplus_seed307"
STATUS = REPORT / "SOFTPLUS_SEED307_STATUS.json"
SOURCE_PAYLOAD = Path(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["source_payload"]["path"])
BEST = ARTIFACT / "checkpoints/final_step22500.ckpt"
LAST = ARTIFACT / "checkpoints/last_recovery.ckpt"
INITIAL = ARTIFACT / "checkpoints/initial_random_seed307.ckpt"


def config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    seed = int(config()["seed"])
    current = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    if state == "RUNNING":
        current.pop("error", None)
        current.pop("error_type", None)
    current.update({
        "schema_version": "lsgoba-v2-softplus-seed-status-v1",
        "status": state,
        "stage": stage,
        "worker_pid": os.getpid(),
        "heartbeat": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
        "xtb_stage": "NOT_STARTED",
        "other_seed_started": seed != 307,
        **extra,
    })
    base.atomic_json(STATUS, current)


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_model(cfg: Mapping[str, Any], device: torch.device) -> JointMagnitudeLSGO:
    model = JointMagnitudeLSGO(
        hidden_dim=cfg["model"]["hidden_dim"],
        layers=cfg["model"]["layers"],
        initial_tau=cfg["model"]["initial_tau_angstrom"],
        tau_max=cfg["model"]["tau_max_angstrom"],
    )
    if parameter_count(model.geometry) != cfg["v1"]["parameter_count"]:
        raise RuntimeError("geometry architecture parameter count changed")
    return model.to(device)


def loss_terms(
    model: JointMagnitudeLSGO,
    graphs: Sequence[Any],
    batch_graph: Any,
    source: torch.Tensor,
    reference: torch.Tensor,
    atom_cap: float = 0.03,
) -> dict[str, torch.Tensor]:
    prediction = model.geometry(batch_graph)
    parameters = {
        **prediction,
        "bond_sigma": batch_graph.bond_fixed[:, 1],
        "angle_sigma": batch_graph.angle_fixed[:, 1],
    }
    prior, prior_groups = structured_objective(reference, batch_graph, parameters)
    direction, state, graph_embedding, gradient_diag = batch_source_directions_and_state(
        source, graphs, parameters, prediction["node_embedding"]
    )
    tau = model.magnitude(graph_embedding, model.normalized_state(state))
    proposal, cap_active, proposal_graph_rms = scaled_proposal(
        source, direction, tau, graphs, atom_cap=atom_cap
    )
    prop_b, prop_a = geometry_values(proposal, batch_graph)
    ref_b, ref_a = geometry_values(reference, batch_graph)
    post_b = ((prop_b - ref_b) / parameters["bond_sigma"]).square().mean()
    post_a = ((prop_a - ref_a) / parameters["angle_sigma"]).square().mean()
    post = 0.5 * (post_b + post_a)
    move = (tau / model.magnitude.tau_max).square().mean()
    return {
        "prior": prior,
        "prior_b": prior_groups["bond"],
        "prior_a": prior_groups["angle"],
        "post": post,
        "post_b": post_b,
        "post_a": post_a,
        "move": move,
        "tau": tau,
        "bond_mu": parameters["bond_mu"],
        "proposal": proposal,
        "direction": direction,
        "state": state,
        "cap_active": cap_active,
        "proposal_graph_rms": proposal_graph_rms,
        "gradient_diag": gradient_diag,
    }


# Reuse the v2 data/sampling/manifests machinery with this lineage's paths and model.
base.CONFIG_PATH = CONFIG_PATH
base.REPORT = REPORT
base.ARTIFACT = ARTIFACT
base.STATUS = STATUS
base.SOURCE_PAYLOAD = SOURCE_PAYLOAD
base.BEST = BEST
base.LAST = LAST
base.config = config
base.status = status
base.make_model = make_model
base.loss_terms = loss_terms


def distribution(values: torch.Tensor, prefix: str) -> dict[str, float]:
    x = values.detach().cpu().double().numpy()
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_p1": float(np.quantile(x, 0.01)),
        f"{prefix}_p10": float(np.quantile(x, 0.10)),
        f"{prefix}_p90": float(np.quantile(x, 0.90)),
        f"{prefix}_p95": float(np.quantile(x, 0.95)),
        f"{prefix}_p99": float(np.quantile(x, 0.99)),
        f"{prefix}_p999": float(np.quantile(x, 0.999)),
        f"{prefix}_max": float(np.max(x)),
    }


def tau_distribution(values: torch.Tensor) -> dict[str, float]:
    x = values.detach().cpu().double().numpy()
    return {
        "tau_mean": float(np.mean(x)),
        "tau_median": float(np.median(x)),
        "tau_p90": float(np.quantile(x, 0.90)),
        "tau_p95": float(np.quantile(x, 0.95)),
        "tau_p99": float(np.quantile(x, 0.99)),
        "tau_ge0099_fraction": float(np.mean(x >= 0.0099)),
    }


def verify_inputs() -> None:
    cfg = config()
    if base.sha256(SOURCE_PAYLOAD) != cfg["source_payload"]["sha256"]:
        raise RuntimeError("shared frozen Source payload SHA changed")
    payload = torch.load(SOURCE_PAYLOAD, map_location="cpu", weights_only=False)
    if len(payload["train"]) != 150000 or len(payload["val"]) != 10000:
        raise RuntimeError("shared Source payload denominator changed")
    if payload.get("formal_test_records_read") != 0 or payload.get("frozen_holdout_records_read") != 0:
        raise RuntimeError("protected records present in Source payload")
    audit = {
        "schema_version": "lsgoba-v2-softplus-source-reuse-v1",
        "status": "PASS",
        "mode": "READ_ONLY_REUSE_OF_V2_FROZEN_SOURCE_BINDING",
        "source_payload": str(SOURCE_PAYLOAD),
        "source_payload_sha256": cfg["source_payload"]["sha256"],
        "train_records": 150000,
        "validation_records": 10000,
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
    }
    base.atomic_json(REPORT / "00_SOURCE_REUSE_AUDIT.json", audit)


def architecture_report() -> None:
    cfg = config()
    base.seed_all(cfg["seed"])
    model = make_model(cfg, torch.device("cpu"))
    seed = int(cfg["seed"])
    train_steps = int(cfg["training"]["optimizer_steps"])
    scheduler_horizon = int(cfg["training"]["scheduler_horizon"])
    text = f"""# LSGO-BA v2 Softplus seed{seed} architecture

- lineage: independent development-only branch from `{cfg['lineage_base_commit']}`
- initialization: entire trainable model randomly initialized with seed{seed}
- old bounded checkpoint loaded: no
- Bond mean: `softplus(z_B)` with mathematical support `(0, infinity)`
- added lower/upper bound, epsilon, penalty, or learned bound: none
- Angle mean, frozen sigma_B/sigma_A, adaptive tau, tau_max=0.010 A, atom cap=0.03 A: unchanged
- objective: `L_prior + L_post + {cfg['objective']['lambda']} * (tau/0.010)^2`
- training: {train_steps:,} steps; cosine horizon {scheduler_horizon:,}; final step fixed as primary checkpoint
- total trainable parameters: {parameter_count(model):,}
- formal test / frozen holdout: 0 / 0
"""
    base.atomic_text(REPORT / "01_SOFTPLUS_ARCHITECTURE.md", text)


def preflight() -> None:
    cfg = config()
    REPORT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    verify_inputs()
    base.freeze_manifests()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prepared = base.load_prepared(cfg)
    source_payload = torch.load(SOURCE_PAYLOAD, map_location="cpu", weights_only=False)
    sources = base.source_index(source_payload, "train")

    # The preflight model is the exact configured-seed random initialization that train()
    # reconstructs; no checkpoint is loaded and no optimizer update occurs here.
    base.seed_all(cfg["seed"])
    model = make_model(cfg, device)
    initial_sha = state_sha256(model)
    generator = torch.Generator().manual_seed(cfg["seed"] + 92000)
    states: list[torch.Tensor] = []
    for i in range(cfg["preflight"]["batches"]):
        graphs, batch_graph, source, reference, _ = base.sample_batch(
            prepared["train"], sources, generator, cfg["preflight"]["batch_molecules"], device
        )
        prediction = model.geometry(batch_graph)
        params = {
            **prediction,
            "bond_sigma": batch_graph.bond_fixed[:, 1],
            "angle_sigma": batch_graph.angle_fixed[:, 1],
        }
        _, state, _, _ = batch_source_directions_and_state(
            source, graphs, params, prediction["node_embedding"]
        )
        states.append(state.detach().cpu())
        if (i + 1) % 16 == 0:
            status("PREFLIGHT_NORMALIZATION", completed_batches=i + 1, expected_batches=cfg["preflight"]["batches"])

    all_state = torch.cat(states)
    state_mean = all_state.mean(0)
    state_std = all_state.std(0, unbiased=False).clamp_min(cfg["preflight"]["normalization_std_floor"])
    model.set_state_normalization(state_mean, state_std)

    generator = torch.Generator().manual_seed(cfg["seed"] + 92000)
    rows: list[dict[str, float]] = []
    mus: list[torch.Tensor] = []
    taus: list[torch.Tensor] = []
    for i in range(cfg["preflight"]["batches"]):
        graphs, batch_graph, source, reference, _ = base.sample_batch(
            prepared["train"], sources, generator, cfg["preflight"]["batch_molecules"], device
        )
        terms = loss_terms(model, graphs, batch_graph, source, reference)
        total = terms["prior"] + terms["post"] + cfg["objective"]["lambda"] * terms["move"]
        grad = torch.autograd.grad(total, tuple(model.parameters()), allow_unused=True)
        grad_norm = torch.sqrt(sum(
            (value.detach().square().sum() for value in grad if value is not None),
            total.new_zeros(()),
        ))
        rows.append({
            "prior": float(terms["prior"].detach()),
            "post": float(terms["post"].detach()),
            "move": float(terms["move"].detach()),
            "total": float(total.detach()),
            "gradient_norm": float(grad_norm),
        })
        mus.append(terms["bond_mu"].detach().cpu())
        taus.append(terms["tau"].detach().cpu())
        if (i + 1) % 16 == 0:
            status("PREFLIGHT", completed_batches=i + 1, expected_batches=cfg["preflight"]["batches"])

    frame = pd.DataFrame(rows)
    mu = torch.cat(mus)
    tau = torch.cat(taus)
    finite = bool(
        np.isfinite(frame.to_numpy()).all()
        and torch.isfinite(mu).all()
        and torch.isfinite(tau).all()
        and (mu > 0).all()
    )
    report = {
        "schema_version": "lsgoba-v2-softplus-preflight-v1",
        "status": "PASS" if finite else "FAIL",
        "train_only": True,
        "batches": len(frame),
        "full_model_random_initialization": True,
        "warm_start": False,
        "initial_model_state_sha256": initial_sha,
        "lambda": cfg["objective"]["lambda"],
        "lambda_reestimated": False,
        "loss": {column: {
            "mean": float(frame[column].mean()),
            "median": float(frame[column].median()),
            "p90": float(frame[column].quantile(0.9)),
        } for column in ("prior", "post", "move", "total", "gradient_norm")},
        "bond_mu": distribution(mu, "bond_mu"),
        "tau": tau_distribution(tau),
        "state_feature_names": list(SOURCE_STATE_FEATURES),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
    }
    base.atomic_json(REPORT / "02_SOFTPLUS_PREFLIGHT.json", report)
    if not finite:
        raise RuntimeError("Softplus preflight produced nonfinite/nonpositive values")
    status("PREFLIGHT", state="PASS", initial_model_state_sha256=initial_sha)


def deterministic_eval(
    model: JointMagnitudeLSGO,
    items: Sequence[dict],
    source_map: Mapping[str, list[dict]],
    device: torch.device,
    selected_ids: set[str],
    lam: float,
    batch_size: int = 64,
) -> dict[str, float]:
    chosen = [item for item in items if str(item["molecule_id"]) in selected_ids]
    weighted: list[tuple[int, float, float, float]] = []
    mus: list[torch.Tensor] = []
    taus: list[torch.Tensor] = []
    rms: list[torch.Tensor] = []
    caps: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(chosen), batch_size):
        group = chosen[start:start + batch_size]
        graphs = [item["graph"] for item in group]
        batch_graph = base.collate_graphs(graphs).to(device)
        source = torch.cat([source_map[str(item["molecule_id"])][0]["source"] for item in group]).to(device)
        reference = torch.cat([item["references"][0] for item in group]).to(device)
        with torch.enable_grad():
            terms = loss_terms(model, graphs, batch_graph, source, reference)
        weighted.append((len(group), float(terms["prior"].detach()), float(terms["post"].detach()), float(terms["move"].detach())))
        mus.append(terms["bond_mu"].detach().cpu())
        taus.append(terms["tau"].detach().cpu())
        rms.append(terms["proposal_graph_rms"].detach().cpu())
        caps.append(terms["cap_active"].detach().cpu())
    total = sum(row[0] for row in weighted)
    avg = lambda index: sum(row[0] * row[index] for row in weighted) / total
    result = {
        "molecules": total,
        "prior": avg(1),
        "post": avg(2),
        "move": avg(3),
        "selection_objective": avg(2) + lam * avg(3),
        "proposal_rms_mean": float(torch.cat(rms).mean()),
        "atom_cap_fraction": float(torch.cat(caps).double().mean()),
    }
    result.update(distribution(torch.cat(mus), "bond_mu"))
    result.update(tau_distribution(torch.cat(taus)))
    return result


def checkpoint_payload(
    model: JointMagnitudeLSGO,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    generator: torch.Generator,
    step: int,
    normalization: Mapping[str, Any],
    validation: Mapping[str, Any],
    log_rows: int,
) -> dict[str, Any]:
    return {
        "schema_version": "lsgoba-v2-softplus-seed-checkpoint-v1",
        "seed": int(config()["seed"]),
        "global_step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "sampler_generator_state": generator.get_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "lambda": config()["objective"]["lambda"],
        "normalization": normalization,
        "validation": dict(validation),
        "log_rows": log_rows,
        "initialization": "FROM_SCRATCH_FULL_TRAINABLE_MODEL",
        "warm_start": False,
        "old_checkpoint_loaded": False,
        "bond_mean_parameterization": "softplus(z_B)",
        "config_sha256": base.sha256(CONFIG_PATH),
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
    }


def train() -> None:
    cfg = config()
    pre = json.loads((REPORT / "02_SOFTPLUS_PREFLIGHT.json").read_text(encoding="utf-8"))
    if pre.get("status") != "PASS":
        raise RuntimeError("preflight is not PASS")
    lam = float(cfg["objective"]["lambda"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prepared = base.load_prepared(cfg)
    source_payload = torch.load(SOURCE_PAYLOAD, map_location="cpu", weights_only=False)
    train_sources = base.source_index(source_payload, "train")
    val_sources = base.source_index(source_payload, "val")
    val_ids = base.manifest_ids("V2_VAL")

    generator = base.seed_all(cfg["seed"])
    model = make_model(cfg, device)
    initial_sha = state_sha256(model)
    if initial_sha != pre["initial_model_state_sha256"]:
        raise RuntimeError("training random initialization differs from the configured-seed preflight initialization")
    model.set_state_normalization(torch.tensor(pre["state_mean"]), torch.tensor(pre["state_std"]))

    backbone_params = list(model.geometry.parameters())
    head_params = list(model.magnitude.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg["training"]["backbone_learning_rate"]},
        {"params": head_params, "lr": cfg["training"]["head_learning_rate"]},
    ], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["scheduler_horizon"]
    )

    logs: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    start = 0
    if LAST.is_file():
        payload = torch.load(LAST, map_location="cpu", weights_only=False)
        if payload.get("initialization") != "FROM_SCRATCH_FULL_TRAINABLE_MODEL" or payload.get("warm_start") is not False:
            raise RuntimeError("recovery checkpoint lineage mismatch")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        generator.set_state(payload["sampler_generator_state"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        start = int(payload["global_step"])
        if (REPORT / "05_TRAIN_LOG.csv").is_file():
            logs = pd.read_csv(REPORT / "05_TRAIN_LOG.csv").to_dict("records")
        if (REPORT / "VALIDATION_CHECKPOINTS.csv").is_file():
            validations = pd.read_csv(REPORT / "VALIDATION_CHECKPOINTS.csv").to_dict("records")
    else:
        base.atomic_torch(INITIAL, {
            "schema_version": "lsgoba-v2-softplus-initialization-v1",
            "seed": cfg["seed"],
            "model_state": model.state_dict(),
            "model_state_sha256": initial_sha,
            "warm_start": False,
            "old_checkpoint_loaded": False,
        })
        base.atomic_json(REPORT / "INITIALIZATION.json", {
            "status": "FROZEN_BEFORE_OPTIMIZER_STEP",
            "mode": "FROM_SCRATCH_FULL_TRAINABLE_MODEL",
            "seed": cfg["seed"],
            "model_state_sha256": initial_sha,
            "checkpoint": str(INITIAL),
            "checkpoint_sha256": base.sha256(INITIAL),
            "warm_start": False,
            "old_checkpoint_loaded": False,
        })

    checkpoints = set(cfg["training"]["checkpoint_steps"])
    expected = int(cfg["training"]["optimizer_steps"])
    started = time.time()
    for step in range(start + 1, expected + 1):
        tick = time.time()
        model.train()
        graphs, batch_graph, source, reference, _ = base.sample_batch(
            prepared["train"], train_sources, generator, cfg["training"]["batch_molecules"], device
        )
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(model, graphs, batch_graph, source, reference)
        total = terms["prior"] + terms["post"] + lam * terms["move"]
        finite_forward = bool(
            torch.isfinite(total)
            and torch.isfinite(terms["bond_mu"]).all()
            and torch.isfinite(terms["tau"]).all()
            and torch.isfinite(terms["proposal"]).all()
            and (terms["bond_mu"] > 0).all()
        )
        if not finite_forward:
            raise RuntimeError(f"nonfinite/nonpositive forward value at step {step}")
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
        if not bool(torch.isfinite(grad)):
            raise RuntimeError(f"nonfinite gradient at step {step}")
        optimizer.step()
        scheduler.step()

        if step == 1 or step % cfg["training"]["log_interval"] == 0:
            row: dict[str, Any] = {
                "step": step,
                "L_prior": float(terms["prior"].detach()),
                "L_post": float(terms["post"].detach()),
                "L_move": float(terms["move"].detach()),
                "L_total": float(total.detach()),
                "proposal_rms_mean": float(terms["proposal_graph_rms"].mean().detach()),
                "atom_cap_fraction": float(terms["cap_active"].double().mean().detach()),
                "gradient_norm_before_clip": float(grad),
                "nonfinite_forward": False,
                "backbone_lr": scheduler.get_last_lr()[0],
                "head_lr": scheduler.get_last_lr()[1],
                "step_time_seconds": time.time() - tick,
            }
            row.update(distribution(terms["bond_mu"], "bond_mu"))
            row.update(tau_distribution(terms["tau"]))
            logs.append(row)

        validation: dict[str, Any] = {}
        if step in checkpoints:
            validation = {
                "step": step,
                **deterministic_eval(model, prepared["val"], val_sources, device, val_ids, lam),
            }
            validations.append(validation)
            print(json.dumps({"stage": "CHECKPOINT", "step": step, "validation": validation}), flush=True)

        if step % cfg["training"]["recovery_checkpoint_interval"] == 0 or step in checkpoints:
            payload = checkpoint_payload(
                model, optimizer, scheduler, generator, step,
                {"mean": pre["state_mean"], "std": pre["state_std"]},
                validation, len(logs),
            )
            base.atomic_torch(LAST, payload)
            if step in checkpoints:
                base.atomic_torch(ARTIFACT / f"checkpoints/step{step:05d}.ckpt", payload)
            pd.DataFrame(logs).to_csv(REPORT / "05_TRAIN_LOG.csv", index=False)
            pd.DataFrame(validations).to_csv(REPORT / "VALIDATION_CHECKPOINTS.csv", index=False)
            status(
                "TRAINING", completed_steps=step, expected_steps=expected,
                elapsed_seconds=time.time() - started,
                latest_loss=float(total.detach()),
                latest_bond_mu_max=float(terms["bond_mu"].max().detach()),
                latest_tau_mean=float(terms["tau"].mean().detach()),
            )
        if step % 250 == 0:
            print(json.dumps({
                "stage": "TRAINING", "step": step, "expected": expected,
                "loss": float(total.detach()),
                "bond_mu_max": float(terms["bond_mu"].max().detach()),
                "tau_mean": float(terms["tau"].mean().detach()),
            }), flush=True)

    final_path = ARTIFACT / f"checkpoints/step{expected:05d}.ckpt"
    if not final_path.is_file():
        raise RuntimeError("fixed final checkpoint missing")
    shutil.copyfile(final_path, BEST)
    status(
        "TRAINING", state="PASS", completed_steps=expected, expected_steps=expected,
        primary_checkpoint_step=expected,
        best_checkpoint=str(BEST),
        best_checkpoint_sha256=base.sha256(BEST),
        selection_external_metrics_used=False,
    )


def write_config_artifact() -> None:
    cfg = config()
    payload = {
        **cfg,
        "config_path": str(CONFIG_PATH),
        "config_sha256": base.sha256(CONFIG_PATH),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "head_at_setup": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "frozen_v1_v2_v2_1_modified": False,
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
    }
    base.atomic_json(REPORT / "03_SOFTPLUS_CONFIG.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "train", "status"))
    args = parser.parse_args()
    if args.phase == "status":
        print(STATUS.read_text(encoding="utf-8") if STATUS.is_file() else json.dumps({"status": "NOT_STARTED"}, indent=2))
        return 0
    try:
        architecture_report()
        write_config_artifact()
        if args.phase == "preflight":
            preflight()
        elif args.phase == "train":
            train()
    except Exception as exc:
        status("FAILED", state="FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
