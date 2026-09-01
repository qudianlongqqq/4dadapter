#!/usr/bin/env python3
"""Resumable six-arm SIXS direct-mu/sigma x action-reliability experiment.

This is development-only.  It deliberately has no import or path to Formal,
the protected large holdout, xTB, Sigma-v2, or seeds 331/353.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, rdBase

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.formal_rdkit_adapter import adapt_formal_cache_record
from etflow.ecir.learned_geometry import geometry_values, safety_accept
from etflow.ecir.musigma_reliability import (
    DirectMuSigmaModel,
    PrimitiveReliabilityHead,
    action_loss,
    action_proposal,
    belief_loss,
)
from scripts.run_mcvr_lsgo import collate_graphs


CONFIG_PATH = ROOT / "configs/sixs_musigma_reliability_factorial.json"
RUN_NAMESPACE = os.environ.get("SIXS_FACTORIAL_RUN_NAMESPACE", "sixs_musigma_reliability_factorial")
EXECUTION_DEVICE_MODE = os.environ.get("SIXS_FACTORIAL_DEVICE", "auto").lower()
if not RUN_NAMESPACE.replace("_", "").replace("-", "").isalnum():
    raise RuntimeError("invalid SIXS_FACTORIAL_RUN_NAMESPACE")
if EXECUTION_DEVICE_MODE not in {"auto", "cpu", "cuda"}:
    raise RuntimeError("SIXS_FACTORIAL_DEVICE must be auto, cpu, or cuda")
REPORT = ROOT / "reports/ecir_mvr" / RUN_NAMESPACE
ARTIFACT = ROOT / "artifacts/ecir_mvr" / RUN_NAMESPACE
STATUS = REPORT / "RUN_STATUS.json"
LIVE_RESULTS = REPORT / "LIVE_RESULTS.csv"
LIVE_SUMMARY = REPORT / "LIVE_SUMMARY.md"
COMMON_INIT = ARTIFACT / "COMMON_INITIALIZATION_V2.pt"
DEV_MANIFEST = REPORT / "DEV_MANIFEST.json"
ARMS = ("J0-R0", "J0-R1", "J1-R0", "J1-R1", "J2-R0", "J2-R1")
REPORT_NAMES = {
    "J0-R0": "02_J0_R0.md", "J0-R1": "03_J0_R1.md",
    "J1-R0": "04_J1_R0.md", "J1-R1": "05_J1_R1.md",
    "J2-R0": "06_J2_R0.md", "J2-R1": "07_J2_R1.md",
}

OFFICIAL_ADAPTER = Path(r"E:\3dconformergenerationcode\4dadapter-lsgo-v2\scripts\run_lsgo_standard_genbench3d.py")
GENBENCH_REPO = Path(r"E:\miniconda\envs\external-validity\src\genbench3d")
LIGBOUNDCONF = Path(r"E:\3dconformergenerationcode\external_data\genbench3d_official\ligboundconf_minimized\S2_LigBoundConf_minimized.sdf")
REFERENCE_ROOT = Path(r"E:\3dconformergenerationcode\4dadapter-lsgo-v2\reports\ecir_mvr\lsgo_standard_eval\genbench3d_reference_cache")
GENBENCH_COMMIT = "0926bc6614509aa10ccf6f69da0405d4be6af6b3"
GENBENCH_ADAPTER_SHA = "6ebc450b4f841a9f6a3b463b7838e50bc2951e92570146cd0a473abbfd970450"
REFERENCE_SDF_SHA = "15e8e4635525f3d9452292e86995d28f6c24eb50baad661eb4c2665274d00fe2"
REFERENCE_VALUE_SHA = "63659acddd04017a4b8fc5f2df767540e48cd36a7849e578ddb6caf6130deadc"
REFERENCE_KERNEL_SHA = "6c098fa5b10c85f12db49df3a35efa33963fc222754e5d2a9d0b64e61c604a19"
EXTERNAL_PYTHON = Path(r"E:\miniconda\envs\external-validity\python.exe")
EXTERNAL_WORKER = ROOT / "scripts/evaluate_sixs_musigma_external.py"


def cfg() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def execution_device() -> torch.device:
    if EXECUTION_DEVICE_MODE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUIRED_BUT_UNAVAILABLE")
        return torch.device("cuda:0")
    if EXECUTION_DEVICE_MODE == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


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
    if path.suffix == ".parquet":
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def markdown_frame(frame: pd.DataFrame) -> str:
    """Render a report table without making tabulate a runtime dependency."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + frame.to_csv(index=False, lineterminator="\n").rstrip() + "\n```"


def protected_fields() -> dict[str, Any]:
    return {
        "FORMAL_READ": "NO", "LARGE_HOLDOUT_READ": "NO", "XTB_STARTED": "NO",
        "SEED331_STARTED": "NO", "SEED353_STARTED": "NO", "SIGMA_V2_MODIFIED": "NO",
        "SIGMA_TEACHER_CREATED": "NO", "RELIABILITY_TEACHER_CREATED": "NO",
        "UPSTREAM_ID_USED_AS_FEATURE": "NO", "REFERENCE_USED_AT_INFERENCE": "NO",
        "V3D_USED_AS_TRAINING_LOSS": "NO", "PB_USED_AS_TRAINING_LOSS": "NO",
        "PROTOCOL_CHANGED_AFTER_RESULTS": "NO", "HYPERPARAMETER_SWEEP": "NO",
        "CHECKPOINT_SELECTED_BY_DEV_OUTCOME": "NO", "PIPELINE_RESUMABLE": "YES",
        "LIVE_RESULTS_ENABLED": "YES",
    }


def update_status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    current = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    if state != "FAIL":
        current.pop("ERROR", None); current.pop("ERROR_TYPE", None)
    current.update({
        "schema_version": "sixs-musigma-reliability-factorial-status-v1",
        "STATUS": state, "CURRENT_STAGE": stage, "WORKER_PID": os.getpid(),
        "UPDATED_AT": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "RUN_NAMESPACE": RUN_NAMESPACE, "EXECUTION_DEVICE": EXECUTION_DEVICE_MODE,
        **protected_fields(), **extra,
    })
    atomic_json(STATUS, current)


def seed_all(seed: int) -> torch.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed + 91000)


def load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    config = cfg(); data = config["data"]
    for key, expected in (
        ("prepared_payload", data["prepared_sha256"]),
        ("source_payload", data["source_payload_sha256"]),
        ("train_manifest", data["train_manifest_sha256"]),
        ("val_manifest", data["val_manifest_sha256"]),
    ):
        if sha256(data[key]) != expected:
            raise RuntimeError(f"frozen input hash changed: {key}")
    prepared = torch.load(data["prepared_payload"], map_location="cpu", weights_only=False)
    sources = torch.load(data["source_payload"], map_location="cpu", weights_only=False)
    if len(prepared["train"]) != 50000 or len(prepared["val"]) != 5000:
        raise RuntimeError("prepared molecule denominator changed")
    if len(sources["train"]) != 150000 or len(sources["val"]) != 10000:
        raise RuntimeError("source-record denominator changed")
    return prepared, sources


def source_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row["molecule_id"]), []).append(dict(row))
    for values in result.values():
        values.sort(key=lambda x: str(x["sample_id"]))
    return result


def freeze_dev_manifest(prepared: Mapping[str, Any], sources: Mapping[str, Any]) -> dict[str, Any]:
    if DEV_MANIFEST.is_file():
        manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("identity_sha256") != canonical_sha(manifest["rows"]):
            raise RuntimeError("DEV manifest identity hash changed")
        return manifest
    config = cfg(); tag = config["data"]["dev_split_tag"]
    ranked = sorted(
        (str(item["molecule_id"]) for item in prepared["val"]),
        key=lambda value: hashlib.sha256(f"{tag}\0{value}".encode()).hexdigest(),
    )
    chosen = set(ranked[2500:])
    by_source = source_index(sources["val"])
    rows = []
    for molecule_id in sorted(chosen):
        samples = [str(row["sample_id"]) for row in by_source[molecule_id]]
        if len(samples) != 2:
            raise RuntimeError("DEV source multiplicity changed")
        rows.append({"molecule_id": molecule_id, "sample_ids": samples})
    manifest = {
        "schema_version": "sixs-musigma-factorial-dev-manifest-v1",
        "status": "FROZEN_BEFORE_TRAINING", "split_tag": tag,
        "molecules": len(rows), "records": sum(len(x["sample_ids"]) for x in rows),
        "rows": rows, "identity_sha256": canonical_sha(rows), **protected_fields(),
    }
    if manifest["molecules"] != 2500 or manifest["records"] != 5000:
        raise RuntimeError("DEV denominator changed")
    atomic_json(DEV_MANIFEST, manifest)
    return manifest


def load_base_geometry(model: DirectMuSigmaModel) -> None:
    config = cfg(); path = Path(config["initialization"]["checkpoint"])
    if sha256(path) != config["initialization"]["checkpoint_sha256"]:
        raise RuntimeError("mu-only initialization checkpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    if all(key.startswith("geometry.") for key in state if key.startswith("geometry.")):
        geometry = {key[len("geometry."):]: value for key, value in state.items() if key.startswith("geometry.")}
    else:
        geometry = state
    model.geometry.load_state_dict(geometry, strict=True)


def freeze_common_initialization() -> dict[str, Any]:
    config = cfg(); seed_all(config["seed"])
    if not COMMON_INIT.is_file():
        model = DirectMuSigmaModel(config["model"]["hidden_dim"], config["model"]["layers"])
        load_base_geometry(model)
        reliability = PrimitiveReliabilityHead(config["model"]["hidden_dim"])
        atomic_torch(COMMON_INIT, {
            "schema_version": "sixs-musigma-common-init-v1", "seed": config["seed"],
            "model_state": model.state_dict(), "reliability_state": reliability.state_dict(),
            "source_checkpoint_sha256": config["initialization"]["checkpoint_sha256"],
            "config_sha256": sha256(CONFIG_PATH), **protected_fields(),
        })
    payload = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != sha256(CONFIG_PATH):
        raise RuntimeError("common initialization config identity changed")
    model_hashes, reliability_hashes = [], []
    for _ in ARMS:
        model = DirectMuSigmaModel(config["model"]["hidden_dim"], config["model"]["layers"])
        model.load_state_dict(payload["model_state"], strict=True)
        reliability = PrimitiveReliabilityHead(config["model"]["hidden_dim"])
        reliability.load_state_dict(payload["reliability_state"], strict=True)
        model_hashes.append(canonical_sha({k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in model.state_dict().items()}))
        reliability_hashes.append(canonical_sha({k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in reliability.state_dict().items()}))
    if len(set(model_hashes)) != 1 or len(set(reliability_hashes)) != 1:
        raise RuntimeError("COMMON_INITIALIZATION_PASS failed")
    return {"path": str(COMMON_INIT), "sha256": sha256(COMMON_INIT), "model_state_sha256": model_hashes[0], "reliability_state_sha256": reliability_hashes[0]}


def freeze_protocol() -> None:
    REPORT.mkdir(parents=True, exist_ok=True); ARTIFACT.mkdir(parents=True, exist_ok=True)
    prepared, sources = load_inputs(); manifest = freeze_dev_manifest(prepared, sources)
    initialization = freeze_common_initialization(); config = cfg()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    protocol = f"""# SIXS direct mu-sigma x Reliability factorial protocol freeze

Status: **FROZEN_BEFORE_FIRST_ARM**. This is a development-only six-arm 3x2 factorial.

- isolated branch: `{branch}`
- run namespace: `{RUN_NAMESPACE}`
- execution device: `{EXECUTION_DEVICE_MODE}`; CUDA mode is fail-closed and cannot fall back to CPU
- baseline/worktree commit: `{head}`
- configuration SHA-256: `{sha256(CONFIG_PATH)}`
- TRAIN: 50,000 molecules / 150,000 ETFlow source records; manifest `{config['data']['train_manifest_sha256']}`
- DEV_TEST: {manifest['molecules']} molecules / {manifest['records']} records; identity `{manifest['identity_sha256']}`
- prepared graph payload: `{config['data']['prepared_sha256']}`
- source-binding payload: `{config['data']['source_payload_sha256']}`
- mu-only initialization: `{config['initialization']['checkpoint_sha256']}`
- common initialization artifact: `{initialization['sha256']}`
- architecture: shared 3-layer 128D invariant GNN, unchanged Bond Softplus mu and cosine-Angle tanh mu heads, separate source-free sigma heads
- sigma: `sigma_stat * exp(6*tanh(raw/6))`; the +/-6 bound is a frozen numerical dynamic-range guard
- likelihoods: J0 Gaussian NLL; J1 beta-NLL beta=0.5 with detached variance weighting; J2 published Faithful Normal gradient semantics
- Reliability: one shared bounded sigmoid gate, initialized at 0.999, using local representation and inference-safe Source primitive features only
- R1 update: one belief update then one action update; action optimizer contains Reliability plus shared backbone, never mu/sigma heads
- family allocation: Bond=0.5, Angle=0.5
- movement: fixed tau `{config['action']['tau_control_angstrom']}` A from `{config['action']['tau_control_source']}`; atom cap 0.03 A
- training: seed307, AdamW, batch64, backbone LR 1.5e-4, head LR 3e-4, WD 1e-6, cosine T_max=22500, 17,500 belief updates, clip1.0
- run order: {', '.join(ARMS)}; final checkpoint only; DEV once per arm
- evaluators: PoseBusters installed mol_fast; GenBench3D commit `{GENBENCH_COMMIT}` through adapter `{GENBENCH_ADAPTER_SHA}`
- bootstrap: molecule clusters, 10,000 resamples, seed20260828
- frozen classification thresholds: mu is DEGRADED if either paired mean primitive MAE exceeds the mu-only base by more than 1%; sigma is GOOD only when both families have z standard deviation in [0.8,1.25], |z|<=1 coverage in [0.60,0.76], |z|<=1.96 coverage in [0.90,0.99], and no pathology; >1% lower/upper log-range saturation is collapse/inflation
- a Reliability effect is POSITIVE only when the molecule-bootstrap lower CI is positive for Proposal V3D and direction improvement and nonnegative for PB; it is NEGATIVE when any corresponding upper CI is negative, otherwise NEUTRAL

Beta-NLL equation: `NLL(y,mu,sigma) * stop_gradient(sigma^2)^0.5`.

Faithful Normal equation: unit-variance Normal NLL for mean plus Gaussian variance NLL with detached mu; variance-head input representation is detached. Consequently variance gradients update only the sigma head, while mean gradients update mu and backbone.

Formal read: NO. Large holdout read: NO. xTB: NO. Sigma-v2 modified: NO.
"""
    target = REPORT / "00_PROTOCOL_FREEZE.md"
    if target.is_file() and target.read_text(encoding="utf-8").rstrip() != protocol.rstrip():
        # A preflight may expose an engineering-only integration defect.  A
        # correction is permitted only before any optimizer recovery/final
        # artifact exists and before any DEV result is visible.
        begun = any(arm_paths(arm)[name].exists() for arm in ARMS for name in ("recovery", "final", "result"))
        if begun: raise RuntimeError("protocol freeze changed after first arm began")
    atomic_text(target, protocol)
    update_status("PROTOCOL_FROZEN", "PASS", CURRENT_ARM=None, CURRENT_STEP=0, TOTAL_STEPS=17500,
                  COMMON_INITIALIZATION_PASS="YES", COMMON_DATA_IDENTITY_PASS="YES",
                  TAU_CONTROL=config["action"]["tau_control_angstrom"],
                  TAU_CONTROL_SOURCE=config["action"]["tau_control_source"])


def sample_batch(items: Sequence[dict[str, Any]], sources: Mapping[str, list[dict[str, Any]]], generator: torch.Generator, batch: int, device: torch.device):
    indices = torch.randint(len(items), (batch,), generator=generator).tolist()
    graphs, source_values, references = [], [], []
    for index in indices:
        item = items[index]; pool = sources[str(item["molecule_id"])]
        source_i = int(torch.randint(len(pool), (1,), generator=generator))
        ref_i = int(torch.randint(len(item["references"]), (1,), generator=generator))
        graphs.append(item["graph"])
        source_values.append(torch.as_tensor(pool[source_i]["source"], dtype=torch.float64))
        references.append(torch.as_tensor(item["references"][ref_i], dtype=torch.float64))
    batch_graph = collate_graphs(graphs).to(device)
    return graphs, batch_graph, torch.cat(source_values).to(device), torch.cat(references).to(device), indices


def parameter_norm(grads: Iterable[torch.Tensor | None], reference: torch.Tensor) -> float:
    total = reference.new_zeros(())
    for grad in grads:
        if grad is not None: total = total + grad.detach().square().sum()
    return float(total.sqrt())


def flat_gradient(grads: Iterable[torch.Tensor | None], params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().reshape(-1).cpu() for g, p in zip(grads, params, strict=True)])


def engineering_preflight() -> None:
    if not (REPORT / "00_PROTOCOL_FREEZE.md").is_file(): freeze_protocol()
    config = cfg(); device = execution_device()
    prepared, source_payload = load_inputs(); sources = source_index(source_payload["train"])
    generator = seed_all(config["seed"] + 11)
    graphs, bg, source, reference, indices = sample_batch(prepared["train"], sources, generator, 8, device)
    reference_bond, reference_angle = geometry_values(reference, bg)
    init = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
    audits = []
    belief_backbone_for_cosine: dict[str, torch.Tensor] = {}
    for method in ("J0", "J1", "J2"):
        model = DirectMuSigmaModel(128, 3).to(device); model.load_state_dict(init["model_state"], strict=True); model.train()
        pred = model(bg, detach_sigma_features=method == "J2")
        total, pieces = belief_loss(method, pred, reference_bond.to(pred["bond_mu"]), reference_angle.to(pred["angle_mu"]), graphs, .5)
        backbone = model.backbone_parameters(); mu = model.mu_head_parameters(); sigma = model.sigma_head_parameters()
        grads = torch.autograd.grad(total, backbone + mu + sigma, retain_graph=False, allow_unused=True)
        gb, gm, gs = grads[:len(backbone)], grads[len(backbone):len(backbone)+len(mu)], grads[len(backbone)+len(mu):]
        norms = {"grad_norm_backbone_from_belief": parameter_norm(gb, total), "grad_norm_mu_head": parameter_norm(gm, total), "grad_norm_sigma_head": parameter_norm(gs, total)}
        if not all(math.isfinite(x) and x > 0 for x in norms.values()): raise RuntimeError(f"{method} belief gradient route failed: {norms}")
        belief_backbone_for_cosine[method] = flat_gradient(gb, backbone)
        audits.append({"method": method, **norms, "belief_loss": float(total.detach()), "mean_loss": float(pieces["mean"].detach()), "variance_loss": float(pieces["variance"].detach())})
        if method == "J2":
            model.zero_grad(set_to_none=True); pred = model(bg, detach_sigma_features=True)
            _, split = belief_loss(method, pred, reference_bond.to(pred["bond_mu"]), reference_angle.to(pred["angle_mu"]), graphs, .5)
            var_grads = torch.autograd.grad(split["variance"], backbone + mu + sigma, allow_unused=True)
            nb, nm, ns = var_grads[:len(backbone)], var_grads[len(backbone):len(backbone)+len(mu)], var_grads[len(backbone)+len(mu):]
            if parameter_norm(nb, split["variance"]) != 0 or parameter_norm(nm, split["variance"]) != 0 or parameter_norm(ns, split["variance"]) <= 0:
                raise RuntimeError("J2 faithful variance gradient isolation failed")
    compatibility = {}
    action_audits = []
    for method in ("J0", "J1", "J2"):
        model = DirectMuSigmaModel(128, 3).to(device); model.load_state_dict(init["model_state"], strict=True); model.train()
        reliability = PrimitiveReliabilityHead(128).to(device); reliability.load_state_dict(init["reliability_state"], strict=True); reliability.train()
        pred = model(bg, detach_sigma_features=method == "J2")
        r0 = action_proposal(source, graphs, pred, tau=config["action"]["tau_control_angstrom"], atom_cap=.03, reliability_head=None)
        r1 = action_proposal(source, graphs, pred, tau=config["action"]["tau_control_angstrom"], atom_cap=.03, reliability_head=reliability)
        max_direction = float((r0.direction - r1.direction).abs().max())
        max_proposal = float((r0.proposal - r1.proposal).abs().max())
        compatibility[method] = {"direction_max_abs": max_direction, "proposal_max_abs": max_proposal, "tolerance": 1e-10}
        if max_direction > 1e-10 or max_proposal > 1e-10: raise RuntimeError(f"INITIAL_R_GATE_COMPATIBILITY failed: {method}")
        loss = action_loss(r1, reference, graphs, pred)
        backbone = model.backbone_parameters(); mu = model.mu_head_parameters(); sigma = model.sigma_head_parameters(); rp = list(reliability.parameters())
        grads = torch.autograd.grad(loss, backbone + mu + sigma + rp, allow_unused=True)
        i0=len(backbone); i1=i0+len(mu); i2=i1+len(sigma)
        gb, gm, gs, gr = grads[:i0], grads[i0:i1], grads[i1:i2], grads[i2:]
        nb, nm, ns, nr = [parameter_norm(x, loss) for x in (gb, gm, gs, gr)]
        if nb <= 0 or nr <= 0 or nm != 0 or ns != 0: raise RuntimeError(f"R1 action gradient route failed: {method}")
        flat_action = flat_gradient(gb, backbone); flat_belief = belief_backbone_for_cosine[method]
        cosine = float(torch.nn.functional.cosine_similarity(flat_action, flat_belief, dim=0))
        action_audits.append({"method": method, "grad_norm_backbone_from_action": nb, "grad_norm_mu_head_from_action": nm, "grad_norm_sigma_head_from_action": ns, "grad_norm_reliability_head": nr, "belief_action_backbone_gradient_cosine": cosine, "action_loss": float(loss.detach())})
    payload = {"sampled_train_indices_sha256": canonical_sha(indices), "belief_routes": audits, "action_routes": action_audits, "initial_gate_compatibility": compatibility, "INITIAL_R_GATE_COMPATIBILITY": "PASS", "SECOND_ORDER_CARTESIAN_GRAPH": "NO", "REFERENCE_IN_INFERENCE_FEATURES": "NO", "UPSTREAM_ID_IN_FEATURES": "NO", **protected_fields()}
    atomic_json(REPORT / "01_IMPLEMENTATION_AUDIT.json", payload)
    lines = ["# SIXS factorial implementation audit", "", "All fail-closed engineering gates passed.", "", markdown_frame(pd.DataFrame(audits)), "", markdown_frame(pd.DataFrame(action_audits)), "", f"Initial R-gate compatibility: PASS at max-absolute tolerance 1e-10. No second-order Cartesian derivatives are used. Sample identity `{payload['sampled_train_indices_sha256']}`."]
    atomic_text(REPORT / "01_IMPLEMENTATION_AUDIT.md", "\n".join(lines))
    update_status("ENGINEERING_PREFLIGHT", "PASS", CURRENT_ARM=None, CURRENT_STEP=0, TOTAL_STEPS=17500, INITIAL_R_GATE_COMPATIBILITY="PASS")


def arm_paths(arm: str) -> dict[str, Path]:
    root = ARTIFACT / arm.replace("-", "_")
    evaluation = root / "dev_evaluation"
    return {
        "root": root, "recovery": root / "recovery.ckpt", "final": root / "final.ckpt",
        "log": root / "TRAIN_LOG.csv", "evaluation": evaluation, "done": root / "DONE.json",
        "payload": evaluation / "EVALUATION_PAYLOAD.pt",
        "coordinates_ready": root / "COORDINATES_READY.json",
        "per_record": evaluation / "PER_RECORD.parquet", "predictive": evaluation / "PREDICTIVE.json",
        "sdf": evaluation / "Proposal.sdf", "pb": evaluation / "POSEBUSTERS.parquet",
        "v3d": evaluation / "VALIDITY3D.parquet", "result": evaluation / "RESULT.json",
    }


def validate_coordinates_ready(arm: str) -> bool:
    paths = arm_paths(arm)
    if not paths["coordinates_ready"].is_file():
        return False
    marker = json.loads(paths["coordinates_ready"].read_text(encoding="utf-8"))
    if marker.get("arm") != arm or marker.get("status") != "COORDINATES_READY" or marker.get("records") != 5000:
        return False
    if not DEV_MANIFEST.is_file():
        return False
    manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    expected_ids = [sample for row in manifest["rows"] for sample in row["sample_ids"]]
    expected_ids_sha = hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()
    if marker.get("record_ids_sha256") != expected_ids_sha:
        return False
    for name in ("final", "sdf", "payload"):
        if not paths[name].is_file() or marker.get(f"{name}_sha256") != sha256(paths[name]):
            return False
    return True


def make_arm(arm: str, device: torch.device):
    init = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
    model = DirectMuSigmaModel(128, 3).to(device); model.load_state_dict(init["model_state"], strict=True)
    reliability = None
    if arm.endswith("R1"):
        reliability = PrimitiveReliabilityHead(128).to(device); reliability.load_state_dict(init["reliability_state"], strict=True)
    return model, reliability


def optimizers(model: DirectMuSigmaModel, reliability: PrimitiveReliabilityHead | None):
    config = cfg()["training"]
    belief = torch.optim.AdamW([
        {"params": model.backbone_parameters(), "lr": config["backbone_learning_rate"]},
        {"params": model.mu_head_parameters() + model.sigma_head_parameters(), "lr": config["head_learning_rate"]},
    ], weight_decay=config["weight_decay"])
    belief_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(belief, T_max=config["scheduler_horizon"])
    action = action_scheduler = None
    if reliability is not None:
        action = torch.optim.AdamW([
            {"params": model.backbone_parameters(), "lr": config["backbone_learning_rate"]},
            {"params": list(reliability.parameters()), "lr": config["head_learning_rate"]},
        ], weight_decay=config["weight_decay"])
        action_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(action, T_max=config["scheduler_horizon"])
    return belief, belief_scheduler, action, action_scheduler


def quantile(values: torch.Tensor, q: float) -> float | None:
    return float(torch.quantile(values.detach().float().cpu(), q)) if values.numel() else None


def train_arm(arm: str) -> None:
    paths = arm_paths(arm); paths["root"].mkdir(parents=True, exist_ok=True)
    if paths["final"].is_file(): return
    config = cfg(); training = config["training"]; method = arm.split("-")[0]
    device = execution_device()
    prepared, source_payload = load_inputs(); sources = source_index(source_payload["train"])
    generator = seed_all(config["seed"]); model, reliability = make_arm(arm, device)
    belief_optimizer, belief_scheduler, action_optimizer, action_scheduler = optimizers(model, reliability)
    start, logs = 0, []
    if paths["recovery"].is_file():
        saved = torch.load(paths["recovery"], map_location="cpu", weights_only=False)
        if saved.get("arm") != arm or saved.get("config_sha256") != sha256(CONFIG_PATH): raise RuntimeError(f"invalid recovery identity: {arm}")
        model.load_state_dict(saved["model_state"], strict=True)
        if reliability is not None: reliability.load_state_dict(saved["reliability_state"], strict=True)
        belief_optimizer.load_state_dict(saved["belief_optimizer"]); belief_scheduler.load_state_dict(saved["belief_scheduler"])
        if action_optimizer is not None:
            action_optimizer.load_state_dict(saved["action_optimizer"]); action_scheduler.load_state_dict(saved["action_scheduler"])
        generator.set_state(saved["generator_state"]); random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"]); torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start = int(saved["step"])
        if paths["log"].is_file(): logs = pd.read_csv(paths["log"]).to_dict("records")
    for step in range(start + 1, training["belief_updates"] + 1):
        tick = time.time(); model.train()
        if reliability is not None: reliability.train()
        graphs, bg, source, reference, _ = sample_batch(prepared["train"], sources, generator, training["batch_molecules"], device)
        ref_b, ref_a = geometry_values(reference, bg)
        belief_optimizer.zero_grad(set_to_none=True)
        pred = model(bg, detach_sigma_features=method == "J2")
        loss_belief, _ = belief_loss(method, pred, ref_b.to(pred["bond_mu"]), ref_a.to(pred["angle_mu"]), graphs, training["beta_nll_beta"])
        if not bool(torch.isfinite(loss_belief)): raise RuntimeError(f"nonfinite belief loss {arm} step {step}")
        loss_belief.backward(); grad_belief = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
        if not bool(torch.isfinite(grad_belief)): raise RuntimeError(f"nonfinite belief gradient {arm} step {step}")
        belief_optimizer.step(); belief_scheduler.step()
        loss_action_value = None; r_b = pred["bond_mu"].new_ones(pred["bond_mu"].shape); r_a = pred["angle_mu"].new_ones(pred["angle_mu"].shape)
        if reliability is not None:
            action_optimizer.zero_grad(set_to_none=True)
            pred_action = model(bg, detach_sigma_features=method == "J2")
            action = action_proposal(source, graphs, pred_action, tau=config["action"]["tau_control_angstrom"], atom_cap=config["action"]["atom_cap_angstrom"], reliability_head=reliability)
            loss_action = action_loss(action, reference, graphs, pred_action)
            if not bool(torch.isfinite(loss_action)): raise RuntimeError(f"nonfinite action loss {arm} step {step}")
            loss_action.backward()
            grad_action = torch.nn.utils.clip_grad_norm_(model.backbone_parameters() + list(reliability.parameters()), training["gradient_clip"])
            if not bool(torch.isfinite(grad_action)): raise RuntimeError(f"nonfinite action gradient {arm} step {step}")
            action_optimizer.step(); action_scheduler.step()
            loss_action_value = float(loss_action.detach()); r_b = action.bond_reliability.detach(); r_a = action.angle_reliability.detach()
        if step == 1 or step % training["log_interval"] == 0:
            sigma = torch.cat((pred["bond_sigma"].detach(), pred["angle_sigma"].detach()))
            r = torch.cat((r_b, r_a)).detach()
            row = {
                "step": step, "LOSS_BELIEF": float(loss_belief.detach()), "LOSS_ACTION": loss_action_value,
                "MU_BOND_MAE_TRAIN": float((pred["bond_mu"].detach() - ref_b.to(pred["bond_mu"])).abs().mean()),
                "MU_ANGLE_MAE_TRAIN": float((pred["angle_mu"].detach() - ref_a.to(pred["angle_mu"])).abs().mean()),
                "SIGMA_MEAN": float(sigma.mean()), "SIGMA_P01": quantile(sigma, .01), "SIGMA_P50": quantile(sigma, .5), "SIGMA_P99": quantile(sigma, .99),
                "R_MEAN": float(r.mean()), "R_P01": quantile(r, .01), "R_P50": quantile(r, .5), "R_P99": quantile(r, .99),
                "NONFINITE_COUNT": int((~torch.isfinite(sigma)).sum() + (~torch.isfinite(r)).sum()),
                "BELIEF_GRAD_NORM": float(grad_belief), "STEP_SECONDS": time.time() - tick,
            }
            logs.append(row)
            update_status("TRAINING", "RUNNING", CURRENT_ARM=arm, CURRENT_STEP=step, TOTAL_STEPS=training["belief_updates"], **{k: row[k] for k in ("LOSS_BELIEF", "LOSS_ACTION", "MU_BOND_MAE_TRAIN", "MU_ANGLE_MAE_TRAIN", "SIGMA_MEAN", "SIGMA_P01", "SIGMA_P50", "SIGMA_P99", "R_MEAN", "R_P01", "R_P50", "R_P99", "NONFINITE_COUNT")})
        if step % training["recovery_interval"] == 0:
            recovery = {"schema_version": "sixs-musigma-arm-recovery-v1", "arm": arm, "step": step, "config_sha256": sha256(CONFIG_PATH), "common_initialization_sha256": sha256(COMMON_INIT), "model_state": model.state_dict(), "reliability_state": reliability.state_dict() if reliability is not None else None, "belief_optimizer": belief_optimizer.state_dict(), "belief_scheduler": belief_scheduler.state_dict(), "action_optimizer": action_optimizer.state_dict() if action_optimizer is not None else None, "action_scheduler": action_scheduler.state_dict() if action_scheduler is not None else None, "generator_state": generator.get_state(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], **protected_fields()}
            atomic_torch(paths["recovery"], recovery); atomic_frame(paths["log"], pd.DataFrame(logs))
        if step % 25 == 0: print(json.dumps({"arm": arm, "step": step, "loss_belief": float(loss_belief.detach()), "loss_action": loss_action_value}), flush=True)
    final = {"schema_version": "sixs-musigma-arm-final-v1", "arm": arm, "step": training["belief_updates"], "config_sha256": sha256(CONFIG_PATH), "common_initialization_sha256": sha256(COMMON_INIT), "model_state": model.state_dict(), "reliability_state": reliability.state_dict() if reliability is not None else None, **protected_fields()}
    atomic_torch(paths["final"], final); atomic_frame(paths["log"], pd.DataFrame(logs))
    update_status("FINAL_CHECKPOINT_FROZEN", "PASS", CURRENT_ARM=arm, CURRENT_STEP=training["belief_updates"], TOTAL_STEPS=training["belief_updates"], FINAL_CHECKPOINT_SHA256=sha256(paths["final"]))


def distribution_summary(values: np.ndarray, prefix: str = "") -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values); clean = values[finite]
    result: dict[str, Any] = {f"{prefix}nonfinite_fraction": float(1.0 - finite.mean())}
    if not len(clean): return result
    result.update({f"{prefix}mean": float(clean.mean()), f"{prefix}std": float(clean.std()), f"{prefix}min": float(clean.min()), f"{prefix}max": float(clean.max())})
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99): result[f"{prefix}p{q:02d}"] = float(np.percentile(clean, q))
    return result


def predictive_family_summary(rows: pd.DataFrame, family: str) -> dict[str, Any]:
    y = np.concatenate(rows[f"{family}_reference"].to_numpy()); mu = np.concatenate(rows[f"{family}_mu"].to_numpy()); sigma = np.concatenate(rows[f"{family}_sigma"].to_numpy()); stat = np.concatenate(rows[f"{family}_sigma_stat"].to_numpy())
    residual = y - mu; z = residual / sigma; absolute_z = np.abs(z); delta = np.log(sigma / stat)
    result = {
        "primitives": int(len(y)), "mu_mae": float(np.mean(np.abs(residual))), "mu_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "gaussian_nll": float(np.mean(.5 * (residual / sigma) ** 2 + np.log(sigma) + .5 * np.log(2 * np.pi))),
        **distribution_summary(sigma, "sigma_"), "z_mean": float(np.mean(z)), "z_std": float(np.std(z)),
        "z_median_abs": float(np.median(absolute_z)), "z_p90_abs": float(np.percentile(absolute_z, 90)), "z_p95_abs": float(np.percentile(absolute_z, 95)), "z_p99_abs": float(np.percentile(absolute_z, 99)),
        "coverage_abs_z_le_1": float(np.mean(absolute_z <= 1)), "coverage_abs_z_le_1_96": float(np.mean(absolute_z <= 1.96)), "coverage_abs_z_le_3": float(np.mean(absolute_z <= 3)),
        "sigma_lower_saturation_fraction": float(np.mean(delta <= -6 + 1e-6)), "sigma_upper_saturation_fraction": float(np.mean(delta >= 6 - 1e-6)),
        "sigma_stat_equal_fraction": float(np.mean(np.abs(delta) <= 1e-7)), "extreme_inflation_fraction_sigma_gt_10x_stat": float(np.mean(sigma > 10 * stat)),
    }
    return result


def reliability_summary(rows: pd.DataFrame, family: str) -> dict[str, Any]:
    r = np.concatenate(rows[f"{family}_reliability"].to_numpy()); defect = np.concatenate(rows[f"{family}_source_abs_defect"].to_numpy()); standardized = np.concatenate(rows[f"{family}_source_abs_standardized_defect"].to_numpy()); sigma = np.concatenate(rows[f"{family}_sigma"].to_numpy())
    def corr(left, right):
        return float(pd.Series(left).corr(pd.Series(right), method="spearman")) if np.std(left) > 0 and np.std(right) > 0 else None
    return {**distribution_summary(r), "fraction_lt_0_1": float(np.mean(r < .1)), "fraction_lt_0_25": float(np.mean(r < .25)), "fraction_lt_0_5": float(np.mean(r < .5)), "fraction_gt_0_9": float(np.mean(r > .9)), "fraction_gt_0_99": float(np.mean(r > .99)), "spearman_abs_source_defect": corr(r, defect), "spearman_abs_standardized_defect": corr(r, standardized), "spearman_sigma": corr(r, sigma), "RELIABILITY_COLLAPSED_TO_ZERO": bool(np.mean(r < .1) > .9), "RELIABILITY_COLLAPSED_TO_ONE": bool(np.mean(r > .99) > .9)}


def write_molecule(writer: Chem.SDWriter, record: Mapping[str, Any], xyz: torch.Tensor, sample_id: str, arm: str) -> None:
    adapted = adapt_formal_cache_record(record); mol = Chem.Mol(adapted["_formal_rdkit_mol"]); conf = Chem.Conformer(mol.GetNumAtoms())
    for index, point in enumerate(xyz.detach().cpu().double().tolist()): conf.SetAtomPosition(index, point)
    mol.RemoveAllConformers(); mol.AddConformer(conf, assignId=True); mol.SetProp("_Name", sample_id); mol.SetProp("sample_id", sample_id); mol.SetProp("method", arm); writer.write(mol)


def selected_pb_columns(installed: Mapping[str, Any]) -> list[str]:
    result = []
    for module in installed.get("modules", []):
        renames = module.get("rename_outputs", {})
        for raw in module.get("chosen_binary_test_output", []):
            name = str(renames.get(raw, raw)).lower().replace(" ", "_")
            if name not in result: result.append(name)
    return result


def run_external_evaluators(arm: str, paths: Mapping[str, Path], ids: Sequence[str]) -> None:
    if not EXTERNAL_PYTHON.is_file(): raise RuntimeError("external-validity Python is missing")
    subprocess.run([str(EXTERNAL_PYTHON),str(EXTERNAL_WORKER),"--arm",arm,"--sdf",str(paths["sdf"]),"--records",str(paths["per_record"]),"--pb",str(paths["pb"]),"--v3d",str(paths["v3d"])],cwd=ROOT,check=True)
    pb=pd.read_parquet(paths["pb"]);v3d=pd.read_parquet(paths["v3d"])
    if len(pb)!=len(ids) or pb.record_id.astype(str).tolist()!=list(ids):raise RuntimeError(f"PoseBusters returned changed identities: {arm}")
    if len(v3d)!=len(ids) or v3d.record_id.astype(str).tolist()!=list(ids):raise RuntimeError(f"Validity3D returned changed identities: {arm}")


def complete_arm_from_existing(arm: str) -> None:
    """Finish result materialization without regenerating frozen coordinates."""
    paths=arm_paths(arm);predictive=json.loads(paths["predictive"].read_text(encoding="utf-8"));pb=pd.read_parquet(paths["pb"]);v3d=pd.read_parquet(paths["v3d"]);records=pd.read_parquet(paths["per_record"]).drop(columns=["PB","V3D"],errors="ignore")
    records=records.merge(pb[["record_id","PB"]],on="record_id",validate="one_to_one").merge(v3d[["record_id","validity3d"]].rename(columns={"validity3d":"V3D"}),on="record_id",validate="one_to_one");atomic_frame(paths["per_record"],records)
    summary={"arm":arm,"records":len(records),"molecules":records.molecule_id.nunique(),"proposal_V3D":float(records.V3D.mean()),"proposal_PB":float(records.PB.mean()),"internal_post":float(records.internal_post.mean()),"direction_improvement":float(records.direction_improvement.mean()),"bond_raw_mae":float(records.bond_raw_mae.mean()),"angle_raw_mae":float(records.angle_raw_mae.mean()),"source_rmsd":float(records.source_rmsd.mean()),"proposal_movement":float(records.proposal_movement.mean()),"atom_cap_fraction":float(records.atom_cap_active.mean()),"rollback_fraction":float(records.rollback.mean()),"mu_bond_mae":predictive["bond"]["mu_mae"],"mu_angle_mae":predictive["angle"]["mu_mae"],"sigma_bond_nll":predictive["bond"]["gaussian_nll"],"sigma_angle_nll":predictive["angle"]["gaussian_nll"],**protected_fields()}
    result={"schema_version":"sixs-musigma-arm-result-v1","status":"COMPLETE","summary":summary,"predictive":predictive,"checkpoint_sha256":sha256(paths["final"]),"artifacts":{"per_record_sha256":sha256(paths["per_record"]),"posebusters_sha256":sha256(paths["pb"]),"validity3d_sha256":sha256(paths["v3d"]),"sdf_sha256":sha256(paths["sdf"])},**protected_fields()};atomic_json(paths["result"],result)
    atomic_text(REPORT/REPORT_NAMES[arm],f"# {arm} frozen development result\n\n{markdown_frame(pd.DataFrame([summary]))}\n\nPredictive diagnostics: `{paths['predictive']}`. This final checkpoint was evaluated on DEV exactly once. Formal/large holdout/xTB were not accessed.")


def materialize_evaluation_payload(arm: str) -> None:
    paths = arm_paths(arm)
    if not validate_coordinates_ready(arm):
        raise RuntimeError(f"invalid coordinate-ready manifest: {arm}")
    payload = torch.load(paths["payload"], map_location="cpu", weights_only=False)
    if payload.get("arm") != arm or payload.get("record_ids") is None or len(payload["record_ids"]) != 5000:
        raise RuntimeError(f"invalid evaluation payload identity: {arm}")
    primitive_rows = payload["primitive_rows"]
    record_rows = payload["record_rows"]
    primitives = pd.DataFrame(primitive_rows)
    records = pd.DataFrame(record_rows)
    if records.record_id.astype(str).tolist() != [str(value) for value in payload["record_ids"]]:
        raise RuntimeError(f"evaluation payload record order changed: {arm}")
    reliability_present = bool(payload["reliability_present"])
    predictive = {
        "bond": predictive_family_summary(primitives, "bond"),
        "angle": predictive_family_summary(primitives, "angle"),
        "mean_protection": {
            "bond_delta_candidate_minus_base": float((records.bond_mu_mae-records.base_bond_mu_mae).mean()),
            "angle_delta_candidate_minus_base": float((records.angle_mu_mae-records.base_angle_mu_mae).mean()),
        },
        "reliability": None if not reliability_present else {
            "bond": reliability_summary(primitives, "bond"),
            "angle": reliability_summary(primitives, "angle"),
        },
    }
    atomic_json(paths["predictive"], predictive)
    atomic_frame(paths["per_record"], records)


def evaluate_arm(arm: str, *, coordinates_only: bool = False) -> None:
    paths = arm_paths(arm)
    if paths["result"].is_file(): return
    if not paths["final"].is_file(): raise RuntimeError(f"missing final checkpoint: {arm}")
    if validate_coordinates_ready(arm) and not (paths["per_record"].is_file() and paths["predictive"].is_file()):
        if coordinates_only:
            return
        update_status("PAYLOAD_MATERIALIZATION", "RUNNING", CURRENT_ARM=arm, CURRENT_STEP=17500, TOTAL_STEPS=17500)
        materialize_evaluation_payload(arm)
    if all(paths[name].is_file() for name in ("per_record","predictive","sdf")):
        if coordinates_only:
            if not validate_coordinates_ready(arm):
                raise RuntimeError(f"coordinate-only stage lacks frozen manifest: {arm}")
            return
        frozen=pd.read_parquet(paths["per_record"]);ids=frozen.record_id.astype(str).tolist()
        if len(ids)!=5000 or len(set(ids))!=5000:raise RuntimeError(f"frozen coordinate identity changed: {arm}")
        if not (paths["pb"].is_file() and paths["v3d"].is_file()):
            update_status("EXTERNAL_EVALUATION_RECOVERY","RUNNING",CURRENT_ARM=arm,CURRENT_STEP=17500,TOTAL_STEPS=17500,EVALUATION_RECORDS=5000,EVALUATION_TOTAL=5000)
            run_external_evaluators(arm,paths,ids)
        complete_arm_from_existing(arm);return
    update_status("DEV_COORDINATES", "RUNNING", CURRENT_ARM=arm, CURRENT_STEP=17500, TOTAL_STEPS=17500)
    config = cfg(); device = execution_device()
    prepared, source_payload = load_inputs(); by_item = {str(item["molecule_id"]): item for item in prepared["val"]}; by_sample = {str(row["sample_id"]): row for row in source_payload["val"]}
    manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8")); ids = [sample for row in manifest["rows"] for sample in row["sample_ids"]]
    final = torch.load(paths["final"], map_location="cpu", weights_only=False); model, reliability = make_arm(arm, device); model.load_state_dict(final["model_state"], strict=True)
    if reliability is not None: reliability.load_state_dict(final["reliability_state"], strict=True)
    model.eval(); reliability.eval() if reliability is not None else None
    base, _ = make_arm("J0-R0", device); base.eval()
    val_manifest = pd.read_parquet(config["data"]["val_manifest"]); manifest_by_sample = {str(row.sample_id): row for row in val_manifest.itertuples(index=False)}
    paths["evaluation"].mkdir(parents=True, exist_ok=True); temp_sdf = Path(str(paths["sdf"]) + f".tmp.{os.getpid()}"); writer = Chem.SDWriter(str(temp_sdf)); primitive_rows, record_rows = [], []
    method = arm.split("-")[0]
    try:
        for start in range(0, len(ids), 64):
            batch_ids = ids[start:start + 64]; source_rows = [by_sample[x] for x in batch_ids]; items = [by_item[str(row["molecule_id"])] for row in source_rows]; graphs = [item["graph"] for item in items]; bg = collate_graphs(graphs).to(device)
            source = torch.cat([torch.as_tensor(row["source"], dtype=torch.float64) for row in source_rows]).to(device); reference = torch.cat([torch.as_tensor(item["references"][0], dtype=torch.float64) for item in items]).to(device)
            with torch.no_grad(): pred = model(bg, detach_sigma_features=method == "J2"); base_pred = base(bg)
            with torch.enable_grad(): action = action_proposal(source, graphs, pred, tau=config["action"]["tau_control_angstrom"], atom_cap=config["action"]["atom_cap_angstrom"], reliability_head=reliability)
            ref_b, ref_a = geometry_values(reference, bg); src_b, src_a = geometry_values(source, bg); prop_b, prop_a = geometry_values(action.proposal, bg)
            bo = ao = atoms = 0
            for sample_id, source_row, item, graph in zip(batch_ids, source_rows, items, graphs, strict=True):
                nb, na, nat = graph.bonds.size(1), graph.angles.size(0), graph.atom_categorical.size(0); bb=slice(bo,bo+nb); aa=slice(ao,ao+na); xx=slice(atoms,atoms+nat); bo+=nb;ao+=na;atoms+=nat
                bmu=pred["bond_mu"][bb].detach().cpu().double().numpy(); amu=pred["angle_mu"][aa].detach().cpu().double().numpy(); bs=pred["bond_sigma"][bb].detach().cpu().double().numpy(); ass=pred["angle_sigma"][aa].detach().cpu().double().numpy(); bstat=bg.bond_fixed[bb,1].detach().cpu().double().numpy(); astat=bg.angle_fixed[aa,1].detach().cpu().double().numpy(); rb=action.bond_reliability[bb].detach().cpu().double().numpy(); ra=action.angle_reliability[aa].detach().cpu().double().numpy()
                yb=ref_b[bb].detach().cpu().numpy(); ya=ref_a[aa].detach().cpu().numpy(); sb=src_b[bb].detach().cpu().numpy(); sa=src_a[aa].detach().cpu().numpy(); pb=prop_b[bb].detach().cpu().numpy(); pa=prop_a[aa].detach().cpu().numpy()
                primitive_rows.append({"record_id":sample_id,"molecule_id":str(source_row["molecule_id"]),"bond_reference":yb,"bond_mu":bmu,"bond_sigma":bs,"bond_sigma_stat":bstat,"bond_reliability":rb,"bond_source_abs_defect":np.abs(sb-bmu),"bond_source_abs_standardized_defect":np.abs((sb-bmu)/bs),"angle_reference":ya,"angle_mu":amu,"angle_sigma":ass,"angle_sigma_stat":astat,"angle_reliability":ra,"angle_source_abs_defect":np.abs(sa-amu),"angle_source_abs_standardized_defect":np.abs((sa-amu)/ass)})
                common_source=.5*(np.mean(((sb-yb)/bstat)**2)+np.mean(((sa-ya)/astat)**2)); common_post=.5*(np.mean(((pb-yb)/bstat)**2)+np.mean(((pa-ya)/astat)**2)); own_post=.5*(np.mean(((pb-yb)/bs)**2)+np.mean(((pa-ya)/ass)**2)); delta=action.proposal[xx]-source[xx]; source_rms=float(delta.square().sum(-1).mean().sqrt()); max_atom=float(torch.linalg.vector_norm(delta,dim=-1).max())
                final_coord, safety = safety_accept(source[xx].detach().cpu(), action.proposal[xx].detach().cpu(), graph)
                base_b=base_pred["bond_mu"][bb].detach().cpu().double().numpy(); base_a=base_pred["angle_mu"][aa].detach().cpu().double().numpy()
                record_rows.append({"record_id":sample_id,"molecule_id":str(source_row["molecule_id"]),"arm":arm,"bond_mu_mae":float(np.mean(np.abs(bmu-yb))),"angle_mu_mae":float(np.mean(np.abs(amu-ya))),"base_bond_mu_mae":float(np.mean(np.abs(base_b-yb))),"base_angle_mu_mae":float(np.mean(np.abs(base_a-ya))),"common_source_objective":common_source,"internal_post":common_post,"own_sigma_post":own_post,"direction_improvement":common_source-common_post,"bond_raw_mae":float(np.mean(np.abs(pb-yb))),"angle_raw_mae":float(np.mean(np.abs(pa-ya))),"source_rmsd":source_rms,"proposal_movement":source_rms,"max_atom_displacement":max_atom,"atom_cap_active":bool(action.cap_active[len(record_rows)%64].detach().cpu()),"rollback":bool(safety["fallback"]),"PB":None,"V3D":None})
                meta = manifest_by_sample[sample_id]; cache = Path(config["data"]["val_cache"]) / Path(str(meta.source_path)).name; raw = cache.read_bytes()
                if hashlib.sha256(raw).hexdigest() != str(meta.source_file_sha256): raise RuntimeError("DEV Source cache hash changed")
                record = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False); write_molecule(writer, record, action.proposal[xx], sample_id, arm)
            update_status("DEV_COORDINATES", "RUNNING", CURRENT_ARM=arm, CURRENT_STEP=17500, TOTAL_STEPS=17500, EVALUATION_RECORDS=min(start+64,len(ids)), EVALUATION_TOTAL=len(ids))
    finally: writer.close()
    os.replace(temp_sdf, paths["sdf"])
    atomic_torch(paths["payload"], {
        "schema_version": "sixs-factorial-evaluation-payload-v1", "arm": arm,
        "record_ids": ids, "primitive_rows": primitive_rows, "record_rows": record_rows,
        "reliability_present": reliability is not None, **protected_fields(),
    })
    atomic_json(paths["coordinates_ready"], {
        "schema_version": "sixs-factorial-coordinates-ready-v1", "status": "COORDINATES_READY",
        "arm": arm, "records": len(ids),
        "record_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "final_sha256": sha256(paths["final"]), "sdf_sha256": sha256(paths["sdf"]),
        "payload_sha256": sha256(paths["payload"]), **protected_fields(),
    })
    if coordinates_only:
        del model, reliability, base, primitive_rows, record_rows
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        update_status("COORDINATES_READY", "PASS", CURRENT_ARM=arm, CURRENT_STEP=17500, TOTAL_STEPS=17500, EVALUATION_RECORDS=len(ids), EVALUATION_TOTAL=len(ids))
        return
    primitives = pd.DataFrame(primitive_rows); records = pd.DataFrame(record_rows)
    predictive = {"bond":predictive_family_summary(primitives,"bond"),"angle":predictive_family_summary(primitives,"angle"),"mean_protection":{"bond_delta_candidate_minus_base":float((records.bond_mu_mae-records.base_bond_mu_mae).mean()),"angle_delta_candidate_minus_base":float((records.angle_mu_mae-records.base_angle_mu_mae).mean())},"reliability":None if reliability is None else {"bond":reliability_summary(primitives,"bond"),"angle":reliability_summary(primitives,"angle")}}
    atomic_json(paths["predictive"], predictive); atomic_frame(paths["per_record"], records)
    del primitives, primitive_rows; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    update_status("POSEBUSTERS", "RUNNING", CURRENT_ARM=arm, CURRENT_STEP=17500, TOTAL_STEPS=17500)
    run_external_evaluators(arm, paths, ids)
    pb = pd.read_parquet(paths["pb"]); v3d = pd.read_parquet(paths["v3d"]); records = pd.read_parquet(paths["per_record"]).drop(columns=["PB","V3D"])
    records = records.merge(pb[["record_id","PB"]],on="record_id",validate="one_to_one").merge(v3d[["record_id","validity3d"]].rename(columns={"validity3d":"V3D"}),on="record_id",validate="one_to_one"); atomic_frame(paths["per_record"],records)
    summary = {"arm":arm,"records":len(records),"molecules":records.molecule_id.nunique(),"proposal_V3D":float(records.V3D.mean()),"proposal_PB":float(records.PB.mean()),"internal_post":float(records.internal_post.mean()),"direction_improvement":float(records.direction_improvement.mean()),"bond_raw_mae":float(records.bond_raw_mae.mean()),"angle_raw_mae":float(records.angle_raw_mae.mean()),"source_rmsd":float(records.source_rmsd.mean()),"proposal_movement":float(records.proposal_movement.mean()),"atom_cap_fraction":float(records.atom_cap_active.mean()),"rollback_fraction":float(records.rollback.mean()),"mu_bond_mae":predictive["bond"]["mu_mae"],"mu_angle_mae":predictive["angle"]["mu_mae"],"sigma_bond_nll":predictive["bond"]["gaussian_nll"],"sigma_angle_nll":predictive["angle"]["gaussian_nll"],**protected_fields()}
    result = {"schema_version":"sixs-musigma-arm-result-v1","status":"COMPLETE","summary":summary,"predictive":predictive,"checkpoint_sha256":sha256(paths["final"]),"artifacts":{"per_record_sha256":sha256(paths["per_record"]),"posebusters_sha256":sha256(paths["pb"]),"validity3d_sha256":sha256(paths["v3d"]),"sdf_sha256":sha256(paths["sdf"])},**protected_fields()}; atomic_json(paths["result"],result)
    table = pd.DataFrame([summary]); atomic_text(REPORT/REPORT_NAMES[arm],f"# {arm} frozen development result\n\n{markdown_frame(table)}\n\nPredictive diagnostics: `{paths['predictive']}`. This final checkpoint was evaluated on DEV exactly once. Formal/large holdout/xTB were not accessed.")


def validate_done(arm: str) -> bool:
    paths=arm_paths(arm)
    if not paths["done"].is_file(): return False
    done=json.loads(paths["done"].read_text(encoding="utf-8"))
    if done.get("arm")!=arm or done.get("status")!="COMPLETE": return False
    for name, expected in done.get("artifact_hashes",{}).items():
        if name not in paths or not paths[name].is_file() or sha256(paths[name])!=expected: return False
    return True


def freeze_done(arm: str) -> None:
    paths=arm_paths(arm); names=("final","log","per_record","predictive","sdf","pb","v3d","result")
    if not all(paths[name].is_file() for name in names): raise RuntimeError(f"incomplete arm cannot be marked DONE: {arm}")
    atomic_json(paths["done"],{"schema_version":"sixs-musigma-arm-done-v1","arm":arm,"status":"COMPLETE","training_status":"PASS","evaluator_status":"PASS","artifact_hashes":{name:sha256(paths[name]) for name in names},**protected_fields()})


def update_live() -> None:
    rows=[]
    for arm in ARMS:
        path=arm_paths(arm)["result"]
        if path.is_file(): rows.append(json.loads(path.read_text(encoding="utf-8"))["summary"])
    frame=pd.DataFrame(rows); atomic_frame(LIVE_RESULTS,frame)
    atomic_text(LIVE_SUMMARY,"# Live frozen six-arm results\n\n"+(markdown_frame(frame) if len(frame) else "No arm has completed DEV evaluation yet.")+"\n\nOnly completed final-checkpoint evaluations appear here; protocol is frozen and unchanged.")


PRIMARY_COMPARISONS = (
    ("J0-R0","J1-R0"),("J0-R0","J2-R0"),("J1-R0","J2-R0"),
    ("J0-R0","J0-R1"),("J1-R0","J1-R1"),("J2-R0","J2-R1"),
)
ACTION_METRICS = ("V3D","PB","internal_post","direction_improvement","bond_raw_mae","angle_raw_mae","source_rmsd")


def cluster_bootstrap(left: pd.DataFrame, right: pd.DataFrame, metric: str, *, resamples: int = 10000, seed: int = 20260828) -> dict[str, Any]:
    joined=left[["record_id","molecule_id",metric]].merge(right[["record_id",metric]],on="record_id",suffixes=("_left","_right"),validate="one_to_one")
    joined["delta"]=joined[f"{metric}_right"].astype(float)-joined[f"{metric}_left"].astype(float)
    cluster=joined.groupby("molecule_id",sort=True).delta.mean().to_numpy(float); rng=np.random.default_rng(seed); n=len(cluster); draws=np.empty(resamples)
    for start in range(0,resamples,500):
        count=min(500,resamples-start); sampled=rng.integers(0,n,size=(count,n)); draws[start:start+count]=cluster[sampled].mean(axis=1)
    return {"delta_candidate_minus_baseline":float(cluster.mean()),"ci95_low":float(np.percentile(draws,2.5)),"ci95_high":float(np.percentile(draws,97.5)),"bootstrap_clusters":n,"bootstrap_resamples":resamples,"seed":seed}


def classify_predictive(result: Mapping[str, Any]) -> tuple[str,str]:
    pred=result["predictive"]; protection=pred["mean_protection"]
    base_records=pd.read_parquet(arm_paths(result["summary"]["arm"])["per_record"])
    bond_base=float(base_records.base_bond_mu_mae.mean()); angle_base=float(base_records.base_angle_mu_mae.mean())
    mu="DEGRADED" if protection["bond_delta_candidate_minus_base"]>.01*bond_base or protection["angle_delta_candidate_minus_base"]>.01*angle_base else "PASS"
    nonfinite=max(pred[fam]["sigma_nonfinite_fraction"] for fam in ("bond","angle")); upper=max(pred[fam]["sigma_upper_saturation_fraction"] for fam in ("bond","angle")); lower=max(pred[fam]["sigma_lower_saturation_fraction"] for fam in ("bond","angle"))
    if nonfinite>0: pathology="NONFINITE"
    elif upper>.01: pathology="INFLATION"
    elif lower>.01: pathology="COLLAPSE"
    else: pathology="NONE"
    good=all(.8<=pred[fam]["z_std"]<=1.25 and .60<=pred[fam]["coverage_abs_z_le_1"]<=.76 and .90<=pred[fam]["coverage_abs_z_le_1_96"]<=.99 for fam in ("bond","angle"))
    quality="GOOD" if good and pathology=="NONE" else ("PARTIAL" if pathology=="NONE" else "BAD")
    return mu, quality+"|"+pathology


def effect_class(row_v3d: Mapping[str,Any], row_pb: Mapping[str,Any], row_direction: Mapping[str,Any]) -> str:
    if row_v3d["ci95_low"]>0 and row_pb["ci95_low"]>=0 and row_direction["ci95_low"]>=0: return "POSITIVE"
    if row_v3d["ci95_high"]<0 or row_pb["ci95_high"]<0 or row_direction["ci95_high"]<0: return "NEGATIVE"
    return "NEUTRAL"


def finalize_analysis() -> None:
    if not all(validate_done(arm) for arm in ARMS): raise RuntimeError("all six valid DONE manifests are required")
    summaries=[]; results={}; records={}
    for arm in ARMS:
        results[arm]=json.loads(arm_paths(arm)["result"].read_text(encoding="utf-8")); summaries.append(results[arm]["summary"]); records[arm]=pd.read_parquet(arm_paths(arm)["per_record"])
    action=pd.DataFrame(summaries); atomic_frame(REPORT/"09_ACTION_COMPARISON.csv",action)
    predictive=[]; classification={}
    for arm in ARMS:
        mu,combined=classify_predictive(results[arm]); quality,pathology=combined.split("|")
        classification[arm]={"MU_QUALITY":mu,"SIGMA_PREDICTIVE_QUALITY":quality,"SIGMA_PATHOLOGY":pathology}
        for family in ("bond","angle"): predictive.append({"arm":arm,"family":family,**results[arm]["predictive"][family],**classification[arm]})
    atomic_frame(REPORT/"08_PREDICTIVE_SIGMA_COMPARISON.csv",pd.DataFrame(predictive))
    boot=[]
    for baseline,candidate in PRIMARY_COMPARISONS:
        for metric in ACTION_METRICS:
            row={"comparison":f"{candidate}_minus_{baseline}","baseline":baseline,"candidate":candidate,"metric":metric,**cluster_bootstrap(records[baseline],records[candidate],metric)}; boot.append(row)
    boot_frame=pd.DataFrame(boot)
    valid_j=[j for j in ("J0","J1","J2") if classification[f"{j}-R0"]["MU_QUALITY"]=="PASS" and classification[f"{j}-R0"]["SIGMA_PREDICTIVE_QUALITY"]!="BAD"]
    if valid_j:
        best_r0=max((f"{j}-R0" for j in valid_j),key=lambda arm:results[arm]["summary"]["proposal_V3D"]); best_r1=max((f"{j}-R1" for j in valid_j),key=lambda arm:results[arm]["summary"]["proposal_V3D"])
        for metric in ACTION_METRICS: boot.append({"comparison":"BEST_R1_minus_BEST_R0","baseline":best_r0,"candidate":best_r1,"metric":metric,**cluster_bootstrap(records[best_r0],records[best_r1],metric)})
        boot_frame=pd.DataFrame(boot)
    else: best_r0=best_r1=None
    atomic_frame(REPORT/"10_PAIRED_BOOTSTRAP.csv",boot_frame)
    factorial=[]
    for metric in ("V3D","PB","internal_post","direction_improvement"):
        r_deltas=[]
        for method in ("J0","J1","J2"):
            row=cluster_bootstrap(records[f"{method}-R0"],records[f"{method}-R1"],metric); factorial.append({"effect":f"R_effect_within_{method}","metric":metric,**row});r_deltas.append(row["delta_candidate_minus_baseline"])
        factorial.append({"effect":"R_main_effect_equal_J_average","metric":metric,"delta_candidate_minus_baseline":float(np.mean(r_deltas)),"ci95_low":None,"ci95_high":None})
        for method in ("J1","J2"):
            factorial.append({"effect":f"JxR_interaction_{method}_vs_J0","metric":metric,"delta_candidate_minus_baseline":float(r_deltas[("J0","J1","J2").index(method)]-r_deltas[0]),"ci95_low":None,"ci95_high":None})
    factorial_frame=pd.DataFrame(factorial)
    effects={}
    for method in ("J0","J1","J2"):
        selected={metric:boot_frame[(boot_frame.baseline==f"{method}-R0")&(boot_frame.candidate==f"{method}-R1")&(boot_frame.metric==metric)].iloc[0].to_dict() for metric in ("V3D","PB","direction_improvement")}
        effects[method]=effect_class(selected["V3D"],selected["PB"],selected["direction_improvement"])
    supported_methods=[j for j in valid_j if effects[j]=="POSITIVE"]
    poor_supported=[j for j in ("J0","J1","J2") if classification[f"{j}-R0"]["SIGMA_PREDICTIVE_QUALITY"]=="BAD" and effects[j]=="POSITIVE"]
    if supported_methods: necessity="SUPPORTED"
    elif poor_supported: necessity="COMPENSATORY_ONLY"
    elif all(effects[j] in {"NEUTRAL","NEGATIVE"} for j in valid_j): necessity="NOT_NEEDED"
    else: necessity="MIXED"
    best_j=best_r0.split("-")[0] if best_r0 else "NONE"
    eligible=[arm for arm in ARMS if classification[arm]["MU_QUALITY"]=="PASS" and classification[arm]["SIGMA_PREDICTIVE_QUALITY"]!="BAD"]
    best_arm=max(eligible,key=lambda arm:results[arm]["summary"]["proposal_V3D"]) if eligible else "NONE"
    def j_effect(method):
        if method=="J0": return "REFERENCE"
        base=results["J0-R0"]["summary"]; cand=results[f"{method}-R0"]["summary"]
        if classification[f"{method}-R0"]["MU_QUALITY"]=="DEGRADED": return "NEGATIVE"
        delta=cand["proposal_V3D"]-base["proposal_V3D"]
        return "POSITIVE" if delta>0 else ("NEGATIVE" if delta<0 else "NEUTRAL")
    standard_confirmed="YES" if classification["J0-R0"]["MU_QUALITY"]=="DEGRADED" and any(classification[f"{j}-R0"]["MU_QUALITY"]=="PASS" for j in ("J1","J2")) else "UNCLEAR"
    predictive_sufficient="NO" if necessity=="SUPPORTED" else ("YES" if necessity=="NOT_NEEDED" else "UNCLEAR")
    decision={"BEST_MUSIGMA_TRAINING":best_j,"STANDARD_NLL_COUPLING_CONFIRMED":standard_confirmed,"BETA_NLL_EFFECT":j_effect("J1"),"FAITHFUL_EFFECT":j_effect("J2"),"RELIABILITY_EFFECT_J0":effects["J0"],"RELIABILITY_EFFECT_J1":effects["J1"],"RELIABILITY_EFFECT_J2":effects["J2"],"RELIABILITY_NECESSITY":necessity,"PREDICTIVE_UNCERTAINTY_SUFFICIENT_AS_ACTION_WEIGHT":predictive_sufficient,"BEST_SIX_ARM_MODEL":best_arm,"NEXT_RECOMMENDED_MODEL_STRUCTURE":("DIRECT_"+best_arm if best_arm!="NONE" else "NONE"),"classifications":classification,**protected_fields()}
    atomic_text(REPORT/"11_FACTORIAL_ANALYSIS.md","# Factorial analysis\n\n"+markdown_frame(factorial_frame)+"\n\nThe R main effect is the equal-J average of the three within-J paired effects. J×R entries are differences of within-J R effects; confidence intervals are provided for the primary within-J paired effects in `10_PAIRED_BOOTSTRAP.csv`.")
    atomic_text(REPORT/"12_CROSS_UPSTREAM.md","# Optional zero-shot cross-upstream diagnostic\n\nStatus: **SKIPPED**. Existing AvgFlow/DiTMC assets are tied to separate evaluator/worktree adapters; executing the optional diagnostic here would require modifying or adapting evaluation assets. The frozen protocol explicitly requires skipping in that circumstance. No AvgFlow or DiTMC coordinates were read or generated.")
    block="\n".join(f"{key} = {value}" for key,value in decision.items() if key!="classifications")
    atomic_text(REPORT/"13_FINAL_DECISION.md",f"# SIXS mu-sigma x Reliability final decision\n\n{markdown_frame(pd.DataFrame(summaries))}\n\n## Scientific classifications\n\n```text\n{block}\n```\n\nThe choice is restricted to models passing the preregistered mean-quality and sigma-pathology gates; it is not a raw largest-V3D selection over invalid arms. No new architecture is started.")
    atomic_json(REPORT/"FINAL_DECISION.json",decision)
    status_fields={f"{arm.replace('-','_')}_COMPLETED":"YES" for arm in ARMS}
    for arm in ARMS: status_fields[f"{arm.replace('-','_')}_PROPOSAL_V3D"]=results[arm]["summary"]["proposal_V3D"];status_fields[f"{arm.replace('-','_')}_PB"]=results[arm]["summary"]["proposal_PB"]
    for method in ("J0","J1","J2"):
        status_fields[f"{method}_MU_BOND_MAE"]=results[f"{method}-R0"]["summary"]["mu_bond_mae"];status_fields[f"{method}_MU_ANGLE_MAE"]=results[f"{method}-R0"]["summary"]["mu_angle_mae"];status_fields[f"{method}_SIGMA_QUALITY"]=classification[f"{method}-R0"]["SIGMA_PREDICTIVE_QUALITY"]
        row=boot_frame[(boot_frame.baseline==f"{method}-R0")&(boot_frame.candidate==f"{method}-R1")&(boot_frame.metric=="V3D")].iloc[0];status_fields[f"DELTA_R_{method}_V3D"]=row.delta_candidate_minus_baseline;status_fields[f"DELTA_R_{method}_V3D_95CI"]=[row.ci95_low,row.ci95_high]
    update_status("SIX_ARM_FINAL_DECISION", "COMPLETE", CURRENT_ARM=None, CURRENT_STEP=17500, TOTAL_STEPS=17500, **status_fields, **{k:v for k,v in decision.items() if k!="classifications"}, STOP_AFTER_SIX_ARM_FINAL_DECISION="YES")


def pipeline() -> None:
    freeze_protocol()
    if not (REPORT/"01_IMPLEMENTATION_AUDIT.json").is_file(): engineering_preflight()
    update_live()
    for arm in ARMS:
        if validate_done(arm): continue
        train_arm(arm); evaluate_arm(arm); freeze_done(arm); update_live()
        update_status("ARM_COMPLETE", "PASS", CURRENT_ARM=arm, CURRENT_STEP=17500, TOTAL_STEPS=17500, COMPLETED_ARMS=[a for a in ARMS if validate_done(a)])
    finalize_analysis()


def queue_status(path: Path, status: str, **extra: Any) -> None:
    atomic_json(path, {
        "schema_version": "sixs-factorial-async-queue-v1", "status": status,
        "pid": os.getpid(), "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "frozen_arm_order": list(ARMS), **protected_fields(), **extra,
    })


def train_coordinates_queue() -> None:
    queue_path = REPORT / "GPU_TRAIN_QUEUE_STATUS.json"
    try:
        queue_status(queue_path, "RUNNING", current_arm=None)
        for arm in ARMS:
            if validate_done(arm) or validate_coordinates_ready(arm):
                continue
            queue_status(queue_path, "TRAINING", current_arm=arm)
            train_arm(arm)
            queue_status(queue_path, "COORDINATE_GENERATION", current_arm=arm)
            evaluate_arm(arm, coordinates_only=True)
            if not validate_coordinates_ready(arm):
                raise RuntimeError(f"GPU queue failed to freeze coordinates: {arm}")
            queue_status(queue_path, "ARM_COORDINATES_READY", current_arm=arm,
                         ready_arms=[value for value in ARMS if validate_coordinates_ready(value) or validate_done(value)])
        queue_status(queue_path, "COMPLETE", current_arm=None,
                     ready_arms=[value for value in ARMS if validate_coordinates_ready(value) or validate_done(value)])
    except Exception as exc:
        queue_status(queue_path, "FAIL", error_type=type(exc).__name__, error=str(exc))
        raise


def evaluation_queue() -> None:
    queue_path = REPORT / "CPU_EVALUATION_QUEUE_STATUS.json"
    gpu_path = REPORT / "GPU_TRAIN_QUEUE_STATUS.json"
    try:
        queue_status(queue_path, "RUNNING", current_arm=None)
        while not all(validate_done(arm) for arm in ARMS):
            progressed = False
            for arm in ARMS:
                if validate_done(arm):
                    continue
                if validate_coordinates_ready(arm):
                    queue_status(queue_path, "EVALUATING", current_arm=arm)
                    evaluate_arm(arm)
                    freeze_done(arm)
                    update_live()
                    queue_status(queue_path, "ARM_COMPLETE", current_arm=arm,
                                 completed_arms=[value for value in ARMS if validate_done(value)])
                    progressed = True
                    break
            if progressed:
                continue
            if gpu_path.is_file():
                gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
                if gpu.get("status") == "FAIL":
                    raise RuntimeError(f"GPU train queue failed: {gpu.get('error')}")
                if gpu.get("status") == "COMPLETE":
                    missing = [arm for arm in ARMS if not validate_done(arm) and not validate_coordinates_ready(arm)]
                    if missing:
                        raise RuntimeError(f"GPU queue completed without coordinate manifests: {missing}")
            time.sleep(2)
        queue_status(queue_path, "COMPLETE", current_arm=None, completed_arms=list(ARMS))
    except Exception as exc:
        queue_status(queue_path, "FAIL", error_type=type(exc).__name__, error=str(exc))
        raise


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("protocol","preflight","pipeline","coordinates","evaluate","train-queue","evaluate-queue","status","finalize"),default="pipeline",nargs="?"); parser.add_argument("--arm",choices=ARMS); args=parser.parse_args()
    if args.phase=="status": print(STATUS.read_text(encoding="utf-8") if STATUS.is_file() else json.dumps({"STATUS":"NOT_STARTED"},indent=2));return 0
    try:
        if args.phase=="protocol": freeze_protocol()
        elif args.phase=="preflight": freeze_protocol();engineering_preflight()
        elif args.phase=="evaluate":
            if args.arm is None: raise RuntimeError("evaluate phase requires --arm")
            if not (REPORT/"00_PROTOCOL_FREEZE.md").is_file(): raise RuntimeError("evaluate phase requires an existing frozen protocol")
            evaluate_arm(args.arm);freeze_done(args.arm);update_live()
            update_status("ARM_COMPLETE","PASS",CURRENT_ARM=args.arm,CURRENT_STEP=17500,TOTAL_STEPS=17500,COMPLETED_ARMS=[a for a in ARMS if validate_done(a)])
        elif args.phase=="coordinates":
            if args.arm is None: raise RuntimeError("coordinates phase requires --arm")
            if not (REPORT/"00_PROTOCOL_FREEZE.md").is_file(): raise RuntimeError("coordinates phase requires an existing frozen protocol")
            train_arm(args.arm);evaluate_arm(args.arm,coordinates_only=True)
        elif args.phase=="train-queue": train_coordinates_queue()
        elif args.phase=="evaluate-queue": evaluation_queue()
        elif args.phase=="finalize": finalize_analysis()
        else: pipeline()
    except Exception as exc:
        current=json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}; update_status("FAILED","FAIL",CURRENT_ARM=current.get("CURRENT_ARM"),CURRENT_STEP=current.get("CURRENT_STEP",0),TOTAL_STEPS=17500,ERROR_TYPE=type(exc).__name__,ERROR=str(exc));raise
    return 0


if __name__=="__main__": raise SystemExit(main())
