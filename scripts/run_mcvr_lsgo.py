#!/usr/bin/env python3
"""Orchestrate the internal, pre-external stages of the LSGO pilot."""

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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

from etflow.ecir.lsgo_io import atomic_json, file_sha256, nearest_reference_metrics
from etflow.ecir.learned_geometry import (
    GraphGeometry,
    LearnedGeometryObjective,
    SIGMA_FLOORS,
    SIGMA_MAX,
    direct_gradient_update,
    distribution_parameters,
    gaussian_nll,
    geometry_values,
    parameter_count,
    precision_projection_update,
    prepare_graph,
    safety_accept,
    structured_objective,
)


CONFIG_PATH = ROOT / "configs/ecir_mvr_lsgo_pilot.yaml"
OUT = ROOT / "reports/ecir_mvr/learned_geometry"
ISOLATION = {
    "formal_test_records_read": 0,
    "frozen_holdout_records_read": 0,
    "full10k_used_for_tuning": False,
    "mvt_coordinates_used": False,
    "source_reference_cartesian_delta_used": False,
    "xtb_training_access": False,
    "posebusters_training_access": False,
    "weighted_bac_selection_access": False,
    "amortized_refiner_training": False,
}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def seed_all(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed + 47000)


def verify_calibration(payload: Mapping[str, Any]) -> None:
    identity = payload.get("identity_sha256")
    bare = {key: value for key, value in payload.items() if key != "identity_sha256"}
    if identity != canonical_sha256(bare):
        raise RuntimeError("frozen DRCSR calibration identity mismatch")
    if int(payload.get("formal_test_records_read", -1)) != 0 or int(payload.get("frozen_holdout_records_read", -1)) != 0:
        raise RuntimeError("protected split access in frozen DRCSR calibration")


def load_inputs(config: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    identity_path = OUT / "DATASET_IDENTITY.json"
    if not identity_path.is_file():
        raise RuntimeError("run build_lsgo_datasets.py before LSGO")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if any(int(identity[key]) != 0 for key in ("formal_test_records_read", "frozen_holdout_records_read")):
        raise RuntimeError("protected split access in LSGO identity")
    train_path = Path(config["dataset"]["training_compact"])
    if file_sha256(train_path) != config["dataset"]["training_compact_sha256"]:
        raise RuntimeError("training compact SHA changed")
    train = torch.load(train_path, map_location="cpu", weights_only=False)
    calibration_path = Path(config["dataset"]["drcsr_calibration"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    verify_calibration(calibration)
    return train, calibration, identity


def deterministic_items(items: Sequence[Mapping[str, Any]], partition: str, count: int | None = None):
    chosen = [item for item in items if str(item["partition"]) == partition]
    chosen.sort(key=lambda item: hashlib.sha256(str(item["molecule_id"]).encode()).hexdigest())
    return chosen if count is None else chosen[: int(count)]


def prepared_cache_path() -> Path:
    return OUT / "manifests/LSGO_PREPARED_GRAPHS.pt"


def prepare_cache(config: Mapping[str, Any]) -> dict[str, Any]:
    train, calibration, identity = load_inputs(config)
    graphs = []
    for index, item in enumerate(train["items"]):
        graphs.append({"molecule_id": item["molecule_id"], "partition": item["partition"], "graph": prepare_graph(item["record"], calibration)})
        if (index + 1) % 250 == 0:
            print(f"LSGO PREPARE {index + 1}/{len(train['items'])}", flush=True)
    payload = {
        "schema_version": "mcvr-lsgo-prepared-graphs-v1",
        "graphs": graphs,
        "dataset_identity_sha256": identity["identity_sha256"],
        "training_compact_sha256": config["dataset"]["training_compact_sha256"],
        "drcsr_calibration_sha256": file_sha256(Path(config["dataset"]["drcsr_calibration"])),
        **ISOLATION,
    }
    atomic_torch(prepared_cache_path(), payload)
    manifest = {
        "schema_version": payload["schema_version"],
        "path": str(prepared_cache_path()), "sha256": file_sha256(prepared_cache_path()),
        "graph_count": len(graphs), "dataset_identity_sha256": identity["identity_sha256"],
        "drcsr_calibration_sha256": payload["drcsr_calibration_sha256"], **ISOLATION,
    }
    atomic_json(OUT / "manifests/LSGO_PREPARED_GRAPHS.json", manifest)
    return manifest


def load_graphs(config: Mapping[str, Any]) -> tuple[list[dict], list[dict], dict]:
    train, calibration, _ = load_inputs(config)
    manifest = json.loads((OUT / "manifests/LSGO_PREPARED_GRAPHS.json").read_text(encoding="utf-8"))
    if file_sha256(prepared_cache_path()) != manifest["sha256"]:
        raise RuntimeError("prepared graph cache SHA changed")
    prepared = torch.load(prepared_cache_path(), map_location="cpu", weights_only=False)
    if len(prepared["graphs"]) != len(train["items"]):
        raise RuntimeError("prepared graph denominator changed")
    for item, row in zip(train["items"], prepared["graphs"], strict=True):
        if str(item["molecule_id"]) != str(row["molecule_id"]):
            raise RuntimeError("prepared graph ordering changed")
    return train["items"], prepared["graphs"], calibration


def collate_graphs(graphs: Sequence[GraphGeometry]) -> GraphGeometry:
    atoms, edges, edge_codes, bonds, bond_codes, angles, angle_codes = [], [], [], [], [], [], []
    bond_fixed, angle_fixed, ring_fixed = [], [], []
    offset = 0
    for graph in graphs:
        atoms.append(graph.atom_categorical)
        edges.append(graph.edge_index + offset)
        edge_codes.append(graph.edge_categorical)
        bonds.append(graph.bonds + offset)
        bond_codes.append(graph.bond_categorical)
        angles.append(graph.angles + offset)
        angle_codes.append(graph.angle_edge_categorical)
        if graph.bond_fixed is not None:
            bond_fixed.append(graph.bond_fixed)
        if graph.angle_fixed is not None:
            angle_fixed.append(graph.angle_fixed)
        if graph.ring_fixed is not None and graph.ring_fixed.numel():
            local = graph.ring_fixed.clone()
            local[:, 0] += len(ring_fixed)
            ring_fixed.append(local)
        offset += graph.atom_categorical.size(0)
    return GraphGeometry(
        atom_categorical=torch.cat(atoms), edge_index=torch.cat(edges, dim=1),
        edge_categorical=torch.cat(edge_codes), bonds=torch.cat(bonds, dim=1),
        bond_categorical=torch.cat(bond_codes), angles=torch.cat(angles),
        angle_edge_categorical=torch.cat(angle_codes), rings=(), chirality=(),
        bond_fixed=torch.cat(bond_fixed) if bond_fixed else None,
        angle_fixed=torch.cat(angle_fixed) if angle_fixed else None,
        ring_fixed=torch.cat(ring_fixed) if ring_fixed else None,
        backoff=None,
    )


def model_from_config(config: Mapping[str, Any], variant: str) -> LearnedGeometryObjective:
    if variant not in {"B", "C"}:
        raise ValueError("only B/C are neural")
    model = LearnedGeometryObjective(
        hidden_dim=int(config["model"]["hidden_dim"]),
        layers=int(config["model"]["layers"]), learned_sigma=variant == "C",
    )
    count = parameter_count(model)
    if count >= int(config["model"]["parameter_limit"]):
        raise RuntimeError(f"LSGO model parameter limit exceeded: {count}")
    return model


def checkpoint_path(variant: str, seed: int, step: int | str) -> Path:
    name = f"step{int(step):04d}.ckpt" if isinstance(step, int) else f"{step}.ckpt"
    return OUT / "checkpoints" / f"variant_{variant.lower()}_seed{seed}" / name


def training_batch(
    items: Sequence[dict], graph_rows: Sequence[dict], indices: Sequence[int],
    generator: torch.Generator, device: torch.device,
) -> tuple[GraphGeometry, Tensor]:
    graphs, coordinates = [], []
    for index in indices:
        item = items[index]
        graph = graph_rows[index]["graph"]
        reference_index = int(torch.randint(len(item["references"]), (1,), generator=generator))
        graphs.append(graph)
        coordinates.append(torch.as_tensor(item["references"][reference_index], dtype=torch.float32))
    return collate_graphs(graphs).to(device), torch.cat(coordinates).to(device)


@torch.no_grad()
def evaluate_likelihood(
    model: LearnedGeometryObjective | None,
    item_rows: Sequence[tuple[dict, GraphGeometry]], variant: str, device: torch.device,
) -> dict[str, Any]:
    bond_nll, angle_nll, bond_abs, angle_abs, bond_z, angle_z, bond_sigma, angle_sigma = ([] for _ in range(8))
    per_molecule = []
    for item, graph_cpu in item_rows:
        graph = graph_cpu.to(device)
        parameters = distribution_parameters(graph, model=model, variant=variant)
        local_bond, local_angle = [], []
        for coordinates_cpu in torch.as_tensor(item["references"], dtype=torch.float32):
            coordinates = coordinates_cpu.to(device)
            bonds, angles = geometry_values(coordinates, graph)
            bn = gaussian_nll(bonds, parameters["bond_mu"], parameters["bond_sigma"])
            an = gaussian_nll(angles, parameters["angle_mu"], parameters["angle_sigma"])
            bond_nll.extend(bn.cpu().tolist()); angle_nll.extend(an.cpu().tolist())
            bond_abs.extend((bonds - parameters["bond_mu"]).abs().cpu().tolist())
            angle_abs.extend((angles - parameters["angle_mu"]).abs().cpu().tolist())
            bond_z.extend(((bonds - parameters["bond_mu"]) / parameters["bond_sigma"]).cpu().tolist())
            angle_z.extend(((angles - parameters["angle_mu"]) / parameters["angle_sigma"]).cpu().tolist())
            local_bond.append(float(bn.mean())); local_angle.append(float(an.mean()))
        bond_sigma.extend(parameters["bond_sigma"].cpu().tolist())
        angle_sigma.extend(parameters["angle_sigma"].cpu().tolist())
        per_molecule.append({
            "molecule_id": item["molecule_id"], "variant": variant,
            "bond_nll": float(np.mean(local_bond)), "angle_nll": float(np.mean(local_angle)),
            "joint_nll": float((np.mean(local_bond) + np.mean(local_angle)) / 2),
        })
    def stats(values):
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()), "std": float(array.std()),
            "median_abs": float(np.median(np.abs(array))),
            "p90_abs": float(np.quantile(np.abs(array), .90)),
            "p95_abs": float(np.quantile(np.abs(array), .95)),
            "p99_abs": float(np.quantile(np.abs(array), .99)),
        }
    def sigma_stats(values, floor, maximum):
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()), "median": float(np.median(array)),
            "p05": float(np.quantile(array, .05)), "p25": float(np.quantile(array, .25)),
            "p75": float(np.quantile(array, .75)), "p95": float(np.quantile(array, .95)),
            "relative_iqr": float((np.quantile(array, .75) - np.quantile(array, .25)) / max(np.median(array), 1e-12)),
            "collapse_fraction": float(np.mean(array <= floor * 1.01)),
            "inflation_fraction": float(np.mean(array >= maximum * .99)),
        }
    result = {
        "variant": variant, "molecules": len(item_rows),
        "reference_conformers": int(sum(len(item["references"]) for item, _ in item_rows)),
        "bond_primitives": len(bond_nll), "angle_primitives": len(angle_nll),
        "bond_nll": float(np.mean(bond_nll)), "angle_nll": float(np.mean(angle_nll)),
        "joint_nll": float((np.mean(bond_nll) + np.mean(angle_nll)) / 2),
        "bond_mu_mae_angstrom": float(np.mean(bond_abs)),
        "angle_mu_mae_cosine": float(np.mean(angle_abs)),
        "bond_z": stats(bond_z), "angle_z": stats(angle_z),
        "bond_sigma": sigma_stats(bond_sigma, SIGMA_FLOORS["bond"], SIGMA_MAX["bond"]),
        "angle_sigma": sigma_stats(angle_sigma, SIGMA_FLOORS["angle"], SIGMA_MAX["angle"]),
        "calibration_error": float(abs(np.mean(bond_z)) + abs(np.std(bond_z) - 1) + abs(np.mean(angle_z)) + abs(np.std(angle_z) - 1)),
        "per_molecule": per_molecule, **ISOLATION,
    }
    return result


def train_one(
    config: Mapping[str, Any], variant: str, seed: int, device: torch.device,
    *, steps_override: int | None = None,
) -> dict[str, Any]:
    items, graph_rows, _ = load_graphs(config)
    train_indices = [i for i, item in enumerate(items) if item["partition"] == "train"]
    dev_a = [(items[i], graph_rows[i]["graph"]) for i in range(len(items)) if items[i]["partition"] == "dev_a"]
    generator = seed_all(seed)
    model = model_from_config(config, variant).to(device)
    training = config["training"]
    steps = int(steps_override or training["steps"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    logs, evaluations = [], []
    started = time.time()
    checkpoints = set(int(value) for value in training["checkpoint_steps"] if int(value) <= steps)
    checkpoints.add(steps)
    for step in range(1, steps + 1):
        selected = torch.randint(
            len(train_indices), (int(training["batch_molecules"]),), generator=generator
        ).tolist()
        indices = [train_indices[index] for index in selected]
        graph, coordinates = training_batch(items, graph_rows, indices, generator, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = distribution_parameters(graph, model=model, variant=variant)
        objective, groups = structured_objective(coordinates, graph, prediction)
        upper_penalty = coordinates.new_zeros(())
        if variant == "C":
            upper_penalty = (
                torch.relu(prediction["bond_sigma_unclamped"] - SIGMA_MAX["bond"]).square().mean()
                + torch.relu(prediction["angle_sigma_unclamped"] - SIGMA_MAX["angle"]).square().mean()
            )
            objective = objective + float(training["sigma_upper_regularization"]) * upper_penalty
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(f"nonfinite LSGO loss at {variant}/{seed}/{step}")
        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step(); scheduler.step()
        if step == 1 or step % 25 == 0:
            logs.append({
                "variant": variant, "seed": seed, "step": step,
                "loss": float(objective.detach()), "bond_nll": float(groups["bond"].detach()),
                "angle_nll": float(groups["angle"].detach()), "sigma_upper_penalty": float(upper_penalty.detach()),
                "gradient_norm": float(gradient_norm), "learning_rate": scheduler.get_last_lr()[0],
            })
        if step in checkpoints:
            model.eval()
            metrics = evaluate_likelihood(model, dev_a, variant, device)
            metrics.update({"seed": seed, "step": step})
            evaluations.append({key: value for key, value in metrics.items() if key != "per_molecule"})
            payload = {
                "schema_version": "mcvr-lsgo-checkpoint-v1", "variant": variant,
                "seed": seed, "step": step, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "sampler_generator_state": generator.get_state(), "validation": metrics,
                "parameter_count": parameter_count(model), "coordinate_teacher": False,
                "config_sha256": file_sha256(CONFIG_PATH), **ISOLATION,
            }
            path = checkpoint_path(variant, seed, step)
            atomic_torch(path, payload)
            model.train()
            atomic_csv(OUT / "logs" / f"TRAIN_{variant}_seed{seed}.csv", pd.DataFrame(logs))
            atomic_csv(OUT / "tables" / f"DEV_A_{variant}_seed{seed}.csv", pd.DataFrame(evaluations))
    best = min(evaluations, key=lambda row: (row["joint_nll"], row["calibration_error"], row["step"]))
    source = checkpoint_path(variant, seed, int(best["step"]))
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    atomic_torch(checkpoint_path(variant, seed, "best"), checkpoint)
    result = {
        "status": "COMPLETED", "variant": variant, "seed": seed, "steps": steps,
        "parameter_count": parameter_count(model), "best_step": int(best["step"]),
        "best_checkpoint": str(checkpoint_path(variant, seed, "best")),
        "best_checkpoint_sha256": file_sha256(checkpoint_path(variant, seed, "best")),
        "best_dev_a": best, "runtime_seconds": time.time() - started, **ISOLATION,
    }
    atomic_json(OUT / "logs" / f"TRAIN_{variant}_seed{seed}.json", result)
    return result


def load_selected(config: Mapping[str, Any], variant: str, seed: int, device: torch.device):
    path = checkpoint_path(variant, seed, "best")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["variant"] != variant or int(checkpoint["seed"]) != int(seed) or checkpoint["coordinate_teacher"] is not False:
        raise RuntimeError("LSGO checkpoint identity mismatch")
    model = model_from_config(config, variant).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint, path


def preregister(config: Mapping[str, Any]) -> dict[str, Any]:
    identity = json.loads((OUT / "DATASET_IDENTITY.json").read_text(encoding="utf-8"))
    prepared = json.loads((OUT / "manifests/LSGO_PREPARED_GRAPHS.json").read_text(encoding="utf-8"))
    if git("status", "--short"):
        # Pre-registration must bind the exact implementation. The caller is
        # expected to commit implementation/tests first.
        raise RuntimeError("working tree must be clean before preregistration")
    payload = {
        "schema_version": "mcvr-lsgo-preregistration-v1", "status": "FROZEN",
        "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
        "config_path": str(CONFIG_PATH), "config_sha256": file_sha256(CONFIG_PATH),
        "dataset_identity_path": str(OUT / "DATASET_IDENTITY.json"),
        "dataset_identity_sha256": identity["identity_sha256"],
        "prepared_graph_sha256": prepared["sha256"],
        "drcsr_calibration_path": config["dataset"]["drcsr_calibration"],
        "drcsr_calibration_sha256": file_sha256(Path(config["dataset"]["drcsr_calibration"])),
        "variants": {
            "A": "frozen DRCSR typed median and scale",
            "B": "neural conditional mean plus frozen DRCSR scale",
            "C": "neural conditional mean and learned heteroscedastic scale",
        },
        "seeds": list(config["seeds"]), "model": config["model"],
        "training": config["training"], "internal_selection": config["internal_selection"],
        "external_evaluation": config["external_evaluation"], "guards": config["guards"],
        "primary_comparison": ["A-G", "B-G", "C-G", "C-P"],
        "external_evaluation_locked_until_internal_gate": True,
        "no_post_external_selection": True, "ring_neuralized": False,
        "remaining_handcrafted_items": [
            "Bond/Angle primitive choice", "Gaussian likelihood", "GNN architecture",
            "sigma floor/max", "ridge/rank tolerance", "trust budget", "one/two steps",
            "frozen DRCSR ring guard", "equal Bond/Angle group aggregation",
        ],
        **ISOLATION,
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    atomic_json(OUT / "LSGO_PREREGISTRATION.json", payload)
    atomic_text(OUT / "LSGO_PREREGISTRATION.md", f"""# LSGO preregistration

Status: **FROZEN before training**.

- Branch/HEAD: `{payload['branch']}` / `{payload['head']}`
- Seeds: `{payload['seeds']}`
- A/B/C and all sigma, optimizer, checkpoint, stationarity, separation, Jacobian, trust and external gates are frozen in `LSGO_PREREGISTRATION.json`.
- Training uses TRAIN Reference local Bond/Angle values only. There is no Cartesian label, MVT coordinate, Source-to-Reference delta, PB/xTB/BAC selection, or amortized refiner.
- External PB/xTB remain locked until checkpoint and coordinate freeze.

Formal test reads=0; frozen holdout reads=0; FULL10K used for tuning=false.
""")
    return payload


def smoke(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    rows = []
    for variant in ("B", "C"):
        result = train_one(config, variant, int(config["seeds"][0]), device, steps_override=20)
        rows.append(result)
    payload = {"status": "PASSED", "runs": rows, **ISOLATION}
    atomic_json(OUT / "manifests/LSGO_SMOKE.json", payload)
    return payload


def train_all(config: Mapping[str, Any], device: torch.device) -> list[dict[str, Any]]:
    rows = []
    for variant in ("B", "C"):
        for seed in config["seeds"]:
            print(f"LSGO TRAIN variant={variant} seed={seed}", flush=True)
            rows.append(train_one(config, variant, int(seed), device))
    atomic_json(OUT / "manifests/LSGO_TRAINING_COMPLETE.json", {
        "status": "COMPLETED", "runs": rows, "run_count": len(rows), **ISOLATION,
    })
    return rows


def sigma_pathology(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    gate = config["internal_selection"]["sigma_pathology"]
    values = (metrics["bond_sigma"], metrics["angle_sigma"])
    if any(row["collapse_fraction"] > float(gate["collapse_fraction_max"]) for row in values):
        return "SIGMA_COLLAPSE"
    if any(row["inflation_fraction"] > float(gate["inflation_fraction_max"]) for row in values):
        return "SIGMA_INFLATION"
    if any(row["relative_iqr"] < float(gate["constant_relative_iqr_min"]) for row in values):
        return "LEARNED_SIGMA_CONSTANT"
    return "UNCERTAINTY_MEANINGFUL"


def reference_source_audit(
    config: Mapping[str, Any], variant: str, seed: int, model: LearnedGeometryObjective,
    items: Sequence[dict], graph_rows: Sequence[dict], partition: str, device: torch.device,
    count: int = 48,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lookup = {str(row["molecule_id"]): row["graph"] for row in graph_rows}
    selected = deterministic_items(items, partition, count)
    rows = []
    for item in selected:
        graph = lookup[str(item["molecule_id"])].to(device)
        parameters = distribution_parameters(graph, model=model, variant=variant)
        reference = torch.as_tensor(item["references"][0], dtype=torch.float64, device=device)
        source = torch.as_tensor(item["sources"][0], dtype=torch.float64, device=device)
        values = {}
        for role, coordinates in (("reference", reference), ("source", source)):
            x = coordinates.detach().clone().requires_grad_(True)
            objective, groups = structured_objective(x, graph, parameters)
            gradient, = torch.autograd.grad(objective, x)
            gradient = gradient - gradient.mean(0, keepdim=True)
            gradient_rms = torch.sqrt(gradient.square().sum(-1).mean())
            update = direct_gradient_update(
                coordinates, graph, parameters, rms_budget=0.001,
                atom_cap=float(config["internal_selection"]["atom_cap_angstrom"]), steps=1,
            )
            moved = update["coordinates"]
            moved_objective = structured_objective(moved, graph, parameters)[0]
            values[role] = {
                "objective": float(objective), "bond_objective": float(groups["bond"]),
                "angle_objective": float(groups["angle"]), "gradient_rms": float(gradient_rms),
                "projected_movement": float(torch.sqrt((moved - coordinates).square().sum(-1).mean())),
                "projected_objective": float(moved_objective),
            }
        rows.append({
            "variant": variant, "seed": seed, "partition": partition,
            "molecule_id": item["molecule_id"],
            **{f"reference_{key}": value for key, value in values["reference"].items()},
            **{f"source_{key}": value for key, value in values["source"].items()},
            "source_minus_reference_objective": values["source"]["objective"] - values["reference"]["objective"],
        })
    frame = pd.DataFrame(rows)
    summary = {
        "variant": variant, "seed": seed, "partition": partition, "molecules": len(frame),
        "reference_gradient_rms_median": float(frame.reference_gradient_rms.median()),
        "source_gradient_rms_median": float(frame.source_gradient_rms.median()),
        "gradient_rms_ratio": float(frame.reference_gradient_rms.median() / max(frame.source_gradient_rms.median(), 1e-12)),
        "reference_projected_movement_median": float(frame.reference_projected_movement.median()),
        "reference_objective_nonincrease_fraction": float((frame.reference_projected_objective <= frame.reference_objective + 1e-10).mean()),
        "source_gt_reference_fraction": float((frame.source_minus_reference_objective > 0).mean()),
        "source_minus_reference_objective_median": float(frame.source_minus_reference_objective.median()),
        **ISOLATION,
    }
    return frame, summary


def direction_diagnostics(
    config: Mapping[str, Any], models: Mapping[tuple[str, int], LearnedGeometryObjective],
    items: Sequence[dict], graph_rows: Sequence[dict], partition: str, device: torch.device,
    count: int = 24, budgets: Sequence[float] | None = None,
) -> pd.DataFrame:
    lookup = {str(row["molecule_id"]): row["graph"] for row in graph_rows}
    selected = deterministic_items(items, partition, count)
    rows = []
    for seed in config["seeds"]:
        for item in selected:
            graph = lookup[str(item["molecule_id"])].to(device)
            for source_index, source_cpu in enumerate(torch.as_tensor(item["sources"])):
                source = source_cpu.to(device=device, dtype=torch.float64)
                source_mode, source_rmsd = nearest_reference_metrics(source.cpu().float(), item["references"])
                for budget in (budgets or config["internal_selection"]["budget_candidates_angstrom"]):
                    for method, variant in (("A-G", "A"), ("B-G", "B"), ("C-G", "C"), ("C-P", "C")):
                        model = None if variant == "A" else models[(variant, int(seed))]
                        parameters = distribution_parameters(graph, model=model, variant=variant)
                        if method.endswith("-P"):
                            candidate = precision_projection_update(
                                source, graph, parameters,
                                ridge=float(config["internal_selection"]["solver"]["ridge"]),
                                rank_tolerance=float(config["internal_selection"]["solver"]["rank_tolerance"]),
                                maximum_condition=float(config["internal_selection"]["solver"]["maximum_condition"]),
                                rms_budget=float(budget),
                                atom_cap=float(config["internal_selection"]["atom_cap_angstrom"]), steps=1,
                            )
                        else:
                            candidate = direct_gradient_update(
                                source, graph, parameters, rms_budget=float(budget),
                                atom_cap=float(config["internal_selection"]["atom_cap_angstrom"]), steps=1,
                            )
                        accepted, safety = safety_accept(source, candidate["coordinates"], graph)
                        before = structured_objective(source, graph, parameters)[0]
                        after = structured_objective(accepted, graph, parameters)[0]
                        output_mode, output_rmsd = nearest_reference_metrics(accepted.cpu().float(), item["references"])
                        delta = accepted - source
                        trace = candidate["trace"][-1] if candidate["trace"] else {}
                        rows.append({
                            "method": method, "variant": variant, "seed": int(seed),
                            "partition": partition, "budget": float(budget),
                            "molecule_id": item["molecule_id"], "sample_id": item["sample_ids"][source_index],
                            "finite": candidate["finite"], "fallback": safety["fallback"],
                            "chirality_preserved": safety["chirality_preserved"],
                            "ring_nonregression": safety["ring_nonregression"],
                            "catastrophic_clash_nonregression": safety["catastrophic_clash_nonregression"],
                            "objective_before": float(before), "objective_after": float(after),
                            "objective_delta": float(after - before),
                            "graph_rms_movement": float(torch.sqrt(delta.square().sum(-1).mean())),
                            "max_atom_movement": float(torch.linalg.vector_norm(delta, dim=-1).max()),
                            "source_reference_rmsd": source_rmsd, "output_reference_rmsd": output_rmsd,
                            "nearest_reference_rmsd_delta": output_rmsd - source_rmsd,
                            "mode_switch": output_mode != source_mode,
                            "condition_number": trace.get("condition_number"),
                            "effective_rank": trace.get("effective_rank"),
                            "singular_min": trace.get("singular_min"),
                            "solver_backend": trace.get("solver_backend"),
                        })
    return pd.DataFrame(rows)


def summarize_direction(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["partition", "method", "seed", "budget"], as_index=False).agg(
        records=("sample_id", "size"), finite_fraction=("finite", "mean"),
        fallback_fraction=("fallback", "mean"), chirality_preserved=("chirality_preserved", "mean"),
        ring_nonregression=("ring_nonregression", "mean"),
        catastrophic_nonregression=("catastrophic_clash_nonregression", "mean"),
        objective_delta=("objective_delta", "mean"), objective_improved=("objective_delta", lambda x: float((x < 0).mean())),
        graph_rms_movement=("graph_rms_movement", "mean"), max_atom_movement=("max_atom_movement", "max"),
        reference_rmsd_delta=("nearest_reference_rmsd_delta", "mean"), mode_switch=("mode_switch", "mean"),
        condition_median=("condition_number", "median"), condition_p95=("condition_number", lambda x: float(x.dropna().quantile(.95)) if x.notna().any() else float("nan")),
        singular_min=("singular_min", "min"),
    )


def internal_finalize(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    items, graph_rows, _ = load_graphs(config)
    models, checkpoint_rows, likelihood_rows = {}, [], []
    baseline_rows = []
    for partition in ("dev_a", "dev_b"):
        subset = [(items[i], graph_rows[i]["graph"]) for i in range(len(items)) if items[i]["partition"] == partition]
        baseline = evaluate_likelihood(None, subset, "A", device)
        baseline_rows.append({"partition": partition, "variant": "A", "seed": None, **{k: v for k, v in baseline.items() if not isinstance(v, (dict, list))}})
    for variant in ("B", "C"):
        for seed in config["seeds"]:
            model, checkpoint, path = load_selected(config, variant, int(seed), device)
            models[(variant, int(seed))] = model
            checkpoint_rows.append({
                "variant": variant, "seed": int(seed), "step": int(checkpoint["step"]),
                "path": str(path), "sha256": file_sha256(path),
                "parameter_count": checkpoint["parameter_count"],
            })
            for partition in ("dev_a", "dev_b"):
                subset = [(items[i], graph_rows[i]["graph"]) for i in range(len(items)) if items[i]["partition"] == partition]
                metrics = evaluate_likelihood(model, subset, variant, device)
                path_metrics = OUT / "per_record" / f"LIKELIHOOD_{variant}_seed{seed}_{partition}.csv"
                atomic_csv(path_metrics, pd.DataFrame(metrics.pop("per_molecule")))
                likelihood_rows.append({"partition": partition, "variant": variant, "seed": int(seed), **{k: v for k, v in metrics.items() if not isinstance(v, dict)}, "bond_z": metrics["bond_z"], "angle_z": metrics["angle_z"], "bond_sigma": metrics["bond_sigma"], "angle_sigma": metrics["angle_sigma"]})

    baseline = {(row["partition"]): row["joint_nll"] for row in baseline_rows}
    comparisons = {}
    for variant in ("B", "C"):
        comparisons[variant] = {}
        for partition in ("dev_a", "dev_b"):
            values = [row["joint_nll"] for row in likelihood_rows if row["variant"] == variant and row["partition"] == partition]
            comparisons[variant][partition] = {
                "seed_values": values, "median": float(np.median(values)),
                "drcsr": baseline[partition], "improvement": float(baseline[partition] - np.median(values)),
                "better_than_drcsr": bool(np.median(values) < baseline[partition]),
            }
    eligible_variants = [
        variant for variant in ("B", "C")
        if all(comparisons[variant][partition]["better_than_drcsr"] for partition in ("dev_a", "dev_b"))
    ]
    c_pathologies = {}
    for seed in config["seeds"]:
        rows = [row for row in likelihood_rows if row["variant"] == "C" and row["seed"] == int(seed)]
        decisions = [sigma_pathology(row, config) for row in rows]
        c_pathologies[str(seed)] = "UNCERTAINTY_MEANINGFUL" if all(value == "UNCERTAINTY_MEANINGFUL" for value in decisions) else next(value for value in decisions if value != "UNCERTAINTY_MEANINGFUL")
    uncertainty_meaningful = all(value == "UNCERTAINTY_MEANINGFUL" for value in c_pathologies.values())

    audit_rows, audit_summaries = [], []
    for variant in ("B", "C"):
        if variant == "C" and not uncertainty_meaningful:
            continue
        for seed in config["seeds"]:
            for partition in ("dev_a", "dev_b"):
                frame, summary = reference_source_audit(
                    config, variant, int(seed), models[(variant, int(seed))],
                    items, graph_rows, partition, device,
                )
                audit_rows.append(frame); audit_summaries.append(summary)
    audit_frame = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()
    if not audit_frame.empty:
        atomic_csv(OUT / "per_record/REFERENCE_SOURCE_AUDIT.csv", audit_frame)
    stationarity_gate = config["internal_selection"]["reference_stationarity"]
    separation_gate = config["internal_selection"]["source_separation"]
    primary_variant = "C" if "C" in eligible_variants and uncertainty_meaningful else ("B" if "B" in eligible_variants else None)
    primary_summaries = [row for row in audit_summaries if row["variant"] == primary_variant] if primary_variant else []
    stationarity_pass = bool(primary_summaries) and all(
        row["gradient_rms_ratio"] <= float(stationarity_gate["median_gradient_rms_ratio_to_source_max"])
        and row["reference_projected_movement_median"] <= float(stationarity_gate["projected_micro_step_max_angstrom"]) + 1e-9
        and row["reference_objective_nonincrease_fraction"] >= float(stationarity_gate["objective_nonincrease_fraction_min"])
        for row in primary_summaries
    )
    separation_pass = bool(primary_summaries) and all(
        row["source_gt_reference_fraction"] >= float(separation_gate["source_gt_reference_fraction_min"])
        and row["source_minus_reference_objective_median"] > float(separation_gate["median_objective_difference_min"])
        for row in primary_summaries
    )

    hard_stop = None
    if not eligible_variants:
        hard_stop = "NEURAL_LIKELIHOOD_NOT_BETTER_THAN_DRCSR"
    elif primary_variant == "C" and not uncertainty_meaningful:
        hard_stop = "LEARNED_SIGMA_PATHOLOGY"
    elif not stationarity_pass:
        hard_stop = "REFERENCE_STATIONARITY_FAIL"
    elif not separation_pass:
        hard_stop = "OBJECTIVE_NOT_INFORMATIVE"

    direction_summary = pd.DataFrame()
    selected_budget = None
    direction_pass = False
    if hard_stop is None:
        dev_a_path = OUT / "per_record/DIRECTION_DEV_A.csv"
        expected_dev_a = 24 * 3 * len(config["seeds"]) * len(config["internal_selection"]["budget_candidates_angstrom"]) * 4
        if dev_a_path.is_file():
            dev_a = pd.read_csv(dev_a_path)
            if len(dev_a) != expected_dev_a or set(dev_a.method) != {"A-G", "B-G", "C-G", "C-P"}:
                raise RuntimeError("incomplete or corrupt cached DEV_A direction diagnostic")
        else:
            dev_a = direction_diagnostics(config, models, items, graph_rows, "dev_a", device)
            atomic_csv(dev_a_path, dev_a)
        direction_summary = summarize_direction(dev_a)
        atomic_csv(OUT / "tables/DIRECTION_DEV_A_SUMMARY.csv", direction_summary)
        budgets = sorted(config["internal_selection"]["budget_candidates_angstrom"], reverse=True)
        for budget in budgets:
            candidate = direction_summary[direction_summary.budget == float(budget)]
            if (
                len(candidate) == 4 * len(config["seeds"])
                and (candidate.finite_fraction >= 1.0).all()
                and (candidate.chirality_preserved >= 1.0).all()
                and (candidate.mode_switch <= .005).all()
                and (candidate.graph_rms_movement <= float(budget) + 1e-8).all()
                and (candidate.objective_delta < 0).all()
            ):
                selected_budget = float(budget); break
        if selected_budget is None:
            hard_stop = "DIRECT_GRADIENT_AND_PROJECTION_FAIL"
        else:
            dev_b = direction_diagnostics(
                config, models, items, graph_rows, "dev_b", device,
                budgets=[selected_budget],
            )
            atomic_csv(OUT / "per_record/DIRECTION_DEV_B_ONE_SHOT.csv", dev_b)
            dev_b_summary = summarize_direction(dev_b)
            atomic_csv(OUT / "tables/DIRECTION_DEV_B_ONE_SHOT_SUMMARY.csv", dev_b_summary)
            direction_summary = pd.concat((direction_summary, dev_b_summary), ignore_index=True)
            direction_pass = bool(
                (dev_b_summary.finite_fraction >= 1.0).all()
                and (dev_b_summary.chirality_preserved >= 1.0).all()
                and (dev_b_summary.mode_switch <= .005).all()
                and (dev_b_summary.objective_delta < 0).all()
            )
            if not direction_pass:
                hard_stop = "DIRECT_GRADIENT_AND_PROJECTION_FAIL"

    freeze = {
        "schema_version": "mcvr-lsgo-checkpoint-freeze-v1", "status": "FROZEN",
        "checkpoints": checkpoint_rows, "count": len(checkpoint_rows),
        "all_selected_without_external_metrics": True, **ISOLATION,
    }
    freeze["identity_sha256"] = canonical_sha256(freeze)
    atomic_json(OUT / "CHECKPOINT_FREEZE_MANIFEST.json", freeze)
    decision = {
        "schema_version": "mcvr-lsgo-internal-selection-v1",
        "status": "STOPPED_BY_PREREGISTERED_RULE" if hard_stop else "INTERNAL_GATE_PASSED",
        "hard_stop": hard_stop, "external_evaluation_authorized": hard_stop is None,
        "drcsr_baseline": baseline, "likelihood_comparisons": comparisons,
        "eligible_neural_variants": eligible_variants, "primary_variant": primary_variant,
        "uncertainty_by_seed": c_pathologies,
        "uncertainty_meaningful": uncertainty_meaningful,
        "reference_stationarity_pass": stationarity_pass,
        "source_reference_separation_pass": separation_pass,
        "reference_source_summaries": audit_summaries,
        "selected_primary_budget_angstrom": selected_budget,
        "primary_steps": 1, "secondary_steps": 2,
        "direction_pass": direction_pass, "checkpoint_freeze_sha256": file_sha256(OUT / "CHECKPOINT_FREEZE_MANIFEST.json"),
        **ISOLATION,
    }
    decision["identity_sha256"] = canonical_sha256(decision)
    atomic_json(OUT / "INTERNAL_SELECTION_REPORT.json", decision)
    write_internal_reports(config, decision, baseline_rows, likelihood_rows, direction_summary)
    return decision


def write_internal_reports(config, decision, baseline_rows, likelihood_rows, direction_summary):
    table = []
    for row in baseline_rows:
        table.append({key: row.get(key) for key in ("partition", "variant", "seed", "joint_nll", "bond_nll", "angle_nll", "bond_mu_mae_angstrom", "angle_mu_mae_cosine")})
    for row in likelihood_rows:
        table.append({key: row.get(key) for key in ("partition", "variant", "seed", "joint_nll", "bond_nll", "angle_nll", "bond_mu_mae_angstrom", "angle_mu_mae_cosine")})
    atomic_csv(OUT / "tables/REFERENCE_LIKELIHOOD_SUMMARY.csv", pd.DataFrame(table))
    atomic_text(OUT / "REFERENCE_CALIBRATION_REPORT.md", "# Reference calibration report\n\n```csv\n" + pd.DataFrame(table).to_csv(index=False, float_format="%.8g") + "```\n")
    atomic_text(OUT / "LEARNED_UNCERTAINTY_REPORT.md", "# Learned uncertainty report\n\nDecision by C seed: `" + json.dumps(decision["uncertainty_by_seed"], sort_keys=True) + "`.\n\nThe full Bond/Angle sigma and z-score distributions are embedded in `INTERNAL_SELECTION_REPORT.json`.\n")
    summaries = pd.DataFrame(decision["reference_source_summaries"])
    atomic_text(OUT / "SOURCE_ERROR_STATE_REPORT.md", "# Learned Source error-state report\n\n```csv\n" + summaries.to_csv(index=False, float_format="%.8g") + "```\n")
    atomic_text(OUT / "GRADIENT_DIAGNOSTIC_REPORT.md", "# Direct learned-gradient diagnostic\n\n" + ("Stopped before Source coordinate diagnostics by: **" + str(decision["hard_stop"]) + "**.\n" if direction_summary.empty else "```csv\n" + direction_summary[direction_summary.method.str.endswith("-G")].to_csv(index=False, float_format="%.8g") + "```\n"))
    atomic_text(OUT / "PROJECTION_DIAGNOSTIC_REPORT.md", "# Precision-weighted projection diagnostic\n\n" + ("Stopped before projection diagnostics by: **" + str(decision["hard_stop"]) + "**.\n" if direction_summary.empty else "```csv\n" + direction_summary[direction_summary.method == "C-P"].to_csv(index=False, float_format="%.8g") + "```\n"))
    if direction_summary.empty:
        jacobian = "No real-Source solve was authorized after the earlier hard stop."
    else:
        cp = direction_summary[direction_summary.method == "C-P"]
        jacobian = "```csv\n" + cp[["partition", "seed", "budget", "finite_fraction", "condition_median", "condition_p95", "singular_min"]].to_csv(index=False, float_format="%.8g") + "```"
    atomic_text(OUT / "JACOBIAN_CONDITIONING_REPORT.md", "# Jacobian conditioning report\n\n" + jacobian + "\n")
    atomic_text(OUT / "INTERNAL_SELECTION_REPORT.md", f"""# LSGO internal selection

Status: **{decision['status']}**  
Hard stop: `{decision['hard_stop']}`  
Eligible neural variants: `{decision['eligible_neural_variants']}`  
Primary variant: `{decision['primary_variant']}`  
Uncertainty meaningful: `{decision['uncertainty_meaningful']}`  
Reference stationarity: `{decision['reference_stationarity_pass']}`  
Source/Reference separation: `{decision['source_reference_separation_pass']}`  
Selected matched budget: `{decision['selected_primary_budget_angstrom']}` A  
External evaluation authorized: `{decision['external_evaluation_authorized']}`

No PB, xTB, weighted-BAC, MVT coordinate or FULL10K result entered this decision.
""")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=(
        "prepare", "preregister", "smoke", "train", "internal-finalize", "all-internal",
    ))
    parser.add_argument("--variant", choices=("B", "C"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    config = load_config()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if arguments.phase in {"prepare", "all-internal"}:
        prepare_cache(config)
    if arguments.phase in {"preregister", "all-internal"}:
        preregister(config)
    if arguments.phase in {"smoke", "all-internal"}:
        smoke(config, device)
    if arguments.phase == "train":
        if (arguments.variant is None) != (arguments.seed is None):
            raise ValueError("provide both --variant and --seed, or neither")
        if arguments.variant is not None:
            train_one(config, arguments.variant, arguments.seed, device)
        else:
            train_all(config, device)
    if arguments.phase == "all-internal":
        train_all(config, device)
    if arguments.phase in {"internal-finalize", "all-internal"}:
        decision = internal_finalize(config, device)
        print(f"LSGO_{decision['status']}", flush=True)
    else:
        print(f"LSGO_{arguments.phase.upper().replace('-', '_')}_COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
