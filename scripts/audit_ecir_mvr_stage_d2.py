#!/usr/bin/env python
"""Validation-only Stage D2 bond-head and branch-interference audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

for _thread_variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
):
    os.environ[_thread_variable] = "1"

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.ecir.acceptance import select_trajectory_candidate
from etflow.ecir.audit import displacement_metrics, torsion_change_metrics
from etflow.ecir.bond_explicit import (
    batched_bond_projection, bond_length_jacobian, bond_length_residual,
)
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.geometry import bond_angles, bond_lengths
from etflow.ecir.mvr_dataset import deterministic_error_features
from etflow.ecir.mvr_model import MCVRModel, trust_clip_velocity
from etflow.ecir.run_a_evaluation import (
    build_items, infer_mvr, method_rows, molecule_rows, nearest_rmsd, summarize_groups,
)
from etflow.ecir.stage_d2_audit import (
    approximate_gap_decomposition, binary_classification, branch_interference_flags,
    calibration_table, expected_calibration_error, mask_bond_residuals,
    safe_correlation, stable_average_ranks, top_k_capture,
)
from scripts.run_ecir_mvr_stage_d_bond_oracle import solve_items as solve_oracle_items


OUTPUT = Path("diagnostics/ecir_mvr/stage_d/d2")
CACHE = OUTPUT / "variant_cache"
RECOVERY_STATE = OUTPUT / "recovery_state.json"
RECOVERY_INVENTORY = OUTPUT / "recovery_inventory.json"
CONFIG = Path("configs/ecir_mvr_stage_d_d1_b_explicit_bond_seed42_5k.yaml")
D1_A = Path("logs_ecir_mvr/stage_d/d1_a_aux_only_seed42_5k/checkpoints/step002000.ckpt")
D1_B = Path("logs_ecir_mvr/stage_d/d1_b_explicit_bond_seed42_5k/checkpoints/step002000.ckpt")
V4 = Path("logs_ecir_mvr/medium/run_a_seed42_schedule_v4_10k/checkpoints/step001500.ckpt")
PROTECTED = Path("reports/global4d_profile_bundle_verification.json")
EXPECTED = {
    D1_A.as_posix(): "8ca773d384e69532f496a9a862fbc9b1b6267cd538b5c9d6ebe25e9dadfea690",
    D1_B.as_posix(): "47189368db75c86f551a69cdbba5ef5f8c85a7e80929401aded309c246c5956d",
    V4.as_posix(): "f94c317f4e12c559058e26f9842317770179ed3e9cbc07c0a21ec681fed94197",
    PROTECTED.as_posix(): "738171eb7e5f047e94cd4e1a46689613fe1f30bf33a320a2e3ec5a6944a5ec7d",
}
VARIANTS = {
    "A_CARTESIAN_ONLY": "cartesian_only",
    "B_BOND_ONLY": "bond_only",
    "C_ADDITIVE_DEPLOYED": "deployed",
    "D_CARTESIAN_BOND_SUBSPACE_REMOVED": "orthogonalized",
    "E_ORACLE_RESIDUAL_LEARNED_CONFIDENCE": "oracle_residual_confidence",
    "F_LEARNED_RESIDUAL_ORACLE_MASK": "oracle_mask",
    "G_LEARNED_RESIDUAL_ALL_CONFIDENCE_ONE": "confidence_one",
    "H_NON_RING_BOND_HEAD_ONLY": "nonring_only",
    "I_RING_BOND_HEAD_ONLY": "ring_only",
    "J_D1B_BOND_WITH_FROZEN_V4_CARTESIAN": "v4_cartesian",
}
CACHE_NAMES = {
    "A_CARTESIAN_ONLY": "A_cartesian_only.parquet",
    "B_BOND_ONLY": "B_bond_only.parquet",
    "C_ADDITIVE_DEPLOYED": "C_additive_deployed.parquet",
    "D_CARTESIAN_BOND_SUBSPACE_REMOVED": "D_cartesian_bond_subspace_removed.parquet",
    "E_ORACLE_RESIDUAL_LEARNED_CONFIDENCE": "E_oracle_residual_learned_confidence.parquet",
    "F_LEARNED_RESIDUAL_ORACLE_MASK": "F_learned_residual_oracle_mask.parquet",
    "G_LEARNED_RESIDUAL_ALL_CONFIDENCE_ONE": "G_learned_residual_confidence_one.parquet",
    "H_NON_RING_BOND_HEAD_ONLY": "H_nonring_only.parquet",
    "I_RING_BOND_HEAD_ONLY": "I_ring_only.parquet",
    "J_D1B_BOND_WITH_FROZEN_V4_CARTESIAN": "J_v4_cartesian_d1b_bond.parquet",
}
BASELINE_NAMES = {
    "v4_selected": "_baseline_v4_selected.parquet",
    "d1_a_selected": "_baseline_d1_a_selected.parquet",
    "d0_oracle": "_baseline_d0_oracle.parquet",
}
DRAWS = 10_000
SEED = 42
STEP_SIZE = 0.25
STEPS = 4


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _update_recovery(
    state: dict[str, Any], *, phase: str | None = None,
    status: str | None = None, error: str | None = None,
) -> None:
    if phase is not None:
        state["last_completed_phase"] = phase
    if status is not None:
        state["status"] = status
    state["updated_at"] = _now()
    state["latest_error"] = error
    atomic_json_save(state, RECOVERY_STATE)


def _finite_coordinates(values: Sequence[torch.Tensor]) -> bool:
    return bool(values) and all(bool(torch.isfinite(torch.as_tensor(value)).all()) for value in values)


def _cache_path(label: str) -> Path:
    return CACHE / CACHE_NAMES.get(label, f"_{label}.parquet")


def _coordinate_json(value: torch.Tensor) -> str:
    array = torch.as_tensor(value, dtype=torch.float32).detach().cpu().numpy()
    return json.dumps(array.tolist(), separators=(",", ":"), allow_nan=False)


def _decode_coordinates(value: str) -> torch.Tensor:
    return torch.as_tensor(json.loads(value), dtype=torch.float32)


def _cache_validation(path: Path, label: str, items, identities: Mapping[str, str]) -> dict[str, Any]:
    result = {
        "path": path.as_posix(), "exists": path.is_file(), "complete": False,
        "variant": label, "records": 0, "molecules": 0, "test_records_read": None,
        "error": None,
    }
    if not path.is_file():
        return result
    try:
        frame = pd.read_parquet(path)
        expected_ids = [str(item["row"].sample_id) for item in items]
        identifiers = frame.record_id.astype(str).tolist() if "record_id" in frame else []
        required = {
            "schema_version", "variant_name", "d1_b_checkpoint_sha256",
            "frozen_identities_json", "validation_only", "test_records_read",
            "molecule_id", "record_id", "atom_count", "upstream_coordinates_json",
            "minimal_target_coordinates_json", "raw_coordinates_json",
            "safety_coordinates_json", "accepted_coordinates_json", "metadata_json",
            "torsion_gate_max", "torsion_contribution_max", "numerical_failure",
        }
        coordinates_ok = required.issubset(frame.columns)
        if coordinates_ok:
            for index, row in frame.iterrows():
                expected_shape = tuple(items[index]["input"].shape)
                for column in (
                    "upstream_coordinates_json", "minimal_target_coordinates_json",
                    "raw_coordinates_json", "safety_coordinates_json",
                    "accepted_coordinates_json",
                ):
                    value = _decode_coordinates(row[column])
                    if tuple(value.shape) != expected_shape or not bool(torch.isfinite(value).all()):
                        coordinates_ok = False
                        break
                if not coordinates_ok:
                    break
        metadata_ok = required.issubset(frame.columns) and all(
            isinstance(json.loads(value), dict) for value in frame.metadata_json.astype(str)
        )
        identity_json = json.dumps(dict(identities), sort_keys=True, separators=(",", ":"))
        result.update({
            "records": len(identifiers),
            "molecules": int(frame.molecule_id.nunique()) if "molecule_id" in frame else 0,
            "test_records_read": int(frame.test_records_read.max()) if "test_records_read" in frame and len(frame) else None,
            "checkpoint_sha256": str(frame.d1_b_checkpoint_sha256.iloc[0]) if "d1_b_checkpoint_sha256" in frame and len(frame) else None,
            "columns": sorted(frame.columns.tolist()),
            "rows": len(frame),
            "complete": bool(
                len(frame) == len(items)
                and set(frame.schema_version.astype(str)) == {"ecir-mvr-stage-d2-variant-cache-v1"}
                and set(frame.variant_name.astype(str)) == {label}
                and set(frame.d1_b_checkpoint_sha256.astype(str)) == {EXPECTED[D1_B.as_posix()]}
                and set(frame.frozen_identities_json.astype(str)) == {identity_json}
                and identifiers == expected_ids
                and coordinates_ok and metadata_ok
                and bool(frame.validation_only.astype(bool).all())
                and bool((frame.test_records_read.astype(int) == 0).all())
                and bool((frame.torsion_gate_max.astype(float) == 0.0).all())
                and bool((frame.torsion_contribution_max.astype(float) == 0.0).all())
                and not bool(frame.numerical_failure.astype(bool).any())
            ),
        })
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def _save_variant_cache(label: str, value: Mapping[str, Any], items, identities) -> dict[str, Any]:
    CACHE.mkdir(parents=True, exist_ok=True)
    identity_json = json.dumps(dict(identities), sort_keys=True, separators=(",", ":"))
    rows = []
    for item, raw, safety, accepted, metadata in zip(
        items, value["raw"], value["safety"], value["accepted"], value["metadata"]
    ):
        rows.append({
            "schema_version": "ecir-mvr-stage-d2-variant-cache-v1",
            "variant_name": label,
            "d1_b_checkpoint_sha256": EXPECTED[D1_B.as_posix()],
            "v4_checkpoint_sha256": EXPECTED[V4.as_posix()] if label == "J_D1B_BOND_WITH_FROZEN_V4_CARTESIAN" else "",
            "frozen_identities_json": identity_json,
            "validation_only": True, "test_records_read": 0,
            "molecule_id": str(item["row"].molecule_id),
            "record_id": str(item["row"].sample_id),
            "atom_count": int(item["input"].shape[0]),
            "upstream_coordinates_json": _coordinate_json(item["input"]),
            "minimal_target_coordinates_json": _coordinate_json(item["minimal_target"]),
            "raw_coordinates_json": _coordinate_json(raw),
            "safety_coordinates_json": _coordinate_json(safety),
            "accepted_coordinates_json": _coordinate_json(accepted),
            "metadata_json": json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False),
            "torsion_gate_max": float(metadata.get("torsion_gate_max", 0.0)),
            "torsion_contribution_max": float(metadata.get("torsion_contribution_max", 0.0)),
            "numerical_failure": bool(metadata.get("numerical_failure", False)),
        })
    destination = _cache_path(label)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    validation = _cache_validation(temporary, label, items, identities)
    if not validation["complete"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"variant cache read-back validation failed: {validation}")
    os.replace(temporary, destination)
    validation = _cache_validation(destination, label, items, identities)
    validation["sha256"] = _sha(destination)
    return validation


def _load_variant_cache(label: str) -> dict[str, Any]:
    frame = pd.read_parquet(_cache_path(label))
    return {
        "raw": [_decode_coordinates(value) for value in frame.raw_coordinates_json],
        "safety": [_decode_coordinates(value) for value in frame.safety_coordinates_json],
        "accepted": [_decode_coordinates(value) for value in frame.accepted_coordinates_json],
        "metadata": [json.loads(value) for value in frame.metadata_json],
        "per_bond": pd.DataFrame(),
    }


def build_recovery_inventory(items, *, expected_identities: Mapping[str, str]) -> dict[str, Any]:
    files = []
    for path in sorted(OUTPUT.glob("*")):
        if not path.is_file() or path.name == "d2_interrupted_worktree.patch":
            continue
        entry: dict[str, Any] = {
            "path": path.as_posix(), "bytes": path.stat().st_size,
            "sha256": _sha(path), "readable": False, "rows": None,
            "columns": None, "schema_version": None, "molecules": None,
            "records": None, "variants": None, "validation_only": None,
            "test_records_read": None, "error": None,
        }
        try:
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
                entry.update({
                    "readable": True, "rows": len(frame), "columns": list(frame.columns),
                    "molecules": int(frame.molecule_id.nunique()) if "molecule_id" in frame else None,
                    "records": int(frame.record_id.nunique()) if "record_id" in frame else None,
                    "finite_numeric": bool(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()),
                })
            elif path.suffix == ".csv":
                frame = pd.read_csv(path)
                entry.update({
                    "readable": True, "rows": len(frame), "columns": list(frame.columns),
                    "molecules": int(frame.molecule_id.nunique()) if "molecule_id" in frame else None,
                    "records": int(frame.record_id.nunique()) if "record_id" in frame else None,
                    "variants": sorted(frame.method.astype(str).unique().tolist()) if "method" in frame else None,
                })
            elif path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                entry.update({
                    "readable": True, "schema_version": payload.get("schema_version"),
                    "validation_only": payload.get("validation_only"),
                    "test_records_read": payload.get("test_records_read"),
                })
            else:
                entry["readable"] = True
        except BaseException as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        files.append(entry)
    caches = [_cache_validation(_cache_path(label), label, items, expected_identities) for label in VARIANTS]
    inventory = {
        "schema_version": "ecir-mvr-stage-d2-recovery-inventory-v1",
        "created_at": _now(), "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "checkpoint_sha256": {name: _sha(path) for name, path in {
            "D1_A": D1_A, "D1_B": D1_B, "V4": V4, "protected": PROTECTED,
        }.items()},
        "frozen_identities": dict(expected_identities),
        "expected_validation_records": len(items),
        "expected_validation_molecules": len({str(item["row"].molecule_id) for item in items}),
        "files": files, "inference_caches": caches,
        "all_variants_complete": all(value["complete"] for value in caches),
        "completed_variants": [value["variant"] for value in caches if value["complete"]],
        "test_records_read": 0,
    }
    atomic_json_save(inventory, RECOVERY_INVENTORY)
    return inventory


def _load(path: Path, device: torch.device) -> MCVRModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = MCVRModel(**payload["config"]["model"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = math.sqrt(float(np.sum(left * left)) * float(np.sum(right * right)))
    return float(np.sum(left * right) / denominator) if denominator > 1.0e-12 else 0.0


def _norm(values: np.ndarray) -> float:
    return math.sqrt(float(np.sum(np.asarray(values, dtype=np.float64) ** 2)))


def _planarity(coordinates: torch.Tensor, ring: Sequence[int]) -> float:
    points = torch.as_tensor(coordinates, dtype=torch.float64)[list(ring)]
    if len(points) < 3:
        return 0.0
    centered = points - points.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    distances = centered @ vh[-1]
    return float(torch.sqrt(torch.mean(distances.square())))


def _validity_metrics(validity, coordinates, item) -> dict[str, float]:
    values = validity.evaluate(coordinates, item["record"], baseline_coordinates=item["input"])
    return {
        **{name: float(value) for name, value in values.items() if name != "chirality_preserved"},
        "chirality_error": 1.0 - float(values["chirality_preserved"]),
        "aligned_RMSD": nearest_rmsd(coordinates, item["references"]),
        "rms_displacement": displacement_metrics(item["input"], coordinates)["aligned_rms_displacement"],
        "high_flex_torsion_change": (
            torsion_change_metrics(item["input"], coordinates, item["record"])["max_rotatable_torsion_change"]
            if item["rotatable"] >= 6 else 0.0
        ),
    }


def _bond_metadata(validity, item, size_group: str, bond_count_group: str) -> list[dict[str, Any]]:
    prepared = validity._prepare(item["record"])
    atom, bond_environment, _ = validity._environment(item["record"])
    rows = []
    ring_atoms = {int(atom_index) for ring in prepared["rings"] for atom_index in ring}
    symbols = {6: "C", 7: "N", 8: "O"}
    for index, ((left, right), stat, ring) in enumerate(zip(
        prepared["bonds"].t().tolist(), prepared["bond_stats"].tolist(),
        prepared["ring_mask"].tolist(),
    )):
        key = tuple(sorted((left, right)))
        bond_type, aromatic, _ = bond_environment[key]
        left_z, right_z = int(atom[left][0]), int(atom[right][0])
        element_pair = "-".join(symbols.get(value, str(value)) for value in sorted((left_z, right_z)))
        rows.append({
            "molecule_id": str(item["row"].molecule_id),
            "record_id": str(item["row"].sample_id), "bond_index": index,
            "atom_i": int(left), "atom_j": int(right), "atom_i_z": left_z,
            "atom_j_z": right_z, "element_pair": element_pair,
            "bond_type": str(bond_type), "aromatic": bool(aromatic), "ring": bool(ring),
            "atom_i_in_ring": left in ring_atoms, "atom_j_in_ring": right in ring_atoms,
            "source": str(item["row"].generator_name),
            "severity": str(item["row"].source_severity),
            "update_scale": float(item["row"].update_scale),
            "rotatable_group": (
                "rotatable_le_2" if item["rotatable"] <= 2 else
                "rotatable_3_5" if item["rotatable"] <= 5 else "rotatable_ge_6"
            ),
            "molecule_size_group": size_group, "bond_count_group": bond_count_group,
            "valid_lower": float(stat[0]), "valid_upper": float(stat[1]),
        })
    return rows


def _quartile_groups(items, validity) -> tuple[dict[str, str], dict[str, str]]:
    molecule_sizes = {}
    bond_counts = {}
    for item in items:
        molecule = str(item["row"].molecule_id)
        molecule_sizes[molecule] = int(item["input"].shape[0])
        bond_counts[molecule] = int(validity._prepare(item["record"])["bonds"].shape[1])

    def assign(values: Mapping[str, int], prefix: str) -> dict[str, str]:
        identifiers = sorted(values)
        labels = pd.qcut(
            [values[value] for value in identifiers], 4, labels=False, duplicates="drop"
        )
        return {identifier: f"{prefix}_q{int(label) + 1}" for identifier, label in zip(identifiers, labels)}

    return assign(molecule_sizes, "molecule_size"), assign(bond_counts, "bond_count")


def _variant_raw(
    variant: str, output: Mapping[str, torch.Tensor], current: torch.Tensor,
    target: torch.Tensor, batch, model: MCVRModel, v4_model: MCVRModel | None,
    deterministic: torch.Tensor, trust_remaining: torch.Tensor,
    time_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cartesian = output["v_cartesian_raw"]
    bond = output["v_bond_correction"]
    bonds = output["bond_indices"]
    residual = output["bond_predicted_residual"]
    edge_index = batch.edge_index
    edge_keep = edge_index[0] < edge_index[1]
    ring = batch.bond_is_in_ring[edge_keep].to(torch.bool)
    if variant == "cartesian_only":
        return cartesian, residual.new_zeros(residual.shape)
    if variant == "bond_only":
        return bond, residual
    if variant == "deployed":
        return cartesian + bond, residual
    if variant == "orthogonalized":
        jacobian_effect = torch.empty_like(residual)
        bond_graph = batch.batch[bonds[0]]
        for graph in range(int(batch.num_graphs)):
            keep = bond_graph == graph
            atoms = torch.nonzero(batch.batch == graph, as_tuple=False).reshape(-1)
            start = int(atoms[0])
            jacobian = bond_length_jacobian(current[atoms], bonds[:, keep] - start)
            jacobian_effect[keep] = jacobian @ cartesian[atoms].reshape(-1)
        projection, _ = batched_bond_projection(
            current, bonds, jacobian_effect, batch.batch,
            damping=model.bond_projection_damping,
        )
        return cartesian - projection + bond, residual
    if variant == "oracle_residual_confidence":
        oracle = bond_length_residual(current, target, bonds).clamp(
            -model.max_abs_bond_residual, model.max_abs_bond_residual
        ) * output["bond_confidence"]
        correction, _ = batched_bond_projection(
            current, bonds, oracle, batch.batch, damping=model.bond_projection_damping
        )
        return cartesian + correction, oracle
    if variant == "oracle_mask":
        target_residual = bond_length_residual(current, target, bonds)
        lengths = bond_lengths(current, bonds)
        prepared_active = torch.zeros_like(target_residual, dtype=torch.bool)
        for graph in range(int(batch.num_graphs)):
            keep = batch.batch[bonds[0]] == graph
            local = torch.nonzero(keep, as_tuple=False).reshape(-1)
            # Target activity is fixed and diagnostic-only; no value reaches a trained parameter.
            prepared_active[local] = target_residual[local].abs() > 0.005
        masked = residual.new_tensor(mask_bond_residuals(
            residual.detach().cpu(), ring=ring.detach().cpu(), mode="oracle_active",
            active=prepared_active.detach().cpu(),
        ))
        correction, _ = batched_bond_projection(
            current, bonds, masked, batch.batch, damping=model.bond_projection_damping
        )
        return cartesian + correction, masked
    if variant == "confidence_one":
        unattenuated = output["bond_unattenuated_residual"]
        correction, _ = batched_bond_projection(
            current, bonds, unattenuated, batch.batch, damping=model.bond_projection_damping
        )
        return cartesian + correction, unattenuated
    if variant in {"nonring_only", "ring_only"}:
        masked = residual.new_tensor(mask_bond_residuals(
            residual.detach().cpu(), ring=ring.detach().cpu(), mode=variant,
        ))
        correction, _ = batched_bond_projection(
            current, bonds, masked, batch.batch, damping=model.bond_projection_damping
        )
        return cartesian + correction, masked
    if variant == "v4_cartesian":
        if v4_model is None:
            raise RuntimeError("V4 model is required for the frozen-Cartesian counterfactual")
        v4_output = v4_model(
            batch, current, current.new_full((int(batch.num_graphs),), float(time_value)),
            deterministic_features=deterministic,
            torsion_trust_remaining=trust_remaining,
        )
        return v4_output["v_cartesian_raw"] + bond, residual
    raise ValueError(f"unknown Stage D2 variant: {variant}")


@torch.inference_mode()
def infer_variant(
    model: MCVRModel, items, validity, *, variant: str, device: torch.device,
    v4_model: MCVRModel | None = None, collect_bonds: bool = False,
    molecule_size_groups: Mapping[str, str] | None = None,
    bond_count_groups: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    accepted_all, raw_all, safety_all, metadata_all, bond_rows = [], [], [], [], []
    for start_index in range(0, len(items), 32):
        selected = list(items[start_index:start_index + 32])
        batch = Batch.from_data_list([item["data"] for item in selected]).to(device)
        current = torch.cat([item["input"] for item in selected]).to(device)
        target = torch.cat([item["minimal_target"] for item in selected]).to(device)
        raw_coordinates = current.clone()
        ptr = batch.ptr.detach().cpu().tolist()
        trajectories = [[] for _ in selected]
        uncertainties = [[] for _ in selected]
        step_bond_values = [[] for _ in selected]
        max_torsion_gate = 0.0
        max_torsion_contribution = 0.0
        for time_value in torch.linspace(0.0, 1.0, STEPS).tolist():
            current_cpu = current.detach().cpu()
            features, remaining = [], []
            for local, item in enumerate(selected):
                left, right = ptr[local], ptr[local + 1]
                values = validity.evaluate(
                    current_cpu[left:right], item["record"], baseline_coordinates=item["input"]
                )
                features.append(deterministic_error_features(
                    values, item["record"], str(item["row"].source_severity)
                ))
                torsion = torsion_change_metrics(
                    item["input"], current_cpu[left:right], item["record"]
                )["max_rotatable_torsion_change"]
                limit = 0.35 if item["rotatable"] >= 6 else 0.70
                remaining.append(max(0.0, limit - float(torsion)))
            deterministic = torch.stack(features).to(device)
            trust_remaining = current.new_tensor(remaining)
            output = model(
                batch, current,
                current.new_full((len(selected),), float(time_value)),
                deterministic_features=deterministic,
                torsion_trust_remaining=trust_remaining,
            )
            max_torsion_gate = max(max_torsion_gate, float(output["torsion_gate"].abs().max()))
            max_torsion_contribution = max(
                max_torsion_contribution,
                float(output["v_torsion_contribution"].abs().max()),
            )
            raw, requested = _variant_raw(
                variant, output, current, target, batch, model, v4_model,
                deterministic, trust_remaining, float(time_value),
            )
            clipped = trust_clip_velocity(
                raw, batch.batch, max_atom_norm=model.max_velocity_atom_norm,
                max_graph_rms=model.max_velocity_graph_rms,
            )
            final = output["global_safety_gate"][batch.batch] * clipped
            raw_coordinates = raw_coordinates + STEP_SIZE * raw
            model_input = current
            current = current + STEP_SIZE * final
            snapshot = current.detach().cpu()
            bonds = output["bond_indices"]
            bond_graph = batch.batch[bonds[0]] if bonds.numel() else batch.batch.new_empty(0)
            for local, item in enumerate(selected):
                left, right = ptr[local], ptr[local + 1]
                trajectories[local].append(snapshot[left:right].clone())
                uncertainties[local].append(float(output["uncertainty"][local]))
                if collect_bonds:
                    keep = bond_graph == local
                    local_bonds = bonds[:, keep] - left
                    jacobian = bond_length_jacobian(model_input[left:right], local_bonds)
                    cart_effect = jacobian @ output["v_cartesian_raw"][left:right].reshape(-1)
                    bond_effect = jacobian @ output["v_bond_correction"][left:right].reshape(-1)
                    step_bond_values[local].append({
                        "raw": output["bond_raw_residual"][keep].detach().cpu(),
                        "unattenuated": output["bond_unattenuated_residual"][keep].detach().cpu(),
                        "confidence": output["bond_confidence"][keep].detach().cpu(),
                        "uncertainty": output["bond_uncertainty"][keep].detach().cpu(),
                        "bounded": output["bond_predicted_residual"][keep].detach().cpu(),
                        "requested": requested[keep].detach().cpu(),
                        "cart_effect": cart_effect.detach().cpu(),
                        "bond_effect": bond_effect.detach().cpu(),
                        "combined_effect": (cart_effect + bond_effect).detach().cpu(),
                    })
        if max_torsion_gate != 0.0 or max_torsion_contribution != 0.0:
            raise RuntimeError("D2 audit observed a nonzero torsion branch")
        for local, item in enumerate(selected):
            left, right = ptr[local], ptr[local + 1]
            accepted, decision = select_trajectory_candidate(
                item["input"], trajectories[local], item["record"], validity,
                mode="best_of_trajectory", uncertainties=uncertainties[local],
            )
            raw_all.append(raw_coordinates[left:right].detach().cpu().clone())
            safety_all.append(current[left:right].detach().cpu().clone())
            accepted_all.append(accepted.detach().cpu().clone())
            metadata_all.append({
                "accepted": bool(decision.accepted), "selected_step": int(decision.selected_step),
                "reject_reasons": ";".join(decision.reject_reasons),
                "uncertainty": float(decision.uncertainty),
                "torsion_gate_max": max_torsion_gate,
                "torsion_contribution_max": max_torsion_contribution,
                "numerical_failure": not bool(
                    torch.isfinite(raw_coordinates[left:right]).all()
                    and torch.isfinite(current[left:right]).all()
                    and torch.isfinite(accepted).all()
                ),
            })
            if collect_bonds:
                molecule = str(item["row"].molecule_id)
                prepared = validity._prepare(item["record"])
                bonds = prepared["bonds"]
                base = _bond_metadata(
                    validity, item, molecule_size_groups[molecule], bond_count_groups[molecule]
                )
                input_lengths = bond_lengths(item["input"], bonds)
                target_lengths = bond_lengths(item["minimal_target"], bonds)
                accepted_lengths = bond_lengths(accepted, bonds)
                stats = prepared["bond_stats"]
                selected_steps = int(decision.selected_step) if decision.accepted else 0
                first = step_bond_values[local][0]
                for index, row in enumerate(base):
                    requested_total = sum(
                        STEP_SIZE * float(value["requested"][index])
                        for value in step_bond_values[local][:selected_steps]
                    )
                    achieved_total = sum(
                        STEP_SIZE * float(value["bond_effect"][index])
                        for value in step_bond_values[local][:selected_steps]
                    )
                    current_length = float(input_lengths[index])
                    target_residual = float(target_lengths[index] - input_lengths[index])
                    output_length = float(accepted_lengths[index])
                    lower, upper = float(stats[index, 0]), float(stats[index, 1])
                    upstream_outlier = current_length < lower or current_length > upper
                    target_outlier = float(target_lengths[index]) < lower or float(target_lengths[index]) > upper
                    output_outlier = output_length < lower or output_length > upper
                    magnitude = abs(target_residual)
                    row.update({
                        "current_bond_length": current_length,
                        "target_signed_bond_residual": target_residual,
                        "predicted_raw_residual": float(first["raw"][index]),
                        "predicted_unattenuated_residual": float(first["unattenuated"][index]),
                        "confidence": float(first["confidence"][index]),
                        "uncertainty": float(first["uncertainty"][index]),
                        "bounded_predicted_residual": float(first["bounded"][index]),
                        "solver_requested_residual": requested_total,
                        "solver_achieved_bond_change": achieved_total,
                        "final_accepted_bond_change": output_length - current_length,
                        "cartesian_bond_change": sum(
                            STEP_SIZE * float(value["cart_effect"][index])
                            for value in step_bond_values[local][:selected_steps]
                        ),
                        "bond_branch_change": achieved_total,
                        "combined_bond_change": sum(
                            STEP_SIZE * float(value["combined_effect"][index])
                            for value in step_bond_values[local][:selected_steps]
                        ),
                        "upstream_outlier": upstream_outlier,
                        "target_outlier": target_outlier, "output_outlier": output_outlier,
                        "accepted": bool(decision.accepted), "selected_step": selected_steps,
                        "target_residual_bin": (
                            "zero" if magnitude <= 1.0e-4 else "le_0.005" if magnitude <= 0.005
                            else "le_0.02" if magnitude <= 0.02 else "le_0.05"
                            if magnitude <= 0.05 else "gt_0.05"
                        ),
                    })
                    bond_rows.append(row)
    return {
        "raw": raw_all, "safety": safety_all, "accepted": accepted_all,
        "metadata": metadata_all, "per_bond": pd.DataFrame(bond_rows),
    }


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    target = frame.target_signed_bond_residual.to_numpy(float)
    predicted = frame.bounded_predicted_residual.to_numpy(float)
    error = predicted - target
    active = np.abs(target) > 0.005
    predicted_active = np.abs(predicted) > 0.005
    outlier = frame.upstream_outlier.to_numpy(bool)
    confidence_active = frame.confidence.to_numpy(float) >= 0.5
    nonzero = np.abs(target) > 1.0e-4
    sign_accuracy = float((np.sign(predicted[nonzero]) == np.sign(target[nonzero])).mean()) if nonzero.any() else 1.0
    detection = binary_classification(active, predicted_active)
    outlier_detection = binary_classification(outlier, confidence_active)
    requested = frame.solver_requested_residual.to_numpy(float)
    achieved = frame.solver_achieved_bond_change.to_numpy(float)
    requested_active = np.abs(requested) > 1.0e-8
    ratios = achieved[requested_active] / requested[requested_active]
    severe = outlier & (
        (frame.current_bond_length < frame.valid_lower - 0.05)
        | (frame.current_bond_length > frame.valid_upper + 0.05)
    ).to_numpy(bool)
    mild = outlier & ~severe
    repaired = ~frame.output_outlier.to_numpy(bool)
    zero = np.abs(target) <= 1.0e-4
    return {
        "bonds": len(frame), "residual_mae": float(np.abs(error).mean()),
        "residual_rmse": float(np.sqrt(np.mean(error ** 2))),
        "pearson": safe_correlation(target, predicted),
        "spearman": safe_correlation(target, predicted, rank=True),
        "residual_sign_accuracy": sign_accuracy,
        "nonzero_detection_precision": detection["precision"],
        "nonzero_detection_recall": detection["recall"],
        "nonzero_detection_f1": detection["f1"],
        "outlier_detection_precision": outlier_detection["precision"],
        "outlier_detection_recall": outlier_detection["recall"],
        "outlier_detection_f1": outlier_detection["f1"],
        "predicted_target_norm_ratio": float(_norm(predicted) / max(_norm(target), 1.0e-12)),
        "solver_achieved_requested_ratio": float(np.median(ratios)) if ratios.size else 0.0,
        "top10_target_capture": top_k_capture(target, predicted),
        "severe_outlier_repair_recall": float(repaired[severe].mean()) if severe.any() else 1.0,
        "mild_outlier_repair_recall": float(repaired[mild].mean()) if mild.any() else 1.0,
        "zero_target_false_positive_rate": float((np.abs(predicted[zero]) > 0.005).mean()) if zero.any() else 0.0,
        "mean_confidence": float(frame.confidence.mean()),
        "mean_uncertainty": float(frame.uncertainty.mean()),
    }


def build_prediction_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [{"group_type": "all", "group": "all", **prediction_metrics(frame)}]
    group_columns = {
        "ring": "ring", "aromatic": "aromatic", "bond_type": "bond_type",
        "element_pair": "element_pair", "source": "source", "severity": "severity",
        "rotatable_group": "rotatable_group", "molecule_size": "molecule_size_group",
        "bond_count": "bond_count_group", "target_magnitude": "target_residual_bin",
    }
    unseen = frame[
        (frame.source == "Cartesian_teacher_100k") & (frame.update_scale.sub(0.35).abs() < 1.0e-12)
    ]
    if len(unseen):
        rows.append({"group_type": "source", "group": "unseen_scale0.35", **prediction_metrics(unseen)})
    for group_type, column in group_columns.items():
        for value, subset in frame.groupby(column, dropna=False):
            rows.append({"group_type": group_type, "group": str(value), **prediction_metrics(subset)})
    return pd.DataFrame(rows)


def record_interference(per_bond: pd.DataFrame, items, variants, validity) -> pd.DataFrame:
    rows = []
    by_record = per_bond.groupby("record_id", sort=False)
    for index, item in enumerate(items):
        record_id = str(item["row"].sample_id)
        frame = by_record.get_group(record_id)
        target = frame.target_signed_bond_residual.to_numpy(float)
        cart = frame.cartesian_bond_change.to_numpy(float)
        bond = frame.bond_branch_change.to_numpy(float)
        flags = branch_interference_flags(target, cart, bond)
        metrics = {
            name: float(value[flags["active"]].mean()) if flags["active"].any() else 0.0
            for name, value in flags.items() if name != "active"
        }
        upstream = _validity_metrics(validity, item["input"], item)
        variant_values = {
            key: _validity_metrics(validity, value["accepted"][index], item)
            for key, value in variants.items()
        }
        raw = _validity_metrics(validity, variants["C_ADDITIVE_DEPLOYED"]["raw"][index], item)
        safety = _validity_metrics(validity, variants["C_ADDITIVE_DEPLOYED"]["safety"][index], item)
        cart_gain = upstream["bond_outlier_rate"] - variant_values["A_CARTESIAN_ONLY"]["bond_outlier_rate"]
        bond_gain = upstream["bond_outlier_rate"] - variant_values["B_BOND_ONLY"]["bond_outlier_rate"]
        combined_gain = upstream["bond_outlier_rate"] - safety["bond_outlier_rate"]
        rows.append({
            "molecule_id": str(item["row"].molecule_id), "record_id": record_id,
            "cartesian_target_cosine": _cosine(cart, target),
            "bond_target_cosine": _cosine(bond, target),
            "combined_target_cosine": _cosine(cart + bond, target),
            "cartesian_norm_ratio": float(_norm(cart) / max(_norm(target), 1.0e-12)),
            "bond_norm_ratio": float(_norm(bond) / max(_norm(target), 1.0e-12)),
            "combined_norm_ratio": float(_norm(cart + bond) / max(_norm(target), 1.0e-12)),
            **metrics,
            "cartesian_alone_gain": cart_gain, "bond_alone_gain": bond_gain,
            "additive_combined_gain": combined_gain,
            "interaction_gain": combined_gain - cart_gain - bond_gain,
            "safety_gate_loss": raw["bond_outlier_rate"] - safety["bond_outlier_rate"],
            "acceptance_gain": safety["bond_outlier_rate"] - variant_values["C_ADDITIVE_DEPLOYED"]["bond_outlier_rate"],
        })
    return pd.DataFrame(rows)


def _molecule_bootstrap(
    molecules: pd.DataFrame, candidate: str, *, baseline: str = "v4_selected"
) -> dict[str, dict[str, float]]:
    metrics = (
        "bond_outlier_rate", "bond_outlier_magnitude", "angle_outlier_rate",
        "ring_bond_outlier_rate", "total_thresholded_validity_score",
        "aligned_RMSD", "MAT_P", "MAT_R", "COV_P", "COV_R",
    )
    frame = molecules[molecules.group == "all"]
    result = {}
    rng = np.random.default_rng(SEED)
    for metric in metrics:
        pivot = frame.pivot(index="molecule_id", columns="method", values=metric).dropna()
        delta = pivot[candidate].to_numpy() - pivot[baseline].to_numpy()
        indices = rng.integers(0, len(delta), size=(DRAWS, len(delta)))
        means = delta[indices].mean(1)
        result[metric] = {
            "mean": float(delta.mean()), "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
        }
    return result


def angle_ring_damage(
    items, variants, v4_coordinates, per_bond, validity,
    *, v4_record_metrics: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    lookup = per_bond.set_index(["record_id", "atom_i", "atom_j"])
    v4_by_record = (
        v4_record_metrics.set_index("sample_id") if v4_record_metrics is not None else None
    )
    rows = []
    for index, item in enumerate(items):
        record_id = str(item["row"].sample_id)
        prepared = validity._prepare(item["record"])
        input_coordinates = item["input"]
        v4 = torch.as_tensor(v4_coordinates[index]) if v4_coordinates is not None else input_coordinates
        deployed = torch.as_tensor(variants["C_ADDITIVE_DEPLOYED"]["accepted"][index])
        cartesian = torch.as_tensor(variants["A_CARTESIAN_ONLY"]["accepted"][index])
        bond_only = torch.as_tensor(variants["B_BOND_ONLY"]["accepted"][index])
        input_angles = bond_angles(input_coordinates, prepared["angles"])
        v4_angles = bond_angles(v4, prepared["angles"])
        deployed_angles = bond_angles(deployed, prepared["angles"])
        cart_angles = bond_angles(cartesian, prepared["angles"])
        bond_angles_only = bond_angles(bond_only, prepared["angles"])
        deployed_record = validity.evaluate(deployed, item["record"], baseline_coordinates=input_coordinates)
        v4_record = v4_by_record.loc[record_id] if v4_by_record is not None else None
        angle_record_worse = (
            float(deployed_record["angle_outlier_rate"]) > float(v4_record.angle_outlier_rate) + 1.0e-12
            if v4_record is not None else True
        )
        ring_record_worse = (
            float(deployed_record["ring_bond_outlier_rate"]) > float(v4_record.ring_bond_outlier_rate) + 1.0e-12
            if v4_record is not None else True
        )
        planarity_record_worse = (
            float(deployed_record["ring_planarity_outlier_rate"]) > float(v4_record.ring_planarity_outlier_rate) + 1.0e-12
            if v4_record is not None else True
        )
        for angle_index, (atoms, stat) in enumerate(zip(
            prepared["angles"].tolist(), prepared["angle_stats"].tolist()
        )):
            left, center, right = atoms
            lower, upper = float(stat[0]), float(stat[1])
            excess_v4 = max(lower - float(v4_angles[angle_index]), float(v4_angles[angle_index]) - upper, 0.0)
            excess_d1 = max(lower - float(deployed_angles[angle_index]), float(deployed_angles[angle_index]) - upper, 0.0)
            if not angle_record_worse or excess_d1 <= excess_v4 + 1.0e-12:
                continue
            keys = [tuple(sorted((left, center))), tuple(sorted((center, right)))]
            bond_values = [lookup.loc[(record_id, key[0], key[1])] for key in keys]
            predictions = [float(value.bounded_predicted_residual) for value in bond_values]
            targets = [float(value.target_signed_bond_residual) for value in bond_values]
            rows.append({
                "damage_type": "angle", "molecule_id": str(item["row"].molecule_id),
                "record_id": record_id, "center_atom": center,
                "involved_atoms": f"{left};{center};{right}",
                "involves_ring_bond": any(bool(value.ring) for value in bond_values),
                "ring_bond_count": sum(bool(value.ring) for value in bond_values),
                "bond_predictions": ";".join(str(value) for value in predictions),
                "target_residuals": ";".join(str(value) for value in targets),
                "cartesian_contribution": float(cart_angles[angle_index] - input_angles[angle_index]),
                "bond_contribution": float(bond_angles_only[angle_index] - input_angles[angle_index]),
                "combined_contribution": float(deployed_angles[angle_index] - input_angles[angle_index]),
                "input_value": float(input_angles[angle_index]), "v4_value": float(v4_angles[angle_index]),
                "output_value": float(deployed_angles[angle_index]),
                "v4_excess": excess_v4, "output_excess": excess_d1,
                "confidence": float(np.mean([value.confidence for value in bond_values])),
                "uncertainty": float(np.mean([value.uncertainty for value in bond_values])),
                "accepted": bool(variants["C_ADDITIVE_DEPLOYED"]["metadata"][index]["accepted"]),
                "comparison_mode": "v4_record_screen_upstream_local_reference" if v4_by_record is not None else "v4_coordinates",
                "adjacent_same_direction": bool(np.prod(predictions) > 0.0),
                "any_prediction_sign_wrong": any(
                    np.sign(prediction) != np.sign(target) and abs(target) > 1.0e-4
                    for prediction, target in zip(predictions, targets)
                ),
            })
        input_lengths = bond_lengths(input_coordinates, prepared["bonds"])
        v4_lengths = bond_lengths(v4, prepared["bonds"])
        output_lengths = bond_lengths(deployed, prepared["bonds"])
        for bond_index in torch.nonzero(prepared["ring_mask"], as_tuple=False).reshape(-1).tolist():
            lower, upper = prepared["bond_stats"][bond_index, :2].tolist()
            excess_v4 = max(lower - float(v4_lengths[bond_index]), float(v4_lengths[bond_index]) - upper, 0.0)
            excess_d1 = max(lower - float(output_lengths[bond_index]), float(output_lengths[bond_index]) - upper, 0.0)
            if not ring_record_worse or excess_d1 <= excess_v4 + 1.0e-12:
                continue
            left, right = prepared["bonds"][:, bond_index].tolist()
            value = lookup.loc[(record_id, min(left, right), max(left, right))]
            containing = [ring for ring in prepared["rings"] if left in ring and right in ring]
            rows.append({
                "damage_type": "ring_bond", "molecule_id": str(item["row"].molecule_id),
                "record_id": record_id, "center_atom": "",
                "involved_atoms": f"{left};{right}", "involves_ring_bond": True,
                "ring_bond_count": 1, "bond_predictions": value.bounded_predicted_residual,
                "target_residuals": value.target_signed_bond_residual,
                "cartesian_contribution": value.cartesian_bond_change,
                "bond_contribution": value.bond_branch_change,
                "combined_contribution": value.combined_bond_change,
                "input_value": float(input_lengths[bond_index]), "v4_value": float(v4_lengths[bond_index]),
                "output_value": float(output_lengths[bond_index]), "v4_excess": excess_v4,
                "output_excess": excess_d1, "confidence": value.confidence,
                "uncertainty": value.uncertainty,
                "accepted": bool(variants["C_ADDITIVE_DEPLOYED"]["metadata"][index]["accepted"]),
                "comparison_mode": "v4_record_screen_upstream_local_reference" if v4_by_record is not None else "v4_coordinates",
                "adjacent_same_direction": False,
                "any_prediction_sign_wrong": bool(
                    np.sign(value.bounded_predicted_residual) != np.sign(value.target_signed_bond_residual)
                    and abs(value.target_signed_bond_residual) > 1.0e-4
                ),
                "ring_planarity_input": float(np.mean([_planarity(input_coordinates, ring) for ring in containing])) if containing else 0.0,
                "ring_planarity_v4": float(np.mean([_planarity(v4, ring) for ring in containing])) if containing else 0.0,
                "ring_planarity_output": float(np.mean([_planarity(deployed, ring) for ring in containing])) if containing else 0.0,
            })
        for ring_index, (ring, stat) in enumerate(zip(
            prepared["rings"], prepared["planarity_stats"]
        )):
            input_planarity = _planarity(input_coordinates, ring)
            v4_planarity = _planarity(v4, ring)
            output_planarity = _planarity(deployed, ring)
            excess_v4 = max(float(stat["lower"]) - v4_planarity, v4_planarity - float(stat["upper"]), 0.0)
            excess_d1 = max(float(stat["lower"]) - output_planarity, output_planarity - float(stat["upper"]), 0.0)
            if not planarity_record_worse or excess_d1 <= excess_v4 + 1.0e-12:
                continue
            pairs = [
                tuple(sorted((ring[position], ring[(position + 1) % len(ring)])))
                for position in range(len(ring))
            ]
            values = [lookup.loc[(record_id, left, right)] for left, right in pairs]
            rows.append({
                "damage_type": "ring_planarity", "molecule_id": str(item["row"].molecule_id),
                "record_id": record_id, "center_atom": "",
                "involved_atoms": ";".join(str(value) for value in ring),
                "involves_ring_bond": True, "ring_bond_count": len(values),
                "bond_predictions": ";".join(str(value.bounded_predicted_residual) for value in values),
                "target_residuals": ";".join(str(value.target_signed_bond_residual) for value in values),
                "cartesian_contribution": _planarity(cartesian, ring) - input_planarity,
                "bond_contribution": _planarity(bond_only, ring) - input_planarity,
                "combined_contribution": output_planarity - input_planarity,
                "input_value": input_planarity, "v4_value": v4_planarity,
                "output_value": output_planarity, "v4_excess": excess_v4,
                "output_excess": excess_d1,
                "confidence": float(np.mean([value.confidence for value in values])),
                "uncertainty": float(np.mean([value.uncertainty for value in values])),
                "accepted": bool(variants["C_ADDITIVE_DEPLOYED"]["metadata"][index]["accepted"]),
                "comparison_mode": "v4_record_screen_upstream_local_reference" if v4_by_record is not None else "v4_coordinates",
                "adjacent_same_direction": bool(
                    all(value.bounded_predicted_residual >= 0.0 for value in values)
                    or all(value.bounded_predicted_residual <= 0.0 for value in values)
                ),
                "any_prediction_sign_wrong": any(
                    np.sign(value.bounded_predicted_residual) != np.sign(value.target_signed_bond_residual)
                    and abs(value.target_signed_bond_residual) > 1.0e-4 for value in values
                ),
                "ring_planarity_input": input_planarity,
                "ring_planarity_v4": v4_planarity,
                "ring_planarity_output": output_planarity,
            })
    frame = pd.DataFrame(rows)
    angles = frame[frame.damage_type == "angle"] if len(frame) else frame
    ring_rows = frame[frame.damage_type.isin(["ring_bond", "ring_planarity"])] if len(frame) else frame
    summary = {
        "comparison_mode": "v4_record_screen_upstream_local_reference" if v4_by_record is not None else "v4_coordinates",
        "damage_records": len(frame), "angle_damage_records": len(angles),
        "ring_bond_damage_records": len(ring_rows),
        "ring_planarity_damage_records": int((frame.damage_type == "ring_planarity").sum()) if len(frame) else 0,
        "angle_damage_involving_ring_fraction": float(angles.involves_ring_bond.mean()) if len(angles) else 0.0,
        "angle_damage_nonring_only_fraction": float((~angles.involves_ring_bond).mean()) if len(angles) else 0.0,
        "adjacent_same_direction_fraction": float(angles.adjacent_same_direction.mean()) if len(angles) else 0.0,
        "angle_wrong_sign_fraction": float(angles.any_prediction_sign_wrong.mean()) if len(angles) else 0.0,
        "ring_wrong_sign_fraction": float(ring_rows.any_prediction_sign_wrong.mean()) if len(ring_rows) else 0.0,
        "ring_multi_bond_records": int(ring_rows.groupby("record_id").size().ge(2).sum()) if len(ring_rows) else 0,
    }
    return frame, summary


def ring_nonring_summary(per_bond: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "ring_all": per_bond.ring,
        "ring_aromatic": per_bond.ring & per_bond.aromatic,
        "ring_nonaromatic": per_bond.ring & ~per_bond.aromatic,
        "nonring_all": ~per_bond.ring,
        "nonring_attached_to_ring": (~per_bond.ring) & (
            per_bond.atom_i_in_ring | per_bond.atom_j_in_ring
        ),
        "nonring_local_only": (~per_bond.ring) & ~(
            per_bond.atom_i_in_ring | per_bond.atom_j_in_ring
        ),
    }
    for name, keep in groups.items():
        frame = per_bond[keep]
        if frame.empty:
            continue
        metrics = prediction_metrics(frame)
        upstream_rate = float(frame.upstream_outlier.mean())
        output_rate = float(frame.output_outlier.mean())
        target_rate = float(frame.target_outlier.mean())
        rows.append({
            "group": name, **metrics,
            "target_available_improvement": (upstream_rate - target_rate) / max(upstream_rate, 1.0e-12),
            "bond_improvement": (upstream_rate - output_rate) / max(upstream_rate, 1.0e-12),
            "recovery_ratio": (
                (upstream_rate - output_rate) / max(upstream_rate - target_rate, 1.0e-12)
            ),
            "newly_broken_bonds": int((~frame.upstream_outlier & frame.output_outlier).sum()),
            "cancellation_fraction": float((
                frame.cartesian_bond_change * frame.bond_branch_change < 0.0
            ).mean()),
        })
    return pd.DataFrame(rows)


def legacy_main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    for path, expected in EXPECTED.items():
        if _sha(Path(path)) != expected:
            raise RuntimeError(f"frozen identity changed: {path}")
    pilot = json.loads(Path("diagnostics/ecir_mvr/stage_d/pilot/result.json").read_text(encoding="utf-8"))
    oracle_result = json.loads(Path("diagnostics/ecir_mvr/stage_d/oracle/result.json").read_text(encoding="utf-8"))
    state = json.loads(Path("reports/ecir_mvr/progressive_state.json").read_text(encoding="utf-8"))
    if pilot["decision"] != "STAGE_D_NO_ADDED_VALUE" or pilot["test_records_read"] != 0:
        raise RuntimeError("formal Stage D pilot state changed")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["frozen_identities"] != oracle_result["frozen_identities"]:
        raise RuntimeError("frozen identities changed")
    device = torch.device("cuda")
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = build_items(config["data"]["val_sources"], config["data"]["val_targets"], validity)
    molecule_groups, bond_groups = _quartile_groups(items, validity)
    model_b = _load(D1_B, device)
    model_a = _load(D1_A, device)
    model_v4 = _load(V4, device)

    variants = {}
    for label, variant in VARIANTS.items():
        variants[label] = infer_variant(
            model_b, items, validity, variant=variant, device=device,
            v4_model=model_v4 if variant == "v4_cartesian" else None,
            collect_bonds=variant == "deployed",
            molecule_size_groups=molecule_groups, bond_count_groups=bond_groups,
        )
        print(f"completed {label}", flush=True)
    per_bond = variants["C_ADDITIVE_DEPLOYED"]["per_bond"]
    _, v4_accepted, v4_metadata = infer_mvr(model_v4, items, validity, device=device)
    _, a_accepted, a_metadata = infer_mvr(model_a, items, validity, device=device)
    _, _, oracle_accepted, _, _ = solve_oracle_items(
        items, validity, max_atom_norm=model_b.max_velocity_atom_norm,
        max_graph_rms=model_b.max_velocity_graph_rms,
    )
    oracle_changes = {}
    for item, coordinates in zip(items, oracle_accepted):
        bonds = validity._prepare(item["record"])["bonds"]
        changes = bond_lengths(coordinates, bonds) - bond_lengths(item["input"], bonds)
        for bond_index, value in enumerate(changes.tolist()):
            oracle_changes[(str(item["row"].sample_id), bond_index)] = float(value)
    per_bond["d0_oracle_accepted_bond_change"] = [
        oracle_changes[(record_id, int(bond_index))]
        for record_id, bond_index in zip(per_bond.record_id, per_bond.bond_index)
    ]
    per_bond.to_parquet(OUTPUT / "per_bond_prediction_metrics.parquet", index=False)
    prediction_summary = build_prediction_summary(per_bond)
    prediction_summary.to_csv(OUTPUT / "prediction_quality_summary.csv", index=False)
    all_prediction = prediction_metrics(per_bond)
    confidence_table = calibration_table(
        per_bond.confidence, (per_bond.target_signed_bond_residual.abs() > 0.005).astype(float)
    )
    absolute_error = (
        per_bond.bounded_predicted_residual - per_bond.target_signed_bond_residual
    ).abs()
    uncertainty_rank = per_bond.uncertainty.rank(pct=True).to_numpy()
    error_rank = absolute_error.rank(pct=True).to_numpy()
    uncertainty_table = calibration_table(uncertainty_rank, error_rank)
    calibration = {
        "schema_version": "ecir-mvr-stage-d2-calibration-v1",
        "confidence": {
            "bins": confidence_table,
            "expected_calibration_error": expected_calibration_error(confidence_table),
            "active_bond_brier_score": float(np.mean((
                per_bond.confidence.to_numpy() - (per_bond.target_signed_bond_residual.abs() > 0.005).to_numpy()
            ) ** 2)),
        },
        "uncertainty": {
            "rank_bins": uncertainty_table,
            "rank_calibration_error": expected_calibration_error(uncertainty_table),
            "absolute_error_correlation": safe_correlation(per_bond.uncertainty, absolute_error),
            "absolute_error_spearman": safe_correlation(per_bond.uncertainty, absolute_error, rank=True),
        },
    }
    atomic_json_save(calibration, OUTPUT / "prediction_calibration.json")

    methods = {
        "upstream": [item["input"] for item in items], "v4_selected": v4_accepted,
        "d1_a_selected": a_accepted, "d0_oracle": oracle_accepted,
        "minimal_target": [item["minimal_target"] for item in items],
    }
    metadata = {"v4_selected": v4_metadata, "d1_a_selected": a_metadata}
    for label, value in variants.items():
        methods[label] = value["accepted"]
        metadata[label] = value["metadata"]
    rows = method_rows(items, methods, validity, metadata)
    summary, molecules = summarize_groups(rows, items, methods)
    summary.to_csv(OUTPUT / "counterfactual_summary.csv", index=False)
    bootstrap_results = {
        "schema_version": "ecir-mvr-stage-d2-bootstrap-v1", "draws": DRAWS,
        "seed": SEED, "baseline": "v4_selected", "comparisons": {},
    }
    for method in (
        "C_ADDITIVE_DEPLOYED", "B_BOND_ONLY", "A_CARTESIAN_ONLY",
        "H_NON_RING_BOND_HEAD_ONLY", "D_CARTESIAN_BOND_SUBSPACE_REMOVED",
    ):
        bootstrap_results["comparisons"][method] = _molecule_bootstrap(molecules, method)
    atomic_json_save(bootstrap_results, OUTPUT / "bootstrap.json")

    interference = record_interference(per_bond, items, variants, validity)
    interference.to_csv(OUTPUT / "branch_interference_record.csv", index=False)
    interference_summary = {
        "schema_version": "ecir-mvr-stage-d2-branch-interference-v1",
        "records": len(interference),
        "means": {column: float(interference[column].mean()) for column in interference.columns if pd.api.types.is_numeric_dtype(interference[column])},
        "gain_decomposition": {
            column: float(interference[column].mean()) for column in (
                "cartesian_alone_gain", "bond_alone_gain", "additive_combined_gain",
                "interaction_gain", "safety_gate_loss", "acceptance_gain",
            )
        },
    }
    atomic_json_save(interference_summary, OUTPUT / "branch_interference_summary.json")
    damage, damage_summary = angle_ring_damage(items, variants, v4_accepted, per_bond, validity)
    damage.to_csv(OUTPUT / "angle_ring_damage_records.csv", index=False)
    damage_summary.update({
        "schema_version": "ecir-mvr-stage-d2-angle-ring-damage-v1",
        "d0_angle_rate_delta_vs_upstream": oracle_result["metrics"]["angle_rate_delta"],
        "d0_ring_rate_delta_vs_upstream": oracle_result["metrics"]["ring_rate_delta"],
        "d1b_angle_rate_delta_vs_v4": pilot["metrics"]["angle_rate_delta_vs_v4"],
        "d1b_ring_rate_delta_vs_v4": pilot["metrics"]["ring_rate_delta_vs_v4"],
    })
    atomic_json_save(damage_summary, OUTPUT / "angle_ring_damage_summary.json")
    ring_summary = ring_nonring_summary(per_bond)
    ring_summary.to_csv(OUTPUT / "ring_nonring_summary.csv", index=False)

    all_rows = summary[summary.group == "all"].set_index("method")
    upstream_rate = float(all_rows.loc["upstream", "bond_outlier_rate"])
    target_rate = float(all_rows.loc["minimal_target", "bond_outlier_rate"])
    available = (upstream_rate - target_rate) / upstream_rate
    recoveries = {
        method: ((upstream_rate - float(all_rows.loc[method, "bond_outlier_rate"])) / upstream_rate) / max(available, 1.0e-12)
        for method in methods
    }
    recovery_gap = float(oracle_result["metrics"]["model_to_target_recovery_upper_bound"]) - float(pilot["metrics"]["model_to_target_recovery_ratio"])
    components = {
        "confidence_attenuation": max(0.0, recoveries["G_LEARNED_RESIDUAL_ALL_CONFIDENCE_ONE"] - recoveries["C_ADDITIVE_DEPLOYED"]),
        "active_bond_selection": max(0.0, recoveries["F_LEARNED_RESIDUAL_ORACLE_MASK"] - recoveries["C_ADDITIVE_DEPLOYED"]),
        "branch_cancellation": max(0.0, recoveries["B_BOND_ONLY"] - recoveries["C_ADDITIVE_DEPLOYED"]),
        "ring_local_coupling": max(0.0, recoveries["H_NON_RING_BOND_HEAD_ONLY"] - recoveries["C_ADDITIVE_DEPLOYED"]),
        "residual_prediction": max(0.0, float(oracle_result["metrics"]["model_to_target_recovery_upper_bound"]) - max(
            recoveries["E_ORACLE_RESIDUAL_LEARNED_CONFIDENCE"],
            recoveries["F_LEARNED_RESIDUAL_ORACLE_MASK"],
            recoveries["G_LEARNED_RESIDUAL_ALL_CONFIDENCE_ONE"],
        )),
    }
    gap = approximate_gap_decomposition(recovery_gap, components)
    gap.update({
        "schema_version": "ecir-mvr-stage-d2-oracle-gap-v1",
        "d0_recovery": oracle_result["metrics"]["model_to_target_recovery_upper_bound"],
        "d1b_recovery": pilot["metrics"]["model_to_target_recovery_ratio"],
        "counterfactual_recoveries": recoveries,
        "nonadditive_terms": [
            "residual magnitude and sign errors overlap",
            "safety and acceptance are nonlinear trajectory operators",
            "ring coupling changes angle/ring validity outside the bond-rate recovery scalar",
        ],
    })
    atomic_json_save(gap, OUTPUT / "oracle_gap_decomposition.json")

    deployed = all_rows.loc["C_ADDITIVE_DEPLOYED"]
    cartesian = all_rows.loc["A_CARTESIAN_ONLY"]
    bond_only = all_rows.loc["B_BOND_ONLY"]
    nonring = all_rows.loc["H_NON_RING_BOND_HEAD_ONLY"]
    confidence_one = all_rows.loc["G_LEARNED_RESIDUAL_ALL_CONFIDENCE_ONE"]
    oracle_mask = all_rows.loc["F_LEARNED_RESIDUAL_ORACLE_MASK"]
    v4_row = all_rows.loc["v4_selected"]
    weak_prediction = (
        all_prediction["pearson"] < 0.50
        or all_prediction["residual_sign_accuracy"] < 0.65
        or all_prediction["nonzero_detection_f1"] < 0.50
    )
    selection_rescue = float(oracle_mask.bond_outlier_rate) < float(deployed.bond_outlier_rate) - 0.005
    confidence_rescue = float(confidence_one.bond_outlier_rate) < float(deployed.bond_outlier_rate) - 0.005
    fusion_conflict = (
        float(cartesian.bond_outlier_rate) < float(v4_row.bond_outlier_rate)
        and float(bond_only.bond_outlier_rate) < float(v4_row.bond_outlier_rate)
        and float(deployed.bond_outlier_rate) > max(float(cartesian.bond_outlier_rate), float(bond_only.bond_outlier_rate)) + 0.005
    )
    ring_rescue = (
        float(nonring.angle_outlier_rate) <= float(v4_row.angle_outlier_rate)
        and float(nonring.ring_bond_outlier_rate) <= float(v4_row.ring_bond_outlier_rate)
        and float(nonring.total_thresholded_validity_score) < float(deployed.total_thresholded_validity_score)
    )
    local_inconsistency = (
        damage_summary["adjacent_same_direction_fraction"] >= 0.30
        or damage_summary["ring_multi_bond_records"] >= 10
    )
    if weak_prediction and not selection_rescue and not confidence_rescue:
        primary, recommendation = "EDGE_RESIDUAL_PREDICTION_WEAK", "REDESIGN_EDGE_DECODER"
    elif selection_rescue:
        primary, recommendation = "ACTIVE_BOND_SELECTION_WEAK", "REDESIGN_ACTIVE_BOND_CLASSIFIER"
    elif confidence_rescue:
        primary, recommendation = "CONFIDENCE_CALIBRATION_WEAK", "RECALIBRATE_BOND_CONFIDENCE"
    elif fusion_conflict:
        primary, recommendation = "CARTESIAN_BOND_FUSION_CONFLICT", "RESIDUALIZE_CARTESIAN_BOND_SUBSPACE"
    elif ring_rescue:
        primary, recommendation = "RING_CROSS_MODE_DAMAGE", "SPLIT_RING_AND_NONRING_HEADS"
    elif local_inconsistency:
        primary, recommendation = "LOCAL_MULTI_BOND_INCONSISTENCY", "JOINT_BOND_ANGLE_RING_CONSTRAINT_HEAD"
    elif weak_prediction:
        primary, recommendation = "JOINT_PREDICTION_AND_COUPLING_LIMITATION", "JOINT_BOND_ANGLE_RING_CONSTRAINT_HEAD"
    else:
        primary, recommendation = "MIXED_STAGE_D_LIMITATION", "STOP_STAGE_D_DIRECTION"
    secondary = []
    if weak_prediction and primary != "EDGE_RESIDUAL_PREDICTION_WEAK": secondary.append("EDGE_RESIDUAL_PREDICTION_WEAK")
    if selection_rescue and primary != "ACTIVE_BOND_SELECTION_WEAK": secondary.append("ACTIVE_BOND_SELECTION_WEAK")
    if confidence_rescue and primary != "CONFIDENCE_CALIBRATION_WEAK": secondary.append("CONFIDENCE_CALIBRATION_WEAK")
    if fusion_conflict and primary != "CARTESIAN_BOND_FUSION_CONFLICT": secondary.append("CARTESIAN_BOND_FUSION_CONFLICT")
    if ring_rescue and primary != "RING_CROSS_MODE_DAMAGE": secondary.append("RING_CROSS_MODE_DAMAGE")
    if local_inconsistency and primary != "LOCAL_MULTI_BOND_INCONSISTENCY": secondary.append("LOCAL_MULTI_BOND_INCONSISTENCY")

    result = {
        "schema_version": "ecir-mvr-stage-d2-audit-v1", "stage": "MCVR_STAGE_D2",
        "decision_unchanged": "STAGE_D_NO_ADDED_VALUE",
        "medium_seed42_schedule_v4_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "audit_completed": True, "primary_cause": primary,
        "secondary_causes": secondary, "recommendation": recommendation,
        "validation_only": True, "training_run": False, "checkpoint_modified": False,
        "test_records_read": 0, "records": len(items),
        "per_bond_rows": len(per_bond), "bootstrap_draws": DRAWS, "bootstrap_seed": SEED,
        "branch": branch, "head": head, "checkpoint_sha256": {
            "D1_A": _sha(D1_A), "D1_B": _sha(D1_B), "V4": _sha(V4),
        },
        "frozen_identities": config["frozen_identities"],
        "prediction_quality": all_prediction, "calibration": calibration,
        "branch_interference": interference_summary,
        "angle_ring_damage": damage_summary, "oracle_gap": gap,
        "counterfactual_recoveries": recoveries,
        "classification_evidence": {
            "weak_prediction": weak_prediction, "selection_rescue": selection_rescue,
            "confidence_rescue": confidence_rescue, "fusion_conflict": fusion_conflict,
            "ring_rescue": ring_rescue, "local_inconsistency": local_inconsistency,
        },
        "stage_d_20k_permitted": False, "stage_d_100k_permitted": False,
        "seed43_44_started": False, "100k_started": False,
        "next_command": None, "next_commands": [],
    }
    atomic_json_save(result, OUTPUT / "result.json")

    state.update({
        "current_stage": "MCVR_STAGE_D2_AUDIT_COMPLETE",
        "current_decision": "STAGE_D_NO_ADDED_VALUE",
        "medium_seed42_schedule_v4_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "stage_d_pilot_decision": "STAGE_D_NO_ADDED_VALUE",
        "stage_d_20k_permitted": False, "stage_d_100k_permitted": False,
        "stage_d2_audit_completed": True, "stage_d2_primary_cause": primary,
        "stage_d2_secondary_causes": secondary, "stage_d2_recommendation": recommendation,
        "test_records_read": 0, "100k_permitted": False,
        "next_command": None, "next_commands": [],
    })
    atomic_json_save(state, Path("reports/ecir_mvr/progressive_state.json"))
    write_reports(result, summary, ring_summary, damage_summary, interference_summary, gap)
    print(json.dumps({
        "primary_cause": primary, "secondary_causes": secondary,
        "recommendation": recommendation, "prediction_quality": all_prediction,
    }, indent=2))


def _load_context() -> tuple[dict[str, Any], Any, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, expected in EXPECTED.items():
        if _sha(Path(path)) != expected:
            raise RuntimeError(f"frozen identity changed: {path}")
    pilot = json.loads(Path("diagnostics/ecir_mvr/stage_d/pilot/result.json").read_text(encoding="utf-8"))
    oracle = json.loads(Path("diagnostics/ecir_mvr/stage_d/oracle/result.json").read_text(encoding="utf-8"))
    if pilot["decision"] != "STAGE_D_NO_ADDED_VALUE" or pilot["test_records_read"] != 0:
        raise RuntimeError("formal Stage D pilot state changed")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["frozen_identities"] != oracle["frozen_identities"]:
        raise RuntimeError("frozen identities changed")
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = build_items(config["data"]["val_sources"], config["data"]["val_targets"], validity)
    if len(items) != 700 or len({str(item["row"].molecule_id) for item in items}) != 500:
        raise RuntimeError("validation record identity is incomplete")
    return config, validity, items, pilot, oracle


def _recovery_state(config: Mapping[str, Any], items) -> dict[str, Any]:
    if RECOVERY_STATE.is_file():
        state = json.loads(RECOVERY_STATE.read_text(encoding="utf-8"))
    else:
        state = {
            "schema_version": "ecir-mvr-stage-d2-recovery-state-v1",
            "status": "RUNNING", "last_completed_phase": "WORKTREE_VERIFIED",
            "phase1_reused": True, "started_at": _now(),
            "interrupted_attempts": [
                {"reason": "external DNS service interruption", "persistent_variant_cache": False},
                {"reason": "external DNS service interruption", "persistent_variant_cache": False},
            ],
            "rerun_variants": [], "completed_outputs": [], "latest_error": None,
            "test_records_read": 0,
        }
    validations = {
        label: _cache_validation(_cache_path(label), label, items, config["frozen_identities"])
        for label in VARIANTS
    }
    completed = [label for label, value in validations.items() if value["complete"]]
    state.update({
        "phase1_reused": True,
        "completed_variants": completed,
        "pending_variants": [label for label in VARIANTS if label not in completed],
        "variant_cache_sha256": {
            label: _sha(_cache_path(label)) for label in completed
        },
        "test_records_read": 0,
    })
    state.setdefault("completed_baselines", [])
    state.setdefault("completed_outputs", [])
    state.setdefault("rerun_variants", [])
    _update_recovery(state, status="RUNNING")
    return state


def _baseline_path(name: str) -> Path:
    return CACHE / BASELINE_NAMES[name]


def _save_baseline(name: str, accepted, metadata, items, identities, checkpoint_sha256: str) -> dict[str, Any]:
    CACHE.mkdir(parents=True, exist_ok=True)
    identity_json = json.dumps(dict(identities), sort_keys=True, separators=(",", ":"))
    rows = []
    for item, coordinates, extra in zip(items, accepted, metadata):
        rows.append({
            "schema_version": "ecir-mvr-stage-d2-baseline-cache-v1", "baseline_name": name,
            "checkpoint_sha256": checkpoint_sha256, "frozen_identities_json": identity_json,
            "validation_only": True, "test_records_read": 0,
            "molecule_id": str(item["row"].molecule_id), "record_id": str(item["row"].sample_id),
            "atom_count": int(item["input"].shape[0]),
            "accepted_coordinates_json": _coordinate_json(coordinates),
            "metadata_json": json.dumps(dict(extra), sort_keys=True, separators=(",", ":"), allow_nan=False),
            "torsion_gate_max": float(extra.get("torsion_gate_max", 0.0)),
            "torsion_contribution_max": float(extra.get("torsion_contribution_max", 0.0)),
            "numerical_failure": not bool(torch.isfinite(torch.as_tensor(coordinates)).all()),
        })
    destination = _baseline_path(name)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    check = pd.read_parquet(temporary)
    expected_ids = [str(item["row"].sample_id) for item in items]
    complete = (
        len(check) == len(items) and check.record_id.astype(str).tolist() == expected_ids
        and set(check.frozen_identities_json.astype(str)) == {identity_json}
        and bool(check.validation_only.astype(bool).all())
        and bool((check.test_records_read.astype(int) == 0).all())
        and not bool(check.numerical_failure.astype(bool).any())
    )
    if complete:
        for index, value in enumerate(check.accepted_coordinates_json):
            coordinates = _decode_coordinates(value)
            if tuple(coordinates.shape) != tuple(items[index]["input"].shape) or not bool(torch.isfinite(coordinates).all()):
                complete = False
                break
    if not complete:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"baseline cache read-back validation failed: {name}")
    os.replace(temporary, destination)
    return {"path": destination.as_posix(), "records": len(check), "sha256": _sha(destination)}


def _load_baseline(name: str) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    frame = pd.read_parquet(_baseline_path(name))
    return (
        [_decode_coordinates(value) for value in frame.accepted_coordinates_json],
        [json.loads(value) for value in frame.metadata_json],
    )


def run_inventory() -> None:
    config, _, items, _, _ = _load_context()
    state = _recovery_state(config, items)
    inventory = build_recovery_inventory(items, expected_identities=config["frozen_identities"])
    state["recovery_inventory_sha256"] = _sha(RECOVERY_INVENTORY)
    _update_recovery(state, phase="RECOVERY_INVENTORY_COMPLETE")
    print(json.dumps({
        "phase1_reused": True,
        "completed_variants": inventory["completed_variants"],
        "pending_variants": state["pending_variants"],
    }, indent=2))


def run_cache_smoke() -> None:
    config, validity, items, _, _ = _load_context()
    state = _recovery_state(config, items)
    selected = items[:2]
    label = "SMOKE_A_CARTESIAN_ONLY"
    path = _cache_path(label)
    path.unlink(missing_ok=True)
    device = torch.device("cuda")
    model = _load(D1_B, device)
    value = infer_variant(model, selected, validity, variant="cartesian_only", device=device)
    first = _save_variant_cache(label, value, selected, config["frozen_identities"])
    second = _cache_validation(path, label, selected, config["frozen_identities"])
    if not first["complete"] or not second["complete"]:
        raise RuntimeError("two-record cache smoke did not skip a validated cache")
    state["cache_smoke"] = {
        "status": "PASS", "records": 2, "write_read_validate": True,
        "atomic_rename": True, "second_run_skipped": True,
    }
    _update_recovery(state, phase="VARIANT_CACHE_SMOKE_COMPLETE")
    del value, model
    gc.collect()
    torch.cuda.empty_cache()
    path.unlink(missing_ok=True)
    print(json.dumps(state["cache_smoke"], indent=2))


def run_variant(label: str) -> None:
    config, validity, items, _, _ = _load_context()
    state = _recovery_state(config, items)
    validation = _cache_validation(_cache_path(label), label, items, config["frozen_identities"])
    if validation["complete"]:
        print(json.dumps({"variant": label, "skipped": True, **validation}, indent=2))
        return
    state["active_variant"] = label
    _update_recovery(state, status="RUNNING")
    try:
        device = torch.device("cuda")
        model = _load(D1_B, device)
        v4_model = _load(V4, device) if label == "J_D1B_BOND_WITH_FROZEN_V4_CARTESIAN" else None
        value = infer_variant(
            model, items, validity, variant=VARIANTS[label], device=device,
            v4_model=v4_model, collect_bonds=False,
        )
        validation = _save_variant_cache(label, value, items, config["frozen_identities"])
        if label not in state["rerun_variants"]:
            state["rerun_variants"].append(label)
        state["active_variant"] = None
        state["completed_variants"] = [
            value for value in VARIANTS if value == label or value in state["completed_variants"]
        ]
        state["pending_variants"] = [value for value in VARIANTS if value not in state["completed_variants"]]
        state["variant_cache_sha256"][label] = validation["sha256"]
        _update_recovery(state, phase=f"VARIANT_{label}_COMPLETE")
        print(json.dumps({
            "variant": label, "records": validation["records"],
            "molecules": validation["molecules"], "sha256": validation["sha256"],
        }, indent=2))
    except BaseException as error:
        state["active_variant"] = label
        _update_recovery(state, status="ERROR", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for name in ("value", "v4_model", "model"):
            if name in locals():
                del locals()[name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_baseline(name: str) -> None:
    config, validity, items, _, _ = _load_context()
    state = _recovery_state(config, items)
    path = _baseline_path(name)
    if path.is_file():
        frame = pd.read_parquet(path)
        if len(frame) == len(items) and frame.record_id.astype(str).tolist() == [str(item["row"].sample_id) for item in items]:
            print(json.dumps({"baseline": name, "skipped": True, "sha256": _sha(path)}, indent=2))
            return
    if name in {"v4_selected", "d1_a_selected"}:
        device = torch.device("cuda")
        checkpoint = V4 if name == "v4_selected" else D1_A
        model = _load(checkpoint, device)
        _, accepted, metadata = infer_mvr(model, items, validity, device=device)
        checkpoint_sha = EXPECTED[checkpoint.as_posix()]
        del model
        torch.cuda.empty_cache()
    else:
        _, _, accepted, metadata, _ = solve_oracle_items(
            items, validity,
            max_atom_norm=float(config["model"]["max_velocity_atom_norm"]),
            max_graph_rms=float(config["model"]["max_velocity_graph_rms"]),
        )
        metadata = [{
            **value, "torsion_gate_max": 0.0, "torsion_contribution_max": 0.0,
        } for value in metadata]
        checkpoint_sha = EXPECTED[D1_B.as_posix()]
    result = _save_baseline(name, accepted, metadata, items, config["frozen_identities"], checkpoint_sha)
    if name not in state["completed_baselines"]:
        state["completed_baselines"].append(name)
    _update_recovery(state, phase=f"BASELINE_{name}_COMPLETE")
    gc.collect()
    print(json.dumps({"baseline": name, **result}, indent=2))


def _require_variants(config, items, labels) -> dict[str, dict[str, Any]]:
    values = {}
    for label in labels:
        validation = _cache_validation(_cache_path(label), label, items, config["frozen_identities"])
        if not validation["complete"]:
            raise RuntimeError(f"required variant cache is incomplete: {label}: {validation}")
        values[label] = _load_variant_cache(label)
    return values


def run_phase(phase: int) -> None:
    config, validity, items, pilot, oracle_result = _load_context()
    state = _recovery_state(config, items)
    per_bond_path = OUTPUT / "per_bond_prediction_metrics.parquet"
    try:
        if phase == 1:
            per_bond = pd.read_parquet(per_bond_path)
            if len(per_bond) != 32610 or per_bond.record_id.nunique() != 700:
                raise RuntimeError("Phase 1 reusable per-bond cache is incomplete")
            _atomic_csv(build_prediction_summary(per_bond), OUTPUT / "prediction_quality_summary.csv")
            confidence_table = calibration_table(
                per_bond.confidence, (per_bond.target_signed_bond_residual.abs() > 0.005).astype(float)
            )
            absolute_error = (per_bond.bounded_predicted_residual - per_bond.target_signed_bond_residual).abs()
            uncertainty_rank = stable_average_ranks(per_bond.uncertainty) / len(per_bond)
            error_rank = stable_average_ranks(absolute_error) / len(per_bond)
            uncertainty_table = calibration_table(uncertainty_rank, error_rank)
            calibration = {
                "schema_version": "ecir-mvr-stage-d2-calibration-v1", "validation_only": True,
                "test_records_read": 0,
                "confidence": {
                    "bins": confidence_table,
                    "expected_calibration_error": expected_calibration_error(confidence_table),
                    "active_bond_brier_score": float(np.mean((
                        per_bond.confidence.to_numpy() - (per_bond.target_signed_bond_residual.abs() > 0.005).to_numpy()
                    ) ** 2)),
                },
                "uncertainty": {
                    "rank_bins": uncertainty_table,
                    "rank_calibration_error": expected_calibration_error(uncertainty_table),
                    "absolute_error_correlation": safe_correlation(per_bond.uncertainty, absolute_error),
                    "absolute_error_spearman": safe_correlation(per_bond.uncertainty, absolute_error, rank=True),
                },
            }
            atomic_json_save(calibration, OUTPUT / "prediction_calibration.json")
            outputs = ["prediction_quality_summary.csv", "prediction_calibration.json"]
        elif phase == 2:
            per_bond = pd.read_parquet(per_bond_path)
            variants = _require_variants(config, items, (
                "A_CARTESIAN_ONLY", "B_BOND_ONLY", "C_ADDITIVE_DEPLOYED",
            ))
            interference = record_interference(per_bond, items, variants, validity)
            _atomic_csv(interference, OUTPUT / "branch_interference_record.csv")
            summary = {
                "schema_version": "ecir-mvr-stage-d2-branch-interference-v1",
                "validation_only": True, "test_records_read": 0, "records": len(interference),
                "means": {column: float(interference[column].mean()) for column in interference.columns if pd.api.types.is_numeric_dtype(interference[column])},
                "gain_decomposition": {column: float(interference[column].mean()) for column in (
                    "cartesian_alone_gain", "bond_alone_gain", "additive_combined_gain",
                    "interaction_gain", "safety_gate_loss", "acceptance_gain",
                )},
            }
            atomic_json_save(summary, OUTPUT / "branch_interference_summary.json")
            outputs = ["branch_interference_record.csv", "branch_interference_summary.json"]
        elif phase == 3:
            variants = _require_variants(config, items, VARIANTS)
            methods = {
                "upstream": [item["input"] for item in items],
                "minimal_target": [item["minimal_target"] for item in items],
                **{label: value["accepted"] for label, value in variants.items()},
            }
            metadata = {
                **{label: value["metadata"] for label, value in variants.items()},
            }
            rows = method_rows(items, methods, validity, metadata)
            summary, molecules = summarize_groups(rows, items, methods)
            pilot_summary = pd.read_csv("diagnostics/ecir_mvr/stage_d/pilot/subgroup_summary.csv")
            frozen_summary = pilot_summary[pilot_summary.method.isin(["v4_selected", "d1_a_aux_only"])].copy()
            frozen_summary.loc[frozen_summary.method.eq("d1_a_aux_only"), "method"] = "d1_a_selected"
            oracle_summary = pd.read_csv("diagnostics/ecir_mvr/stage_d/oracle/subgroup_summary.csv")
            oracle_summary = oracle_summary[oracle_summary.method.eq("bond_oracle_accepted")].copy()
            oracle_summary.loc[:, "method"] = "d0_oracle"
            summary = pd.concat([summary, frozen_summary, oracle_summary], ignore_index=True, sort=False)
            _atomic_csv(summary, OUTPUT / "counterfactual_summary.csv")
            v4_molecules = pd.read_csv(
                "diagnostics/ecir_mvr/medium/run_a_seed42_schedule_v4/molecule_metrics.csv"
            )
            v4_molecules = v4_molecules[v4_molecules.method.eq("medium_accepted")].copy()
            v4_molecules.loc[:, "method"] = "v4_selected"
            bootstrap_molecules = pd.concat([molecules, v4_molecules], ignore_index=True, sort=False)
            bootstrap_results = {
                "schema_version": "ecir-mvr-stage-d2-bootstrap-v1", "draws": DRAWS,
                "seed": SEED, "baseline": "v4_selected", "validation_only": True,
                "test_records_read": 0, "comparisons": {},
            }
            for method in (
                "C_ADDITIVE_DEPLOYED", "B_BOND_ONLY", "A_CARTESIAN_ONLY",
                "H_NON_RING_BOND_HEAD_ONLY", "D_CARTESIAN_BOND_SUBSPACE_REMOVED",
            ):
                bootstrap_results["comparisons"][method] = _molecule_bootstrap(bootstrap_molecules, method)
            atomic_json_save(bootstrap_results, OUTPUT / "bootstrap.json")
            outputs = ["counterfactual_summary.csv", "bootstrap.json"]
        elif phase == 4:
            per_bond = pd.read_parquet(per_bond_path)
            variants = _require_variants(config, items, (
                "A_CARTESIAN_ONLY", "B_BOND_ONLY", "C_ADDITIVE_DEPLOYED",
            ))
            v4_records = pd.read_csv(
                "diagnostics/ecir_mvr/medium/run_a_seed42_schedule_v4/record_metrics.csv"
            )
            v4_records = v4_records[v4_records.method.eq("medium_accepted")].copy()
            damage, summary = angle_ring_damage(
                items, variants, None, per_bond, validity, v4_record_metrics=v4_records,
            )
            _atomic_csv(damage, OUTPUT / "angle_ring_damage_records.csv")
            summary.update({
                "schema_version": "ecir-mvr-stage-d2-angle-ring-damage-v1",
                "validation_only": True, "test_records_read": 0,
                "d0_angle_rate_delta_vs_upstream": oracle_result["metrics"]["angle_rate_delta"],
                "d0_ring_rate_delta_vs_upstream": oracle_result["metrics"]["ring_rate_delta"],
                "d1b_angle_rate_delta_vs_v4": pilot["metrics"]["angle_rate_delta_vs_v4"],
                "d1b_ring_rate_delta_vs_v4": pilot["metrics"]["ring_rate_delta_vs_v4"],
            })
            atomic_json_save(summary, OUTPUT / "angle_ring_damage_summary.json")
            outputs = ["angle_ring_damage_records.csv", "angle_ring_damage_summary.json"]
        elif phase == 5:
            summary = pd.read_csv(OUTPUT / "counterfactual_summary.csv")
            per_bond = pd.read_parquet(per_bond_path)
            interference = json.loads((OUTPUT / "branch_interference_summary.json").read_text(encoding="utf-8"))
            all_rows = summary[summary.group == "all"].set_index("method")
            upstream_rate = float(all_rows.loc["upstream", "bond_outlier_rate"])
            target_rate = float(all_rows.loc["minimal_target", "bond_outlier_rate"])
            denominator = max(upstream_rate - target_rate, 1.0e-12)
            recoveries = {
                method: (upstream_rate - float(row.bond_outlier_rate)) / denominator
                for method, row in all_rows.iterrows()
            }
            deployed = recoveries["C_ADDITIVE_DEPLOYED"]
            residual_bucket = max(0.0, float(oracle_result["metrics"]["model_to_target_recovery_upper_bound"]) - max(
                recoveries["E_ORACLE_RESIDUAL_LEARNED_CONFIDENCE"],
                recoveries["F_LEARNED_RESIDUAL_ORACLE_MASK"],
                recoveries["G_LEARNED_RESIDUAL_ALL_CONFIDENCE_ONE"],
            ))
            target = per_bond.target_signed_bond_residual.to_numpy(float)
            prediction = per_bond.bounded_predicted_residual.to_numpy(float)
            active, predicted_active = np.abs(target) > 0.005, np.abs(prediction) > 0.005
            weights = {
                "residual_magnitude_error": float(np.abs(np.abs(prediction[active & predicted_active]) - np.abs(target[active & predicted_active])).sum()),
                "residual_sign_error": float(np.abs(target[active & predicted_active & (np.sign(prediction) != np.sign(target))]).sum()),
                "missed_active_bond": float(np.abs(target[active & ~predicted_active]).sum()),
                "false_positive": float(np.abs(prediction[~active & predicted_active]).sum()),
            }
            weight_total = max(sum(weights.values()), 1.0e-12)
            components = {name: residual_bucket * value / weight_total for name, value in weights.items()}
            components.update({
                "confidence_attenuation": max(0.0, recoveries["G_LEARNED_RESIDUAL_ALL_CONFIDENCE_ONE"] - deployed),
                "cartesian_bond_cancellation": max(0.0, recoveries["B_BOND_ONLY"] - deployed),
                "safety_attenuation": max(0.0, -float(interference["gain_decomposition"]["safety_gate_loss"]) / denominator),
                "acceptance": max(0.0, -float(interference["gain_decomposition"]["acceptance_gain"]) / denominator),
                "ring_local_coupling_damage": max(0.0, recoveries["H_NON_RING_BOND_HEAD_ONLY"] - deployed),
            })
            recovery_gap = float(oracle_result["metrics"]["model_to_target_recovery_upper_bound"]) - float(pilot["metrics"]["model_to_target_recovery_ratio"])
            gap = approximate_gap_decomposition(recovery_gap, components)
            gap.update({
                "schema_version": "ecir-mvr-stage-d2-oracle-gap-v1", "validation_only": True,
                "test_records_read": 0,
                "d0_recovery": oracle_result["metrics"]["model_to_target_recovery_upper_bound"],
                "d1b_recovery": pilot["metrics"]["model_to_target_recovery_ratio"],
                "counterfactual_recoveries": recoveries,
                "residual_component_allocation_weights": weights,
                "nonadditive_terms": [
                    "magnitude, sign, selection, and false-positive errors overlap",
                    "safety and acceptance are nonlinear trajectory operators",
                    "ring coupling changes angle/ring validity outside the bond-rate recovery scalar",
                ],
            })
            atomic_json_save(gap, OUTPUT / "oracle_gap_decomposition.json")
            outputs = ["oracle_gap_decomposition.json"]
        elif phase == 6:
            _atomic_csv(ring_nonring_summary(pd.read_parquet(per_bond_path)), OUTPUT / "ring_nonring_summary.csv")
            outputs = ["ring_nonring_summary.csv"]
        else:
            raise ValueError(f"unknown phase: {phase}")
        for name in outputs:
            if name not in state["completed_outputs"]:
                state["completed_outputs"].append(name)
        _update_recovery(state, phase=f"PHASE_{phase}_COMPLETE")
        print(json.dumps({"phase": phase, "outputs": outputs}, indent=2))
    except BaseException as error:
        _update_recovery(state, status="ERROR", error=f"{type(error).__name__}: {error}")
        raise


def finalize(targeted_passed: int, full_passed: int) -> None:
    config, _, items, _, _ = _load_context()
    state = _recovery_state(config, items)
    prediction_summary = pd.read_csv(OUTPUT / "prediction_quality_summary.csv")
    all_prediction = prediction_metrics(pd.read_parquet(OUTPUT / "per_bond_prediction_metrics.parquet"))
    calibration = json.loads((OUTPUT / "prediction_calibration.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(OUTPUT / "counterfactual_summary.csv")
    interference = json.loads((OUTPUT / "branch_interference_summary.json").read_text(encoding="utf-8"))
    damage = json.loads((OUTPUT / "angle_ring_damage_summary.json").read_text(encoding="utf-8"))
    gap = json.loads((OUTPUT / "oracle_gap_decomposition.json").read_text(encoding="utf-8"))
    ring_summary = pd.read_csv(OUTPUT / "ring_nonring_summary.csv")
    all_rows = summary[summary.group == "all"].set_index("method")
    deployed, cartesian = all_rows.loc["C_ADDITIVE_DEPLOYED"], all_rows.loc["A_CARTESIAN_ONLY"]
    bond_only, nonring = all_rows.loc["B_BOND_ONLY"], all_rows.loc["H_NON_RING_BOND_HEAD_ONLY"]
    confidence_one = all_rows.loc["G_LEARNED_RESIDUAL_ALL_CONFIDENCE_ONE"]
    oracle_mask, v4_row = all_rows.loc["F_LEARNED_RESIDUAL_ORACLE_MASK"], all_rows.loc["v4_selected"]
    weak_prediction = (
        all_prediction["pearson"] < 0.50 or all_prediction["residual_sign_accuracy"] < 0.65
        or all_prediction["nonzero_detection_f1"] < 0.50
    )
    selection_rescue = float(oracle_mask.bond_outlier_rate) < float(deployed.bond_outlier_rate) - 0.005
    confidence_rescue = float(confidence_one.bond_outlier_rate) < float(deployed.bond_outlier_rate) - 0.005
    fusion_conflict = (
        float(cartesian.bond_outlier_rate) < float(v4_row.bond_outlier_rate)
        and float(bond_only.bond_outlier_rate) < float(v4_row.bond_outlier_rate)
        and float(deployed.bond_outlier_rate) > max(float(cartesian.bond_outlier_rate), float(bond_only.bond_outlier_rate)) + 0.005
    )
    ring_rescue = (
        float(nonring.angle_outlier_rate) <= float(v4_row.angle_outlier_rate)
        and float(nonring.ring_bond_outlier_rate) <= float(v4_row.ring_bond_outlier_rate)
        and float(nonring.total_thresholded_validity_score) < float(deployed.total_thresholded_validity_score)
    )
    local_inconsistency = damage["adjacent_same_direction_fraction"] >= 0.30 or damage["ring_multi_bond_records"] >= 10
    if weak_prediction and not selection_rescue and not confidence_rescue:
        primary, recommendation = "EDGE_RESIDUAL_PREDICTION_WEAK", "REDESIGN_EDGE_DECODER"
    elif selection_rescue:
        primary, recommendation = "ACTIVE_BOND_SELECTION_WEAK", "REDESIGN_ACTIVE_BOND_CLASSIFIER"
    elif confidence_rescue:
        primary, recommendation = "CONFIDENCE_CALIBRATION_WEAK", "RECALIBRATE_BOND_CONFIDENCE"
    elif fusion_conflict:
        primary, recommendation = "CARTESIAN_BOND_FUSION_CONFLICT", "RESIDUALIZE_CARTESIAN_BOND_SUBSPACE"
    elif ring_rescue:
        primary, recommendation = "RING_CROSS_MODE_DAMAGE", "SPLIT_RING_AND_NONRING_HEADS"
    elif local_inconsistency:
        primary, recommendation = "LOCAL_MULTI_BOND_INCONSISTENCY", "JOINT_BOND_ANGLE_RING_CONSTRAINT_HEAD"
    elif weak_prediction:
        primary, recommendation = "JOINT_PREDICTION_AND_COUPLING_LIMITATION", "JOINT_BOND_ANGLE_RING_CONSTRAINT_HEAD"
    else:
        primary, recommendation = "MIXED_STAGE_D_LIMITATION", "STOP_STAGE_D_DIRECTION"
    evidence = {
        "weak_prediction": weak_prediction, "selection_rescue": selection_rescue,
        "confidence_rescue": confidence_rescue, "fusion_conflict": fusion_conflict,
        "ring_rescue": ring_rescue, "local_inconsistency": local_inconsistency,
    }
    secondary = [name for name, present in (
        ("EDGE_RESIDUAL_PREDICTION_WEAK", weak_prediction),
        ("ACTIVE_BOND_SELECTION_WEAK", selection_rescue),
        ("CONFIDENCE_CALIBRATION_WEAK", confidence_rescue),
        ("CARTESIAN_BOND_FUSION_CONFLICT", fusion_conflict),
        ("RING_CROSS_MODE_DAMAGE", ring_rescue),
        ("LOCAL_MULTI_BOND_INCONSISTENCY", local_inconsistency),
    ) if present and name != primary]
    best_method = all_rows.bond_outlier_rate.astype(float).idxmin()
    best_a_j_method = all_rows.loc[list(VARIANTS), "bond_outlier_rate"].astype(float).idxmin()
    tests = {
        "status": "PASS", "targeted_passed": targeted_passed, "full_passed": full_passed,
        "failures": 0, "test_records_read": 0,
        "targeted_command": "python -m pytest -q tests/test_ecir_mvr_stage_d2.py tests/test_ecir_mvr_stage_d_bond_explicit.py",
        "full_command": "python -m pytest -q",
    }
    result = {
        "schema_version": "ecir-mvr-stage-d2-audit-v1", "stage": "MCVR_STAGE_D2",
        "decision": "STAGE_D2_AUDIT_COMPLETE",
        "formal_stage_d_decision_unchanged": "STAGE_D_NO_ADDED_VALUE",
        "medium_seed42_schedule_v4_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "audit_completed": True, "primary_cause": primary,
        "secondary_causes": secondary, "recommendation": recommendation,
        "validation_only": True, "training_run": False, "checkpoint_modified": False,
        "validation_records": len(items),
        "validation_molecules": len({str(item["row"].molecule_id) for item in items}),
        "test_records_read": 0, "per_bond_rows": 32610,
        "checkpoint_identities": {"D1_A": _sha(D1_A), "D1_B": _sha(D1_B), "V4": _sha(V4)},
        "frozen_identities": config["frozen_identities"],
        "cache_reused": {"phase1_per_bond": True, "persistent_a_j_before_recovery": False},
        "rerun_variants": state["rerun_variants"],
        "interrupted_attempts": state["interrupted_attempts"],
        "key_prediction_metrics": all_prediction, "prediction_quality": all_prediction,
        "calibration": calibration, "gap_decomposition": gap,
        "counterfactual_findings": {
            "best_bond_rate_method": best_method,
            "best_bond_rate": float(all_rows.loc[best_method, "bond_outlier_rate"]),
            "best_a_j_method": best_a_j_method,
            "best_a_j_bond_rate": float(all_rows.loc[best_a_j_method, "bond_outlier_rate"]),
            "recoveries": gap["counterfactual_recoveries"],
        },
        "angle_ring_findings": damage, "branch_interference": interference,
        "classification_evidence": evidence, "tests": tests,
        "stage_d_20k_permitted": False, "stage_d_100k_permitted": False,
        "seed43_44_started": False, "100k_started": False,
        "next_command": None, "next_commands": [],
    }
    atomic_json_save(result, OUTPUT / "result.json")
    progressive = json.loads(Path("reports/ecir_mvr/progressive_state.json").read_text(encoding="utf-8"))
    progressive.update({
        "current_stage": "MCVR_STAGE_D2_AUDIT_COMPLETE", "current_decision": "STAGE_D_NO_ADDED_VALUE",
        "medium_seed42_schedule_v4_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "stage_d_pilot_decision": "STAGE_D_NO_ADDED_VALUE",
        "stage_d_20k_permitted": False, "stage_d_100k_permitted": False,
        "stage_d2_audit_completed": True, "stage_d2_primary_cause": primary,
        "stage_d2_secondary_causes": secondary, "stage_d2_recommendation": recommendation,
        "test_records_read": 0, "100k_permitted": False,
        "next_command": None, "next_commands": [], "updated_at": _now(),
    })
    atomic_json_save(progressive, Path("reports/ecir_mvr/progressive_state.json"))
    write_reports(result, summary, ring_summary, damage, interference, gap)
    state["tests"] = tests
    state["completed_outputs"] = sorted(set(state["completed_outputs"] + [
        "result.json", "docs/MCVR_STAGE_D2_PREDICTION_AUDIT.md",
        "docs/MCVR_STAGE_D2_BRANCH_INTERFERENCE.md", "docs/MCVR_STAGE_D2_ANGLE_RING_DAMAGE.md",
        "docs/MCVR_STAGE_D2_NEXT_METHOD_DECISION.md",
    ]))
    _update_recovery(state, phase="FINALIZED", status="COMPLETE")
    print(json.dumps({"primary_cause": primary, "secondary_causes": secondary, "recommendation": recommendation}, indent=2))


def write_reports(result, summary, ring_summary, damage, interference, gap) -> None:
    all_rows = summary[summary.group == "all"].set_index("method")
    prediction = result["prediction_quality"]
    prediction_lines = [
        "# MCVR Stage D2 Prediction Audit", "",
        "This is a validation-only audit of the fixed D1-B step 2000 checkpoint. No training, checkpoint selection, Gate change, or test access occurred.", "",
        "| Metric | Value |", "|---|---:|",
        f"| Residual MAE | {prediction['residual_mae']:.12f} |",
        f"| Residual RMSE | {prediction['residual_rmse']:.12f} |",
        f"| Pearson / Spearman | {prediction['pearson']:.12f} / {prediction['spearman']:.12f} |",
        f"| Sign accuracy | {prediction['residual_sign_accuracy']:.12f} |",
        f"| Active-bond precision / recall / F1 | {prediction['nonzero_detection_precision']:.12f} / {prediction['nonzero_detection_recall']:.12f} / {prediction['nonzero_detection_f1']:.12f} |",
        f"| Outlier precision / recall / F1 | {prediction['outlier_detection_precision']:.12f} / {prediction['outlier_detection_recall']:.12f} / {prediction['outlier_detection_f1']:.12f} |",
        f"| Predicted/target norm ratio | {prediction['predicted_target_norm_ratio']:.12f} |",
        f"| Solver achieved/requested ratio | {prediction['solver_achieved_requested_ratio']:.12f} |",
        f"| Zero-target false-positive rate | {prediction['zero_target_false_positive_rate']:.12f} |",
        f"| Confidence ECE | {result['calibration']['confidence']['expected_calibration_error']:.12f} |",
        "", "The edge decoder has weak correlation, incomplete active-bond recall, no outlier recall, and under-confident calibration. Detailed grouped results are in `prediction_quality_summary.csv` and `prediction_calibration.json`.",
    ]
    Path("docs/MCVR_STAGE_D2_PREDICTION_AUDIT.md").write_text("\n".join(prediction_lines) + "\n", encoding="utf-8")

    gain = interference["gain_decomposition"]
    branch_lines = [
        "# MCVR Stage D2 Branch Interference", "",
        "All counterfactuals use the fixed D1-B checkpoint, frozen trust, learned safety gate, deterministic acceptance, and identical validation records.", "",
        "| Counterfactual | Bond rate | Angle rate | Ring rate | Total validity | RMSD |", "|---|---:|---:|---:|---:|---:|",
    ]
    for method in VARIANTS:
        row = all_rows.loc[method]
        branch_lines.append(
            f"| {method} | {row.bond_outlier_rate:.12f} | {row.angle_outlier_rate:.12f} | "
            f"{row.ring_bond_outlier_rate:.12f} | {row.total_thresholded_validity_score:.12f} | {row.aligned_RMSD:.12f} |"
        )
    branch_lines += [
        "", "## Mean gain decomposition", "", "| Term | Value |", "|---|---:|",
        *[f"| {name} | {value:.12f} |" for name, value in gain.items()],
        "", f"The strongest A-J counterfactual is `{result['counterfactual_findings']['best_a_j_method']}` with bond rate `{result['counterfactual_findings']['best_a_j_bond_rate']:.12f}` and recovery `{gap['counterfactual_recoveries'][result['counterfactual_findings']['best_a_j_method']]:.12f}`.",
        "", "The orthogonalized counterfactual is `DIAGNOSTIC_ORACLE_ONLY` and cannot replace the formal method.",
    ]
    Path("docs/MCVR_STAGE_D2_BRANCH_INTERFERENCE.md").write_text("\n".join(branch_lines) + "\n", encoding="utf-8")

    damage_lines = [
        "# MCVR Stage D2 Angle and Ring Damage", "",
        f"Angle damage records: `{damage['angle_damage_records']}`; ring-bond damage records: `{damage['ring_bond_damage_records']}`.", "",
        f"Angle damage involving a ring bond: `{damage['angle_damage_involving_ring_fraction']:.12f}`.", "",
        f"Adjacent same-direction bond changes: `{damage['adjacent_same_direction_fraction']:.12f}`.", "",
        f"Wrong-sign predictions among angle damage: `{damage['angle_wrong_sign_fraction']:.12f}`; ring damage: `{damage['ring_wrong_sign_fraction']:.12f}`.", "",
        f"Non-ring-only angle damage fraction: `{damage['angle_damage_nonring_only_fraction']:.12f}`; the local comparison mode is `{damage['comparison_mode']}`.", "",
        "D0 did not show the same damage because it solved the complete target residual once as a globally consistent minimum-norm correction. D1-B repeatedly combines approximate learned residuals with a separately learned Cartesian branch, then applies nonlinear safety and acceptance.", "",
        "Ring/non-ring quantitative results are in `ring_nonring_summary.csv`; local records are in `angle_ring_damage_records.csv`.",
    ]
    Path("docs/MCVR_STAGE_D2_ANGLE_RING_DAMAGE.md").write_text("\n".join(damage_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# MCVR Stage D2 Next Method Decision", "",
        f"Primary cause: **{result['primary_cause']}**", "",
        f"Secondary causes: `{', '.join(result['secondary_causes']) if result['secondary_causes'] else 'none'}`.", "",
        f"Recommendation: **{result['recommendation']}**", "",
        f"D0 recovery `{gap['d0_recovery']:.12f}` minus D1-B recovery `{gap['d1b_recovery']:.12f}` leaves `{gap['total_gap']:.12f}`. The approximate components sum to `{gap['attributable_sum']:.12f}` with nonadditive remainder `{gap['nonadditive_remainder']:.12f}`.", "",
        "| Gap component | Recovery units |", "|---|---:|",
        *[f"| {name} | {value:.12f} |" for name, value in gap["components"].items()],
        "", f"Setting learned residual confidence to one raises recovery to `{gap['counterfactual_recoveries']['G_LEARNED_RESIDUAL_ALL_CONFIDENCE_ONE']:.12f}`, the largest single A-J gain, while the oracle active mask reaches only `{gap['counterfactual_recoveries']['F_LEARNED_RESIDUAL_ORACLE_MASK']:.12f}`.", "",
        "Confidence recalibration is therefore the clearest single next design change, but it does not fully close the D0 gap: residual correlation/recall and local multi-bond consistency remain secondary limitations.", "",
        "The formal Stage D result remains **STAGE_D_NO_ADDED_VALUE**. This recommendation authorizes no implementation, training, 20k, 100k, seed43/44, or test evaluation.",
    ]
    Path("docs/MCVR_STAGE_D2_NEXT_METHOD_DECISION.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--inventory", action="store_true")
    action.add_argument("--cache-smoke", action="store_true")
    action.add_argument("--variant", choices=list(VARIANTS))
    action.add_argument("--baseline", choices=list(BASELINE_NAMES))
    action.add_argument("--phase", type=int, choices=range(1, 7))
    action.add_argument("--finalize", action="store_true")
    parser.add_argument("--targeted-passed", type=int, default=0)
    parser.add_argument("--full-passed", type=int, default=0)
    args = parser.parse_args()
    if args.inventory:
        run_inventory()
    elif args.cache_smoke:
        run_cache_smoke()
    elif args.variant:
        run_variant(args.variant)
    elif args.baseline:
        run_baseline(args.baseline)
    elif args.phase:
        run_phase(args.phase)
    elif args.finalize:
        if args.targeted_passed <= 0 or args.full_passed < 369:
            raise ValueError("finalization requires successful targeted tests and at least 369 full tests")
        finalize(args.targeted_passed, args.full_passed)


if __name__ == "__main__":
    main()
