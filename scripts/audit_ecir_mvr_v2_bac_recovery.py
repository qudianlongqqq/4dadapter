#!/usr/bin/env python3
"""Read-only Phase-1 diagnostics for the V2-BAC recovery cohort."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import torch
import yaml
from torch import Tensor
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.bac_constraints import (  # noqa: E402
    angle_equivariant_directions,
    canonical_constraint_fields,
    sparse_clash_edges,
    standardized_interval_residual,
)
from etflow.ecir.bac_safety import BACSafetyConfig, evaluate_bac_proposal  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.geometry import bond_angles, bond_lengths  # noqa: E402
from etflow.ecir.mvr_dataset import MCVRMixedDataset  # noqa: E402
from etflow.ecir.mvr_v2_bac import MCVRBACModel  # noqa: E402
from etflow.ecir.mvr_v2_bac_loss import MCVRBACLoss  # noqa: E402
from etflow.ecir.run_a_evaluation import build_items  # noqa: E402


SCHEMA_VERSION = "mcvr-v2-bac-recovery-audit-v1"
EXPECTED_DIAGNOSTIC_RECORDS = 256
EXPECTED_DIAGNOSTIC_MOLECULES = 128
EXPECTED_MANIFEST_IDENTITY = (
    "3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _summary(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if not tensor.numel():
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "max": float(tensor.max()),
    }


def _norm(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().to(torch.float64)))


def _cosine(left: Tensor, right: Tensor) -> float:
    left = left.detach().reshape(-1).to(torch.float64)
    right = right.detach().reshape(-1).to(torch.float64)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 1.0e-15:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _scatter_components(
    atom_count: int,
    indices: Tensor,
    directions: tuple[Tensor, ...],
    weights: Tensor,
    template: Tensor,
) -> Tensor:
    result = torch.zeros_like(template)
    counts = template.new_zeros(atom_count)
    for column, direction in enumerate(directions):
        atoms = indices[:, column]
        result.index_add_(0, atoms, weights[:, None] * direction)
        counts.index_add_(0, atoms, torch.ones_like(weights))
    return result / counts.clamp_min(1.0)[:, None]


def _local_target_components(item: Mapping[str, Any], validity: ChemicalValidity):
    source = item["input"]
    static = canonical_constraint_fields(
        validity,
        item["record"],
        source_identity_sha256="recovery-diagnostic-only",
    )
    bonds = static["active_bond_constraint_index"]
    bond_ranges = static["bond_allowed_range"]
    angles = static["active_angle_constraint_index"].t()
    angle_ranges = static["angle_allowed_range"]
    bond_residual, bond_severity = standardized_interval_residual(
        bond_lengths(source, bonds), bond_ranges
    )
    angle_residual, angle_severity = standardized_interval_residual(
        bond_angles(source, angles), angle_ranges
    )

    bond_vector = torch.zeros_like(source)
    if bonds.numel():
        left, right = bonds
        relative = source[left] - source[right]
        distance = torch.linalg.vector_norm(relative, dim=-1).clamp_min(1.0e-8)
        direction = relative / distance[:, None]
        signed_angstrom = bond_residual * bond_ranges[:, 2]
        bond_vector.index_add_(0, left, -0.5 * signed_angstrom[:, None] * direction)
        bond_vector.index_add_(0, right, 0.5 * signed_angstrom[:, None] * direction)

    angle_vector = torch.zeros_like(source)
    if angles.numel():
        signed_radian = angle_residual * angle_ranges[:, 2]
        angle_vector = _scatter_components(
            source.size(0),
            angles,
            angle_equivariant_directions(source, angles),
            signed_radian,
            source,
        )

    clash = sparse_clash_edges(
        source,
        bonds,
        cutoff=2.0,
        allowed_contact=1.0,
        exclude_topology_distance=2,
        max_edges_per_graph=128,
    )
    clash_vector = torch.zeros_like(source)
    if clash["edge_index"].numel():
        left, right = clash["edge_index"]
        weights = clash["penetration"] * clash["active_mask"].to(source.dtype)
        counts = source.new_zeros(source.size(0))
        clash_vector.index_add_(0, left, weights[:, None] * clash["direction"])
        clash_vector.index_add_(0, right, -weights[:, None] * clash["direction"])
        counts.index_add_(0, left, torch.ones_like(weights))
        counts.index_add_(0, right, torch.ones_like(weights))
        clash_vector = clash_vector / counts.clamp_min(1.0)[:, None]
    return {
        "static": static,
        "bond_residual": bond_residual,
        "bond_severity": bond_severity,
        "angle_residual": angle_residual,
        "angle_severity": angle_severity,
        "clash": clash,
        "bond_vector": bond_vector,
        "angle_vector": angle_vector,
        "clash_vector": clash_vector,
    }


def _topology_distances(atom_count: int, bonds: Tensor) -> list[list[int]]:
    distance = [[atom_count + 1] * atom_count for _ in range(atom_count)]
    neighbors = [[] for _ in range(atom_count)]
    for left, right in bonds.t().tolist():
        neighbors[left].append(right)
        neighbors[right].append(left)
    for atom in range(atom_count):
        distance[atom][atom] = 0
        queue = [atom]
        for current in queue:
            for neighbor in neighbors[current]:
                if distance[atom][neighbor] > distance[atom][current] + 1:
                    distance[atom][neighbor] = distance[atom][current] + 1
                    queue.append(neighbor)
    return distance


def _clash_definition_audit(source: Tensor, bonds: Tensor, threshold: float):
    topology = _topology_distances(source.size(0), bonds)
    distances = torch.cdist(source.to(torch.float64), source.to(torch.float64))
    counts = Counter()
    penetration = Counter()
    for left in range(source.size(0)):
        for right in range(left + 1, source.size(0)):
            relation = (
                "bond_1_2"
                if topology[left][right] == 1
                else "angle_1_3"
                if topology[left][right] == 2
                else "target_nonbonded"
            )
            value = max(float(threshold) - float(distances[left, right]), 0.0)
            if value > 0:
                counts[relation] += 1
                penetration[relation] += value
    return counts, penetration


def _coverage_and_targets(
    items: list[dict[str, Any]], validity: ChemicalValidity
) -> tuple[dict[str, Any], dict[str, Any]]:
    coverage_rows = []
    target_rows = []
    combinations = Counter()
    mismatch_counts = Counter()
    mismatch_penetration = Counter()
    clash_threshold = float(validity.config["clash_distance_angstrom"])
    for item in items:
        components = _local_target_components(item, validity)
        static = components["static"]
        bonds = static["active_bond_constraint_index"]
        angles = static["active_angle_constraint_index"].t()
        clash = components["clash"]
        active = {
            "B": int((components["bond_severity"] > 0).sum()),
            "A": int((components["angle_severity"] > 0).sum()),
            "C": int(clash["active_mask"].sum()),
        }
        key = "+".join(name for name in ("B", "A", "C") if active[name]) or "none"
        combinations[key] += 1
        counts, penetration = _clash_definition_audit(
            item["input"], bonds, clash_threshold
        )
        mismatch_counts.update(counts)
        mismatch_penetration.update(penetration)
        coverage_rows.append(
            {
                "sample_id": str(item["row"].sample_id),
                "active_bond_count": active["B"],
                "active_angle_count": active["A"],
                "active_clash_count": active["C"],
                "total_bond_count": int(bonds.size(1)),
                "total_angle_count": int(angles.size(0)),
                "candidate_nonbond_pair_count": int(clash["edge_index"].size(1)),
                "active_bond_ratio": active["B"] / max(int(bonds.size(1)), 1),
                "active_angle_ratio": active["A"] / max(int(angles.size(0)), 1),
                "active_clash_ratio": active["C"] / max(int(clash["edge_index"].size(1)), 1),
                "active_combination": key,
            }
        )
        bond = components["bond_vector"]
        angle = components["angle_vector"]
        clash_vector = components["clash_vector"]
        unified = bond + angle + clash_vector
        denominator = _norm(bond) + _norm(angle) + _norm(clash_vector)
        actual = item["minimal_target"] - item["input"]
        target_rows.append(
            {
                "sample_id": str(item["row"].sample_id),
                "bond_target_norm": _norm(bond),
                "angle_target_norm": _norm(angle),
                "clash_target_norm": _norm(clash_vector),
                "unified_local_target_norm": _norm(unified),
                "materialized_target_norm": _norm(actual),
                "bond_angle_cosine": _cosine(bond, angle),
                "bond_clash_cosine": _cosine(bond, clash_vector),
                "angle_clash_cosine": _cosine(angle, clash_vector),
                "bond_unified_cosine": _cosine(bond, unified),
                "angle_unified_cosine": _cosine(angle, unified),
                "clash_unified_cosine": _cosine(clash_vector, unified),
                "materialized_local_cosine": _cosine(actual, unified),
                "cancellation_ratio": 1.0 - _norm(unified) / max(denominator, 1.0e-15),
            }
        )
    coverage = {
        "schema_version": SCHEMA_VERSION,
        "records": len(coverage_rows),
        "coverage": {
            name: _summary(row[name] for row in coverage_rows)
            for name in (
                "active_bond_count",
                "active_angle_count",
                "active_clash_count",
                "total_bond_count",
                "total_angle_count",
                "candidate_nonbond_pair_count",
                "active_bond_ratio",
                "active_angle_ratio",
                "active_clash_ratio",
            )
        },
        "graph_active_combinations": dict(combinations),
        "clash_definition_mismatch": {
            "metric_excludes": "1-2 only",
            "target_and_model_exclude": "1-2 and 1-3",
            "penetrating_pair_counts_by_relation": dict(mismatch_counts),
            "penetration_sum_by_relation": dict(mismatch_penetration),
        },
        "records_detail": coverage_rows,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
    }
    targets = {
        "schema_version": SCHEMA_VERSION,
        "component_definition": (
            "diagnostic local negative-residual directions at x_input; these are not "
            "serialized training targets and do not change target semantics"
        ),
        "materialized_target_definition": "existing offline minimal target minus source",
        "statistics": {
            name: _summary(row[name] for row in target_rows)
            for name in target_rows[0]
            if name != "sample_id"
        },
        "records": target_rows,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
    }
    return coverage, targets


def _parameter_group(name: str) -> str:
    if name.startswith(("backbone.", "error_encoder.")):
        return "shared_backbone"
    if name.startswith(("angle_constraint_", "clash_constraint_", "constraint_fusion.")):
        return "bac_output"
    return "other"


def _gradient_vector(
    term: Tensor, named_parameters: list[tuple[str, torch.nn.Parameter]]
) -> tuple[Tensor, dict[str, float]]:
    if not term.requires_grad:
        size = sum(parameter.numel() for _, parameter in named_parameters)
        return term.new_zeros(size), {"global": 0.0, "shared_backbone": 0.0, "bac_output": 0.0, "other": 0.0}
    gradients = torch.autograd.grad(
        term,
        [parameter for _, parameter in named_parameters],
        retain_graph=True,
        allow_unused=True,
    )
    chunks = []
    groups: dict[str, float] = {"global": 0.0, "shared_backbone": 0.0, "bac_output": 0.0, "other": 0.0}
    for (name, parameter), gradient in zip(named_parameters, gradients, strict=True):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        chunks.append(value.detach().reshape(-1))
        squared = float(value.detach().square().sum())
        groups[_parameter_group(name)] += squared
        groups["global"] += squared
    return torch.cat(chunks), {name: math.sqrt(value) for name, value in groups.items()}


def _stage_row(
    sample_id: str,
    target: Tensor,
    base_raw: Tensor,
    fused: Tensor,
    clipped: Tensor,
    final: Tensor,
    accepted: Tensor,
    atom_limit: float,
) -> dict[str, Any]:
    target_norm = _norm(target)
    base_norm = _norm(base_raw)
    fused_norm = _norm(fused)
    clipped_norm = _norm(clipped)
    final_norm = _norm(final)
    return {
        "sample_id": sample_id,
        "target_delta_norm": target_norm,
        "raw_model_delta_norm": base_norm,
        "fused_delta_norm": fused_norm,
        "clipped_delta_norm": clipped_norm,
        "accepted_delta_norm": _norm(accepted),
        "target_to_raw_scale": base_norm / max(target_norm, 1.0e-15),
        "raw_to_fused_scale": fused_norm / max(base_norm, 1.0e-15),
        "fused_to_clipped_scale": clipped_norm / max(fused_norm, 1.0e-15),
        "clipped_to_final_gate_scale": final_norm / max(clipped_norm, 1.0e-15),
        "clipped_to_accepted_scale": _norm(accepted) / max(clipped_norm, 1.0e-15),
        "cosine_target_raw": _cosine(target, base_raw),
        "cosine_target_fused": _cosine(target, fused),
        "atom_clipping_fraction": float(
            (torch.linalg.vector_norm(fused, dim=-1) > atom_limit + 1.0e-12)
            .to(torch.float32)
            .mean()
        ),
        "graph_clipping_flag": bool(not torch.allclose(fused, clipped, atol=1.0e-12, rtol=0.0)),
    }


def _run_model_diagnostics(
    model: MCVRBACModel,
    loss_fn: MCVRBACLoss,
    loader: DataLoader,
    items_by_sample: Mapping[str, dict[str, Any]],
    validity: ChemicalValidity,
    device: torch.device,
    maximum_batches: int,
    step_size: float,
    teacher_steps: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    gradient_batches = []
    pipeline_rows = []
    rollback_records = []
    precision_rows = []
    safety = BACSafetyConfig()
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        batch = batch.to(device)
        model.train()
        losses = loss_fn(model, batch)
        selected_terms = {
            "L_bond": losses["weighted_bond_residual"],
            "L_angle": losses["weighted_angle_residual"],
            "L_clash": losses["weighted_clash_penetration"],
            "L_preserve": losses["weighted_preservation"],
            "L_total": losses["loss"],
        }
        vectors = {}
        norms = {}
        for name, term in selected_terms.items():
            vectors[name], norms[name] = _gradient_vector(term, named_parameters)
        gradient_batches.append(
            {
                "batch": batch_index,
                "losses": {name: float(value.detach()) for name, value in selected_terms.items()},
                "gradient_norms": norms,
                "gradient_cosines": {
                    f"{left}__{right}": _cosine(vectors[left], vectors[right])
                    for left, right in itertools.combinations(("L_bond", "L_angle", "L_clash"), 2)
                },
            }
        )

        model.eval()
        with torch.inference_mode():
            t = batch.x_input.new_zeros(batch.num_graphs)
            output = model(batch, batch.x_input, t)
            branch_fused = output["v_angle_fused"] + output["v_clash_fused"]
            base_raw = output["v_raw"] - branch_fused
            ptr = batch.ptr.detach().cpu().tolist()
            sample_ids = list(batch.sample_id)
            for local, sample_id in enumerate(sample_ids):
                left, right = ptr[local], ptr[local + 1]
                item = items_by_sample[str(sample_id)]
                source = item["input"]
                target = (batch.x_target[left:right] - batch.x_input[left:right]).cpu()
                base_delta = (step_size * base_raw[left:right]).cpu()
                fused_delta = (step_size * output["v_raw"][left:right]).cpu()
                clipped_delta = (step_size * output["v_trust_clipped"][left:right]).cpu()
                final_delta = (step_size * output["v_final"][left:right]).cpu()
                decision = evaluate_bac_proposal(
                    source, source + final_delta, item["record"], validity, safety
                )
                accepted = final_delta if decision["accepted"] else torch.zeros_like(final_delta)
                pipeline_rows.append(
                    _stage_row(
                        str(sample_id), target, base_delta, fused_delta, clipped_delta,
                        final_delta, accepted, step_size * model.max_velocity_atom_norm,
                    )
                )
                precision_rows.append(
                    {
                        "sample_id": str(sample_id),
                        "float32_input_vs_float64_input_bac_delta_max_abs": max(
                            abs(
                                evaluate_bac_proposal(
                                    source.to(torch.float64),
                                    (source + final_delta).to(torch.float64),
                                    item["record"],
                                    validity,
                                    safety,
                                )["bac_deltas"][name]
                                - decision["bac_deltas"][name]
                            )
                            for name in ("bond", "angle", "clash", "ring")
                        ),
                    }
                )

            current = batch.x_input.clone()
            trajectories: list[list[Tensor]] = [[] for _ in range(batch.num_graphs)]
            for step in range(teacher_steps):
                time_value = current.new_full(
                    (batch.num_graphs,), 1.0 - step / max(teacher_steps, 1)
                )
                step_output = model(batch, current, time_value)
                current = current + step_size * step_output["v_final"]
                for local in range(batch.num_graphs):
                    left, right = ptr[local], ptr[local + 1]
                    trajectories[local].append(current[left:right].cpu().clone())
            for local, sample_id in enumerate(sample_ids):
                item = items_by_sample[str(sample_id)]
                decisions = [
                    evaluate_bac_proposal(
                        item["input"], proposal, item["record"], validity, safety
                    )
                    for proposal in trajectories[local]
                ]
                safe = [decision for decision in decisions if decision["accepted"]]
                selected = max(safe, key=lambda value: value["bac_gain"]) if safe else decisions[-1]
                rollback_records.append(
                    {
                        "sample_id": str(sample_id),
                        "accepted": bool(safe),
                        "reasons": [] if safe else list(selected["reasons"]),
                        "bac_deltas": dict(selected["bac_deltas"]),
                        "bac_gain": float(selected["bac_gain"]),
                        "displacement": dict(selected["displacement"]),
                        "chirality_preserved": float(selected["after"]["chirality_preserved"]),
                        "stereocenter_before": float(selected["before"]["stereocenter_degenerate_rate"]),
                        "stereocenter_after": float(selected["after"]["stereocenter_degenerate_rate"]),
                    }
                )

    gradient_statistics = {
        "schema_version": SCHEMA_VERSION,
        "optimizer_created": False,
        "optimizer_state_mutated": False,
        "batches": gradient_batches,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
    }
    pipeline_statistics = {
        "schema_version": SCHEMA_VERSION,
        "records": len(pipeline_rows),
        "teacher_steps": teacher_steps,
        "step_size": step_size,
        "statistics": {
            name: _summary(float(row[name]) for row in pipeline_rows)
            for name in pipeline_rows[0]
            if name != "sample_id" and not isinstance(pipeline_rows[0][name], bool)
        },
        "graph_clipping_fraction": sum(row["graph_clipping_flag"] for row in pipeline_rows)
        / max(len(pipeline_rows), 1),
        "records_detail": pipeline_rows,
        "precision_audit": {
            "ChemicalValidity_coordinate_dtype": "float32 (explicit cast in evaluate)",
            "input_float32_vs_float64": _summary(
                row["float32_input_vs_float64_input_bac_delta_max_abs"]
                for row in precision_rows
            ),
        },
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
    }
    rollback_statistics = _rollback_statistics(rollback_records, safety)
    return gradient_statistics, pipeline_statistics, rollback_statistics


def _rollback_statistics(
    records: list[dict[str, Any]], safety: BACSafetyConfig
) -> dict[str, Any]:
    expected = (
        "nonfinite",
        "molecule_trust_radius",
        "atom_trust_radius",
        "new_bond_violation",
        "new_angle_violation",
        "new_clash",
        "weighted_bac_objective_worse",
        "new_ring_violation",
        "chirality_changed",
        "stereocenter_degenerated",
        "identity_shape_changed",
        "rmsd_safeguard_failure",
        "no_bac_gain",
        "other",
    )
    counts = Counter()
    cooccurrence = Counter()
    for row in records:
        reasons = row["reasons"]
        counts.update(reasons)
        for left, right in itertools.combinations_with_replacement(sorted(set(reasons)), 2):
            cooccurrence[(left, right)] += 1
    tolerances = {}
    for tolerance in (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6):
        accepted = 0
        for row in records:
            deltas = row["bac_deltas"]
            displacement = row["displacement"]
            hard_safe = (
                displacement["max_atom_displacement"] <= safety.max_atom_displacement
                and displacement["aligned_rms_displacement"]
                <= safety.max_molecule_rms_displacement
                and all(deltas[name] <= tolerance for name in ("bond", "angle", "clash", "ring"))
                and row["chirality_preserved"] >= 1.0
                and row["stereocenter_after"] <= row["stereocenter_before"] + tolerance
                and row["bac_gain"] >= safety.minimum_bac_gain
            )
            accepted += int(hard_safe)
        tolerances[f"absolute_tolerance_{tolerance:.0e}"] = {
            "accepted": accepted,
            "fraction": accepted / max(len(records), 1),
        }
    rows = [
        {"reason_left": left, "reason_right": right, "count": count}
        for (left, right), count in sorted(cooccurrence.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "records": len(records),
        "accepted": sum(row["accepted"] for row in records),
        "rejected": sum(not row["accepted"] for row in records),
        "reason_counts": {name: int(counts.get(name, 0)) for name in expected},
        "reason_cooccurrence": rows,
        "absolute_tolerance_sensitivity_diagnostic_only": tolerances,
        "missing_current_checks": [
            "weighted_bac_objective_worse separate from no meaningful improvement",
            "identity/topology content mismatch beyond shape",
            "RMSD safeguard",
        ],
        "records_detail": records,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "diagnostics/ecir_mvr/v2_bac_overnight/runs/"
            "v2_d_bond_angle_clash_2000step_seed44/checkpoint_final.ckpt"
        ),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_recovery/manifests"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_recovery"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--maximum-model-batches", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.maximum_model_batches <= 2:
        raise ValueError("Phase-1 diagnostic is frozen to one or two model batches")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested diagnostic CUDA device is unavailable")
    formal_root = args.formal_root.expanduser().resolve()
    source_cache_root = args.source_cache_root.expanduser().resolve()
    manifest_dir = args.manifest_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    recovery_manifest = json.loads(
        (manifest_dir / "recovery_development_manifest.json").read_text(encoding="utf-8")
    )
    if recovery_manifest["identity_sha256"] != EXPECTED_MANIFEST_IDENTITY:
        raise RuntimeError("recovery manifest identity changed after preregistration")
    for field in ("test_records_read", "frozen_holdout_records_opened"):
        if int(recovery_manifest[field]) != 0:
            raise RuntimeError(f"recovery manifest violates isolation: {field}")
    if bool(recovery_manifest["test_assets_opened"]):
        raise RuntimeError("recovery manifest reports formal test access")
    source_manifest = manifest_dir / "diagnostic_sources.parquet"
    target_manifest = manifest_dir / "diagnostic_targets.parquet"
    source_rows = pd.read_parquet(source_manifest)
    target_rows = pd.read_parquet(target_manifest)
    if len(source_rows) != EXPECTED_DIAGNOSTIC_RECORDS:
        raise RuntimeError("diagnostic source count changed")
    if source_rows.molecule_id.nunique() != EXPECTED_DIAGNOSTIC_MOLECULES:
        raise RuntimeError("diagnostic molecule count changed")
    if set(source_rows.sample_id) != set(target_rows.sample_id):
        raise RuntimeError("diagnostic source/target sample identities differ")

    validity = ChemicalValidity("data/ecir_mvr/validity_reference_stats.json")
    items = build_items(
        source_manifest,
        target_manifest,
        validity,
        source_cache_root=source_cache_root,
        target_cache_root=formal_root / "minimal_targets",
    )
    coverage, targets = _coverage_and_targets(items, validity)
    _write_json(output_dir / "target_statistics.json", {"data_coverage": coverage, **targets})

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != 2000:
        raise RuntimeError("diagnostic checkpoint is not the frozen D0 step-2000 checkpoint")
    config = checkpoint["config"]
    if config.get("mode") != "V2_D_BOND_ANGLE_CLASH":
        raise RuntimeError("diagnostic checkpoint is not the original unified D0 candidate")
    model = MCVRBACModel(**config["model"]).to(torch.device(args.device))
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("D0 diagnostic checkpoint strict-load failed")
    loss_fn = MCVRBACLoss(config["base_loss"], config["bac_loss"])
    dataset = MCVRMixedDataset(
        source_manifest,
        target_manifest,
        validity,
        length=args.batch_size * args.maximum_model_batches,
        seed=43017,
        formal_adapter_lru_size=0,
        precompute_training_topology=True,
        source_cache_root=source_cache_root,
        target_cache_root=formal_root / "minimal_targets",
        canonical_constraints=True,
        constraint_source_identity_sha256=json.loads(
            (formal_root / "real_sources" / "metadata.json").read_text(encoding="utf-8")
        )["formal_source_identity_sha256"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    items_by_sample = {str(item["row"].sample_id): item for item in items}
    gradients, pipeline, rollback = _run_model_diagnostics(
        model,
        loss_fn,
        loader,
        items_by_sample,
        validity,
        torch.device(args.device),
        args.maximum_model_batches,
        float(config["inference"]["step_size"]),
        int(config["inference"]["teacher_steps"]),
    )
    _write_json(output_dir / "gradient_statistics.json", gradients)
    _write_json(output_dir / "proposal_pipeline_statistics.json", pipeline)
    _write_json(output_dir / "rollback_reason_distribution.json", rollback)
    pd.DataFrame(rollback["reason_cooccurrence"]).to_csv(
        output_dir / "rollback_reason_cooccurrence.csv", index=False
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PHASE1_AUDIT_COMPLETE_NO_REPAIR_SELECTED",
        "development_manifest_identity_sha256": recovery_manifest["identity_sha256"],
        "diagnostic_records_loaded": len(items),
        "diagnostic_model_records": int(args.batch_size * args.maximum_model_batches),
        "model_batches": args.maximum_model_batches,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_mode": config["mode"],
        "model_config": yaml.safe_dump(config["model"], sort_keys=True),
        "known_semantic_mismatches": [
            "ChemicalValidity clash excludes 1-2; sparse target/model clash excludes 1-2 and 1-3",
            "BAC branch loss uses a t=0 second forward while base loss uses sampled flow time",
            "BAC branch loss proposal excludes the base field while inference integrates the full field",
            "inference does not call the existing select_safe_bac_proposal backtracking helper",
            "safety uses zero per-type non-regression tolerances",
            "constraint_type_embedding is registered but unused in forward",
        ],
        "repair_candidates_frozen_before_selection": [
            "align clash objective and evaluator topology exclusions",
            "use finite absolute/relative tolerance only if precision evidence proves false rejection",
            "activate bounded backtracking in inference with unchanged hard ring/chirality checks",
            "remove duplicated branch attenuation or train/eval proposal mismatch",
        ],
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
    }
    _write_json(output_dir / "audit_summary.json", audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "diagnostic_records_loaded": len(items),
                "diagnostic_model_records": audit["diagnostic_model_records"],
                "test_records_read": 0,
                "test_assets_opened": False,
                "frozen_holdout_records_opened": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
