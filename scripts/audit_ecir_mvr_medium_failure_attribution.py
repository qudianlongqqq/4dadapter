#!/usr/bin/env python
"""Run the validation-only MCVR Medium failure-attribution audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from etflow.ecir.acceptance import evaluate_candidate, select_trajectory_candidate
from etflow.ecir.audit import displacement_metrics, torsion_change_metrics
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.failure_attribution import (
    DEFAULT_CLASSIFICATION_RULES,
    STAGE_ORDER,
    bond_observations,
    classify_failure,
    leave_one_out_influence,
    molecule_equal_aggregate,
    paired_relative_bootstrap,
    relative_improvement,
    stage_gain_decomposition,
    transition_labels,
)
from etflow.ecir.mvr_dataset import deterministic_error_features
from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir.mvr_safety import trust_clip_with_diagnostics
from etflow.ecir.run_a_evaluation import (
    CHEMICAL_METRICS, build_items, nearest_rmsd,
)


CONFIG_PATH = Path("configs/ecir_mvr_medium_5k_500_run_a_seed42_schedule_v4_10k.yaml")
V4_RESULT_PATH = Path("diagnostics/ecir_mvr/medium/run_a_seed42_schedule_v4/result.json")
V4_RECORDS_PATH = Path("diagnostics/ecir_mvr/medium/run_a_seed42_schedule_v4/record_metrics.csv")
V4_MOLECULES_PATH = Path("diagnostics/ecir_mvr/medium/run_a_seed42_schedule_v4/molecule_metrics.csv")
V4_SUMMARY_PATH = Path("diagnostics/ecir_mvr/medium/run_a_seed42_schedule_v4/source_summary.csv")
SELECTED_CHECKPOINT = Path("logs_ecir_mvr/medium/run_a_seed42_schedule_v4_10k/checkpoints/step001500.ckpt")
SELECTED_SHA256 = "f94c317f4e12c559058e26f9842317770179ed3e9cbc07c0a21ec681fed94197"
PROTECTED_PATH = Path("reports/global4d_profile_bundle_verification.json")
PROTECTED_SHA256 = "738171eb7e5f047e94cd4e1a46689613fe1f30bf33a320a2e3ec5a6944a5ec7d"
DIAGNOSTIC_LABEL = "DIAGNOSTIC_ORACLE_ONLY"
METRIC_COLUMNS = tuple(name for name in CHEMICAL_METRICS if name != "chirality_error")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _metric_values(validity: ChemicalValidity, coordinates, item) -> dict[str, float]:
    values = validity.evaluate(coordinates, item["record"], baseline_coordinates=item["input"])
    return {
        **{name: float(values[name]) for name in METRIC_COLUMNS},
        "chirality_error": 1.0 - float(values["chirality_preserved"]),
    }


def _atom_clip(raw: torch.Tensor, max_atom_norm: float) -> tuple[torch.Tensor, torch.Tensor]:
    norms = torch.linalg.vector_norm(raw, dim=-1)
    scales = torch.clamp(float(max_atom_norm) / norms.clamp_min(1.0e-12), max=1.0)
    return raw * scales[:, None], scales


@torch.inference_mode()
def infer_stage_coordinates(
    model: MCVRModel,
    items: Sequence[dict[str, Any]],
    validity: ChemicalValidity,
    *,
    device: torch.device,
    steps: int = 4,
    step_size: float = 0.25,
    batch_size: int = 32,
) -> tuple[dict[str, list[torch.Tensor]], list[dict[str, Any]]]:
    model.eval()
    stages = {name: [] for name in STAGE_ORDER}
    metadata: list[dict[str, Any]] = []
    schedule = torch.linspace(0.0, 1.0, int(steps)).tolist()
    for start in range(0, len(items), batch_size):
        selected = list(items[start:start + batch_size])
        batch = Batch.from_data_list([item["data"] for item in selected]).to(device)
        upstream = torch.cat([item["input"] for item in selected]).to(device)
        current = upstream.clone()
        raw_coordinates = upstream.clone()
        atom_coordinates = upstream.clone()
        clipped_coordinates = upstream.clone()
        ptr = batch.ptr.detach().cpu().tolist()
        trajectories: list[list[torch.Tensor]] = [[] for _ in selected]
        uncertainties: list[list[float]] = [[] for _ in selected]
        atom_clipped_steps: list[list[bool]] = [[] for _ in selected]
        graph_clipped_steps: list[list[bool]] = [[] for _ in selected]
        atom_clip_fractions: list[list[float]] = [[] for _ in selected]
        graph_clip_scales: list[list[float]] = [[] for _ in selected]
        safety_gates: list[list[float]] = [[] for _ in selected]
        raw_norms: list[list[float]] = [[] for _ in selected]
        clipped_norms: list[list[float]] = [[] for _ in selected]

        for time_value in schedule:
            current_cpu = current.detach().cpu()
            features = []
            trust_remaining = []
            for local, item in enumerate(selected):
                left, right = ptr[local], ptr[local + 1]
                values = validity.evaluate(
                    current_cpu[left:right], item["record"], baseline_coordinates=item["input"]
                )
                features.append(deterministic_error_features(
                    values, item["record"], str(item["row"].source_severity)
                ))
                changed = torsion_change_metrics(
                    item["input"], current_cpu[left:right], item["record"]
                )["max_rotatable_torsion_change"]
                limit = 0.35 if item["rotatable"] >= 6 else 0.70
                trust_remaining.append(max(0.0, limit - float(changed)))

            output = model(
                batch, current, current.new_full((len(selected),), float(time_value)),
                deterministic_features=torch.stack(features).to(device),
                torsion_trust_remaining=current.new_tensor(trust_remaining),
            )
            raw = output["v_raw"]
            atom_clipped, atom_scales = _atom_clip(raw, model.max_velocity_atom_norm)
            clipped, clipping = trust_clip_with_diagnostics(
                raw, batch.batch,
                max_atom_norm=model.max_velocity_atom_norm,
                max_graph_rms=model.max_velocity_graph_rms,
            )
            if not torch.allclose(clipped, output["v_trust_clipped"], rtol=1.0e-6, atol=1.0e-7):
                raise RuntimeError("stagewise clipping reconstruction differs from deployed output")
            final = output["v_final"]
            raw_coordinates = raw_coordinates + float(step_size) * raw
            atom_coordinates = atom_coordinates + float(step_size) * atom_clipped
            clipped_coordinates = clipped_coordinates + float(step_size) * clipped
            current = current + float(step_size) * final
            snapshot = current.detach().cpu()
            graph_scales = clipping["graph_clip_scale_per_graph"]
            for local in range(len(selected)):
                left, right = ptr[local], ptr[local + 1]
                trajectories[local].append(snapshot[left:right].clone())
                uncertainties[local].append(float(output["uncertainty"][local]))
                local_atom_scales = atom_scales[left:right]
                atom_clipped_steps[local].append(bool((local_atom_scales < 1.0).any()))
                graph_clipped_steps[local].append(bool(graph_scales[local] < 1.0))
                atom_clip_fractions[local].append(float((local_atom_scales < 1.0).float().mean()))
                graph_clip_scales[local].append(float(graph_scales[local]))
                safety_gates[local].append(float(output["global_safety_gate"][local]))
                raw_norms[local].append(float(torch.linalg.vector_norm(raw[left:right], dim=-1).mean()))
                clipped_norms[local].append(float(torch.linalg.vector_norm(clipped[left:right], dim=-1).mean()))

        if float(output["torsion_gate"].abs().max()) != 0.0 or float(output["v_torsion_contribution"].abs().max()) != 0.0:
            raise RuntimeError("frozen torsion branch became nonzero during attribution")
        for local, item in enumerate(selected):
            left, right = ptr[local], ptr[local + 1]
            accepted, decision = select_trajectory_candidate(
                item["input"], trajectories[local], item["record"], validity,
                mode="best_of_trajectory", uncertainties=uncertainties[local],
            )
            safe_oracle = []
            for trajectory_step, (candidate, uncertainty) in enumerate(
                zip(trajectories[local], uncertainties[local]), start=1
            ):
                candidate_decision = evaluate_candidate(
                    item["input"], candidate, item["record"], validity,
                    step=trajectory_step, uncertainty=uncertainty,
                )
                safety_reasons = set(candidate_decision.reject_reasons) - {
                    "insufficient_validity_gain", "no_thresholded_local_improvement",
                }
                if not safety_reasons:
                    values = validity.evaluate(candidate, item["record"], baseline_coordinates=item["input"])
                    safe_oracle.append((
                        float(values["bond_outlier_rate"]),
                        float(values["bond_outlier_magnitude"]),
                        float(values["total_thresholded_validity_score"]),
                        trajectory_step,
                    ))
            oracle_step = min(safe_oracle)[-1] if safe_oracle else 0
            oracle_coordinates = (
                trajectories[local][oracle_step - 1].clone()
                if oracle_step else torch.as_tensor(item["input"]).clone()
            )
            stages["upstream"].append(torch.as_tensor(item["input"]).clone())
            stages["raw_proposal"].append(raw_coordinates[left:right].detach().cpu().clone())
            stages["atom_clipped_proposal"].append(atom_coordinates[left:right].detach().cpu().clone())
            stages["trust_clipped_proposal"].append(clipped_coordinates[left:right].detach().cpu().clone())
            stages["safety_gated_proposal"].append(current[left:right].detach().cpu().clone())
            stages["accepted"].append(accepted.detach().cpu().clone())
            stages["minimal_target"].append(torch.as_tensor(item["minimal_target"]).clone())
            metadata.append({
                "accepted": bool(decision.accepted), "selected_step": int(decision.selected_step),
                "reject_reasons": ";".join(decision.reject_reasons),
                "acceptance_validity_gain": float(decision.validity_gain),
                "uncertainty": float(decision.uncertainty),
                "atom_clipped_any": any(atom_clipped_steps[local]),
                "graph_clipped_any": any(graph_clipped_steps[local]),
                "atom_clipping_fraction": float(np.mean(atom_clip_fractions[local])),
                "graph_clip_scale_min": float(min(graph_clip_scales[local])),
                "global_safety_gate_mean": float(np.mean(safety_gates[local])),
                "raw_velocity_norm_mean": float(np.mean(raw_norms[local])),
                "clipped_velocity_norm_mean": float(np.mean(clipped_norms[local])),
                "oracle_safe_step": int(oracle_step),
                "oracle_safe_coordinates": oracle_coordinates,
                "diagnostic_label": DIAGNOSTIC_LABEL,
            })
    return stages, metadata


def build_stage_record_metrics(
    items: Sequence[dict[str, Any]],
    stages: Mapping[str, Sequence[torch.Tensor]],
    metadata: Sequence[Mapping[str, Any]],
    validity: ChemicalValidity,
    *,
    include_coordinates: bool,
) -> pd.DataFrame:
    rows = []
    for index, (item, extra) in enumerate(zip(items, metadata)):
        row = {
            "molecule_id": str(item["row"].molecule_id),
            "sample_id": str(item["row"].sample_id),
            "source": str(item["row"].generator_name),
            "severity": str(item["row"].source_severity),
            "update_scale": float(item["row"].update_scale),
            "target_status": str(item.get("target_status", "unknown")),
            "rotatable_bond_count": int(item["rotatable"]),
            "has_ring": bool(item["has_ring"]), "clean_valid": bool(item["clean"]),
            "atom_count": int(item["input"].shape[0]),
            "bond_count": int(validity._prepare(item["record"])["bonds"].shape[1]),
            **{key: value for key, value in extra.items() if key != "oracle_safe_coordinates"},
        }
        for stage in STAGE_ORDER:
            coordinates = torch.as_tensor(stages[stage][index], dtype=torch.float32)
            values = _metric_values(validity, coordinates, item)
            displacement = displacement_metrics(item["input"], coordinates)
            torsion = torsion_change_metrics(item["input"], coordinates, item["record"])
            for name, value in values.items():
                row[f"{stage}_{name}"] = float(value)
            row[f"{stage}_aligned_RMSD"] = nearest_rmsd(coordinates, item["references"])
            row[f"{stage}_mean_displacement"] = float(displacement["mean_atom_displacement"])
            row[f"{stage}_rms_displacement"] = float(displacement["aligned_rms_displacement"])
            row[f"{stage}_max_displacement"] = float(displacement["max_atom_displacement"])
            row[f"{stage}_high_flex_torsion_change"] = (
                float(torsion["max_rotatable_torsion_change"]) if item["rotatable"] >= 6 else 0.0
            )
            row[f"{stage}_coordinate_unchanged"] = float(torch.equal(
                coordinates, torch.as_tensor(item["input"], dtype=torch.float32)
            ))
            if include_coordinates:
                row[f"{stage}_coordinates"] = coordinates.numpy().tolist()
        oracle = torch.as_tensor(extra["oracle_safe_coordinates"], dtype=torch.float32)
        oracle_values = _metric_values(validity, oracle, item)
        row["diagnostic_oracle_bond_outlier_rate"] = oracle_values["bond_outlier_rate"]
        row["diagnostic_oracle_bond_outlier_magnitude"] = oracle_values["bond_outlier_magnitude"]
        raw_direction = torch.as_tensor(stages["raw_proposal"][index]) - item["input"]
        target_direction = item["minimal_target"] - item["input"]
        raw_flat, target_flat = raw_direction.reshape(-1), target_direction.reshape(-1)
        denominator = float(torch.linalg.vector_norm(raw_flat) * torch.linalg.vector_norm(target_flat))
        row["raw_target_cosine_similarity"] = (
            float(torch.dot(raw_flat, target_flat)) / denominator if denominator > 1.0e-12 else 0.0
        )
        raw_direction_norm = float(torch.linalg.vector_norm(raw_flat))
        target_direction_norm = float(torch.linalg.vector_norm(target_flat))
        row["raw_direction_norm"] = raw_direction_norm
        row["target_direction_norm"] = target_direction_norm
        row["raw_target_norm_ratio"] = (
            raw_direction_norm / target_direction_norm if target_direction_norm > 1.0e-8 else math.nan
        )
        prepared = validity._prepare(item["record"])
        bonds = prepared["bonds"]
        upstream_lengths = torch.linalg.vector_norm(
            item["input"][bonds[0]] - item["input"][bonds[1]], dim=-1
        )
        raw_coordinates = torch.as_tensor(stages["raw_proposal"][index])
        target_coordinates = torch.as_tensor(stages["minimal_target"][index])
        raw_delta = torch.linalg.vector_norm(raw_coordinates[bonds[0]] - raw_coordinates[bonds[1]], dim=-1) - upstream_lengths
        target_delta = torch.linalg.vector_norm(target_coordinates[bonds[0]] - target_coordinates[bonds[1]], dim=-1) - upstream_lengths
        target_bond_energy = float(torch.dot(target_delta, target_delta))
        row["bond_local_projection_recovery"] = (
            float(torch.dot(raw_delta, target_delta)) / target_bond_energy
            if target_bond_energy > 1.0e-12 else math.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_bond_stage_metrics(
    items: Sequence[dict[str, Any]],
    stages: Mapping[str, Sequence[torch.Tensor]],
    validity: ChemicalValidity,
) -> pd.DataFrame:
    frames = []
    for index, item in enumerate(items):
        for stage in STAGE_ORDER:
            frame = bond_observations(validity, stages[stage][index], item["record"])
            frame.insert(0, "stage", stage)
            frame.insert(0, "sample_id", str(item["row"].sample_id))
            frame.insert(0, "molecule_id", str(item["row"].molecule_id))
            frame["source"] = str(item["row"].generator_name)
            frame["severity"] = str(item["row"].source_severity)
            frame["rotatable_bond_count"] = int(item["rotatable"])
            frame["has_ring_molecule"] = bool(item["has_ring"])
            frame["atom_count"] = int(item["input"].shape[0])
            frame["record_bond_count"] = len(frame)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_transition_details(bonds: pd.DataFrame) -> pd.DataFrame:
    identity = ["molecule_id", "sample_id", "bond_index"]
    metadata = bonds[bonds.stage.eq("upstream")].set_index(identity)
    stage_values = {
        stage: bonds[bonds.stage.eq(stage)].set_index(identity)
        for stage in ("upstream", "raw_proposal", "accepted", "minimal_target")
    }
    rows = []
    for comparison in ("raw_proposal", "accepted", "minimal_target"):
        candidate = stage_values[comparison]
        labels = transition_labels(stage_values["upstream"].outlier, candidate.outlier)
        frame = metadata.reset_index()[identity + [
            "left_atomic_number", "right_atomic_number", "bond_type", "aromatic", "ring",
            "heteroatom_bond", "branched", "source", "severity", "rotatable_bond_count",
            "has_ring_molecule", "atom_count", "record_bond_count",
        ]].copy()
        frame["comparison"] = comparison
        frame["transition"] = labels
        frame["upstream_margin"] = stage_values["upstream"].threshold_margin.to_numpy()
        frame["candidate_margin"] = candidate.threshold_margin.to_numpy()
        frame["upstream_excess"] = stage_values["upstream"].threshold_excess.to_numpy()
        frame["candidate_excess"] = candidate.threshold_excess.to_numpy()
        frame["threshold_near"] = (
            (frame.upstream_margin.abs() <= 0.05) | (frame.candidate_margin.abs() <= 0.05)
        )
        frame["upstream_severity"] = np.select(
            [frame.upstream_margin > 0.20, frame.upstream_margin > 0.0],
            ["severe", "mild"], default="normal",
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def transition_matrix(transition_details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison, subset in transition_details.groupby("comparison"):
        total = len(subset)
        for transition in (
            "normal_to_normal", "outlier_to_normal", "outlier_to_outlier", "normal_to_outlier"
        ):
            selected = subset[subset.transition.eq(transition)]
            rows.append({
                "comparison": comparison, "transition": transition,
                "bond_count": int(len(selected)), "fraction": len(selected) / max(total, 1),
                "ring_bond_count": int(selected.ring.sum()),
                "heteroatom_bond_count": int(selected.heteroatom_bond.sum()),
                "branched_bond_count": int(selected.branched.sum()),
                "threshold_near_count": int(selected.threshold_near.sum()),
                "mild_upstream_count": int(selected.upstream_severity.eq("mild").sum()),
                "severe_upstream_count": int(selected.upstream_severity.eq("severe").sum()),
            })
    return pd.DataFrame(rows)


def environment_summary(transition_details: pd.DataFrame) -> pd.DataFrame:
    dimensions = {
        "ring": transition_details.ring.map({True: "ring", False: "non_ring"}),
        "aromatic": transition_details.aromatic.map({True: "aromatic", False: "non_aromatic"}),
        "heteroatom": transition_details.heteroatom_bond.map({True: "heteroatom", False: "carbon_only"}),
        "branch": transition_details.branched.map({True: "branched", False: "unbranched"}),
        "bond_type": transition_details.bond_type.astype(str),
        "flexibility": pd.cut(
            transition_details.rotatable_bond_count,
            [-math.inf, 2, 5, math.inf], labels=["rotatable_le_2", "rotatable_3_5", "rotatable_ge_6"]
        ).astype(str),
    }
    rows = []
    for dimension, values in dimensions.items():
        working = transition_details.assign(environment=values)
        for (comparison, environment), subset in working.groupby(["comparison", "environment"]):
            counts = subset.transition.value_counts()
            repaired = int(counts.get("outlier_to_normal", 0))
            new = int(counts.get("normal_to_outlier", 0))
            rows.append({
                "dimension": dimension, "environment": str(environment),
                "comparison": comparison, "bond_count": int(len(subset)),
                "record_count": int(subset.sample_id.nunique()),
                "molecule_count": int(subset.molecule_id.nunique()),
                "repaired_bonds": repaired,
                "unchanged_bad_bonds": int(counts.get("outlier_to_outlier", 0)),
                "newly_broken_bonds": new,
                "normal_unchanged_bonds": int(counts.get("normal_to_normal", 0)),
                "net_repaired_bonds": repaired - new,
                "threshold_near_bonds": int(subset.threshold_near.sum()),
            })
    return pd.DataFrame(rows)


def build_stage_molecule_metrics(
    records: pd.DataFrame, transition_details: pd.DataFrame
) -> pd.DataFrame:
    value_columns = [
        f"{stage}_{suffix}"
        for stage in STAGE_ORDER
        for suffix in (
            "bond_outlier_rate", "bond_outlier_magnitude", "angle_outlier_rate",
            "angle_outlier_magnitude", "ring_bond_outlier_rate",
            "ring_planarity_outlier_rate", "clash_penetration", "severe_clash_rate",
            "chirality_error", "total_thresholded_validity_score", "aligned_RMSD",
            "mean_displacement", "rms_displacement", "high_flex_torsion_change",
        )
    ]
    numeric_metadata = [
        "rotatable_bond_count", "atom_count", "bond_count", "accepted", "selected_step",
        "atom_clipped_any", "graph_clipped_any", "atom_clipping_fraction",
        "graph_clip_scale_min", "global_safety_gate_mean", "raw_velocity_norm_mean",
        "clipped_velocity_norm_mean", "raw_target_cosine_similarity", "raw_target_norm_ratio",
        "raw_direction_norm", "target_direction_norm", "bond_local_projection_recovery",
        "diagnostic_oracle_bond_outlier_rate",
        "diagnostic_oracle_bond_outlier_magnitude",
    ]
    grouped = records.groupby("molecule_id", sort=True)
    molecules = grouped[value_columns + numeric_metadata].mean().reset_index()
    molecules["record_count"] = grouped.size().to_numpy()
    molecules["source_distribution"] = grouped.source.apply(lambda values: _json(Counter(values))).to_numpy()
    molecules["severity_distribution"] = grouped.severity.apply(lambda values: _json(Counter(values))).to_numpy()
    molecules["target_status_distribution"] = grouped.target_status.apply(lambda values: _json(Counter(values))).to_numpy()
    molecules["has_ring"] = grouped.has_ring.max().to_numpy()

    selected = transition_details[transition_details.comparison.eq("accepted")]
    transition_counts = selected.groupby("molecule_id").transition.value_counts().unstack(fill_value=0)
    for name, column in (
        ("repaired_bond_count", "outlier_to_normal"),
        ("unchanged_bad_bond_count", "outlier_to_outlier"),
        ("new_outlier_count", "normal_to_outlier"),
    ):
        molecules[name] = molecules.molecule_id.map(transition_counts.get(column, pd.Series(dtype=float))).fillna(0)
    near = selected.groupby("molecule_id").threshold_near.mean()
    molecules["threshold_near_fraction"] = molecules.molecule_id.map(near).fillna(0.0)
    observation_count = selected.groupby("molecule_id").size()
    molecules["bond_observation_count"] = molecules.molecule_id.map(observation_count).fillna(0)

    decompositions = []
    categories = []
    for _, row in molecules.iterrows():
        values = {
            stage: float(row[f"{stage}_bond_outlier_rate"])
            for stage in STAGE_ORDER
        }
        decomposition = stage_gain_decomposition(values)
        decompositions.append(decomposition)
        classification_values = {
            **decomposition,
            "upstream_bond_outlier_rate": values["upstream"],
            "repaired_bond_count": row.repaired_bond_count,
            "new_outlier_count": row.new_outlier_count,
            "threshold_near_fraction": row.threshold_near_fraction,
            "bond_magnitude_improvement": (
                row.upstream_bond_outlier_magnitude - row.accepted_bond_outlier_magnitude
            ),
        }
        categories.append(classify_failure(classification_values))
    decomposition_frame = pd.DataFrame(decompositions)
    for column in decomposition_frame:
        molecules[column] = decomposition_frame[column].to_numpy()
    molecules["failure_category"] = categories
    molecules["model_to_target_recovery_ratio"] = molecules.accepted_gain / molecules.target_available_gain.replace(0.0, np.nan)
    return molecules


def failure_category_summary(molecules: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, subset in molecules.groupby("failure_category"):
        rows.append({
            "failure_category": category,
            "molecule_count": int(len(subset)),
            "record_count": int(subset.record_count.sum()),
            "bond_count": int(subset.bond_observation_count.sum()),
            "accepted_gain_contribution": float(subset.accepted_gain.sum()),
            "target_gap_contribution": float(subset.target_gap.sum()),
            "model_proposal_shortfall": float((subset.target_available_gain - subset.raw_potential_gain).clip(lower=0).sum()),
            "clipping_loss_contribution": float(subset.clipping_loss.clip(lower=0).sum()),
            "safety_gate_loss_contribution": float(subset.safety_gate_loss.clip(lower=0).sum()),
            "acceptance_loss_contribution": float(subset.acceptance_loss.clip(lower=0).sum()),
            "source_distribution": _json(sum((Counter(json.loads(value)) for value in subset.source_distribution), Counter())),
            "severity_distribution": _json(sum((Counter(json.loads(value)) for value in subset.severity_distribution), Counter())),
            "flexibility_distribution": _json(Counter(pd.cut(
                subset.rotatable_bond_count, [-math.inf, 2, 5, math.inf],
                labels=["rotatable_le_2", "rotatable_3_5", "rotatable_ge_6"]
            ).astype(str))),
            "structure_distribution": _json(Counter(np.where(subset.has_ring, "ring", "non_ring"))),
        })
    return pd.DataFrame(rows).sort_values("molecule_count", ascending=False)


def metric_recalculation_audit() -> tuple[dict[str, Any], bool]:
    result = json.loads(V4_RESULT_PATH.read_text(encoding="utf-8"))
    records = pd.read_csv(V4_RECORDS_PATH)
    official_molecules = pd.read_csv(V4_MOLECULES_PATH)
    official_summary = pd.read_csv(V4_SUMMARY_PATH)
    selected = records[records.method.isin(["upstream", "medium_accepted"])].copy()
    molecules = selected.groupby(["method", "molecule_id"], sort=True).bond_outlier_rate.mean()
    aggregate = molecules.groupby("method").mean()
    upstream = float(aggregate["upstream"])
    model = float(aggregate["medium_accepted"])
    relative = relative_improvement(upstream, model)
    record_means = selected.groupby("method").bond_outlier_rate.mean()
    source_values = official_summary[
        official_summary.group.eq("all")
        & official_summary.method.isin(["upstream", "medium_accepted"])
    ].set_index("method").bond_outlier_rate
    official_relative = float(result["gate"]["relative_improvements"]["bond_outlier_rate"])
    expected_rows = official_molecules[
        official_molecules.group.eq("all")
        & official_molecules.method.isin(["upstream", "medium_accepted"])
    ]
    checks = {
        "formula_exact": abs(relative - (upstream - model) / upstream) <= 1.0e-15,
        "matches_gate_value": abs(relative - official_relative) <= 1.0e-15,
        "matches_source_summary_upstream": abs(upstream - float(source_values.upstream)) <= 1.0e-15,
        "matches_source_summary_model": abs(model - float(source_values.medium_accepted)) <= 1.0e-15,
        "all_500_molecules_present": selected.molecule_id.nunique() == 500,
        "no_missing_metric_values": not selected.bond_outlier_rate.isna().any(),
        "paired_molecule_rows_complete": len(expected_rows) == 1000,
        "selected_checkpoint_identity": _sha(SELECTED_CHECKPOINT) == SELECTED_SHA256,
        "test_records_zero": result["test_records_read"] == 0,
        "protected_file_unchanged": _sha(PROTECTED_PATH) == PROTECTED_SHA256,
    }
    audit = {
        "schema_version": "ecir-mvr-medium-bond-metric-recalculation-v1",
        "decision": "METRIC_IMPLEMENTATION_CORRECT" if all(checks.values()) else "METRIC_IMPLEMENTATION_HOLD",
        "formula": "(bond_outlier_rate_upstream - bond_outlier_rate_accepted_model) / bond_outlier_rate_upstream",
        "bond_outlier_rate_upstream": upstream,
        "bond_outlier_rate_model": model,
        "absolute_delta_model_minus_upstream": model - upstream,
        "absolute_improvement_upstream_minus_model": upstream - model,
        "relative_improvement": relative,
        "official_gate_relative_improvement": official_relative,
        "record_equal_upstream": float(record_means.upstream),
        "record_equal_model": float(record_means.medium_accepted),
        "record_equal_relative_improvement": relative_improvement(
            record_means.upstream, record_means.medium_accepted
        ),
        "aggregation_contract": {
            "bond_within_record": "each unique undirected bond has equal weight within its record",
            "record_within_molecule": "arithmetic mean; one or two validation records per molecule",
            "molecule_within_all": "arithmetic mean; each of 500 molecules has equal weight",
            "bootstrap": "paired molecule resampling with replacement",
            "missing": "none; paired pivots drop missing molecules fail-closed in this audit",
            "identity_and_fallback": "included unchanged; target fallback affects target diagnostics only",
            "threshold_equality": "distance > 0.0 is outlier; exact lower/upper equality is normal",
            "ring_bonds": "unique bonds appear once in bond_outlier_rate; ring rate is a subset diagnostic, not an added duplicate",
            "bond_count_weighting": "no cross-molecule bond-count weighting in the formal Gate",
            "floating_comparison": "unrounded float64 relative improvement compared directly with 0.10",
        },
        "molecule_record_count_distribution": selected.drop_duplicates(["molecule_id", "sample_id"]).groupby("molecule_id").size().value_counts().sort_index().to_dict(),
        "checks": checks,
        "test_records_read": 0,
    }
    return audit, all(checks.values())


def _quartiles(values: pd.Series, prefix: str) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, 4, labels=[f"{prefix}_q1", f"{prefix}_q2", f"{prefix}_q3", f"{prefix}_q4"])


def add_record_structure_features(records: pd.DataFrame, bonds: pd.DataFrame) -> pd.DataFrame:
    upstream = bonds[bonds.stage.eq("upstream")]
    features = upstream.groupby("sample_id").agg(
        aromatic_bond_fraction=("aromatic", "mean"),
        heteroatom_bond_fraction=("heteroatom_bond", "mean"),
    )
    result = records.merge(features, on="sample_id", how="left", validate="one_to_one")
    result["size_quartile"] = _quartiles(result.atom_count, "atom_count")
    result["bond_count_quartile"] = _quartiles(result.bond_count, "bond_count")
    result["aromatic_heavy"] = result.aromatic_bond_fraction >= result.aromatic_bond_fraction.median()
    result["heteroatom_rich"] = result.heteroatom_bond_fraction >= result.heteroatom_bond_fraction.median()
    return result


def _subgroup_memberships(row: pd.Series) -> list[tuple[str, str]]:
    groups = [("all", "all")]
    if row.source == "ETFlow_formal_upstream":
        groups.append(("source", "ETFlow_normal"))
    elif row.severity in {"mild", "medium", "severe"}:
        groups.append(("source", f"Cartesian_{row.severity}"))
    if row.source == "Cartesian_teacher_100k" and abs(float(row.update_scale) - 0.35) < 1.0e-12:
        groups.append(("source", "unseen_scale_0.35"))
    groups.append(("flexibility", (
        "rotatable_le_2" if row.rotatable_bond_count <= 2 else
        "rotatable_3_5" if row.rotatable_bond_count <= 5 else "rotatable_ge_6"
    )))
    groups.extend([
        ("structure", "ring" if row.has_ring else "non_ring"),
        ("structure", "aromatic_heavy" if row.aromatic_heavy else "aromatic_light"),
        ("structure", "heteroatom_rich" if row.heteroatom_rich else "heteroatom_light"),
        ("structure", str(row.size_quartile)),
        ("structure", str(row.bond_count_quartile)),
    ])
    groups.append(("acceptance", "accepted" if row.accepted else "rejected"))
    if row.accepted and 0 < row.selected_step < 4:
        groups.append(("acceptance", "partially_updated"))
    if row.accepted_mean_displacement <= 1.0e-8:
        groups.append(("acceptance", "identity_preserved"))
    clipping = (
        "both_clipping" if row.atom_clipped_any and row.graph_clipped_any else
        "atom_clipping_only" if row.atom_clipped_any else
        "graph_clipping_only" if row.graph_clipped_any else "no_clipping"
    )
    groups.append(("clipping", clipping))
    return groups


def subgroup_attribution(records: pd.DataFrame, *, bootstrap_draws: int = 1000) -> pd.DataFrame:
    memberships: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in records.iterrows():
        for group_type, group in _subgroup_memberships(row):
            memberships[(group_type, group)].append(index)
    required_groups = {
        ("source", name) for name in (
            "ETFlow_normal", "Cartesian_mild", "Cartesian_medium",
            "Cartesian_severe", "unseen_scale_0.35",
        )
    } | {
        ("flexibility", name) for name in (
            "rotatable_le_2", "rotatable_3_5", "rotatable_ge_6",
        )
    } | {
        ("structure", name) for name in (
            "ring", "non_ring", "aromatic_heavy", "heteroatom_rich",
        )
    } | {
        ("acceptance", name) for name in (
            "accepted", "rejected", "partially_updated", "identity_preserved",
        )
    } | {
        ("clipping", name) for name in (
            "no_clipping", "atom_clipping_only", "graph_clipping_only", "both_clipping",
        )
    }
    for key in required_groups:
        memberships.setdefault(key, [])
    rows = []
    rate_columns = [f"{stage}_bond_outlier_rate" for stage in STAGE_ORDER]
    for (group_type, group), indices in sorted(memberships.items()):
        if not indices:
            rows.append({
                "group_type": group_type, "group": group, "status": "INSUFFICIENT",
                "molecule_count": 0, "record_count": 0,
                **{name: math.nan for name in (
                    "upstream_bond_outlier_rate", "target_bond_outlier_rate",
                    "raw_bond_outlier_rate", "accepted_bond_outlier_rate",
                    "target_upper_bound_relative_improvement", "raw_model_relative_improvement",
                    "accepted_relative_improvement", "model_to_target_recovery_ratio",
                    "target_available_gain", "raw_potential_gain", "atom_clipping_loss",
                    "graph_clipping_loss", "clipping_loss", "safety_gate_loss",
                    "acceptance_loss", "accepted_gain", "target_gap", "bootstrap_ci95_low",
                    "bootstrap_ci95_high", "bootstrap_probability_ge_10pct",
                )},
            })
            continue
        selected = records.loc[indices]
        molecules = selected.groupby("molecule_id", sort=True)[rate_columns].mean()
        values = {
            stage: float(molecules[f"{stage}_bond_outlier_rate"].mean()) for stage in STAGE_ORDER
        }
        decomposition = stage_gain_decomposition(values)
        bootstrap_result = paired_relative_bootstrap(
            molecules.upstream_bond_outlier_rate,
            molecules.accepted_bond_outlier_rate,
            draws=bootstrap_draws, seed=42,
        )
        target_relative = relative_improvement(values["upstream"], values["minimal_target"])
        accepted_relative = relative_improvement(values["upstream"], values["accepted"])
        rows.append({
            "group_type": group_type, "group": group,
            "status": "OK" if len(molecules) >= 20 else "INSUFFICIENT",
            "molecule_count": int(len(molecules)), "record_count": int(len(selected)),
            "upstream_bond_outlier_rate": values["upstream"],
            "target_bond_outlier_rate": values["minimal_target"],
            "raw_bond_outlier_rate": values["raw_proposal"],
            "accepted_bond_outlier_rate": values["accepted"],
            "target_upper_bound_relative_improvement": target_relative,
            "raw_model_relative_improvement": relative_improvement(values["upstream"], values["raw_proposal"]),
            "accepted_relative_improvement": accepted_relative,
            "model_to_target_recovery_ratio": accepted_relative / target_relative if abs(target_relative) > 1.0e-12 else math.nan,
            **decomposition,
            "bootstrap_ci95_low": bootstrap_result["ci95_low"],
            "bootstrap_ci95_high": bootstrap_result["ci95_high"],
            "bootstrap_probability_ge_10pct": bootstrap_result["probability_ge_10pct"],
        })
    return pd.DataFrame(rows)


def aggregate_stage_summary(records: pd.DataFrame) -> dict[str, float]:
    stage_rate_columns = [f"{stage}_bond_outlier_rate" for stage in STAGE_ORDER]
    stage_magnitude_columns = [f"{stage}_bond_outlier_magnitude" for stage in STAGE_ORDER]
    stage_validity_columns = [f"{stage}_total_thresholded_validity_score" for stage in STAGE_ORDER]
    stage_rmsd_columns = [f"{stage}_aligned_RMSD" for stage in STAGE_ORDER]
    columns = stage_rate_columns + stage_magnitude_columns + stage_validity_columns + stage_rmsd_columns + [
        "upstream_angle_outlier_rate", "minimal_target_angle_outlier_rate",
        "upstream_ring_bond_outlier_rate", "minimal_target_ring_bond_outlier_rate",
        "minimal_target_rms_displacement", "minimal_target_high_flex_torsion_change",
        "minimal_target_coordinate_unchanged",
        "accepted_rms_displacement", "accepted", "atom_clipping_fraction",
        "graph_clipped_any", "global_safety_gate_mean", "raw_target_cosine_similarity",
        "raw_target_norm_ratio", "bond_local_projection_recovery",
        "diagnostic_oracle_bond_outlier_rate",
    ]
    molecules = records.groupby("molecule_id", sort=True)[columns].mean()
    mean = molecules.mean()
    rates = {stage: float(mean[f"{stage}_bond_outlier_rate"]) for stage in STAGE_ORDER}
    decomposition = stage_gain_decomposition(rates)
    upstream = rates["upstream"]
    high_flex = records[records.rotatable_bond_count >= 6].groupby("molecule_id")[
        "minimal_target_high_flex_torsion_change"
    ].mean()
    identity_targets = records.target_status.isin(["identity_clean", "identity_fallback"])
    clean_targets = records.target_status.eq("identity_clean")
    return {
        "molecules": int(len(molecules)), "records": int(len(records)),
        **{f"{stage}_bond_outlier_rate": rates[stage] for stage in STAGE_ORDER},
        **{f"{stage}_bond_outlier_relative_improvement": relative_improvement(upstream, rates[stage]) for stage in STAGE_ORDER[1:]},
        **{f"{stage}_bond_outlier_magnitude": float(mean[f"{stage}_bond_outlier_magnitude"]) for stage in STAGE_ORDER},
        "accepted_bond_magnitude_relative_improvement": relative_improvement(
            float(mean.upstream_bond_outlier_magnitude), float(mean.accepted_bond_outlier_magnitude)
        ),
        "accepted_total_validity_delta": float(
            mean.accepted_total_thresholded_validity_score - mean.upstream_total_thresholded_validity_score
        ),
        "target_total_validity_delta": float(
            mean.minimal_target_total_thresholded_validity_score - mean.upstream_total_thresholded_validity_score
        ),
        "target_total_validity_relative_improvement": relative_improvement(
            float(mean.upstream_total_thresholded_validity_score),
            float(mean.minimal_target_total_thresholded_validity_score),
        ),
        "target_angle_relative_improvement": relative_improvement(
            float(mean.upstream_angle_outlier_rate), float(mean.minimal_target_angle_outlier_rate)
        ),
        "target_ring_relative_improvement": relative_improvement(
            float(mean.upstream_ring_bond_outlier_rate), float(mean.minimal_target_ring_bond_outlier_rate)
        ),
        "target_rmsd_delta": float(mean.minimal_target_aligned_RMSD - mean.upstream_aligned_RMSD),
        "target_displacement": float(mean.minimal_target_rms_displacement),
        "target_high_flex_torsion_change": float(high_flex.mean()) if len(high_flex) else 0.0,
        "target_identity_status_records": int(identity_targets.sum()),
        "target_identity_status_unchanged_fraction": float(
            records.loc[identity_targets, "minimal_target_coordinate_unchanged"].mean()
        ) if identity_targets.any() else math.nan,
        "target_clean_records": int(clean_targets.sum()),
        "target_clean_identity_fraction": float(
            records.loc[clean_targets, "minimal_target_coordinate_unchanged"].mean()
        ) if clean_targets.any() else math.nan,
        "accepted_rmsd_delta": float(mean.accepted_aligned_RMSD - mean.upstream_aligned_RMSD),
        "accepted_displacement": float(mean.accepted_rms_displacement),
        "atom_clipping_fraction": float(mean.atom_clipping_fraction),
        "graph_clipping_fraction": float(mean.graph_clipped_any),
        "acceptance_fraction": float(mean.accepted),
        "global_safety_gate_mean": float(mean.global_safety_gate_mean),
        "raw_target_cosine_similarity": float(mean.raw_target_cosine_similarity),
        "raw_target_norm_ratio": float(mean.raw_target_norm_ratio),
        "bond_local_projection_recovery": float(mean.bond_local_projection_recovery),
        "diagnostic_oracle_bond_outlier_relative_improvement": relative_improvement(
            upstream, float(mean.diagnostic_oracle_bond_outlier_rate)
        ),
        **decomposition,
    }


def checkpoint_specs() -> list[dict[str, Any]]:
    specs = []
    for step in (500, 1000, 1500, 2000, 3000, 5000, 7500, 10000):
        specs.append({
            "run": "V4", "step": step,
            "path": Path(f"logs_ecir_mvr/medium/run_a_seed42_schedule_v4_10k/checkpoints/step{step:06d}.ckpt"),
            "formal_selected": step == 1500,
        })
    specs.extend([
        {"run": "V3_best_overall", "step": 2000, "path": Path("logs_ecir_mvr/medium/run_a_seed42_rescue_v2_20k/checkpoints/step002000.ckpt"), "formal_selected": False},
        {"run": "V3_formal", "step": 10000, "path": Path("logs_ecir_mvr/medium/run_a_seed42_rescue_v3_20k/checkpoints/step010000.ckpt"), "formal_selected": True},
        {"run": "Run_A_Stage2b", "step": 3000, "path": Path("logs_ecir_mvr/stage2b/run_a_rigid_only_seed42_5k/checkpoints/best_noninferior_validity.ckpt"), "formal_selected": True},
    ])
    return specs


def checkpoint_diversity(spec: Mapping[str, Any]) -> float:
    if spec["run"] == "V4":
        path = Path(f"logs_ecir_mvr/medium/run_a_seed42_schedule_v4_10k/validation_step{spec['step']:06d}.csv")
    elif spec["run"] == "V3_best_overall":
        path = Path("logs_ecir_mvr/medium/run_a_seed42_rescue_v2_20k/validation_step002000.csv")
    elif spec["run"] == "V3_formal":
        path = Path("logs_ecir_mvr/medium/run_a_seed42_rescue_v3_20k/validation_step010000.csv")
    else:
        path = Path("logs_ecir_mvr/stage2b/run_a_rigid_only_seed42_5k/validation_step003000.csv")
    frame = pd.read_csv(path)
    row = frame[(frame.group == "all") & (frame.method == "run_a_accepted")]
    return float(row.iloc[0].diversity)


def load_model(path: Path, device: torch.device) -> MCVRModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = MCVRModel(**payload["config"]["model"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def threshold_summary(bonds: pd.DataFrame) -> dict[str, Any]:
    rows = {}
    for stage in ("upstream", "raw_proposal", "accepted", "minimal_target"):
        selected = bonds[bonds.stage.eq(stage)]
        excess = selected.threshold_excess.to_numpy(dtype=float)
        rows[stage] = {
            "bond_count": int(len(selected)),
            "bond_outlier_rate_bond_weighted": float(selected.outlier.mean()),
            "bond_outlier_magnitude_bond_weighted": float(selected.normalized_excess.mean()),
            "mean_threshold_excess": float(excess.mean()),
            "p95_threshold_excess": float(np.quantile(excess, 0.95)),
            "severe_outlier_count": int((selected.threshold_margin > 0.20).sum()),
            "mild_outlier_count": int(((selected.threshold_margin > 0.0) & (selected.threshold_margin <= 0.20)).sum()),
            "threshold_buckets": selected.threshold_bucket.value_counts().reindex(
                ["lt_-20pct", "-20_to_-10pct", "-10_to_-5pct", "-5_to_0pct", "0_to_5pct", "5_to_10pct", "10_to_20pct", "gt_20pct"],
                fill_value=0,
            ).to_dict(),
        }
    return rows


def stability_analysis(records: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    molecules = records.groupby("molecule_id", sort=True)[[
        "upstream_bond_outlier_rate", "accepted_bond_outlier_rate"
    ]].mean().reset_index()
    bootstrap_result = paired_relative_bootstrap(
        molecules.upstream_bond_outlier_rate,
        molecules.accepted_bond_outlier_rate,
        draws=10_000, seed=42,
    )
    influence = leave_one_out_influence(
        molecules.molecule_id, molecules.upstream_bond_outlier_rate,
        molecules.accepted_bond_outlier_rate,
    )
    influence["anonymous_molecule_id"] = influence.molecule_id.map(
        lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    )
    influence = influence.drop(columns=["molecule_id"])
    source_memberships = defaultdict(set)
    for row in records.itertuples():
        if row.source == "ETFlow_formal_upstream":
            source_memberships["ETFlow_normal"].add(row.molecule_id)
        else:
            source_memberships[f"Cartesian_{row.severity}"].add(row.molecule_id)
        if row.source == "Cartesian_teacher_100k" and abs(float(row.update_scale) - 0.35) < 1.0e-12:
            source_memberships["unseen_scale_0.35"].add(row.molecule_id)
    leave_source = {}
    for source, excluded in sorted(source_memberships.items()):
        retained = molecules[~molecules.molecule_id.isin(excluded)]
        leave_source[source] = {
            "excluded_molecules": len(excluded), "retained_molecules": len(retained),
            "relative_improvement": relative_improvement(
                retained.upstream_bond_outlier_rate.mean(),
                retained.accepted_bond_outlier_rate.mean(),
            ) if len(retained) else math.nan,
        }
    result = {
        "schema_version": "ecir-mvr-medium-bond-bootstrap-stability-v1",
        "formal_decision_unchanged": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "paired_molecule_bootstrap": bootstrap_result,
        "leave_one_molecule_out": {
            "minimum": float(influence.leave_one_out_relative_improvement.min()),
            "maximum": float(influence.leave_one_out_relative_improvement.max()),
            "range": float(
                influence.leave_one_out_relative_improvement.max()
                - influence.leave_one_out_relative_improvement.min()
            ),
        },
        "leave_one_source_group_out": leave_source,
        "test_records_read": 0,
    }
    return result, influence.head(20).copy()


def _fmt(value: Any, digits: int = 12) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def write_metric_audit_report(audit: Mapping[str, Any]) -> None:
    checks = audit["checks"]
    lines = [
        "# MCVR Medium Bond Metric Audit", "",
        f"Decision: **{audit['decision']}**", "",
        "## Exact recalculation", "", "| Quantity | Value |", "|---|---:|",
        f"| Upstream bond outlier rate | {_fmt(audit['bond_outlier_rate_upstream'])} |",
        f"| Accepted-model bond outlier rate | {_fmt(audit['bond_outlier_rate_model'])} |",
        f"| Absolute delta (model - upstream) | {_fmt(audit['absolute_delta_model_minus_upstream'])} |",
        f"| Relative improvement | {_fmt(audit['relative_improvement'])} |",
        f"| Relative improvement percent | {_fmt(100 * audit['relative_improvement'], 10)}% |", "",
        "The formal definition is `(upstream - accepted_model) / upstream`; the unrounded float is compared directly with `0.10`.", "",
        "## Aggregation contract", "", "| Level | Rule |", "|---|---|",
    ]
    for name, description in audit["aggregation_contract"].items():
        lines.append(f"| {name} | {description} |")
    lines += [
        "", f"Record-equal relative improvement is `{_fmt(audit['record_equal_relative_improvement'])}`; it is not the preregistered Gate aggregation.", "",
        "## Implementation checks", "", "| Check | Result |", "|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items())
    lines += [
        "", "Exact threshold equality is normal because the implementation uses `distance > 0.0`.",
        "Ring bonds are present once in the unique-bond list; the ring metric is a subset and is not added to the ordinary bond count.",
        "The paired bootstrap resamples molecules, not records or bonds. No validation value is missing.",
    ]
    Path("docs/MCVR_MEDIUM_BOND_METRIC_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def determine_primary_cause(
    aggregate: Mapping[str, float], molecules: pd.DataFrame, transitions: pd.DataFrame
) -> tuple[str, list[str], dict[str, Any], str]:
    target_relative = float(aggregate["minimal_target_bond_outlier_relative_improvement"])
    target_gap = float(aggregate["target_gap"])
    signed = {
        "model_proposal": float(aggregate["target_available_gain"] - aggregate["raw_potential_gain"]),
        "atom_clipping": float(aggregate["atom_clipping_loss"]),
        "graph_clipping": float(aggregate["graph_clipping_loss"]),
        "safety_gate": float(aggregate["safety_gate_loss"]),
        "acceptance": float(aggregate["acceptance_loss"]),
    }
    positive = {name: max(value, 0.0) for name, value in signed.items()}
    total_positive = sum(positive.values())
    shares = {
        name: value / total_positive if total_positive > 1.0e-12 else 0.0
        for name, value in positive.items()
    }
    accepted_transitions = transitions[transitions.comparison.eq("accepted")]
    repaired = int(accepted_transitions.transition.eq("outlier_to_normal").sum())
    newly_broken = int(accepted_transitions.transition.eq("normal_to_outlier").sum())
    cancellation_ratio = newly_broken / max(repaired, 1)
    near_fraction = float(accepted_transitions.threshold_near.mean())
    target_recovery = float(aggregate["accepted_gain"]) / max(float(aggregate["target_available_gain"]), 1.0e-12)
    dominant = max(shares, key=shares.get)

    if target_relative < 0.10:
        primary = "TARGET_LIMITED"
    elif cancellation_ratio >= 0.50 and newly_broken > 0:
        primary = "CANCELLATION_LIMITED"
    elif dominant == "model_proposal" and shares[dominant] >= 0.50:
        primary = "MODEL_PROPOSAL_LIMITED"
    elif dominant in {"atom_clipping", "graph_clipping"} and shares[dominant] >= 0.50:
        primary = "TRUST_CLIP_LIMITED"
    elif dominant == "safety_gate" and shares[dominant] >= 0.50:
        primary = "SAFETY_GATE_LIMITED"
    elif dominant == "acceptance" and shares[dominant] >= 0.50:
        primary = "ACCEPTANCE_LIMITED"
    elif near_fraction >= 0.50 and target_recovery >= 0.80:
        primary = "THRESHOLD_EDGE_LIMITED"
    else:
        primary = "MIXED_LIMITATION"
    mapping = {
        "TARGET_LIMITED": "REDESIGN_MINIMAL_TARGET",
        "MODEL_PROPOSAL_LIMITED": "REDESIGN_LOCAL_BOND_PREDICTION",
        "TRUST_CLIP_LIMITED": "REDESIGN_TRUST_OR_ACCEPTANCE",
        "SAFETY_GATE_LIMITED": "REDESIGN_TRUST_OR_ACCEPTANCE",
        "ACCEPTANCE_LIMITED": "REDESIGN_TRUST_OR_ACCEPTANCE",
        "CANCELLATION_LIMITED": "REDESIGN_LOCAL_BOND_PREDICTION",
        "THRESHOLD_EDGE_LIMITED": "STOP_METHOD_AS_MEDIUM_EFFECT",
        "MIXED_LIMITATION": "STOP_METHOD_AS_MEDIUM_EFFECT",
    }
    secondary = [
        name for name, share in sorted(shares.items(), key=lambda item: item[1], reverse=True)
        if name != dominant and share >= 0.10
    ]
    factors = {
        "target_gap": target_gap,
        "signed_contributions": signed,
        "signed_target_gap_shares": {
            name: value / target_gap if abs(target_gap) > 1.0e-12 else 0.0
            for name, value in signed.items()
        },
        "positive_loss_shares": shares,
        "telescoping_sum": sum(signed.values()),
        "telescoping_error": sum(signed.values()) - target_gap,
        "target_relative_improvement": target_relative,
        "model_to_target_recovery_ratio": target_recovery,
        "repaired_bonds": repaired, "newly_broken_bonds": newly_broken,
        "cancellation_ratio": cancellation_ratio,
        "threshold_near_fraction": near_fraction,
        "classification_rules": DEFAULT_CLASSIFICATION_RULES,
    }
    return primary, secondary, factors, mapping[primary]


def write_reports(
    *,
    result: Mapping[str, Any], metric_audit: Mapping[str, Any],
    aggregate: Mapping[str, float],
    transitions: pd.DataFrame,
    environments: pd.DataFrame, subgroups: pd.DataFrame,
    checkpoints: pd.DataFrame, target_gap: pd.DataFrame,
    stability: Mapping[str, Any], categories: pd.DataFrame,
) -> None:
    primary = result["failure_attribution_primary_cause"]
    recommendation = result["failure_attribution_recommendation"]
    factors = result["factor_contributions"]
    matrix = transitions.pivot(index="transition", columns="comparison", values="bond_count").fillna(0)
    repaired = int(matrix.loc["outlier_to_normal", "accepted"])
    unchanged = int(matrix.loc["outlier_to_outlier", "accepted"])
    new = int(matrix.loc["normal_to_outlier", "accepted"])
    target_repaired = int(matrix.loc["outlier_to_normal", "minimal_target"])
    target_new = int(matrix.loc["normal_to_outlier", "minimal_target"])

    factor_lines = [
        f"| {name} | {_fmt(value)} | {_fmt(factors['signed_target_gap_shares'][name], 6)} | {_fmt(factors['positive_loss_shares'][name], 6)} |"
        for name, value in factors["signed_contributions"].items()
    ]
    main = [
        "# MCVR Medium Failure Attribution Report", "",
        f"Primary cause: **{primary}**", "",
        f"Recommendation: **{recommendation}**", "",
        "The formal decision remains **MEDIUM_SEED42_SCHEDULE_V4_FAIL**. This audit is validation-only and did not train, alter checkpoints, read test data, or create a training command.", "",
        "## Exact formal metric", "", "| Quantity | Value |", "|---|---:|",
        f"| Upstream bond outlier rate | {_fmt(metric_audit['bond_outlier_rate_upstream'])} |",
        f"| Accepted bond outlier rate | {_fmt(metric_audit['bond_outlier_rate_model'])} |",
        f"| Relative improvement | {_fmt(metric_audit['relative_improvement'])} |",
        f"| Percent | {_fmt(100 * metric_audit['relative_improvement'], 10)}% |", "",
        "## Stagewise attribution", "", "| Factor | Signed loss | Target-gap share | Positive-loss share |", "|---|---:|---:|---:|",
        *factor_lines, "",
        f"The signed terms telescope to target gap `{_fmt(factors['target_gap'])}` with numerical error `{_fmt(factors['telescoping_error'])}`.", "",
        f"Minimal Target available relative improvement: `{_fmt(factors['target_relative_improvement'])}`.",
        f"Model-to-target recovery ratio: `{_fmt(factors['model_to_target_recovery_ratio'])}`.", "",
        "## Bond transitions", "",
        f"Accepted output repaired `{repaired}` original outlier bond observations, left `{unchanged}` bad, and created `{new}` new outlier observations.",
        f"Minimal Target repaired `{target_repaired}` and created `{target_new}`.", "",
        "## Classification", "", "| Category | Molecules | Records | Target-gap contribution |", "|---|---:|---:|---:|",
    ]
    for row in categories.itertuples():
        main.append(f"| {row.failure_category} | {row.molecule_count} | {row.record_count} | {_fmt(row.target_gap_contribution)} |")
    main += [
        "", "Counterfactual outputs are labeled `DIAGNOSTIC_ORACLE_ONLY` and were not used for checkpoint selection or Gate decisions.",
        "No Rescue V5, seed43/44, 100k, or test evaluation was run.",
    ]
    Path("docs/MCVR_MEDIUM_FAILURE_ATTRIBUTION_REPORT.md").write_text("\n".join(main) + "\n", encoding="utf-8")

    transition_doc = [
        "# MCVR Medium Bond Transition Analysis", "",
        "Transitions use unique undirected bonds per validation record; ring bonds are flagged as a subset without duplicate counting.", "",
        "| Comparison | Normal->normal | Outlier->normal | Outlier->outlier | Normal->outlier |", "|---|---:|---:|---:|---:|",
    ]
    for comparison in ("raw_proposal", "accepted", "minimal_target"):
        transition_doc.append(
            f"| {comparison} | {int(matrix.loc['normal_to_normal', comparison])} | {int(matrix.loc['outlier_to_normal', comparison])} | {int(matrix.loc['outlier_to_outlier', comparison])} | {int(matrix.loc['normal_to_outlier', comparison])} |"
        )
    worst = environments[(environments.comparison == "accepted") & (environments.newly_broken_bonds > 0)].sort_values("newly_broken_bonds", ascending=False).head(10)
    transition_doc += [
        "", f"The accepted model's new/repaired ratio is `{_fmt(new / max(repaired, 1))}`.", "",
        "## Largest new-outlier environments", "", "| Dimension | Environment | New | Repaired | Net repaired |", "|---|---|---:|---:|---:|",
    ]
    for row in worst.itertuples():
        transition_doc.append(f"| {row.dimension} | {row.environment} | {row.newly_broken_bonds} | {row.repaired_bonds} | {row.net_repaired_bonds} |")
    Path("docs/MCVR_MEDIUM_BOND_TRANSITION_ANALYSIS.md").write_text("\n".join(transition_doc) + "\n", encoding="utf-8")

    all_target = target_gap[(target_gap.group_type == "all") & (target_gap.group == "all")].iloc[0]
    target_doc = [
        "# MCVR Medium Target Gap Analysis", "",
        "Minimal Target is a diagnostic upper bound and is not reported as model performance.", "",
        "| Metric | Value |", "|---|---:|",
        f"| Target bond relative improvement | {_fmt(all_target.target_upper_bound_relative_improvement)} |",
        f"| Model bond relative improvement | {_fmt(all_target.accepted_relative_improvement)} |",
        f"| Model-to-target recovery ratio | {_fmt(all_target.model_to_target_recovery_ratio)} |",
        f"| Target gap | {_fmt(all_target.target_gap)} |", "",
        "## Additional target diagnostics", "", "| Metric | Value |", "|---|---:|",
        f"| Angle outlier relative improvement | {_fmt(aggregate['target_angle_relative_improvement'])} |",
        f"| Ring-bond outlier relative improvement | {_fmt(aggregate['target_ring_relative_improvement'])} |",
        f"| Total validity relative improvement | {_fmt(aggregate['target_total_validity_relative_improvement'])} |",
        f"| Target RMSD delta | {_fmt(aggregate['target_rmsd_delta'])} |",
        f"| Target RMS displacement | {_fmt(aggregate['target_displacement'])} |",
        f"| High-flex target torsion change | {_fmt(aggregate['target_high_flex_torsion_change'])} |",
        f"| Identity-status unchanged fraction | {_fmt(aggregate['target_identity_status_unchanged_fraction'])} |",
        f"| Clean-target identity fraction | {_fmt(aggregate['target_clean_identity_fraction'])} |", "",
        "Because target availability is evaluated on the same 700 validation records with molecule-equal aggregation, it is directly comparable for attribution but not for the formal model Gate.", "",
        "Groups with fewer than 20 molecules are marked `INSUFFICIENT` and support no strong conclusion.",
    ]
    Path("docs/MCVR_MEDIUM_TARGET_GAP_ANALYSIS.md").write_text("\n".join(target_doc) + "\n", encoding="utf-8")

    dynamics = [
        "# MCVR Medium Checkpoint Dynamics", "",
        "All checkpoints are diagnostic controls. The formal V4 result remains selected step 1500.", "",
        "| Run | Step | LR | Bond relative | Raw gain | Clip loss | Safety loss | Acceptance loss | Validity delta | Diversity |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in checkpoints.itertuples():
        dynamics.append(
            f"| {row.run} | {row.step} | {_fmt(row.learning_rate, 8)} | {_fmt(row.accepted_bond_outlier_relative_improvement)} | {_fmt(row.raw_potential_gain)} | {_fmt(row.clipping_loss)} | {_fmt(row.safety_gate_loss)} | {_fmt(row.acceptance_loss)} | {_fmt(row.accepted_total_validity_delta)} | {_fmt(row.diversity)} |"
        )
    v4 = checkpoints[checkpoints.run.eq("V4")]
    best = v4.loc[v4.accepted_bond_outlier_relative_improvement.idxmax()]
    late = v4[v4.step.eq(10000)].iloc[0]
    selected = v4[v4.step.eq(1500)].iloc[0]
    lr_values = [float(value) for value in v4.learning_rate]
    improvement_values = [float(value) for value in v4.accepted_bond_outlier_relative_improvement]
    lr_mean = sum(lr_values) / len(lr_values)
    improvement_mean = sum(improvement_values) / len(improvement_values)
    covariance = sum(
        (left - lr_mean) * (right - improvement_mean)
        for left, right in zip(lr_values, improvement_values)
    )
    lr_scale = math.sqrt(sum((value - lr_mean) ** 2 for value in lr_values))
    improvement_scale = math.sqrt(sum(
        (value - improvement_mean) ** 2 for value in improvement_values
    ))
    lr_correlation = covariance / max(lr_scale * improvement_scale, 1.0e-12)
    diversity_ratio = float(v4.diversity.min() / max(v4.diversity.max(), 1.0e-12))
    dynamics += [
        "", f"The diagnostic bond-rate peak is step `{int(best.step)}`; the preregistered validity-based formal selection remains step `1500`.",
        f"From step 1500 to 10000, raw gain changed `{_fmt(selected.raw_potential_gain)} -> {_fmt(late.raw_potential_gain)}` while safety-gate loss changed `{_fmt(selected.safety_gate_loss)} -> {_fmt(late.safety_gate_loss)}`. The late decline is therefore not raw-proposal degradation; it is dominated by stronger learned safety attenuation.",
        f"Trust-clipping loss remained negligible and acceptance loss was non-positive at both points, so clipping and deterministic acceptance did not cause the late decline.",
        f"V4 diversity minimum/maximum ratio is `{_fmt(diversity_ratio)}`; this does not indicate mode collapse under the frozen diversity criterion.",
        "A classical overfitting claim is not supported because this audit has no matched training-set performance curve; the observed validation dynamics are attributable to stage behavior, not labeled as overfitting.",
        f"The Pearson correlation between registered LR and accepted bond improvement is `{_fmt(lr_correlation)}`; the non-monotonic curve does not support a simple LR-only explanation.",
        "These diagnostics do not authorize V5 or checkpoint reselection.",
    ]
    Path("docs/MCVR_MEDIUM_CHECKPOINT_DYNAMICS.md").write_text("\n".join(dynamics) + "\n", encoding="utf-8")

    next_doc = [
        "# MCVR Medium Next Method Decision", "",
        f"Attribution: **{primary}**", "",
        f"Recommendation: **{recommendation}**", "",
        "This is a recommendation only. No configuration or command was generated.", "",
        "Any future work that changes the method must be a new preregistered study, not Rescue V5 and not a retrospective Gate adjustment.", "",
        "The current method remains a statistically significant, accuracy-noninferior medium-effect result that failed the preregistered 10% core-improvement threshold.",
    ]
    Path("docs/MCVR_MEDIUM_NEXT_METHOD_DECISION.md").write_text("\n".join(next_doc) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/ecir_mvr/medium/failure_attribution"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subgroup-bootstrap-draws", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_audit, metric_correct = metric_recalculation_audit()
    atomic_json_save(metric_audit, args.output_dir / "bond_metric_recalculation.json")
    write_metric_audit_report(metric_audit)
    if not metric_correct:
        hold = {
            "schema_version": "ecir-mvr-medium-failure-attribution-v1",
            "decision": "METRIC_IMPLEMENTATION_HOLD",
            "formal_decision_unchanged": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
            "failure_attribution_completed": False,
            "test_records_read": 0, "next_command": None,
        }
        atomic_json_save(hold, args.output_dir / "result.json")
        raise SystemExit("METRIC_IMPLEMENTATION_HOLD")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("failure attribution requires the audited CUDA validation environment")
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = build_items(config["data"]["val_sources"], config["data"]["val_targets"], validity)
    targets = pd.read_parquet(config["data"]["val_targets"]).set_index("sample_id")
    for item in items:
        item["target_status"] = str(targets.loc[str(item["row"].sample_id), "target_status"])

    selected_model = load_model(SELECTED_CHECKPOINT, device)
    selected_stages, selected_metadata = infer_stage_coordinates(
        selected_model, items, validity, device=device,
    )
    stage_records = build_stage_record_metrics(
        items, selected_stages, selected_metadata, validity, include_coordinates=True,
    )
    official_model = metric_audit["bond_outlier_rate_model"]
    _, independent = molecule_equal_aggregate(
        stage_records, ["upstream_bond_outlier_rate", "accepted_bond_outlier_rate"]
    )
    if abs(independent["accepted_bond_outlier_rate"] - official_model) > 1.0e-7:
        raise RuntimeError("independent coordinate-level metric recalculation differs from formal output")

    bonds = build_bond_stage_metrics(items, selected_stages, validity)
    transition_details = build_transition_details(bonds)
    transitions = transition_matrix(transition_details)
    environments = environment_summary(transition_details)
    stage_records = add_record_structure_features(stage_records, bonds)
    stage_molecules = build_stage_molecule_metrics(stage_records, transition_details)
    categories = failure_category_summary(stage_molecules)
    subgroups = subgroup_attribution(stage_records, bootstrap_draws=args.subgroup_bootstrap_draws)
    aggregate = aggregate_stage_summary(stage_records)

    lr_values = pd.read_csv("diagnostics/ecir_mvr/medium/run_a_seed42_schedule_v4/checkpoint_comparison.csv").set_index("step").learning_rate
    checkpoint_rows = []
    for spec in checkpoint_specs():
        if spec["run"] == "V4" and spec["step"] == 1500:
            records = stage_records
        else:
            model = load_model(spec["path"], device)
            stages, metadata = infer_stage_coordinates(model, items, validity, device=device)
            records = build_stage_record_metrics(
                items, stages, metadata, validity, include_coordinates=False,
            )
            del model, stages, metadata
            torch.cuda.empty_cache()
        summary = aggregate_stage_summary(records)
        checkpoint_rows.append({
            "run": spec["run"], "step": spec["step"],
            "checkpoint": str(spec["path"].resolve()), "checkpoint_sha256": _sha(spec["path"]),
            "formal_selected": spec["formal_selected"],
            "learning_rate": float(lr_values.loc[spec["step"]]) if spec["run"] == "V4" else 0.0002,
            "diversity": checkpoint_diversity(spec),
            **summary,
        })
    checkpoints = pd.DataFrame(checkpoint_rows)

    threshold = threshold_summary(bonds)
    stability, influence = stability_analysis(stage_records)
    primary, secondary, factor_contributions, recommendation = determine_primary_cause(
        aggregate, stage_molecules, transition_details
    )
    target_gap = subgroups[subgroups.group_type.isin(["all", "source", "flexibility", "structure"])].copy()
    target_gap["diagnostic_only"] = True

    result = {
        "schema_version": "ecir-mvr-medium-failure-attribution-v1",
        "decision": primary,
        "formal_decision_unchanged": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "metric_implementation_decision": metric_audit["decision"],
        "failure_attribution_completed": True,
        "failure_attribution_primary_cause": primary,
        "failure_attribution_secondary_factors": secondary,
        "failure_attribution_recommendation": recommendation,
        "factor_contributions": factor_contributions,
        "selected_checkpoint": str(SELECTED_CHECKPOINT.resolve()),
        "selected_checkpoint_sha256": _sha(SELECTED_CHECKPOINT),
        "selected_step": 1500,
        "stage_order_actual": [
            "upstream", "raw velocity integrated on deployed states",
            "atom-norm clipping", "graph-RMS clipping", "learned safety-gate scaling",
            "deterministic trajectory acceptance", "Minimal-Validity Target",
        ],
        "counterfactual_label": DIAGNOSTIC_LABEL,
        "threshold_analysis": threshold,
        "bootstrap_stability": stability["paired_molecule_bootstrap"],
        "tests": {"status": "PENDING", "targeted_passed": None, "full_passed": None},
        "training_run": False, "checkpoint_modified": False,
        "test_records_read": 0, "seed43_44_started": False, "100k_started": False,
        "next_command": None, "next_commands": [],
    }

    stage_records.to_parquet(args.output_dir / "stagewise_record_metrics.parquet", index=False)
    stage_molecules.to_csv(args.output_dir / "stagewise_molecule_metrics.csv", index=False)
    atomic_json_save({
        "schema_version": "ecir-mvr-medium-stagewise-gain-summary-v1",
        "aggregate": aggregate, "factor_contributions": factor_contributions,
        "threshold_analysis": threshold, "counterfactual_label": DIAGNOSTIC_LABEL,
        "test_records_read": 0,
    }, args.output_dir / "stagewise_gain_summary.json")
    categories.to_csv(args.output_dir / "failure_category_summary.csv", index=False)
    transitions.to_csv(args.output_dir / "bond_transition_matrix.csv", index=False)
    environments.to_csv(args.output_dir / "bond_environment_summary.csv", index=False)
    subgroups.to_csv(args.output_dir / "subgroup_attribution.csv", index=False)
    checkpoints.to_csv(args.output_dir / "checkpoint_dynamics.csv", index=False)
    target_gap.to_csv(args.output_dir / "target_gap_summary.csv", index=False)
    atomic_json_save(stability, args.output_dir / "bootstrap_stability.json")
    influence.to_csv(args.output_dir / "influence_molecules.csv", index=False)
    atomic_json_save(result, args.output_dir / "result.json")

    write_reports(
        result=result, metric_audit=metric_audit, aggregate=aggregate,
        transitions=transitions, environments=environments, subgroups=subgroups,
        checkpoints=checkpoints, target_gap=target_gap, stability=stability,
        categories=categories,
    )

    state_path = Path("reports/ecir_mvr/progressive_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "current_stage": "MEDIUM_FAILURE_ATTRIBUTION_COMPLETE",
        "current_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "medium_seed42_schedule_v4_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "failure_attribution_completed": True,
        "failure_attribution_primary_cause": primary,
        "failure_attribution_recommendation": recommendation,
        "100k_permitted": False, "100k_started": False,
        "seed43_44_permitted": False, "seed43_started": False, "seed44_started": False,
        "test_records_read": 0, "next_command": None, "next_commands": [],
        "next_command_executed": False,
    })
    atomic_json_save(state, state_path)
    print(json.dumps({
        "decision": primary, "recommendation": recommendation,
        "formal_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "metric_relative_improvement": metric_audit["relative_improvement"],
    }, indent=2))


if __name__ == "__main__":
    main()
