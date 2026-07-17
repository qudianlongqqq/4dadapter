#!/usr/bin/env python
"""Evaluate the fixed Stage E0 calibrator once on validation data."""

from __future__ import annotations

import argparse
import json
import math
import os
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
from etflow.ecir.acceptance import select_trajectory_candidate
from etflow.ecir.audit import torsion_change_metrics
from etflow.ecir.bond_explicit import batched_bond_projection, bond_length_jacobian
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.confidence_calibration import (
    DIAGNOSTIC_ALL_ONE, confidence_for_mode, load_calibrator,
    molecule_paired_bootstrap, stage_e0_decision, strict_load_frozen_model,
)
from etflow.ecir.failure_attribution import bond_observations, relative_improvement
from etflow.ecir.mvr_dataset import deterministic_error_features
from etflow.ecir.mvr_model import trust_clip_velocity
from etflow.ecir.run_a_evaluation import (
    BOOTSTRAP_METRICS, build_clean_control_items, graph_data, method_rows,
    nearest_rmsd, summarize_groups,
)


def _source_coordinates(row) -> tuple[dict[str, Any], torch.Tensor]:
    record = torch.load(Path(row.source_path), map_location="cpu", weights_only=False)
    if row.coordinate_path is not None and not pd.isna(row.coordinate_path):
        payload = torch.load(Path(row.coordinate_path), map_location="cpu", weights_only=False)
        coordinates = torch.as_tensor(payload[row.coordinate_key], dtype=torch.float32)
    else:
        coordinates = torch.as_tensor(record[row.coordinate_key], dtype=torch.float32)
    return record, coordinates


def load_validation_items(
    source_path: str | Path, target_path: str | Path, validity,
    *, limit_records: int = 0,
) -> list[dict[str, Any]]:
    source = pd.read_parquet(source_path).sort_values(["molecule_id", "sample_id"])
    if set(source.split.astype(str).unique()) != {"val"}:
        raise ValueError("Stage E0 evaluation requires validation sources only")
    if limit_records:
        source = source.groupby("molecule_id", sort=True).head(1).head(int(limit_records))
    targets = pd.read_parquet(target_path).set_index("sample_id")
    items = []
    for row in source.itertuples(index=False):
        record, coordinates = _source_coordinates(row)
        target_payload = torch.load(
            Path(targets.loc[row.sample_id].target_cache_path), map_location="cpu", weights_only=False
        )
        minimal = torch.as_tensor(target_payload["x_target"], dtype=torch.float32)
        references = torch.as_tensor(
            record.get("x_ref_candidates", record.get("x_ref_aligned")), dtype=torch.float32
        )
        if references.ndim == 2:
            references = references.unsqueeze(0)
        input_validity = validity.evaluate(coordinates, record, baseline_coordinates=coordinates)
        rotatable = int(record.get("num_rotatable_bonds", 0))
        has_ring = bool(torch.as_tensor(record.get("bond_is_in_ring", [])).any())
        clean = all(float(input_validity[name]) <= 0.0 for name in (
            "bond_outlier_rate", "angle_outlier_rate", "ring_bond_outlier_rate",
            "ring_planarity_outlier_rate", "clash_penetration", "severe_clash_rate",
            "stereocenter_degenerate_rate",
        ))
        active = torch.tensor([
            float(input_validity["bond_outlier_rate"] > 0),
            float(input_validity["angle_outlier_rate"] > 0),
            float(input_validity["ring_bond_outlier_rate"] > 0 or input_validity["ring_planarity_outlier_rate"] > 0),
            float(input_validity["clash_penetration"] > 0 or input_validity["severe_clash_rate"] > 0),
            float(input_validity["torsion_prior_outlier_score"] > 4.0), float(clean),
        ])
        groups = ["all"]
        if row.generator_name == "ETFlow_formal_upstream":
            groups.append("ETFlow_normal")
        elif row.source_severity in {"mild", "medium", "severe"}:
            groups.append(f"Cartesian_{row.source_severity}")
        if row.generator_name == "Cartesian_teacher_100k" and abs(float(row.update_scale) - 0.35) < 1.0e-12:
            groups.append("unseen_update_scale_0.35")
        groups.append("rotatable_le_2" if rotatable <= 2 else "rotatable_3_5" if rotatable <= 5 else "rotatable_ge_6")
        groups.append("ring" if has_ring else "non_ring")
        if clean:
            groups.append("clean_valid")
        items.append({
            "row": row, "record": record, "input": coordinates,
            "minimal_target": minimal, "references": references,
            "input_validity": input_validity, "input_rmsd": nearest_rmsd(coordinates, references),
            "data": graph_data(record, coordinates, row, active_mode_mask=active),
            "groups": groups, "rotatable": rotatable, "has_ring": has_ring, "clean": clean,
        })
    return items


@torch.inference_mode()
def infer_confidence_mode(
    model, items, validity, *, mode: str, calibrator, device: torch.device,
    inference: Mapping[str, Any], diagnostic_oracle_only: bool = False,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Run deployment inference without reading Minimal Target or reference coordinates."""
    accepted_all, metadata_all = [], []
    schedule = torch.linspace(0.0, 1.0, int(inference["teacher_steps"])).tolist()
    for start in range(0, len(items), 32):
        selected = items[start:start + 32]
        batch = Batch.from_data_list([item["data"] for item in selected]).to(device)
        current = torch.cat([item["input"] for item in selected]).to(device)
        ptr = batch.ptr.detach().cpu().tolist()
        trajectories = [[] for _ in selected]
        uncertainties = [[] for _ in selected]
        cancellations = [[] for _ in selected]
        torsion_gate_max = torsion_contribution_max = 0.0
        for time_value in schedule:
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
                changed = torsion_change_metrics(
                    item["input"], current_cpu[left:right], item["record"]
                )["max_rotatable_torsion_change"]
                limit = 0.35 if item["rotatable"] >= 6 else 0.70
                remaining.append(max(0.0, limit - float(changed)))
            output = model(
                batch, current, current.new_full((len(selected),), float(time_value)),
                deterministic_features=torch.stack(features).to(device),
                torsion_trust_remaining=current.new_tensor(remaining),
            )
            torsion_gate_max = max(torsion_gate_max, float(output["torsion_gate"].abs().max()))
            torsion_contribution_max = max(
                torsion_contribution_max, float(output["v_torsion_contribution"].abs().max())
            )
            bonds = output["bond_indices"]
            if mode == "deployed":
                bond_correction = output["v_bond_correction"]
                raw = output["v_raw"]
            else:
                confidence = confidence_for_mode(
                    output["bond_confidence_logit"], mode=mode, calibrator=calibrator,
                    diagnostic_oracle_only=diagnostic_oracle_only,
                )
                requested = output["bond_unattenuated_residual"] * confidence
                bond_correction, _ = batched_bond_projection(
                    current, bonds, requested, batch.batch, damping=model.bond_projection_damping
                )
                raw = output["v_cartesian_raw"] + model.bond_explicit_alpha * bond_correction
            clipped = trust_clip_velocity(
                raw, batch.batch, max_atom_norm=model.max_velocity_atom_norm,
                max_graph_rms=model.max_velocity_graph_rms,
            )
            final = output["global_safety_gate"][batch.batch] * clipped
            bond_graph = batch.batch[bonds[0]]
            for local in range(len(selected)):
                left, right = ptr[local], ptr[local + 1]
                keep = bond_graph == local
                local_bonds = bonds[:, keep] - left
                jacobian = bond_length_jacobian(current[left:right], local_bonds)
                cart_effect = jacobian @ output["v_cartesian_raw"][left:right].reshape(-1)
                bond_effect = jacobian @ bond_correction[left:right].reshape(-1)
                cancellations[local].append(float((cart_effect * bond_effect < 0.0).float().mean()))
            current = current + float(inference["step_size"]) * final
            snapshot = current.detach().cpu()
            for local in range(len(selected)):
                left, right = ptr[local], ptr[local + 1]
                trajectories[local].append(snapshot[left:right].clone())
                uncertainties[local].append(float(output["uncertainty"][local]))
        if torsion_gate_max != 0.0 or torsion_contribution_max != 0.0:
            raise RuntimeError("Stage E0 observed a nonzero torsion branch")
        for local, item in enumerate(selected):
            accepted, decision = select_trajectory_candidate(
                item["input"], trajectories[local], item["record"], validity,
                mode=str(inference["acceptance_mode"]), uncertainties=uncertainties[local],
            )
            accepted_all.append(accepted.detach().cpu())
            metadata_all.append({
                "accepted": bool(decision.accepted), "selected_step": int(decision.selected_step),
                "reject_reasons": ";".join(decision.reject_reasons),
                "uncertainty": float(decision.uncertainty),
                "bond_cancellation_fraction": float(np.mean(cancellations[local])),
                "torsion_gate_max": torsion_gate_max,
                "torsion_contribution_max": torsion_contribution_max,
            })
    return accepted_all, metadata_all


def _transition(items, coordinates, validity, method: str) -> dict[str, Any]:
    repaired = broken = total = 0
    for item, candidate in zip(items, coordinates):
        before = bond_observations(validity, item["input"], item["record"]).outlier.to_numpy(bool)
        after = bond_observations(validity, candidate, item["record"]).outlier.to_numpy(bool)
        repaired += int((before & ~after).sum())
        broken += int((~before & after).sum())
        total += len(before)
    return {"method": method, "bonds": total, "repaired_bonds": repaired,
            "newly_broken_bonds": broken, "cancellation_ratio": broken / max(repaired, 1)}


def _row(summary: pd.DataFrame, group: str, method: str):
    frame = summary[(summary.group == group) & (summary.method == method)]
    return None if frame.empty or int(frame.iloc[0].get("molecules", 0)) == 0 else frame.iloc[0]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    pd.read_csv(temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ecir_mvr_stage_e0_confidence_calibration.yaml"))
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--bootstrap-draws", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or Path(config["output_dir"])
    calibrator_path = args.calibrator or output / "calibrator.json"
    calibrator_payload = json.loads(calibrator_path.read_text(encoding="utf-8"))
    if calibrator_payload["checkpoint_sha256"] != config["checkpoint"]["sha256"]:
        raise RuntimeError("calibrator and frozen checkpoint identities differ")
    calibrator = load_calibrator(calibrator_payload).to(torch.device(args.device))
    model, checkpoint = strict_load_frozen_model(
        config["checkpoint"]["path"], expected_sha256=config["checkpoint"]["sha256"],
        device=torch.device(args.device),
    )
    if checkpoint["frozen_identities"] != config["frozen_identities"]:
        raise RuntimeError("Stage E0 frozen identities changed")
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = load_validation_items(
        config["data"]["val_sources"], config["data"]["val_targets"], validity,
        limit_records=args.limit_records,
    )
    deployed, deployed_metadata = infer_confidence_mode(
        model, items, validity, mode="deployed", calibrator=None,
        device=torch.device(args.device), inference=config["inference"],
    )
    calibrated, calibrated_metadata = infer_confidence_mode(
        model, items, validity, mode="calibrated", calibrator=calibrator,
        device=torch.device(args.device), inference=config["inference"],
    )
    all_one, all_one_metadata = infer_confidence_mode(
        model, items, validity, mode="confidence_all_one", calibrator=None,
        device=torch.device(args.device), inference=config["inference"],
        diagnostic_oracle_only=True,
    )
    methods = {
        "upstream": [item["input"] for item in items],
        "d1_b_deployed": deployed, "stage_e0_calibrated": calibrated,
        "confidence_all_one_DIAGNOSTIC_ORACLE_ONLY": all_one,
        "minimal_target": [item["minimal_target"] for item in items],
    }
    metadata = {
        "d1_b_deployed": deployed_metadata, "stage_e0_calibrated": calibrated_metadata,
        "confidence_all_one_DIAGNOSTIC_ORACLE_ONLY": all_one_metadata,
    }
    rows = method_rows(items, methods, validity, metadata)
    summary, molecules = summarize_groups(rows, items, methods)
    draws = int(args.bootstrap_draws or config["gate"]["bootstrap_draws"])
    bootstrap_vs_deployed = molecule_paired_bootstrap(
        molecules[molecules.group.eq("all")], candidate="stage_e0_calibrated",
        baseline="d1_b_deployed", metrics=BOOTSTRAP_METRICS, draws=draws,
        seed=int(config["gate"]["bootstrap_seed"]),
    )
    bootstrap_vs_upstream = molecule_paired_bootstrap(
        molecules[molecules.group.eq("all")], candidate="stage_e0_calibrated",
        baseline="upstream", metrics=BOOTSTRAP_METRICS, draws=draws,
        seed=int(config["gate"]["bootstrap_seed"]),
    )
    transitions = [
        _transition(items, deployed, validity, "d1_b_deployed"),
        _transition(items, calibrated, validity, "stage_e0_calibrated"),
    ]
    clean_items = build_clean_control_items(items, validity, limit=min(20, len(items)))
    clean_identity = math.nan
    if clean_items:
        clean_accepted, _ = infer_confidence_mode(
            model, clean_items, validity, mode="calibrated", calibrator=calibrator,
            device=torch.device(args.device), inference=config["inference"],
        )
        clean_identity = float(np.mean([
            torch.equal(torch.as_tensor(value), torch.as_tensor(item["input"]))
            for value, item in zip(clean_accepted, clean_items)
        ]))
    all_rows = summary[summary.group.eq("all")].set_index("method")
    upstream, baseline = all_rows.loc["upstream"], all_rows.loc["d1_b_deployed"]
    candidate, target = all_rows.loc["stage_e0_calibrated"], all_rows.loc["minimal_target"]
    relative = relative_improvement(upstream.bond_outlier_rate, candidate.bond_outlier_rate)
    target_relative = relative_improvement(upstream.bond_outlier_rate, target.bond_outlier_rate)
    recovery = relative / max(target_relative, 1.0e-12)
    v4 = pd.read_csv("diagnostics/ecir_mvr/stage_d/pilot/subgroup_summary.csv")
    v4 = v4[(v4.group.eq("all")) & (v4.method.eq("v4_selected"))].iloc[0]
    high, high_up = _row(summary, "rotatable_ge_6", "stage_e0_calibrated"), _row(summary, "rotatable_ge_6", "upstream")
    unseen, unseen_up = _row(summary, "unseen_update_scale_0.35", "stage_e0_calibrated"), _row(summary, "unseen_update_scale_0.35", "upstream")
    source_improvements = []
    for group in ("ETFlow_normal", "Cartesian_mild", "Cartesian_medium", "Cartesian_severe"):
        value, base = _row(summary, group, "stage_e0_calibrated"), _row(summary, group, "d1_b_deployed")
        if value is not None and base is not None and value.total_thresholded_validity_score < base.total_thresholded_validity_score:
            source_improvements.append(group)
    gate = config["gate"]
    accuracy = all(
        bootstrap_vs_upstream[name]["mean"] <= float(gate["mat_mean_delta_max"])
        and bootstrap_vs_upstream[name]["ci95_high"] <= float(gate["mat_ci_upper_max"])
        for name in ("MAT_P", "MAT_R")
    )
    criteria = {
        "01_model_to_target_recovery_ge_0p30": recovery >= float(gate["model_to_target_recovery_min"]),
        "02_bond_relative_improvement_ge_13pct": relative >= float(gate["bond_relative_improvement_min"]),
        "03_bond_vs_d1b_paired_ci_improves": bootstrap_vs_deployed["bond_outlier_rate"]["ci95_high"] < 0.0,
        "04_total_validity_vs_d1b_not_worse": candidate.total_thresholded_validity_score <= baseline.total_thresholded_validity_score,
        "05_angle_vs_v4_not_clearly_worse": candidate.angle_outlier_rate <= v4.angle_outlier_rate + float(gate["chemical_metric_noninferiority_margin"]),
        "06_ring_vs_v4_not_clearly_worse": candidate.ring_bond_outlier_rate <= v4.ring_bond_outlier_rate + float(gate["chemical_metric_noninferiority_margin"]),
        "07_newly_broken_not_above_d1b": transitions[1]["newly_broken_bonds"] <= transitions[0]["newly_broken_bonds"],
        "08_cancellation_ratio_le_20pct": float(np.mean([x["bond_cancellation_fraction"] for x in calibrated_metadata])) <= float(gate["cancellation_ratio_max"]),
        "09_rmsd_mean_delta_le_0p003": bootstrap_vs_upstream["aligned_RMSD"]["mean"] <= float(gate["rmsd_mean_delta_max"]),
        "10_rmsd_ci_upper_le_0p005": bootstrap_vs_upstream["aligned_RMSD"]["ci95_high"] <= float(gate["rmsd_ci_upper_max"]),
        "11_mat_p_mat_r_noninferior": accuracy,
        "12_cov_p_cov_r_no_material_drop": candidate.COV_P >= upstream.COV_P - float(gate["cov_absolute_drop_max"]) and candidate.COV_R >= upstream.COV_R - float(gate["cov_absolute_drop_max"]),
        "13_high_flex_validity_pass": high is not None and high_up is not None and high.total_thresholded_validity_score < high_up.total_thresholded_validity_score,
        "14_high_flex_torsion_controlled": high is not None and high.high_flex_torsion_change <= float(gate["high_flex_torsion_max"]),
        "15_clean_identity_ge_90pct": math.isfinite(clean_identity) and clean_identity >= float(gate["clean_identity_fraction_min"]),
        "16_clash_chirality_not_worse": candidate.severe_clash_rate <= upstream.severe_clash_rate and candidate.chirality_error <= upstream.chirality_error and candidate.clash_penetration <= upstream.clash_penetration + float(gate["chemical_metric_noninferiority_margin"]),
        "17_unseen_validity_accuracy_pass": unseen is not None and unseen_up is not None and unseen.total_thresholded_validity_score < unseen_up.total_thresholded_validity_score and unseen.aligned_RMSD - unseen_up.aligned_RMSD <= 0.003,
        "18_improvement_not_single_source": len(source_improvements) >= 2,
        "19_test_records_read_zero": True,
    }
    criteria = {name: bool(value) for name, value in criteria.items()}
    harms = bool(
        candidate.total_thresholded_validity_score > baseline.total_thresholded_validity_score
        or bootstrap_vs_upstream["aligned_RMSD"]["ci95_high"] > float(gate["rmsd_ci_upper_max"])
        or not criteria["16_clash_chirality_not_worse"]
    )
    decision = stage_e0_decision(criteria, harms=harms)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_save({
        "schema_version": "ecir-mvr-stage-e0-bootstrap-v1", "draws": draws,
        "seed": int(gate["bootstrap_seed"]), "candidate": "stage_e0_calibrated",
        "vs_d1_b_deployed": bootstrap_vs_deployed, "vs_upstream": bootstrap_vs_upstream,
        "test_records_read": 0,
    }, output / "bootstrap.json")
    _atomic_csv(pd.DataFrame(transitions), output / "bond_transition.csv")
    _atomic_csv(summary[summary.group.isin(["ETFlow_normal", "Cartesian_mild", "Cartesian_medium", "Cartesian_severe"])], output / "source_summary.csv")
    _atomic_csv(summary[summary.group.isin(["ring", "non_ring"])], output / "ring_nonring_summary.csv")
    result = {
        "schema_version": "ecir-mvr-stage-e0-validation-v1", "decision": decision,
        "smoke": bool(args.limit_records), "validation_only": True,
        "validation_records_read": len(items), "test_records_read": 0,
        "neural_training_run": False, "calibrator_parameters": ["raw_a", "b"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "calibration_identity_sha256": calibrator_payload["calibration_identity_sha256"],
        "diagnostic_control": {"method": "confidence_all_one", "label": DIAGNOSTIC_ALL_ONE},
        "metrics": {
            "model_to_target_recovery": recovery, "bond_relative_improvement": relative,
            "clean_identity_fraction": clean_identity,
            "cancellation_ratio": float(np.mean([x["bond_cancellation_fraction"] for x in calibrated_metadata])),
            "newly_broken_bonds": transitions[1]["newly_broken_bonds"],
            "d1b_newly_broken_bonds": transitions[0]["newly_broken_bonds"],
            "improving_sources": source_improvements,
        },
        "method_comparison": {
            method: {
                name: float(all_rows.loc[method, name]) for name in (
                    "bond_outlier_rate", "bond_outlier_magnitude", "angle_outlier_rate",
                    "ring_bond_outlier_rate", "total_thresholded_validity_score",
                    "aligned_RMSD", "MAT_P", "MAT_R", "COV_P", "COV_R",
                )
            } for method in (
                "d1_b_deployed", "stage_e0_calibrated",
                "confidence_all_one_DIAGNOSTIC_ORACLE_ONLY",
            )
        },
        "criteria": criteria, "pass": decision == "STAGE_E0_CONFIDENCE_CALIBRATION_PASS",
        "stage_e0_20k_permitted": False, "stage_e0_100k_permitted": False,
        "next_command": None, "next_commands": [],
    }
    atomic_json_save(result, output / "validation_result.json")
    print(json.dumps({
        "decision": decision, "smoke": bool(args.limit_records),
        "validation_records_read": len(items), "test_records_read": 0,
        "model_to_target_recovery": recovery, "bond_relative_improvement": relative,
    }, indent=2))


if __name__ == "__main__":
    main()
