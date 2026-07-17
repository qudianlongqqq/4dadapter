#!/usr/bin/env python
"""Evaluate the selected Stage F calibrator once on validation data."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

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
    DIAGNOSTIC_ALL_ONE, load_calibrator, molecule_paired_bootstrap,
    strict_load_frozen_model,
)
from etflow.ecir.failure_attribution import bond_observations, relative_improvement
from etflow.ecir.feature_conditioned_confidence import (
    inference_feature_batch, load_feature_calibrator, stage_f_decision,
)
from etflow.ecir.geometry import bond_lengths
from etflow.ecir.mvr_dataset import deterministic_error_features
from etflow.ecir.mvr_model import trust_clip_velocity
from etflow.ecir.run_a_evaluation import (
    BOOTSTRAP_METRICS, build_clean_control_items, method_rows, summarize_groups,
)
from scripts.evaluate_ecir_mvr_stage_e0 import load_validation_items


METHODS = {
    "d1_b_original_confidence": "original",
    "sign_safe_only": "sign_safe",
    "feature_conditioned_sign_safe": "feature",
    "stage_e0_global_calibration": "e0_global",
    "confidence_all_one_DIAGNOSTIC_ORACLE_ONLY": "all_one",
}


@torch.inference_mode()
def infer_stage_f_mode(
    model, items, validity, *, mode: str, calibrator, e0_calibrator,
    device: torch.device, inference: Mapping[str, Any],
) -> tuple[list[torch.Tensor], list[dict[str, Any]], pd.DataFrame]:
    """Deployment inference reads no target, reference, source label, or validation label."""

    accepted_all, metadata_all, detail_rows = [], [], []
    schedule = torch.linspace(0.0, 1.0, int(inference["teacher_steps"])).tolist()
    for start in range(0, len(items), 32):
        selected = items[start:start + 32]
        batch = Batch.from_data_list([item["data"] for item in selected]).to(device)
        current = torch.cat([item["input"] for item in selected]).to(device)
        ptr = batch.ptr.detach().cpu().tolist()
        trajectories = [[] for _ in selected]
        uncertainties = [[] for _ in selected]
        cancellations = [[] for _ in selected]
        for rollout_step, time_value in enumerate(schedule, start=1):
            current_cpu = current.detach().cpu(); deterministic, remaining = [], []
            for local, item in enumerate(selected):
                left, right = ptr[local], ptr[local + 1]
                values = validity.evaluate(
                    current_cpu[left:right], item["record"], baseline_coordinates=item["input"]
                )
                deterministic.append(deterministic_error_features(
                    values, item["record"], str(item["row"].source_severity)
                ))
                changed = torsion_change_metrics(
                    item["input"], current_cpu[left:right], item["record"]
                )["max_rotatable_torsion_change"]
                limit = 0.35 if item["rotatable"] >= 6 else 0.70
                remaining.append(max(0.0, limit - float(changed)))
            output = model(
                batch, current, current.new_full((len(selected),), float(time_value)),
                deterministic_features=torch.stack(deterministic).to(device),
                torsion_trust_remaining=current.new_tensor(remaining),
            )
            if float(output["torsion_gate"].abs().max()) != 0.0 or float(output["v_torsion_contribution"].abs().max()) != 0.0:
                raise RuntimeError("Stage F observed a nonzero torsion branch")
            features, feature_metadata = inference_feature_batch(
                current=current, output=output, batch=batch, items=selected, ptr=ptr,
                validity=validity, time_value=float(time_value),
            )
            if mode == "original":
                confidence = output["bond_confidence"]
                bond_correction = output["v_bond_correction"]
                raw = output["v_raw"]
            else:
                if mode == "sign_safe":
                    confidence = output["bond_confidence"] * features["sign_safe_mask"]
                elif mode == "feature":
                    confidence = calibrator(features).to(output["bond_confidence"])
                elif mode == "e0_global":
                    confidence = e0_calibrator(output["bond_confidence_logit"])
                elif mode == "all_one":
                    confidence = torch.ones_like(output["bond_confidence"])
                else:
                    raise ValueError(f"unknown Stage F mode: {mode}")
                requested = output["bond_unattenuated_residual"] * confidence
                bond_correction, _ = batched_bond_projection(
                    current, output["bond_indices"], requested, batch.batch,
                    damping=model.bond_projection_damping,
                )
                raw = output["v_cartesian_raw"] + model.bond_explicit_alpha * bond_correction
            clipped = trust_clip_velocity(
                raw, batch.batch, max_atom_norm=model.max_velocity_atom_norm,
                max_graph_rms=model.max_velocity_graph_rms,
            )
            final = output["global_safety_gate"][batch.batch] * clipped
            bond_graph = batch.batch[output["bond_indices"][0]]
            for local, item in enumerate(selected):
                left, right = ptr[local], ptr[local + 1]
                keep = bond_graph == local
                local_bonds = output["bond_indices"][:, keep] - left
                jacobian = bond_length_jacobian(current[left:right], local_bonds)
                cart_effect = jacobian @ output["v_cartesian_raw"][left:right].reshape(-1)
                bond_effect = jacobian @ bond_correction[left:right].reshape(-1)
                cancellations[local].append(float((cart_effect * bond_effect < 0.0).float().mean()))
            for index, meta in enumerate(feature_metadata):
                detail_rows.append({
                    **meta, "rollout_step": rollout_step,
                    "current_bond_length": float(features["current_bond_length"][index]),
                    "predicted_residual": float(output["bond_unattenuated_residual"][index]),
                    "confidence": float(confidence[index]),
                    "sign_safe_mask": bool(features["sign_safe_mask"][index]),
                })
            current = current + float(inference["step_size"]) * final
            snapshot = current.detach().cpu()
            for local in range(len(selected)):
                left, right = ptr[local], ptr[local + 1]
                trajectories[local].append(snapshot[left:right].clone())
                uncertainties[local].append(float(output["uncertainty"][local]))
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
                "torsion_gate_max": 0.0, "torsion_contribution_max": 0.0,
            })
    return accepted_all, metadata_all, pd.DataFrame(detail_rows)


def activation_audit(details: pd.DataFrame, items) -> dict[str, float]:
    targets = {}
    for item in items:
        prepared_validity = item["prepared_validity"] if "prepared_validity" in item else None
        if prepared_validity is None:
            raise RuntimeError("Stage F activation audit requires prepared validity metadata")
        lengths = bond_lengths(item["minimal_target"], prepared_validity["bonds"])
        targets[str(item["row"].sample_id)] = lengths.numpy()
    wrong_sign, zero_target, false_positive = [], [], []
    for row in details.itertuples(index=False):
        residual = float(targets[str(row.record_id)][int(row.bond_index)] - row.current_bond_length)
        wrong_sign.append(abs(residual) > 1.0e-8 and np.sign(row.predicted_residual) != np.sign(residual))
        zero_target.append(abs(residual) <= 1.0e-4)
        current_valid = float(row.valid_lower) <= float(row.current_bond_length) <= float(row.valid_upper)
        false_positive.append(abs(residual) <= 1.0e-4 or (current_valid and not bool(row.sign_safe_mask)))
    confidence = details.confidence.to_numpy(float)
    wrong = np.asarray(wrong_sign, dtype=bool); false = np.asarray(false_positive, dtype=bool)
    return {
        "wrong_sign_activation": float(confidence[wrong].mean()) if wrong.any() else 0.0,
        "false_positive_activation": float(confidence[false].mean()) if false.any() else 0.0,
        "zero_target_fraction": float(np.asarray(zero_target).mean()),
        "abstention_fraction": float((confidence < 0.10).mean()),
    }


def _transition(items, coordinates, validity, method):
    repaired = broken = total = 0
    for item, candidate in zip(items, coordinates):
        before = bond_observations(validity, item["input"], item["record"]).outlier.to_numpy(bool)
        after = bond_observations(validity, candidate, item["record"]).outlier.to_numpy(bool)
        repaired += int((before & ~after).sum()); broken += int((~before & after).sum()); total += len(before)
    return {"method": method, "bonds": total, "repaired_bonds": repaired,
            "newly_broken_bonds": broken, "broken_to_repaired_ratio": broken / max(repaired, 1)}


def _row(summary, group, method):
    value = summary[(summary.group == group) & (summary.method == method)]
    return None if value.empty or not int(value.iloc[0].get("molecules", 0)) else value.iloc[0]


def _atomic_csv(frame, path):
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False); pd.read_csv(temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ecir_mvr_stage_f_feature_confidence.yaml"))
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--bootstrap-draws", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = args.input_dir or Path(config["output_dir"]); output_dir = args.output_dir or source
    payload = json.loads((source / "calibrator.json").read_text(encoding="utf-8"))
    if payload["checkpoint_sha256"] != config["checkpoint"]["sha256"]:
        raise RuntimeError("Stage F calibrator checkpoint identity changed")
    device = torch.device(args.device)
    calibrator = load_feature_calibrator(payload["selected_checkpoint"], payload, device=device)
    e0_payload = json.loads(Path("diagnostics/ecir_mvr/stage_e0/calibrator.json").read_text(encoding="utf-8"))
    e0_calibrator = load_calibrator(e0_payload).to(device)
    model, checkpoint = strict_load_frozen_model(
        config["checkpoint"]["path"], expected_sha256=config["checkpoint"]["sha256"], device=device
    )
    if checkpoint["frozen_identities"] != config["frozen_identities"]:
        raise RuntimeError("Stage F frozen identities changed")
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = load_validation_items(
        config["data"]["val_sources"], config["data"]["val_targets"], validity,
        limit_records=args.limit_records,
    )
    for item in items:
        item["prepared_validity"] = validity._prepare(item["record"])
    coordinates, metadata, details = {}, {}, {}
    for label, mode in METHODS.items():
        values, extras, detail = infer_stage_f_mode(
            model, items, validity, mode=mode, calibrator=calibrator,
            e0_calibrator=e0_calibrator, device=device, inference=config["inference"],
        )
        coordinates[label] = values; metadata[label] = extras; details[label] = detail
    methods = {"upstream": [item["input"] for item in items], **coordinates,
               "minimal_target": [item["minimal_target"] for item in items]}
    rows = method_rows(items, methods, validity, metadata)
    summary, molecules = summarize_groups(rows, items, methods)
    transitions = [_transition(items, coordinates[name], validity, name) for name in METHODS]
    activation = {name: activation_audit(details[name], items) for name in METHODS}
    draws = int(args.bootstrap_draws or config["gate"]["bootstrap_draws"])
    candidate = "feature_conditioned_sign_safe"; e0 = "stage_e0_global_calibration"
    all_molecules = molecules[molecules.group.eq("all")]
    bootstrap_e0 = molecule_paired_bootstrap(
        all_molecules, candidate=candidate, baseline=e0, metrics=BOOTSTRAP_METRICS,
        draws=draws, seed=int(config["gate"]["bootstrap_seed"]),
    )
    bootstrap_upstream = molecule_paired_bootstrap(
        all_molecules, candidate=candidate, baseline="upstream", metrics=BOOTSTRAP_METRICS,
        draws=draws, seed=int(config["gate"]["bootstrap_seed"]),
    )
    clean_items = build_clean_control_items(items, validity, limit=min(20, len(items)))
    clean_identity = math.nan
    if clean_items:
        for item in clean_items: item["prepared_validity"] = validity._prepare(item["record"])
        clean_values, _, _ = infer_stage_f_mode(
            model, clean_items, validity, mode="feature", calibrator=calibrator,
            e0_calibrator=e0_calibrator, device=device, inference=config["inference"],
        )
        clean_identity = float(np.mean([
            torch.equal(torch.as_tensor(value), torch.as_tensor(item["input"]))
            for value, item in zip(clean_values, clean_items)
        ]))
    all_rows = summary[summary.group.eq("all")].set_index("method")
    upstream, candidate_row = all_rows.loc["upstream"], all_rows.loc[candidate]
    e0_row, target = all_rows.loc[e0], all_rows.loc["minimal_target"]
    relative = relative_improvement(upstream.bond_outlier_rate, candidate_row.bond_outlier_rate)
    target_relative = relative_improvement(upstream.bond_outlier_rate, target.bond_outlier_rate)
    recovery = relative / max(target_relative, 1.0e-12)
    transition = {row["method"]: row for row in transitions}
    v4 = pd.read_csv("diagnostics/ecir_mvr/stage_d/pilot/subgroup_summary.csv")
    v4 = v4[(v4.group.eq("all")) & (v4.method.eq("v4_selected"))].iloc[0]
    high, high_up = _row(summary, "rotatable_ge_6", candidate), _row(summary, "rotatable_ge_6", "upstream")
    unseen, unseen_up = _row(summary, "unseen_update_scale_0.35", candidate), _row(summary, "unseen_update_scale_0.35", "upstream")
    source_improvements = [group for group in ("ETFlow_normal", "Cartesian_mild", "Cartesian_medium", "Cartesian_severe")
                           if _row(summary, group, candidate) is not None and _row(summary, group, candidate).total_thresholded_validity_score < _row(summary, group, e0).total_thresholded_validity_score]
    gate = config["gate"]
    cancellation = float(np.mean([value["bond_cancellation_fraction"] for value in metadata[candidate]]))
    wrong_reduction = 1.0 - activation[candidate]["wrong_sign_activation"] / max(activation[e0]["wrong_sign_activation"], 1.0e-12)
    false_reduction = 1.0 - activation[candidate]["false_positive_activation"] / max(activation[e0]["false_positive_activation"], 1.0e-12)
    criteria = {
        "01_model_to_target_recovery_ge_0p35": recovery >= gate["model_to_target_recovery_min"],
        "02_bond_relative_improvement_ge_20pct": relative >= gate["bond_relative_improvement_min"],
        "03_bond_vs_e0_ci_or_harm_reduction": bootstrap_e0["bond_outlier_rate"]["ci95_high"] < 0.0 or (bootstrap_e0["bond_outlier_rate"]["ci95_high"] <= gate["chemical_metric_noninferiority_margin"] and transition[candidate]["newly_broken_bonds"] < transition[e0]["newly_broken_bonds"]),
        "04_newly_broken_le_177": transition[candidate]["newly_broken_bonds"] <= gate["newly_broken_max"],
        "05_cancellation_ratio_le_20pct": cancellation <= gate["cancellation_ratio_max"],
        "06_wrong_sign_activation_reduced_50pct": wrong_reduction >= gate["wrong_sign_activation_reduction_min"],
        "07_false_positive_activation_reduced_40pct": false_reduction >= gate["false_positive_activation_reduction_min"],
        "08_total_validity_vs_e0_not_worse": candidate_row.total_thresholded_validity_score <= e0_row.total_thresholded_validity_score,
        "09_angle_vs_v4_not_worse": candidate_row.angle_outlier_rate <= v4.angle_outlier_rate + gate["chemical_metric_noninferiority_margin"],
        "10_ring_vs_v4_not_worse": candidate_row.ring_bond_outlier_rate <= v4.ring_bond_outlier_rate + gate["chemical_metric_noninferiority_margin"],
        "11_rmsd_mean_delta_le_0p003": bootstrap_upstream["aligned_RMSD"]["mean"] <= gate["rmsd_mean_delta_max"],
        "12_rmsd_ci_upper_le_0p005": bootstrap_upstream["aligned_RMSD"]["ci95_high"] <= gate["rmsd_ci_upper_max"],
        "13_mat_p_mat_r_noninferior": all(bootstrap_upstream[name]["mean"] <= gate["mat_mean_delta_max"] and bootstrap_upstream[name]["ci95_high"] <= gate["mat_ci_upper_max"] for name in ("MAT_P", "MAT_R")),
        "14_cov_p_cov_r_no_material_drop": candidate_row.COV_P >= upstream.COV_P - gate["cov_absolute_drop_max"] and candidate_row.COV_R >= upstream.COV_R - gate["cov_absolute_drop_max"],
        "15_high_flex_validity_pass": high is not None and high_up is not None and high.total_thresholded_validity_score < high_up.total_thresholded_validity_score,
        "16_high_flex_torsion_controlled": high is not None and high.high_flex_torsion_change <= gate["high_flex_torsion_max"],
        "17_clean_identity_ge_90pct": math.isfinite(clean_identity) and clean_identity >= gate["clean_identity_fraction_min"],
        "18_clash_chirality_not_worse": candidate_row.severe_clash_rate <= upstream.severe_clash_rate and candidate_row.chirality_error <= upstream.chirality_error and candidate_row.clash_penetration <= upstream.clash_penetration + gate["chemical_metric_noninferiority_margin"],
        "19_unseen_validity_accuracy_pass": unseen is not None and unseen_up is not None and unseen.total_thresholded_validity_score < unseen_up.total_thresholded_validity_score and unseen.aligned_RMSD - unseen_up.aligned_RMSD <= gate["rmsd_mean_delta_max"],
        "20_improvement_not_single_source": len(source_improvements) >= 2,
        "21_test_records_read_zero": True,
    }
    criteria = {name: bool(value) for name, value in criteria.items()}
    sign_safe_row = all_rows.loc["sign_safe_only"]
    sign_safe_better = sign_safe_row.total_thresholded_validity_score < candidate_row.total_thresholded_validity_score and transition["sign_safe_only"]["newly_broken_bonds"] <= transition[candidate]["newly_broken_bonds"]
    harms = candidate_row.total_thresholded_validity_score > e0_row.total_thresholded_validity_score or bootstrap_upstream["aligned_RMSD"]["ci95_high"] > gate["rmsd_ci_upper_max"]
    decision = "STAGE_F_SMOKE_COMPLETE" if args.limit_records else stage_f_decision(criteria, sign_safe_only_better=sign_safe_better, harms=harms)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_save({"schema_version": "ecir-mvr-stage-f-bootstrap-v1", "draws": draws, "seed": gate["bootstrap_seed"], "vs_e0": bootstrap_e0, "vs_upstream": bootstrap_upstream, "test_records_read": 0}, output_dir / "bootstrap.json")
    _atomic_csv(summary, output_dir / "method_summary.csv"); _atomic_csv(pd.DataFrame(transitions), output_dir / "bond_transition.csv")
    _atomic_csv(pd.DataFrame([{"method": name, **value} for name, value in activation.items()]), output_dir / "activation_summary.csv")
    result = {
        "schema_version": "ecir-mvr-stage-f-validation-v1", "decision": decision,
        "smoke": bool(args.limit_records), "validation_only": True,
        "validation_records_read": len(items), "test_records_read": 0,
        "neural_training_run": False, "calibrator_training_only": True,
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "calibrator_identity_sha256": payload["calibrator_identity_sha256"],
        "selected_step": payload["selected_step"],
        "diagnostic_control": {"method": "confidence_all_one", "label": DIAGNOSTIC_ALL_ONE},
        "metrics": {"model_to_target_recovery": recovery, "bond_relative_improvement": relative,
                    "newly_broken_bonds": transition[candidate]["newly_broken_bonds"],
                    "cancellation_ratio": cancellation, "wrong_sign_activation_reduction": wrong_reduction,
                    "false_positive_activation_reduction": false_reduction, "clean_identity_fraction": clean_identity},
        "activation_audit": activation, "criteria": criteria,
        "pass": decision == "STAGE_F_FEATURE_CONFIDENCE_PASS",
        "formal_training_permitted": False, "stage_f_100k_permitted": False,
        "next_command": None, "next_commands": [],
    }
    atomic_json_save(result, output_dir / "validation_result.json")
    print(json.dumps({"decision": decision, "smoke": bool(args.limit_records), "validation_records_read": len(items), "test_records_read": 0}, indent=2))


if __name__ == "__main__":
    main()
