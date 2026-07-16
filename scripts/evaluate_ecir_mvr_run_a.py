#!/usr/bin/env python
"""Run the frozen, validation-only Stage 2b Run A comparison and decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

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
from etflow.ecir.acceptance import evaluate_candidate
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.model import ECIRFlowSystem
from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir.run_a_evaluation import (
    CHEMICAL_METRICS,
    accuracy_gate,
    build_clean_control_items,
    build_items,
    infer_mvr,
    method_rows,
    paired_bootstrap,
    summarize_groups,
)
from etflow.ecir.target_building import restrained_force_field_relaxation


OLD_ECIR_SHA256 = "232e47865d01a71543cf2cd16ede577764fd3d94ac843d78dcdcf8c9789fa98d"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


@torch.inference_mode()
def infer_old_ecir(model, items, validity, *, device, rescued: bool, batch_size: int = 32):
    model.eval()
    coordinates = []
    metadata = []
    steps = 2 if rescued else 4
    trust = 0.5 if rescued else 1.0
    for start in range(0, len(items), batch_size):
        selected = items[start:start + batch_size]
        batch = Batch.from_data_list([item["data"] for item in selected]).to(device)
        source = torch.cat([item["input"] for item in selected]).to(device)
        refined, diagnostics = model.refine(
            batch, coordinates=source, steps=steps, update_scale=1.0,
            trust_radius_scale=trust, gate_threshold=0.0,
            time_schedule_mode="train_range", strict_training_range=True,
            return_trajectory=True,
        )
        refined = refined.detach().cpu()
        ptr = batch.ptr.detach().cpu().tolist()
        last = diagnostics[-1]
        uncertainties = last["graph_uncertainty"].detach().cpu().tolist()
        gates = last["graph_gate"].detach().cpu().tolist()
        for local, item in enumerate(selected):
            left, right = ptr[local], ptr[local + 1]
            raw = refined[left:right].clone()
            if rescued:
                decision = evaluate_candidate(
                    item["input"], raw, item["record"], validity, step=steps,
                    uncertainty=float(uncertainties[local]),
                    input_validity_override=item["input_validity"],
                )
                candidate = raw if decision.accepted else item["input"]
                accepted = decision.accepted
                reasons = ";".join(decision.reject_reasons)
            else:
                candidate = raw
                accepted = True
                reasons = ""
            coordinates.append(candidate)
            metadata.append({
                "accepted": accepted, "selected_step": steps if accepted else 0,
                "uncertainty": float(uncertainties[local]),
                "rigid_gate_mean": float(gates[local]), "reject_reasons": reasons,
            })
    return coordinates, metadata


def old_restrained_targets(items):
    coordinates = []
    metadata = []
    for item in items:
        result = restrained_force_field_relaxation(
            item["record"], item["input"], max_steps=50
        )
        accepted = result.coordinates is not None
        coordinates.append(
            torch.as_tensor(result.coordinates, dtype=torch.float32)
            if accepted else item["input"]
        )
        metadata.append({
            "accepted": accepted, "selected_step": int(result.steps or 0),
            "reject_reasons": "" if accepted else str(
                result.rejection_reason or result.method
            ),
        })
    return coordinates, metadata


def _row(summary: pd.DataFrame, group: str, method: str):
    selected = summary[(summary.group == group) & (summary.method == method)]
    return None if selected.empty or int(selected.iloc[0].get("molecules", 0)) == 0 else selected.iloc[0]


def _group_accuracy_neutral(candidate, baseline, rmsd=0.015, mat=0.015, cov=0.005):
    if candidate is None or baseline is None:
        return False
    return bool(
        candidate.aligned_RMSD - baseline.aligned_RMSD <= rmsd
        and candidate.MAT_P - baseline.MAT_P <= mat
        and candidate.MAT_R - baseline.MAT_R <= mat
        and candidate.COV_P >= baseline.COV_P - cov
        and candidate.COV_R >= baseline.COV_R - cov
    )


def make_decision(summary, bootstrap, margins, clean_identity):
    candidate = _row(summary, "all", "run_a_accepted")
    upstream = _row(summary, "all", "upstream")
    raw = _row(summary, "all", "run_a_raw")
    high = _row(summary, "rotatable_ge_6", "run_a_accepted")
    high_up = _row(summary, "rotatable_ge_6", "upstream")
    high_old = _row(summary, "rotatable_ge_6", "old_ecir_4step")
    et = _row(summary, "ETFlow_normal", "run_a_accepted")
    et_up = _row(summary, "ETFlow_normal", "upstream")
    unseen = _row(summary, "unseen_update_scale_0.35", "run_a_accepted")
    unseen_up = _row(summary, "unseen_update_scale_0.35", "upstream")
    ci_improved = [
        metric for metric in CHEMICAL_METRICS
        if metric not in {"total_thresholded_validity_score"}
        and bootstrap[metric]["ci95_high"] < 0.0
    ]
    relative_candidates = (
        "bond_outlier_rate", "angle_outlier_rate", "ring_bond_outlier_rate",
        "ring_planarity_outlier_rate", "clash_penetration", "severe_clash_rate",
    )
    relative_improvements = {}
    for metric in relative_candidates:
        base = float(upstream[metric])
        value = float(candidate[metric])
        relative_improvements[metric] = (base - value) / base if base > 1e-12 else 0.0
    accuracy = {
        name: bool(value)
        for name, value in accuracy_gate(summary, bootstrap, margins).items()
    }
    conditions = {
        "01_two_real_chemical_metrics_ci_improve": len(ci_improved) >= 2,
        "02_one_core_validity_metric_relative_improvement_ge_10pct": max(relative_improvements.values()) >= 0.10,
        "03_total_validity_improves": candidate.total_thresholded_validity_score < upstream.total_thresholded_validity_score,
        "04_validity_worsened_fraction_not_increased": candidate.validity_worsened_fraction <= upstream.validity_worsened_fraction + 1e-12,
        "05_rmsd_mean_delta": accuracy["rmsd_mean"],
        "06_rmsd_ci_upper": accuracy["rmsd_ci"],
        "07_mat_p_mean_delta": accuracy["mat_p_mean"],
        "08_mat_p_ci_upper": accuracy["mat_p_ci"],
        "09_mat_r_mean_delta": accuracy["mat_r_mean"],
        "10_mat_r_ci_upper": accuracy["mat_r_ci"],
        "11_cov_p_drop": accuracy["cov_p"],
        "12_cov_r_drop": accuracy["cov_r"],
        "13_diversity_not_collapsed": candidate.diversity >= upstream.diversity - 0.02,
        "14_etflow_normal_accuracy_neutral": _group_accuracy_neutral(et, et_up),
        "15_high_flex_rmsd_not_directionally_worse": high is not None and high_up is not None and high.aligned_RMSD - high_up.aligned_RMSD <= 0.02,
        "16_high_flex_torsion_controlled": high is not None and high_old is not None and high.high_flex_torsion_change < high_old.high_flex_torsion_change,
        "17_clean_identity_ge_90pct": _finite(clean_identity) and clean_identity >= 0.90,
        "18_severe_clash_not_increased": candidate.severe_clash_rate <= upstream.severe_clash_rate + 1e-12,
        "19_chirality_not_worse": candidate.chirality_error <= upstream.chirality_error + 1e-12,
        "20_acceptance_reduces_rmsd_worsened_fraction": raw is not None and candidate.RMSD_worsened_fraction < raw.RMSD_worsened_fraction,
        "21_unseen_validity_improves": unseen is not None and unseen_up is not None and unseen.total_thresholded_validity_score < unseen_up.total_thresholded_validity_score,
        "22_unseen_accuracy_noninferior": _group_accuracy_neutral(unseen, unseen_up, rmsd=0.02, mat=0.02, cov=0.005),
    }
    conditions = {name: bool(value) for name, value in conditions.items()}
    hard_accuracy = all(conditions[f"{index:02d}_{name}"] for index, name in (
        (5, "rmsd_mean_delta"), (6, "rmsd_ci_upper"), (7, "mat_p_mean_delta"),
        (8, "mat_p_ci_upper"), (9, "mat_r_mean_delta"), (10, "mat_r_ci_upper"),
        (11, "cov_p_drop"), (12, "cov_r_drop"),
    ))
    hard_failure = (
        not hard_accuracy
        or not conditions["01_two_real_chemical_metrics_ci_improve"]
        or not conditions["03_total_validity_improves"]
        or not conditions["17_clean_identity_ge_90pct"]
        or not conditions["18_severe_clash_not_increased"]
        or not conditions["19_chirality_not_worse"]
    )
    decision = "RUN_A_PASS" if all(conditions.values()) else (
        "RUN_A_FAIL" if hard_failure else "RUN_A_PARTIAL"
    )
    return decision, conditions, ci_improved, relative_improvements, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_audit", type=Path, required=True)
    parser.add_argument("--old_ecir_checkpoint", type=Path, default=Path("logs_ecir/stage2_heterogeneous_500_100_5k/step005000.ckpt"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap_draws", type=int, default=1000)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit = json.loads(args.data_audit.read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["test_records_read"] != 0:
        raise RuntimeError("data audit is not a test-free PASS")
    if audit["identities"] != config["frozen_identities"]:
        raise RuntimeError("frozen identities differ from audited identities")
    if _sha(args.old_ecir_checkpoint) != OLD_ECIR_SHA256:
        raise RuntimeError("old ECIR checkpoint identity mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Run A evaluation requires CUDA")
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = build_items(config["data"]["val_sources"], config["data"]["val_targets"], validity)
    clean_items = build_clean_control_items(items, validity, limit=20)
    if len(clean_items) < 10:
        raise RuntimeError("insufficient clean validation reference controls")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "MCVRModel" or checkpoint.get("run_mode") != "rigid_only":
        raise RuntimeError("checkpoint is not the authorized rigid-only MCVR model")
    model = MCVRModel(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    raw, accepted, run_metadata = infer_mvr(
        model, items, validity, device=device,
        steps=int(config["inference"]["teacher_steps"]),
        step_size=float(config["inference"]["step_size"]),
        acceptance_mode=config["inference"]["acceptance_mode"],
    )

    old_payload = torch.load(args.old_ecir_checkpoint, map_location=device, weights_only=False)
    old_model = ECIRFlowSystem(**old_payload["config"]["model"]).to(device)
    old_model.load_state_dict(old_payload["model_state_dict"], strict=True)
    historical, historical_meta = infer_old_ecir(old_model, items, validity, device=device, rescued=False)
    rescued, rescued_meta = infer_old_ecir(old_model, items, validity, device=device, rescued=True)
    restrained, restrained_meta = old_restrained_targets(items)

    method_coordinates = {
        "upstream": [item["input"] for item in items],
        "old_ecir_4step": historical,
        "stage_b_rescued": rescued,
        "minimal_validity_target": [item["minimal_target"] for item in items],
        "run_a_raw": raw,
        "run_a_accepted": accepted,
        "old_restrained_mmff_target": restrained,
    }
    metadata = {
        "old_ecir_4step": historical_meta, "stage_b_rescued": rescued_meta,
        "run_a_raw": run_metadata, "run_a_accepted": run_metadata,
        "old_restrained_mmff_target": restrained_meta,
    }
    records = method_rows(items, method_coordinates, validity, metadata)
    summary, molecules = summarize_groups(records, items, method_coordinates)
    bootstrap_results = {
        method: paired_bootstrap(
            molecules, candidate=method, baseline="upstream",
            draws=int(args.bootstrap_draws), seed=42,
        )
        for method in method_coordinates if method != "upstream"
    }

    _, clean_accepted, clean_meta = infer_mvr(
        model, clean_items, validity, device=device,
        steps=int(config["inference"]["teacher_steps"]),
        step_size=float(config["inference"]["step_size"]),
        acceptance_mode=config["inference"]["acceptance_mode"],
    )
    clean_coordinates = {
        "upstream": [item["input"] for item in clean_items],
        "run_a_accepted": clean_accepted,
    }
    clean_records = method_rows(
        clean_items, clean_coordinates, validity,
        {"run_a_accepted": clean_meta},
    )
    clean_summary, _ = summarize_groups(clean_records, clean_items, clean_coordinates)
    clean_row = _row(clean_summary, "clean_valid", "run_a_accepted")
    clean_identity = float(clean_row.unchanged_fraction) if clean_row is not None else math.nan
    summary = pd.concat([
        summary[summary.group != "clean_valid"],
        clean_summary[clean_summary.group == "clean_valid"],
    ], ignore_index=True)

    decision, conditions, improved, relative, accuracy = make_decision(
        summary, bootstrap_results["run_a_accepted"],
        config["noninferiority"], clean_identity,
    )
    accepted_fraction = float(np.mean([value["accepted"] for value in run_metadata]))
    result = {
        "schema_version": "ecir-mvr-stage2b-run-a-result-v1",
        "decision": decision, "manual_recommendation": "DO_NOT_START_RUN_B_AUTOMATICALLY",
        "next_command": None, "20k_permitted": False, "100k_permitted": False,
        "validation_only": True, "test_records_read": 0,
        "config_sha256": _sha(args.config), "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha(args.checkpoint),
        "old_ecir_checkpoint_sha256": _sha(args.old_ecir_checkpoint),
        "frozen_identities": config["frozen_identities"],
        "records": len(items), "molecules": len({str(item['row'].molecule_id) for item in items}),
        "clean_control_molecules": len(clean_items), "clean_identity_fraction": clean_identity,
        "acceptance_fraction": accepted_fraction,
        "torsion_gate_max": float(max(value["torsion_gate_max"] for value in run_metadata)),
        "torsion_contribution_max": float(max(value["torsion_contribution_max"] for value in run_metadata)),
        "accuracy_gate": accuracy, "conditions": conditions,
        "chemical_metrics_with_fully_improved_ci": improved,
        "core_relative_improvements": relative,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_save(result, args.output_dir / "result.json")
    atomic_json_save({
        "schema_version": "paired-molecule-bootstrap-v1", "draws": int(args.bootstrap_draws),
        "seed": 42, "baseline": "upstream", "methods": bootstrap_results,
    }, args.output_dir / "bootstrap.json")
    summary.to_csv(args.output_dir / "source_summary.csv", index=False)
    summary[summary.group.isin([
        "rotatable_le_2", "rotatable_3_5", "rotatable_ge_6", "ring", "non_ring", "clean_valid"
    ])].to_csv(args.output_dir / "flexibility_summary.csv", index=False)
    summary[summary.method.isin(["upstream", "run_a_raw", "run_a_accepted"])][[
        "group", "method", "molecules", "records", "accepted_fraction", "rejected_fraction",
        "unchanged_fraction", "validity_improved_fraction", "validity_worsened_fraction",
        "RMSD_improved_fraction", "RMSD_worsened_fraction", "mean_displacement",
        "p95_displacement", "max_displacement", "mean_torsion_change",
        "high_flex_torsion_change", "selected_step", "uncertainty",
    ]].to_csv(args.output_dir / "acceptance_summary.csv", index=False)
    records.to_csv(args.output_dir / "record_metrics.csv", index=False)
    molecules.to_csv(args.output_dir / "molecule_metrics.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
