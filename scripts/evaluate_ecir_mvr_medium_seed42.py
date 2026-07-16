#!/usr/bin/env python
"""Validation-only Medium Gate 2 evaluation for the seed42 rigid-only model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from _bootstrap import bootstrap as bootstrap_path
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap as bootstrap_path

bootstrap_path()

import numpy as np
import pandas as pd
import torch
import yaml

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.commons.run_timing import RunTiming
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.model import ECIRFlowSystem
from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir.run_a_evaluation import (
    BOOTSTRAP_METRICS, CHEMICAL_METRICS, accuracy_gate, build_clean_control_items,
    build_items, infer_mvr, method_rows, summarize_groups,
)
from scripts.evaluate_ecir_mvr_run_a import OLD_ECIR_SHA256, infer_old_ecir


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(summary: pd.DataFrame, group: str, method: str):
    rows = summary[(summary.group == group) & (summary.method == method)]
    return None if rows.empty or int(rows.iloc[0].get("molecules", 0)) == 0 else rows.iloc[0]


def _neutral(candidate, baseline, *, rmsd=0.015, mat=0.015, cov=0.005) -> bool:
    if candidate is None or baseline is None:
        return False
    return bool(
        candidate.aligned_RMSD - baseline.aligned_RMSD <= rmsd
        and candidate.MAT_P - baseline.MAT_P <= mat
        and candidate.MAT_R - baseline.MAT_R <= mat
        and candidate.COV_P >= baseline.COV_P - cov
        and candidate.COV_R >= baseline.COV_R - cov
    )


def _group_bootstrap(molecules: pd.DataFrame, group: str, candidate: str, draws: int) -> dict:
    frame = molecules[molecules.group == group]
    result = {}
    for metric in BOOTSTRAP_METRICS:
        pivot = frame.pivot(index="molecule_id", columns="method", values=metric).dropna()
        if candidate not in pivot or "upstream" not in pivot:
            result[metric] = {"mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
            continue
        delta = pivot[candidate].to_numpy() - pivot["upstream"].to_numpy()
        rng = np.random.default_rng(42)
        means = np.asarray([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(draws)])
        result[metric] = {
            "mean": float(delta.mean()),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
        }
    return result


def _gate(summary, bootstrap_all, margins, clean_identity, training_completed):
    method = "medium_accepted"
    candidate, upstream = _row(summary, "all", method), _row(summary, "all", "upstream")
    raw = _row(summary, "all", "medium_raw")
    et, et_up = _row(summary, "ETFlow_normal", method), _row(summary, "ETFlow_normal", "upstream")
    mild, mild_up = _row(summary, "Cartesian_mild", method), _row(summary, "Cartesian_mild", "upstream")
    medium, medium_up = _row(summary, "Cartesian_medium", method), _row(summary, "Cartesian_medium", "upstream")
    severe, severe_up = _row(summary, "Cartesian_severe", method), _row(summary, "Cartesian_severe", "upstream")
    high, high_up = _row(summary, "rotatable_ge_6", method), _row(summary, "rotatable_ge_6", "upstream")
    unseen, unseen_up = _row(summary, "unseen_update_scale_0.35", method), _row(summary, "unseen_update_scale_0.35", "upstream")
    nonring, nonring_up = _row(summary, "non_ring", method), _row(summary, "non_ring", "upstream")
    improved_ci = [
        metric for metric in CHEMICAL_METRICS
        if metric != "total_thresholded_validity_score" and bootstrap_all[metric]["ci95_high"] < 0.0
    ]
    core = ("bond_outlier_rate", "angle_outlier_rate", "ring_bond_outlier_rate", "clash_penetration")
    relative = {
        metric: ((float(upstream[metric]) - float(candidate[metric])) / float(upstream[metric]) if float(upstream[metric]) > 1e-12 else 0.0)
        for metric in core
    }
    accuracy = accuracy_gate(summary.rename(columns={}), bootstrap_all, margins, method=method)
    source_improvements = []
    for group in ("ETFlow_normal", "Cartesian_mild", "Cartesian_medium", "Cartesian_severe"):
        a, b = _row(summary, group, method), _row(summary, group, "upstream")
        if a is not None and b is not None and a.total_thresholded_validity_score < b.total_thresholded_validity_score:
            source_improvements.append(group)
    conditions = {
        "01_two_real_chemical_metrics_ci_improve": len(improved_ci) >= 2,
        "02_one_core_metric_relative_improvement_ge_10pct": max(relative.values()) >= 0.10,
        "03_total_validity_ci_improves": bootstrap_all["total_thresholded_validity_score"]["ci95_high"] < 0.0,
        "04_validity_worsened_fraction_not_increased": candidate.validity_worsened_fraction <= upstream.validity_worsened_fraction + 1e-12,
        "05_rmsd_mean": accuracy["rmsd_mean"], "06_rmsd_ci": accuracy["rmsd_ci"],
        "07_mat_p_mean": accuracy["mat_p_mean"], "08_mat_p_ci": accuracy["mat_p_ci"],
        "09_mat_r_mean": accuracy["mat_r_mean"], "10_mat_r_ci": accuracy["mat_r_ci"],
        "11_cov_p_cov_r": accuracy["cov_p"] and accuracy["cov_r"],
        "12_diversity_not_collapsed": candidate.diversity >= upstream.diversity - 0.02,
        "13_etflow_accuracy_neutral": _neutral(et, et_up),
        "14_mild_and_medium_validity_improve": mild is not None and medium is not None and mild.total_thresholded_validity_score < mild_up.total_thresholded_validity_score and medium.total_thresholded_validity_score < medium_up.total_thresholded_validity_score,
        "15_severe_accuracy_no_collapse_if_available": severe is None or _neutral(severe, severe_up, rmsd=0.02, mat=0.02),
        "16_high_flex_validity_improves": high is not None and high.total_thresholded_validity_score < high_up.total_thresholded_validity_score,
        "17_high_flex_rmsd_noninferior": high is not None and high.aligned_RMSD - high_up.aligned_RMSD <= 0.02,
        "18_high_flex_torsion_controlled": high is not None and high.high_flex_torsion_change <= 0.05,
        "19_clean_identity_ge_90pct": math.isfinite(clean_identity) and clean_identity >= 0.90,
        "20_severe_clash_not_increased": candidate.severe_clash_rate <= upstream.severe_clash_rate + 1e-12,
        "21_chirality_not_worse": candidate.chirality_error <= upstream.chirality_error + 1e-12,
        "22_acceptance_reduces_validity_worsened": raw is not None and candidate.validity_worsened_fraction < raw.validity_worsened_fraction,
        "23_acceptance_reduces_rmsd_worsened": raw is not None and candidate.RMSD_worsened_fraction < raw.RMSD_worsened_fraction,
        "24_unseen_validity_improves": unseen is not None and unseen.total_thresholded_validity_score < unseen_up.total_thresholded_validity_score,
        "25_unseen_accuracy_noninferior": _neutral(unseen, unseen_up, rmsd=0.02, mat=0.02),
        "26_non_ring_no_abnormal_failure": nonring is None or (_neutral(nonring, nonring_up, rmsd=0.02, mat=0.02) and nonring.severe_clash_rate <= nonring_up.severe_clash_rate + 1e-12),
        "27_improvement_not_single_source": len(source_improvements) >= 2,
    }
    return {
        "training_completed": bool(training_completed),
        "conditions": {key: bool(value) for key, value in conditions.items()},
        "accuracy_gate": {key: bool(value) for key, value in accuracy.items()},
        "improved_ci_metrics": improved_ci, "relative_improvements": relative,
        "improving_sources": source_improvements,
        "pass": bool(training_completed and all(conditions.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--run_a_checkpoint", type=Path, required=True)
    parser.add_argument("--old_ecir_checkpoint", type=Path, default=Path("logs_ecir/stage2_heterogeneous_500_100_5k/step005000.ckpt"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap_draws", type=int, default=1000)
    parser.add_argument("--timing_dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    timing = RunTiming(args.timing_dir) if args.timing_dir else None
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight["status"] != "PASS" or preflight["test_records_read"] != 0 or preflight["identities"] != config["frozen_identities"]:
        raise RuntimeError("medium preflight identity is not a test-free PASS")
    if _sha(args.run_a_checkpoint) != "ac3e7e3b1fa4189e8ccdfeb45ea7c799a7130c213aeed017c301218b71487070":
        raise RuntimeError("Run A pilot checkpoint changed")
    if _sha(args.old_ecir_checkpoint) != OLD_ECIR_SHA256:
        raise RuntimeError("Stage B rescued checkpoint changed")
    if timing:
        timing.mark("final_evaluation_start", checkpoint=str(args.checkpoint))
    device = torch.device(args.device)
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = build_items(config["data"]["val_sources"], config["data"]["val_targets"], validity)
    clean_items = build_clean_control_items(items, validity, limit=20)

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    medium_model = MCVRModel(**config["model"]).to(device)
    medium_model.load_state_dict(payload["model_state_dict"], strict=True)
    medium_raw, medium_accepted, medium_meta = infer_mvr(
        medium_model, items, validity, device=device, steps=4, step_size=0.25,
        acceptance_mode="best_of_trajectory",
    )
    run_a_payload = torch.load(args.run_a_checkpoint, map_location=device, weights_only=False)
    run_a_model = MCVRModel(**run_a_payload["config"]["model"]).to(device)
    run_a_model.load_state_dict(run_a_payload["model_state_dict"], strict=True)
    _, run_a_accepted, run_a_meta = infer_mvr(
        run_a_model, items, validity, device=device, steps=4, step_size=0.25,
        acceptance_mode="best_of_trajectory",
    )
    old_payload = torch.load(args.old_ecir_checkpoint, map_location=device, weights_only=False)
    old_model = ECIRFlowSystem(**old_payload["config"]["model"]).to(device)
    old_model.load_state_dict(old_payload["model_state_dict"], strict=True)
    rescued, rescued_meta = infer_old_ecir(old_model, items, validity, device=device, rescued=True)
    methods = {
        "upstream": [item["input"] for item in items],
        "stage_b_rescued": rescued,
        "run_a_stage2b": run_a_accepted,
        "minimal_target": [item["minimal_target"] for item in items],
        "medium_raw": medium_raw,
        "medium_accepted": medium_accepted,
    }
    metadata = {
        "stage_b_rescued": rescued_meta, "run_a_stage2b": run_a_meta,
        "medium_raw": medium_meta, "medium_accepted": medium_meta,
    }
    records = method_rows(items, methods, validity, metadata)
    summary, molecules = summarize_groups(records, items, methods)

    _, clean_accepted, clean_meta = infer_mvr(
        medium_model, clean_items, validity, device=device, steps=4, step_size=0.25,
        acceptance_mode="best_of_trajectory",
    )
    clean_methods = {"upstream": [item["input"] for item in clean_items], "medium_accepted": clean_accepted}
    clean_records = method_rows(clean_items, clean_methods, validity, {"medium_accepted": clean_meta})
    clean_summary, _ = summarize_groups(clean_records, clean_items, clean_methods)
    clean_row = _row(clean_summary, "clean_valid", "medium_accepted")
    clean_identity = float(clean_row.unchanged_fraction) if clean_row is not None else math.nan
    summary = pd.concat([summary[summary.group != "clean_valid"], clean_summary[clean_summary.group == "clean_valid"]], ignore_index=True)

    if timing:
        timing.mark("final_evaluation_end", checkpoint_step=int(payload["step"]))
        timing.mark("bootstrap_start", draws=args.bootstrap_draws)
    groups = [group for group in summary.group.unique() if _row(summary, group, "medium_accepted") is not None]
    bootstraps = {group: _group_bootstrap(molecules, group, "medium_accepted", args.bootstrap_draws) for group in groups if group != "clean_valid"}
    if timing:
        timing.mark("bootstrap_end", draws=args.bootstrap_draws)
    run_metadata = json.loads((Path(config["output_dir"]) / "run_metadata.json").read_text(encoding="utf-8"))
    training_completed = run_metadata["status"] == "COMPLETED" and run_metadata.get("completed_steps") == 20000
    gate = _gate(summary, bootstraps["all"], config["noninferiority"], clean_identity, training_completed)
    decision = "MEDIUM_SEED42_PASS" if gate["pass"] else "MEDIUM_SEED42_FAIL"
    result = {
        "schema_version": "ecir-mvr-medium-seed42-result-v1", "decision": decision,
        "current_stage": (
            "MEDIUM_SEED42_RESCUE_V3_COMPLETE" if "rescue_v3" in config["experiment_name"]
            else "MEDIUM_SEED42_RESCUE_V2_COMPLETE" if "rescue_v2" in config["experiment_name"]
            else "MEDIUM_SEED42_COMPLETE"
        ), "validation_only": True,
        "test_records_read": 0, "20k_started": True, "20k_completed": training_completed,
        "100k_permitted": False, "100k_started": False, "next_commands": [],
        "training_status": run_metadata["status"], "completed_steps": run_metadata["completed_steps"],
        "stop_reason": run_metadata["stop_reason"], "config_sha256": _sha(args.config),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha(args.checkpoint),
        "run_a_checkpoint_sha256": _sha(args.run_a_checkpoint),
        "medium_real_source_identity_sha256": config["frozen_identities"]["medium_real_source_identity_sha256"],
        "medium_target_identity_sha256": config["frozen_identities"]["medium_target_identity_sha256"],
        "records": len(items), "molecules": len({str(item["row"].molecule_id) for item in items}),
        "clean_control_molecules": len(clean_items), "clean_identity_fraction": clean_identity,
        "acceptance_fraction": float(np.mean([value["accepted"] for value in medium_meta])),
        "torsion_gate_max": float(max(value["torsion_gate_max"] for value in medium_meta)),
        "torsion_contribution_max": float(max(value["torsion_contribution_max"] for value in medium_meta)),
        "gate": gate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_save(result, args.output_dir / "result.json")
    atomic_json_save({"schema_version": "paired-molecule-bootstrap-v1", "draws": args.bootstrap_draws, "seed": 42, "candidate": "medium_accepted", "baseline": "upstream", "groups": bootstraps}, args.output_dir / "bootstrap.json")
    summary.to_csv(args.output_dir / "source_summary.csv", index=False)
    summary[summary.group.isin(["rotatable_le_2", "rotatable_3_5", "rotatable_ge_6", "ring", "non_ring", "clean_valid"])].to_csv(args.output_dir / "flexibility_summary.csv", index=False)
    summary[summary.method.isin(["upstream", "medium_raw", "medium_accepted"])].to_csv(args.output_dir / "acceptance_summary.csv", index=False)
    records.to_csv(args.output_dir / "record_metrics.csv", index=False)
    molecules.to_csv(args.output_dir / "molecule_metrics.csv", index=False)
    pd.DataFrame([{
        "step": int(payload["step"]), "checkpoint": str(args.checkpoint.resolve()),
        "scheduled_validation_checkpoint": False, "training_completed": training_completed,
        "accuracy_noninferior": all(gate["accuracy_gate"].values()),
        "gate2_pass": gate["pass"], "stop_reason": run_metadata["stop_reason"],
        "checkpoint_sha256": _sha(args.checkpoint),
    }]).to_csv(args.output_dir / "checkpoint_comparison.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
