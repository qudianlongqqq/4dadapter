#!/usr/bin/env python
"""Evaluate Run B against both upstream and frozen Run A, then apply Gate 1."""

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

import pandas as pd
import torch
import yaml

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir.run_a_evaluation import build_clean_control_items, build_items
from etflow.ecir.run_b_evaluation import (
    evaluate_three_way, incremental_accuracy_gate, paired_group_bootstrap,
)
from scripts.evaluate_ecir_mvr_run_a import make_decision


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _row(summary, group, method):
    values = summary[(summary.group == group) & (summary.method == method)]
    return None if values.empty or int(values.iloc[0].molecules) == 0 else values.iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap_draws", type=int, default=1000)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit = json.loads(args.data_audit.read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["test_records_read"] != 0:
        raise RuntimeError("Run B evaluation requires test-free data audit PASS")
    if audit["identities"] != config["frozen_identities"]:
        raise RuntimeError("frozen identity mismatch")
    run_a_result = json.loads(Path(
        "diagnostics/ecir_mvr/stage2b/run_a/result.json"
    ).read_text(encoding="utf-8"))
    if run_a_result["decision"] != "RUN_A_PASS":
        raise RuntimeError("Run A frozen PASS is unavailable")
    if _sha(config["run_a_checkpoint"]) != config["run_a_checkpoint_sha256"]:
        raise RuntimeError("Run A checkpoint changed")
    device = torch.device(args.device)
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = build_items(config["data"]["val_sources"], config["data"]["val_targets"], validity)
    clean = build_clean_control_items(items, validity, limit=20)
    run_a_payload = torch.load(config["run_a_checkpoint"], map_location=device, weights_only=False)
    run_a_model = MCVRModel(**run_a_payload["config"]["model"]).to(device)
    run_a_model.load_state_dict(run_a_payload["model_state_dict"], strict=True)
    run_b_payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if run_b_payload.get("run_mode") != "rigid_plus_conservative_torsion":
        raise RuntimeError("selected checkpoint is not Run B")
    run_b_model = MCVRModel(**config["model"]).to(device)
    run_b_model.load_state_dict(run_b_payload["model_state_dict"], strict=True)
    full = evaluate_three_way(
        run_b_model, run_a_model, items, validity, device=device,
        inference=config["inference"], upstream_margins=config["noninferiority"],
        incremental_margins=config["incremental_noninferiority"],
        bootstrap_draws=int(args.bootstrap_draws), clean_control_items=clean,
    )
    summary = full["summary"]
    molecules = full["molecule_rows"]
    bootstrap_up = full["bootstrap_vs_upstream"]
    bootstrap_a = full["bootstrap_vs_run_a"]
    high = full["bootstrap_high_flex_vs_run_a"]
    unseen = paired_group_bootstrap(
        molecules, candidate="run_b_accepted", baseline="run_a_accepted",
        group="unseen_update_scale_0.35", draws=int(args.bootstrap_draws),
    )

    # Reuse the frozen Run A 22-condition definitions with Run B substituted as candidate.
    gate_summary = summary[~summary.method.eq("run_a_accepted")].copy()
    gate_summary.loc[gate_summary.method.eq("run_b_accepted"), "method"] = "run_a_accepted"
    gate_summary.loc[gate_summary.method.eq("run_b_raw"), "method"] = "run_a_raw"
    old = pd.read_csv("diagnostics/ecir_mvr/stage2b/run_a/source_summary.csv")
    gate_summary = pd.concat([gate_summary, old[old.method.eq("old_ecir_4step")]], ignore_index=True)
    _, upstream_conditions, _, _, upstream_accuracy = make_decision(
        gate_summary, bootstrap_up, config["noninferiority"],
        full["clean_identity_fraction"],
    )
    upstream_22_pass = all(upstream_conditions.values())

    margins = config["incremental_noninferiority"]
    incremental_accuracy = incremental_accuracy_gate(bootstrap_a, margins)
    high_accuracy = {
        "rmsd_mean": high["aligned_RMSD"]["mean"] <= margins["high_flex_rmsd_mean_delta_max"],
        "rmsd_ci": high["aligned_RMSD"]["ci95_high"] <= margins["high_flex_rmsd_ci_upper_max"],
    }
    all_a = _row(summary, "all", "run_a_accepted"); all_b = _row(summary, "all", "run_b_accepted")
    high_b = _row(summary, "rotatable_ge_6", "run_b_accepted")
    unseen_b = _row(summary, "unseen_update_scale_0.35", "run_b_accepted")
    unseen_a = _row(summary, "unseen_update_scale_0.35", "run_a_accepted")
    torsion_ci = bootstrap_a["torsion_prior_outlier_score"]["ci95_high"] < 0.0
    high_total_ci = high["total_thresholded_validity_score"]["ci95_high"] < 0.0
    winning = bootstrap_a["torsion_prior_outlier_score"] if torsion_ci else high["total_thresholded_validity_score"]
    not_single = winning.get("improved_molecules", 0) >= 2
    relative_total = (
        (all_a.total_thresholded_validity_score - all_b.total_thresholded_validity_score)
        / max(all_a.total_thresholded_validity_score, 1e-12)
    )
    high_torsion_ci = high["torsion_prior_outlier_score"]["ci95_high"] < 0.0
    safety = bool(
        all_b.severe_clash_rate <= all_a.severe_clash_rate + 1e-12
        and all_b.chirality_error <= all_a.chirality_error + 1e-12
        and all_b.ring_bond_outlier_rate <= all_a.ring_bond_outlier_rate + 1e-12
        and all_b.ring_planarity_outlier_rate <= all_a.ring_planarity_outlier_rate + 1e-12
    )
    acceptance_ok = bool(
        all_b.accepted_fraction >= 0.8 * all_a.accepted_fraction
        and 0.05 < all_b.accepted_fraction < 0.95
    )
    unseen_accuracy = incremental_accuracy_gate(unseen, margins)
    unseen_ok = bool(
        unseen_b is not None and unseen_a is not None
        and unseen_b.total_thresholded_validity_score <= unseen_a.total_thresholded_validity_score + 1e-12
        and all(unseen_accuracy.values())
    )
    conditions = {
        "01_run_b_passes_run_a_22_upstream_conditions": upstream_22_pass,
        "02_torsion_or_high_flex_ci_improves": torsion_ci or high_total_ci,
        "03_increment_not_single_molecule": not_single,
        "04_minimum_effect": relative_total >= 0.02 or high_torsion_ci or high_total_ci,
        "05_rmsd_mean": incremental_accuracy["rmsd_mean"],
        "06_rmsd_ci": incremental_accuracy["rmsd_ci"],
        "07_mat_p_mean": incremental_accuracy["mat_p_mean"],
        "08_mat_p_ci": incremental_accuracy["mat_p_ci"],
        "09_mat_r_mean": incremental_accuracy["mat_r_mean"],
        "10_mat_r_ci": incremental_accuracy["mat_r_ci"],
        "11_cov_no_drop": incremental_accuracy["cov_p"] and incremental_accuracy["cov_r"],
        "12_high_flex_rmsd_mean": high_accuracy["rmsd_mean"],
        "13_high_flex_rmsd_ci": high_accuracy["rmsd_ci"],
        "14_high_flex_mean_torsion_limit": high_b.high_flex_torsion_change <= margins["high_flex_mean_torsion_change_max"],
        "15_high_flex_p95_torsion_limit": high_b.high_flex_p95_torsion_change <= margins["high_flex_p95_torsion_change_max"],
        "16_clean_identity_preserved": full["clean_identity_fraction"] >= 1.0,
        "17_safety_not_worse": safety,
        "18_acceptance_not_collapsed": acceptance_ok,
        "19_unseen_pass": unseen_ok,
    }
    conditions = {name: bool(value) for name, value in conditions.items()}
    harm = not all([
        upstream_22_pass, all(incremental_accuracy.values()), all(high_accuracy.values()),
        conditions["14_high_flex_mean_torsion_limit"],
        conditions["15_high_flex_p95_torsion_limit"],
        conditions["16_clean_identity_preserved"], safety, acceptance_ok,
    ])
    decision = (
        "RUN_B_HARMS" if harm else
        ("RUN_B_ADDS_VALUE" if all(conditions.values()) else "RUN_B_NO_ADDED_VALUE")
    )
    selected = "RUN_B" if decision == "RUN_B_ADDS_VALUE" else "RUN_A"
    gate = f"GO_20K_WITH_{selected}"
    next_command = (
        "python scripts/train_ecir_mvr_medium_20k.py --config "
        f"configs/ecir_mvr_medium_20k_{selected.lower()}_selected.yaml"
    )
    result = {
        "schema_version": "ecir-mvr-stage2b-run-b-result-v1",
        "run_a_decision": "RUN_A_PASS", "run_b_decision": decision,
        "selected_medium_configuration": selected, "current_decision": gate,
        "20k_permitted": True, "100k_permitted": False,
        "next_command": next_command, "next_command_executed": False,
        "validation_only": True, "test_records_read": 0,
        "config_sha256": _sha(args.config), "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha(args.checkpoint),
        "run_a_checkpoint_sha256": config["run_a_checkpoint_sha256"],
        "frozen_identities": config["frozen_identities"],
        "records": len(items), "molecules": len({str(item['row'].molecule_id) for item in items}),
        "conditions": conditions, "run_b_upstream_22_conditions": upstream_conditions,
        "upstream_accuracy_gate": upstream_accuracy,
        "incremental_accuracy_gate": incremental_accuracy,
        "high_flex_accuracy_gate": high_accuracy,
        "unseen_accuracy_gate": unseen_accuracy,
        "relative_total_validity_improvement_vs_run_a": float(relative_total),
        "clean_identity_fraction": full["clean_identity_fraction"],
        "acceptance_fraction": full["acceptance_fraction"],
        "torsion_gate_mean": full["torsion_gate_mean"],
        "torsion_gate_active_fraction": full["torsion_gate_active_fraction"],
        "torsion_velocity_fraction": full["torsion_velocity_fraction"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_save(result, args.output_dir / "result.json")
    atomic_json_save({"draws": args.bootstrap_draws, "seed": 42, "baseline": "upstream",
                      "run_b": bootstrap_up}, args.output_dir / "bootstrap_vs_upstream.json")
    atomic_json_save({"draws": args.bootstrap_draws, "seed": 42, "baseline": "run_a_accepted",
                      "all": bootstrap_a, "high_flex": high, "unseen_0.35": unseen},
                     args.output_dir / "bootstrap_vs_run_a.json")
    summary.to_csv(args.output_dir / "source_summary.csv", index=False)
    summary[summary.group.isin([
        "rotatable_le_2", "rotatable_3_5", "rotatable_ge_6", "ring", "non_ring", "clean_valid"
    ])].to_csv(args.output_dir / "flexibility_summary.csv", index=False)
    torsion_columns = [
        "group", "method", "molecules", "records", "torsion_prior_outlier_score",
        "mean_torsion_change", "p95_torsion_change", "high_flex_torsion_change",
        "high_flex_p95_torsion_change", "torsion_gate_mean",
        "torsion_gate_active_fraction", "torsion_velocity_norm", "torsion_velocity_fraction",
    ]
    summary[[name for name in torsion_columns if name in summary]].to_csv(
        args.output_dir / "torsion_summary.csv", index=False
    )
    full["record_rows"].to_csv(args.output_dir / "record_metrics.csv", index=False)
    molecules.to_csv(args.output_dir / "molecule_metrics.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
