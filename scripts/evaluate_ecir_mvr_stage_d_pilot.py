#!/usr/bin/env python
"""Select and evaluate the preregistered validation-only MCVR Stage D pilot."""

from __future__ import annotations

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

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.failure_attribution import bond_observations, relative_improvement
from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir.run_a_evaluation import (
    BOOTSTRAP_METRICS, build_clean_control_items, build_items, infer_mvr,
    method_rows, molecule_rows, paired_bootstrap, summarize_groups,
)
from scripts.audit_ecir_mvr_medium_failure_attribution import infer_stage_coordinates


OUTPUT = Path("diagnostics/ecir_mvr/stage_d/pilot")
V4_CONFIG = Path("configs/ecir_mvr_medium_5k_500_run_a_seed42_schedule_v4_10k.yaml")
CONFIG_A = Path("configs/ecir_mvr_stage_d_d1_a_aux_only_seed42_5k.yaml")
CONFIG_B = Path("configs/ecir_mvr_stage_d_d1_b_explicit_bond_seed42_5k.yaml")
RUN_A = Path("logs_ecir_mvr/stage2b/run_a_rigid_only_seed42_5k/checkpoints/best_noninferior_validity.ckpt")
V4 = Path("logs_ecir_mvr/medium/run_a_seed42_schedule_v4_10k/checkpoints/step001500.ckpt")
DRAWS = 10_000
SEED = 42


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(summary: pd.DataFrame, group: str, method: str):
    return summary[(summary.group == group) & (summary.method == method)].iloc[0]


def _selected(config: dict) -> tuple[Path, pd.Series]:
    comparison = pd.read_csv(Path(config["diagnostics_dir"]) / "checkpoint_comparison.csv")
    qualified = comparison[comparison.safety_qualified.astype(bool)].copy()
    if qualified.empty:
        raise RuntimeError(f"no qualified checkpoint for {config['stage_d_method']}")
    selected = qualified.sort_values(["validity_delta", "mean_displacement", "step"]).iloc[0]
    return Path(selected.checkpoint), selected


def _load(path: Path, device: torch.device) -> tuple[MCVRModel, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    model = MCVRModel(**config["model"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def _transitions(items, baseline, candidate, validity, label: str) -> dict[str, float]:
    repaired = newly_broken = total = 0
    for item, upstream, value in zip(items, baseline, candidate):
        before = bond_observations(validity, upstream, item["record"]).outlier.to_numpy(bool)
        after = bond_observations(validity, value, item["record"]).outlier.to_numpy(bool)
        repaired += int((before & ~after).sum())
        newly_broken += int((~before & after).sum())
        total += len(before)
    return {
        "method": label, "bonds": total, "repaired_bonds": repaired,
        "newly_broken_bonds": newly_broken,
        "cancellation_ratio": newly_broken / max(repaired, 1),
    }


def _size_quartiles(items, rows, methods) -> pd.DataFrame:
    molecule_size = {}
    for item in items:
        molecule_size[str(item["row"].molecule_id)] = int(item["input"].shape[0])
    ids = sorted(molecule_size)
    labels = pd.qcut([molecule_size[value] for value in ids], 4, labels=False, duplicates="drop")
    mapping = {molecule: f"size_q{int(label) + 1}" for molecule, label in zip(ids, labels)}
    for item in items:
        item["groups"].append(mapping[str(item["row"].molecule_id)])
    frames = []
    for group in sorted(set(mapping.values())):
        values = molecule_rows(rows, items, methods, group=group)
        for method, subset in values.groupby("method"):
            numeric = subset.select_dtypes(include=[np.number]).mean().to_dict()
            frames.append({"group": group, "method": method, "molecules": len(subset), **numeric})
    return pd.DataFrame(frames)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_v4 = yaml.safe_load(V4_CONFIG.read_text(encoding="utf-8"))
    config_a = yaml.safe_load(CONFIG_A.read_text(encoding="utf-8"))
    config_b = yaml.safe_load(CONFIG_B.read_text(encoding="utf-8"))
    if not (config_a["frozen_identities"] == config_b["frozen_identities"] == config_v4["frozen_identities"]):
        raise RuntimeError("Stage D frozen identities differ from V4")
    checkpoint_a, selected_a = _selected(config_a)
    checkpoint_b, selected_b = _selected(config_b)
    device = torch.device("cuda")
    validity = ChemicalValidity(config_b["data"]["validity_statistics"])
    items = build_items(config_b["data"]["val_sources"], config_b["data"]["val_targets"], validity)

    model_a, payload_a = _load(checkpoint_a, device)
    model_b, payload_b = _load(checkpoint_b, device)
    model_v4, _ = _load(V4, device)
    model_run_a, _ = _load(RUN_A, device)
    _, accepted_a, metadata_a = infer_mvr(model_a, items, validity, device=device)
    _, accepted_v4, metadata_v4 = infer_mvr(model_v4, items, validity, device=device)
    _, accepted_run_a, metadata_run_a = infer_mvr(model_run_a, items, validity, device=device)
    stages_b, metadata_b = infer_stage_coordinates(model_b, items, validity, device=device)

    methods = {
        "upstream": [item["input"] for item in items],
        "run_a_stage2b": accepted_run_a,
        "v4_selected": accepted_v4,
        "d1_a_aux_only": accepted_a,
        "d1_b_raw": stages_b["raw_proposal"],
        "d1_b_clipped": stages_b["trust_clipped_proposal"],
        "d1_b_safety_gated": stages_b["safety_gated_proposal"],
        "d1_b_accepted": stages_b["accepted"],
        "minimal_target": [item["minimal_target"] for item in items],
    }
    metadata = {
        "run_a_stage2b": metadata_run_a, "v4_selected": metadata_v4,
        "d1_a_aux_only": metadata_a, "d1_b_accepted": metadata_b,
    }
    rows = method_rows(items, methods, validity, metadata)
    summary, molecules = summarize_groups(rows, items, methods)
    size_summary = _size_quartiles(items, rows, methods)
    pd.concat([summary, size_summary], ignore_index=True).to_csv(
        OUTPUT / "subgroup_summary.csv", index=False
    )
    bootstrap_upstream = paired_bootstrap(
        molecules, candidate="d1_b_accepted", baseline="upstream", draws=DRAWS, seed=SEED
    )
    bootstrap_v4 = paired_bootstrap(
        molecules, candidate="d1_b_accepted", baseline="v4_selected", draws=DRAWS, seed=SEED
    )
    atomic_json_save({
        "schema_version": "ecir-mvr-stage-d-bootstrap-v1", "draws": DRAWS,
        "seed": SEED, "candidate": "d1_b_accepted", "baseline": "v4_selected",
        "metrics": bootstrap_v4,
    }, OUTPUT / "bootstrap_vs_v4.json")

    upstream = _row(summary, "all", "upstream")
    v4 = _row(summary, "all", "v4_selected")
    a = _row(summary, "all", "d1_a_aux_only")
    b = _row(summary, "all", "d1_b_accepted")
    target = _row(summary, "all", "minimal_target")
    high = _row(summary, "rotatable_ge_6", "d1_b_accepted")
    high_up = _row(summary, "rotatable_ge_6", "upstream")
    unseen = _row(summary, "unseen_update_scale_0.35", "d1_b_accepted")
    unseen_up = _row(summary, "unseen_update_scale_0.35", "upstream")
    bond_relative = relative_improvement(upstream.bond_outlier_rate, b.bond_outlier_rate)
    target_relative = relative_improvement(upstream.bond_outlier_rate, target.bond_outlier_rate)
    recovery = bond_relative / max(target_relative, 1.0e-12)

    transitions = [
        _transitions(items, methods["upstream"], accepted_v4, validity, "v4_selected"),
        _transitions(items, methods["upstream"], stages_b["accepted"], validity, "d1_b_accepted"),
    ]
    pd.DataFrame(transitions).to_csv(OUTPUT / "bond_transition.csv", index=False)
    transition_v4, transition_b = transitions
    clean_items = build_clean_control_items(items, validity, limit=20)
    _, clean_accepted, _ = infer_mvr(model_b, clean_items, validity, device=device)
    clean_identity = float(np.mean([
        torch.equal(torch.as_tensor(candidate), torch.as_tensor(item["input"]))
        for candidate, item in zip(clean_accepted, clean_items)
    ]))
    source_improvements = []
    for group in ("ETFlow_normal", "Cartesian_mild", "Cartesian_medium", "Cartesian_severe"):
        if _row(summary, group, "d1_b_accepted").total_thresholded_validity_score < _row(summary, group, "v4_selected").total_thresholded_validity_score:
            source_improvements.append(group)
    cancellation = float(np.mean([value["bond_cancellation_fraction"] for value in metadata_b]))
    solver_failure = float(np.mean([value["bond_solver_failure_fraction"] for value in metadata_b]))
    criteria = {
        "01_bond_relative_improvement_ge_12pct": bond_relative >= 0.12,
        "02_bond_vs_v4_paired_ci_improves": bootstrap_v4["bond_outlier_rate"]["ci95_high"] < 0.0,
        "03_bond_magnitude_vs_v4_not_worse": b.bond_outlier_magnitude <= v4.bond_outlier_magnitude,
        "04_model_to_target_recovery_ge_0p22": recovery >= 0.22,
        "05_total_validity_vs_v4_not_worse": b.total_thresholded_validity_score <= v4.total_thresholded_validity_score,
        "06_angle_vs_v4_not_worse": b.angle_outlier_rate <= v4.angle_outlier_rate,
        "07_ring_vs_v4_not_worse": b.ring_bond_outlier_rate <= v4.ring_bond_outlier_rate,
        "08_newly_broken_bonds_not_above_v4": transition_b["newly_broken_bonds"] <= transition_v4["newly_broken_bonds"],
        "09_cancellation_ratio_le_20pct": transition_b["cancellation_ratio"] <= 0.20 and cancellation <= 0.20,
        "10_rmsd_mean_delta_le_0p003": bootstrap_upstream["aligned_RMSD"]["mean"] <= 0.003,
        "11_rmsd_ci_upper_le_0p005": bootstrap_upstream["aligned_RMSD"]["ci95_high"] <= 0.005,
        "12_mat_p_mat_r_limits": all(bootstrap_upstream[name]["mean"] <= 0.003 and bootstrap_upstream[name]["ci95_high"] <= 0.005 for name in ("MAT_P", "MAT_R")),
        "13_cov_p_cov_r_no_material_drop": b.COV_P >= upstream.COV_P - 0.005 and b.COV_R >= upstream.COV_R - 0.005,
        "14_high_flex_validity_improves": high.total_thresholded_validity_score < high_up.total_thresholded_validity_score,
        "15_high_flex_torsion_controlled": high.high_flex_torsion_change <= 0.05,
        "16_clean_identity_ge_90pct": clean_identity >= 0.90,
        "17_clash_chirality_not_worse": b.severe_clash_rate <= v4.severe_clash_rate + 1.0e-12 and b.chirality_error <= v4.chirality_error + 1.0e-12 and b.clash_penetration <= v4.clash_penetration + 0.005,
        "18_unseen_validity_accuracy_pass": unseen.total_thresholded_validity_score < unseen_up.total_thresholded_validity_score and unseen.aligned_RMSD - unseen_up.aligned_RMSD <= 0.003 and unseen.MAT_P - unseen_up.MAT_P <= 0.003 and unseen.MAT_R - unseen_up.MAT_R <= 0.003 and unseen.COV_P >= unseen_up.COV_P - 0.005 and unseen.COV_R >= unseen_up.COV_R - 0.005,
        "19_improvement_not_single_source": len(source_improvements) >= 2,
        "20_solver_numerical_failure_lt_1pct": solver_failure < 0.01,
    }
    criteria = {name: bool(value) for name, value in criteria.items()}
    if all(criteria.values()):
        decision, selected_method = "STAGE_D_BOND_EXPLICIT_PASS", "D1_B_EXPLICIT_BOND"
    elif a.bond_outlier_rate < b.bond_outlier_rate and a.total_thresholded_validity_score <= b.total_thresholded_validity_score:
        decision, selected_method = "STAGE_D_AUX_ONLY_BETTER", "D1_A_AUX_ONLY"
    elif b.total_thresholded_validity_score > v4.total_thresholded_validity_score or bootstrap_upstream["aligned_RMSD"]["ci95_high"] > 0.005:
        decision, selected_method = "STAGE_D_HARMS", None
    else:
        decision, selected_method = "STAGE_D_NO_ADDED_VALUE", None

    comparison = pd.concat([
        pd.read_csv(Path(config_a["diagnostics_dir"]) / "checkpoint_comparison.csv").assign(method="D1_A"),
        pd.read_csv(Path(config_b["diagnostics_dir"]) / "checkpoint_comparison.csv").assign(method="D1_B"),
    ], ignore_index=True)
    comparison["selected"] = ((comparison.method == "D1_A") & (comparison.step == int(selected_a.step))) | ((comparison.method == "D1_B") & (comparison.step == int(selected_b.step)))
    comparison.to_csv(OUTPUT / "checkpoint_comparison.csv", index=False)
    recovery_rows = [{
        "method": method,
        "bond_relative_improvement": relative_improvement(upstream.bond_outlier_rate, _row(summary, "all", method).bond_outlier_rate),
        "target_recovery": relative_improvement(upstream.bond_outlier_rate, _row(summary, "all", method).bond_outlier_rate) / max(target_relative, 1.0e-12),
    } for method in methods]
    pd.DataFrame(recovery_rows).to_csv(OUTPUT / "proposal_recovery.csv", index=False)
    solver_rows = pd.DataFrame(metadata_b)[[
        "bond_solver_failure_fraction", "cartesian_bond_subspace_cosine",
        "bond_cancellation_fraction", "atom_clipping_fraction", "graph_clip_scale_min",
    ]]
    solver_rows.insert(0, "sample_id", [str(item["row"].sample_id) for item in items])
    solver_rows.to_csv(OUTPUT / "solver_diagnostics.csv", index=False)

    result = {
        "schema_version": "ecir-mvr-stage-d-pilot-result-v1", "decision": decision,
        "formal_v4_decision_unchanged": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "training_completed": {"D1_A": True, "D1_B": True},
        "completed_steps": {"D1_A": 5000, "D1_B": 5000},
        "selected_checkpoints": {
            "D1_A": {"step": int(selected_a.step), "path": str(checkpoint_a.resolve()), "sha256": _sha(checkpoint_a)},
            "D1_B": {"step": int(selected_b.step), "path": str(checkpoint_b.resolve()), "sha256": _sha(checkpoint_b)},
        },
        "metrics": {
            "bond_relative_improvement": bond_relative, "target_relative_improvement": target_relative,
            "model_to_target_recovery_ratio": recovery,
            "d1_b_total_validity_delta_vs_v4": float(b.total_thresholded_validity_score - v4.total_thresholded_validity_score),
            "d1_a_total_validity_delta_vs_v4": float(a.total_thresholded_validity_score - v4.total_thresholded_validity_score),
            "angle_rate_delta_vs_v4": float(b.angle_outlier_rate - v4.angle_outlier_rate),
            "ring_rate_delta_vs_v4": float(b.ring_bond_outlier_rate - v4.ring_bond_outlier_rate),
            "clean_identity_fraction": clean_identity, "cancellation_ratio": transition_b["cancellation_ratio"],
            "stagewise_cancellation_fraction": cancellation, "solver_failure_fraction": solver_failure,
            "cartesian_bond_subspace_cosine": float(np.mean([value["cartesian_bond_subspace_cosine"] for value in metadata_b])),
            "newly_broken_bonds": transition_b["newly_broken_bonds"],
            "v4_newly_broken_bonds": transition_v4["newly_broken_bonds"],
            "improving_sources": source_improvements,
        },
        "criteria": criteria, "gate_pass_count": sum(criteria.values()),
        "gate_condition_count": len(criteria), "pass": all(criteria.values()),
        "stage_d_selected_method": selected_method,
        "stage_d_20k_permitted": False, "stage_d_100k_permitted": False,
        "validation_only": True, "test_records_read": 0,
        "seed43_44_started": False, "20k_started": False, "100k_started": False,
        "next_command": None, "next_commands": [],
    }
    atomic_json_save(result, OUTPUT / "result.json")
    report = [
        "# MCVR Stage D Pilot Report", "", f"Decision: **{decision}**", "",
        f"D1-A selected step {int(selected_a.step)}; D1-B selected step {int(selected_b.step)}.", "",
        "| Metric | Value |", "|---|---:|",
        f"| D1-B bond relative improvement | {bond_relative:.12f} |",
        f"| Model-to-target recovery | {recovery:.12f} |",
        f"| D1-B validity delta vs V4 | {b.total_thresholded_validity_score-v4.total_thresholded_validity_score:.12f} |",
        f"| RMSD mean delta vs upstream | {bootstrap_upstream['aligned_RMSD']['mean']:.12f} |",
        f"| Newly broken bonds D1-B / V4 | {transition_b['newly_broken_bonds']} / {transition_v4['newly_broken_bonds']} |",
        f"| Cancellation ratio | {transition_b['cancellation_ratio']:.12f} |",
        f"| Solver failure fraction | {solver_failure:.12f} |", "", "## Gate", "",
        "| Condition | Result |", "|---|---|",
    ]
    report.extend(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in criteria.items())
    report += ["", "No test, seed43/44, 20k, or 100k execution occurred."]
    Path("docs/MCVR_STAGE_D_PILOT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision_doc = [
        "# MCVR Stage D Decision", "", f"Final decision: **{decision}**", "",
        "The formal Medium Schedule V4 decision remains **MEDIUM_SEED42_SCHEDULE_V4_FAIL**.", "",
        f"Selected method: `{selected_method}`." if selected_method else "No Stage D method is selected.", "",
        "Stage D 20k requires a separate manual decision even after PASS; it was not run.",
        "Stage D 100k, seed43/44, and test remain prohibited and were not run.",
    ]
    Path("docs/MCVR_STAGE_D_DECISION.md").write_text("\n".join(decision_doc) + "\n", encoding="utf-8")
    state_path = Path("reports/ecir_mvr/progressive_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "stage_d_status": "PILOT_COMPLETE", "stage_d_oracle_decision": "PASS",
        "stage_d_pilot_decision": decision, "stage_d_selected_method": selected_method,
        "stage_d_20k_permitted": False, "stage_d_100k_permitted": False,
        "medium_seed42_schedule_v4_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "test_records_read": 0, "seed43_44_permitted": False,
        "100k_permitted": False, "100k_started": False,
        "next_command": None, "next_commands": [],
    })
    atomic_json_save(state, state_path)
    print(json.dumps({"decision": decision, "criteria": criteria, "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
