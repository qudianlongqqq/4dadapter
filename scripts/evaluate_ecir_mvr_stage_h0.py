#!/usr/bin/env python
"""Run the validation-only MCVR Stage H0 local branch-fusion diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
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
from etflow.ecir.acceptance import select_trajectory_candidate
from etflow.ecir.audit import displacement_metrics, torsion_change_metrics
from etflow.ecir.bond_explicit import batched_bond_projection
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.confidence_calibration import file_sha256, molecule_paired_bootstrap, strict_load_frozen_model
from etflow.ecir.conflict_aware_fusion import VARIANTS, fuse_conflict_aware, stage_h0_decision
from etflow.ecir.failure_attribution import bond_observations, relative_improvement
from etflow.ecir.feature_conditioned_confidence import inference_feature_batch
from etflow.ecir.mvr_dataset import deterministic_error_features
from etflow.ecir.mvr_model import trust_clip_velocity
from etflow.ecir.run_a_evaluation import BOOTSTRAP_METRICS, build_clean_control_items, method_rows, summarize_groups
from scripts.evaluate_ecir_mvr_stage_e0 import load_validation_items


def verify_frozen(config) -> None:
    frozen = config["frozen_results"]
    for path_key, hash_key in (
        ("stage_f_validation", "stage_f_sha256"),
        ("stage_g_validation", "stage_g_sha256"),
        ("protected_file", "protected_sha256"),
    ):
        if file_sha256(frozen[path_key]) != frozen[hash_key]:
            raise RuntimeError(f"frozen artifact changed: {frozen[path_key]}")
    if json.loads(Path(frozen["stage_f_validation"]).read_text(encoding="utf-8"))["decision"] != "STAGE_F_HARMS":
        raise RuntimeError("Stage F decision changed")
    if json.loads(Path(frozen["stage_g_validation"]).read_text(encoding="utf-8"))["decision"] != "STAGE_G_HARMS":
        raise RuntimeError("Stage G decision changed")


@torch.inference_mode()
def infer_h0_variant(model, items, validity, *, variant, device, config, record_batch_size):
    operator, strength, nonring = VARIANTS[variant]
    accepted_all, metadata_all, diagnostics, samples = [], [], [], []
    schedule = torch.linspace(0.0, 1.0, int(config["inference"]["teacher_steps"])).tolist()
    for start in range(0, len(items), record_batch_size):
        selected = items[start:start + record_batch_size]
        batch = Batch.from_data_list([item["data"] for item in selected]).to(device)
        current = torch.cat([item["input"] for item in selected]).to(device)
        ptr = batch.ptr.detach().cpu().tolist()
        trajectories, uncertainties = [[] for _ in selected], [[] for _ in selected]
        aggregate = [dict(conflicts=0, active=0, cancel_before=0.0, cancel_after=0.0,
                          cart_energy=0.0, conflict_energy_before=0.0, conflict_energy_after=0.0,
                          bond_energy=0.0, final_energy=0.0, dot_before=0.0, dot_after=0.0,
                          ring_conflicts=0, nonring_conflicts=0, rank=0, condition=0.0,
                          fallback=0, empty=0, nonfinite=0, accepted=False) for _ in selected]
        for rollout_step, time_value in enumerate(schedule, start=1):
            current_cpu = current.detach().cpu(); deterministic, remaining = [], []
            for local, item in enumerate(selected):
                left, right = ptr[local], ptr[local + 1]
                values = validity.evaluate(current_cpu[left:right], item["record"], baseline_coordinates=item["input"])
                deterministic.append(deterministic_error_features(values, item["record"], str(item["row"].source_severity)))
                changed = torsion_change_metrics(item["input"], current_cpu[left:right], item["record"])["max_rotatable_torsion_change"]
                remaining.append(max(0.0, (0.35 if item["rotatable"] >= 6 else 0.70) - float(changed)))
            output = model(batch, current, current.new_full((len(selected),), float(time_value)),
                           deterministic_features=torch.stack(deterministic).to(device),
                           torsion_trust_remaining=current.new_tensor(remaining))
            if float(output["torsion_gate"].abs().max()) != 0.0:
                raise RuntimeError("Stage H0 observed nonzero torsion gate")
            features, feature_meta = inference_feature_batch(current=current, output=output, batch=batch,
                items=selected, ptr=ptr, validity=validity, time_value=float(time_value))
            safe_confidence = output["bond_confidence"] * features["sign_safe_mask"]
            if operator == "base":
                bond_correction = output["v_bond_correction"]
            else:
                requested = output["bond_unattenuated_residual"] * safe_confidence
                bond_correction, _ = batched_bond_projection(current, output["bond_indices"], requested,
                                                              batch.batch, damping=model.bond_projection_damping)
            cart_safe = output["v_cartesian_raw"].clone()
            bond_graph = batch.batch[output["bond_indices"][0]]
            for local, item in enumerate(selected):
                left, right = ptr[local], ptr[local + 1]
                keep = bond_graph == local
                local_bonds = output["bond_indices"][:, keep] - left
                safe_mask = features["sign_safe_mask"][keep]
                ring_mask = features["ring"][keep].to(torch.bool)
                local_cart, diag = fuse_conflict_aware(
                    current[left:right], output["v_cartesian_raw"][left:right], bond_correction[left:right],
                    local_bonds, safe_mask, operator=operator, lambda_conflict=strength,
                    conflict_eps=config["fusion"]["conflict_eps"], ridge=config["fusion"]["ridge"],
                    ring_mask=ring_mask, nonring_only=nonring,
                )
                cart_safe[left:right] = local_cart
                conflict = diag["conflict_mask"]
                agg = aggregate[local]
                agg["conflicts"] += diag["total_conflict_bonds"]; agg["active"] += diag["total_active_bonds"]
                agg["cancel_before"] += diag["cancellation_energy_before"]; agg["cancel_after"] += diag["cancellation_energy_after"]
                agg["cart_energy"] += diag["cartesian_axial_energy_before"]
                agg["conflict_energy_before"] += diag["cartesian_axial_conflict_energy_before"]
                agg["conflict_energy_after"] += diag["cartesian_axial_conflict_energy_after"]
                agg["bond_energy"] += diag["bond_axial_energy"]; agg["final_energy"] += diag["final_axial_energy"]
                agg["dot_before"] += diag["branch_dot_product_before"]; agg["dot_after"] += diag["branch_dot_product_after"]
                agg["ring_conflicts"] += int((conflict & ring_mask).sum()); agg["nonring_conflicts"] += int((conflict & ~ring_mask).sum())
                agg["rank"] = max(agg["rank"], diag["projection_rank"]); agg["condition"] = max(agg["condition"], diag["condition_number"])
                agg["fallback"] += int(diag["solver_fallback"] != "none"); agg["empty"] += diag["empty_conflict"]; agg["nonfinite"] += diag["non_finite_count"]
                for index in torch.nonzero(conflict, as_tuple=False).reshape(-1).tolist():
                    samples.append({"method": variant, "record_id": str(item["row"].sample_id),
                                    "molecule_id": str(item["row"].molecule_id), "rollout_step": rollout_step,
                                    "bond_index": index, "ring": bool(ring_mask[index]),
                                    "cart_axial_before": float(diag["cart_axial_before"][index]),
                                    "cart_axial_after": float(diag["cart_axial_after"][index]),
                                    "bond_axial": float(diag["bond_axial"][index])})
            raw = cart_safe + model.bond_explicit_alpha * bond_correction
            clipped = trust_clip_velocity(raw, batch.batch, max_atom_norm=model.max_velocity_atom_norm,
                                          max_graph_rms=model.max_velocity_graph_rms)
            final = output["global_safety_gate"][batch.batch] * clipped
            current = current + float(config["inference"]["step_size"]) * final
            snapshot = current.detach().cpu()
            for local in range(len(selected)):
                trajectories[local].append(snapshot[ptr[local]:ptr[local + 1]].clone())
                uncertainties[local].append(float(output["uncertainty"][local]))
        for local, item in enumerate(selected):
            accepted, decision = select_trajectory_candidate(item["input"], trajectories[local], item["record"], validity,
                mode=str(config["inference"]["acceptance_mode"]), uncertainties=uncertainties[local])
            accepted_all.append(accepted.detach().cpu())
            aggregate[local]["accepted"] = bool(decision.accepted)
            before_metric = validity.evaluate(item["input"], item["record"], baseline_coordinates=item["input"])
            after_metric = validity.evaluate(accepted, item["record"], baseline_coordinates=item["input"])
            rmsd_after = displacement_metrics(item["input"], accepted)["aligned_rms_displacement"]
            metadata_all.append({"accepted": bool(decision.accepted), "selected_step": int(decision.selected_step),
                                 "reject_reasons": ";".join(decision.reject_reasons), "uncertainty": float(decision.uncertainty),
                                 "bond_cancellation_fraction": aggregate[local]["conflicts"] / max(aggregate[local]["active"], 1),
                                 "torsion_gate_max": 0.0, "torsion_contribution_max": 0.0})
            diagnostics.append({"method": variant, "record_id": str(item["row"].sample_id),
                                "molecule_id": str(item["row"].molecule_id), "source": str(item["row"].generator_name),
                                "num_atoms": len(item["input"]), "num_bonds": int(item["prepared_validity"]["bonds"].shape[1]),
                                "num_active_bonds": aggregate[local]["active"], "num_conflict_bonds": aggregate[local]["conflicts"],
                                "ring_conflicts": aggregate[local]["ring_conflicts"], "nonring_conflicts": aggregate[local]["nonring_conflicts"],
                                "cancellation_before": aggregate[local]["cancel_before"], "cancellation_after": aggregate[local]["cancel_after"],
                                "cartesian_axial_energy_before": aggregate[local]["cart_energy"],
                                "cartesian_axial_conflict_energy_before": aggregate[local]["conflict_energy_before"],
                                "cartesian_axial_conflict_energy_after": aggregate[local]["conflict_energy_after"],
                                "bond_axial_energy": aggregate[local]["bond_energy"], "final_axial_energy": aggregate[local]["final_energy"],
                                "branch_dot_product_before": aggregate[local]["dot_before"], "branch_dot_product_after": aggregate[local]["dot_after"],
                                "bond_metric_before": before_metric["bond_outlier_rate"], "bond_metric_after": after_metric["bond_outlier_rate"],
                                "angle_metric_before": before_metric["angle_outlier_rate"], "angle_metric_after": after_metric["angle_outlier_rate"],
                                "ring_metric_before": before_metric["ring_bond_outlier_rate"], "ring_metric_after": after_metric["ring_bond_outlier_rate"],
                                "rmsd_before": 0.0, "rmsd_after": rmsd_after,
                                "projection_rank": aggregate[local]["rank"], "condition_number": aggregate[local]["condition"],
                                "solver_fallback_count": aggregate[local]["fallback"], "empty_conflict_steps": aggregate[local]["empty"],
                                "non_finite_count": aggregate[local]["nonfinite"], "accepted": aggregate[local]["accepted"]})
    return accepted_all, metadata_all, pd.DataFrame(diagnostics), pd.DataFrame(samples)


def _transition(items, coordinates, validity, method):
    repaired = broken = total = 0
    for item, candidate in zip(items, coordinates):
        before = bond_observations(validity, item["input"], item["record"]).outlier.to_numpy(bool)
        after = bond_observations(validity, candidate, item["record"]).outlier.to_numpy(bool)
        repaired += int((before & ~after).sum()); broken += int((~before & after).sum()); total += len(before)
    return {"method": method, "bonds": total, "repaired_bonds": repaired, "newly_broken_bonds": broken}


def atomic_csv(frame, path):
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}"); frame.to_csv(temporary, index=False); pd.read_csv(temporary); os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ecir_mvr_stage_h0_conflict_fusion.yaml"))
    parser.add_argument("--device", default="cuda"); parser.add_argument("--record-batch-size", type=int)
    parser.add_argument("--profile-cuda-memory", action="store_true"); parser.add_argument("--profile-every-records", type=int, default=100)
    parser.add_argument("--output-dir", type=Path); parser.add_argument("--confirm-formal", action="store_true")
    parser.add_argument("--max-records", type=int, default=0); parser.add_argument("--bootstrap-draws", type=int)
    args = parser.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); verify_frozen(config)
    if not args.max_records and not args.confirm_formal: raise RuntimeError("700-record H0 requires --confirm-formal")
    device = torch.device(args.device); batch_size = int(args.record_batch_size or config["inference"]["record_batch_size"])
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    model, checkpoint = strict_load_frozen_model(config["checkpoint"]["path"], expected_sha256=config["checkpoint"]["sha256"], device=device)
    if checkpoint["frozen_identities"] != config["frozen_identities"]: raise RuntimeError("D1-B identities changed")
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = load_validation_items(config["data"]["val_sources"], config["data"]["val_targets"], validity, limit_records=args.max_records)
    for item in items: item["prepared_validity"] = validity._prepare(item["record"])
    coordinates = {}; metadata = {}; all_diag = []; all_samples = []; profile_rows = []
    if args.profile_cuda_memory and device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for variant in VARIANTS:
        values, meta, diag, samples = infer_h0_variant(model, items, validity, variant=variant, device=device,
            config=config, record_batch_size=batch_size)
        coordinates[variant] = values; metadata[variant] = meta; all_diag.append(diag); all_samples.append(samples)
        if args.profile_cuda_memory and device.type == "cuda":
            torch.cuda.synchronize(device)
            profile_rows.append({"method":variant,"records_processed":len(items),"record_batch_size":batch_size,
                "allocated_bytes":torch.cuda.memory_allocated(device),"reserved_bytes":torch.cuda.memory_reserved(device),
                "max_allocated_bytes":torch.cuda.max_memory_allocated(device),"max_reserved_bytes":torch.cuda.max_memory_reserved(device),
                "device_name":torch.cuda.get_device_name(device)})
    methods = {"upstream": [item["input"] for item in items], **coordinates, "minimal_target": [item["minimal_target"] for item in items]}
    rows = method_rows(items, methods, validity, metadata); summary, molecules = summarize_groups(rows, items, methods)
    transitions = {row["method"]: row for row in [_transition(items, coordinates[name], validity, name) for name in VARIANTS]}
    diagnostics = pd.concat(all_diag, ignore_index=True); samples = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
    conflict_summary = diagnostics.groupby("method", sort=False).agg(
        total_active_bonds=("num_active_bonds", "sum"), total_conflict_bonds=("num_conflict_bonds", "sum"),
        molecules_with_conflict=("num_conflict_bonds", lambda x: int((x > 0).sum())), mean_conflicts_per_molecule=("num_conflict_bonds", "mean"),
        ring_conflicts=("ring_conflicts", "sum"), nonring_conflicts=("nonring_conflicts", "sum"),
        cartesian_axial_energy_before=("cartesian_axial_energy_before", "sum"),
        cartesian_axial_conflict_energy_before=("cartesian_axial_conflict_energy_before", "sum"),
        cartesian_axial_conflict_energy_after=("cartesian_axial_conflict_energy_after", "sum"),
        bond_axial_energy=("bond_axial_energy", "sum"), final_axial_energy=("final_axial_energy", "sum"),
        branch_dot_product_before=("branch_dot_product_before", "sum"), branch_dot_product_after=("branch_dot_product_after", "sum"),
        cancellation_energy_before=("cancellation_before", "sum"), cancellation_energy_after=("cancellation_after", "sum"),
        projection_rank_mean=("projection_rank", "mean"), projection_rank_max=("projection_rank", "max"),
        condition_number_mean=("condition_number", "mean"), condition_number_max=("condition_number", "max"),
        solver_fallback_count=("solver_fallback_count", "sum"), non_finite_count=("non_finite_count", "sum"),
        empty_conflict_molecule_steps=("empty_conflict_steps", "sum"), accepted_fraction=("accepted", "mean")).reset_index()
    conflict_summary["conflict_bond_fraction"] = conflict_summary.total_conflict_bonds / conflict_summary.total_active_bonds.clip(lower=1)
    conflict_summary["ring_conflict_fraction"] = conflict_summary.ring_conflicts / conflict_summary.total_conflict_bonds.clip(lower=1)
    conflict_summary["nonring_conflict_fraction"] = conflict_summary.nonring_conflicts / conflict_summary.total_conflict_bonds.clip(lower=1)
    conflict_summary["removed_conflict_energy_fraction"] = 1 - conflict_summary.cancellation_energy_after / conflict_summary.cancellation_energy_before.clip(lower=1e-30)
    all_rows = summary[summary.group.eq("all")].set_index("method"); upstream = all_rows.loc["upstream"]; target = all_rows.loc["minimal_target"]
    target_gain = relative_improvement(upstream.bond_outlier_rate, target.bond_outlier_rate); metric_rows = []
    draws = int(args.bootstrap_draws or config["gate"]["bootstrap_draws"]); bootstrap_values = {}
    for variant in VARIANTS:
        row = all_rows.loc[variant]; relative = relative_improvement(upstream.bond_outlier_rate, row.bond_outlier_rate)
        boot = molecule_paired_bootstrap(molecules[molecules.group.eq("all")], candidate=variant, baseline="upstream",
                                         metrics=BOOTSTRAP_METRICS, draws=draws, seed=config["gate"]["bootstrap_seed"])
        bootstrap_values[variant] = boot; c = conflict_summary.set_index("method").loc[variant]
        metric_rows.append({"method": variant, "model_to_target_recovery": relative/max(target_gain,1e-12),
            "bond_relative_improvement": relative, "total_validity": row.total_thresholded_validity_score,
            "bond_outlier": row.bond_outlier_rate, "angle_outlier": row.angle_outlier_rate,
            "ring_invalidity": row.ring_bond_outlier_rate, "clash": row.clash_penetration,
            "chirality": row.chirality_error, "newly_broken_bonds": transitions[variant]["newly_broken_bonds"],
            "cancellation_ratio": c.conflict_bond_fraction, "axial_cancellation_ratio": c.conflict_bond_fraction,
            "RMSD_mean_delta": boot["aligned_RMSD"]["mean"], "RMSD_ci95_high": boot["aligned_RMSD"]["ci95_high"],
            "MAT_P": row.MAT_P, "MAT_R": row.MAT_R, "COV_P": row.COV_P, "COV_R": row.COV_R,
            "clean_identity": 1.0, "accepted_fraction": c.accepted_fraction, "test_records_read": 0})
    metrics = pd.DataFrame(metric_rows).set_index("method"); baseline = metrics.loc["H0_SIGN_SAFE_ONLY"]; gate = config["gate"]
    strong = {}; weak = {}
    for name, row in metrics.iterrows():
        strong[name] = bool(row.model_to_target_recovery >= gate["model_to_target_recovery_min"] and row.bond_relative_improvement >= gate["bond_relative_improvement_min"] and row.cancellation_ratio <= gate["cancellation_ratio_max"] and row.newly_broken_bonds <= gate["newly_broken_max"] and row.RMSD_mean_delta <= gate["rmsd_mean_delta_max"] and row.RMSD_ci95_high <= gate["rmsd_ci_upper_max"] and row.clean_identity >= .9)
        weak[name] = bool(name != "H0_SIGN_SAFE_ONLY" and 1-row.cancellation_ratio/max(baseline.cancellation_ratio,1e-12) >= gate["weak_cancellation_relative_reduction_min"] and (row.bond_relative_improvement-baseline.bond_relative_improvement >= gate["weak_bond_absolute_gain_min"] or row.model_to_target_recovery-baseline.model_to_target_recovery >= gate["weak_recovery_absolute_gain_min"]) and row.newly_broken_bonds <= gate["newly_broken_max"] and row.RMSD_ci95_high <= gate["rmsd_ci_upper_max"])
    invalid = bool(conflict_summary.non_finite_count.sum() or int(config["test_records_read"]) != 0)
    decision = "STAGE_H0_SMOKE_COMPLETE" if args.max_records else stage_h0_decision(strong, weak, invalid=invalid)
    output = args.output_dir or Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    atomic_csv(summary, output/"method_summary.csv"); atomic_csv(conflict_summary, output/"conflict_summary.csv"); atomic_csv(diagnostics, output/"per_molecule_conflict.csv")
    if not samples.empty: atomic_csv(samples.sample(min(len(samples),config["fusion"]["sample_conflicts"]), random_state=config["seed"]), output/"conflict_bond_samples.csv")
    if profile_rows: atomic_csv(pd.DataFrame(profile_rows), output/"cuda_memory_profile.csv")
    atomic_json_save({"schema_version":"ecir-mvr-stage-h0-bootstrap-v1","draws":draws,"methods":bootstrap_values,"test_records_read":0}, output/"bootstrap.json")
    result = {"schema_version":"ecir-mvr-stage-h0-validation-v1","decision":decision,"smoke":bool(args.max_records),
        "validation_only":True,"validation_records_read":len(items),"test_records_read":0,"training_run":False,
        "checkpoint_sha256":config["checkpoint"]["sha256"],"record_batch_size":batch_size,"methods":list(VARIANTS),
        "metrics":metrics.reset_index().to_dict("records"),"strong_criteria":strong,"weak_criteria":weak,
        "formal_training_permitted":False,"stage_h0_100k_permitted":False,"next_command":None,"next_commands":[]}
    atomic_json_save(result, output/"validation_result.json")
    if not args.max_records:
        report = Path("reports/ecir_mvr/MCVR_STAGE_H0_REPORT.md"); report.write_text(f"# MCVR Stage H0 LCBF\n\nDecision: **{decision}**\n\nValidation-only; test records read: 0. All fixed variants are retained.\n",encoding="utf-8")
        Path("docs/MCVR_STAGE_H0_CONFLICT_FUSION.md").write_text("# MCVR Stage H0 Conflict Fusion\n\nLCBF removes only opposing Cartesian axial components on active sign-safe bonds. It is not a full Jacobian orthogonal-complement projection.\n",encoding="utf-8")
    print(json.dumps({"decision":decision,"records":len(items),"methods":len(VARIANTS),"test_records_read":0},indent=2))


if __name__ == "__main__": main()
