#!/usr/bin/env python
"""Run the preregistered validation-only Stage D0 bond-Jacobian oracle."""

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
from etflow.ecir.acceptance import select_trajectory_candidate
from etflow.ecir.bond_explicit import (
    bond_length_residual, solve_bond_cartesian_correction,
)
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.failure_attribution import paired_relative_bootstrap, relative_improvement
from etflow.ecir.mvr_safety import trust_clip_with_diagnostics
from etflow.ecir.run_a_evaluation import (
    build_clean_control_items, build_items, method_rows, paired_bootstrap,
    summarize_groups,
)


CONFIG_PATH = Path("configs/ecir_mvr_medium_5k_500_run_a_seed42_schedule_v4_10k.yaml")
OUTPUT_DIR = Path("diagnostics/ecir_mvr/stage_d/oracle")
DAMPING = 1.0e-4
MAX_CONDITION = 1.0e10
BOOTSTRAP_DRAWS = 10_000
SEED = 42
CHEMICAL_RATE_NONINFERIORITY_MARGIN = 0.005
SELECTED_SHA256 = "f94c317f4e12c559058e26f9842317770179ed3e9cbc07c0a21ec681fed94197"
PROTECTED_SHA256 = "738171eb7e5f047e94cd4e1a46689613fe1f30bf33a320a2e3ec5a6944a5ec7d"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(summary: pd.DataFrame, group: str, method: str):
    rows = summary[(summary.group == group) & (summary.method == method)]
    return rows.iloc[0]


def _fmt(value: float, digits: int = 12) -> str:
    return f"{float(value):.{digits}f}"


def solve_items(items, validity, *, max_atom_norm: float, max_graph_rms: float):
    raw, trusted, accepted, metadata, diagnostics = [], [], [], [], []
    for item in items:
        prepared = validity._prepare(item["record"])
        coordinates = torch.as_tensor(item["input"], dtype=torch.float64)
        target = torch.as_tensor(item["minimal_target"], dtype=torch.float64)
        bonds = prepared["bonds"]
        residual = bond_length_residual(coordinates, target, bonds)
        correction, solver = solve_bond_cartesian_correction(
            coordinates, bonds, residual, damping=DAMPING, max_condition=MAX_CONDITION,
        )
        correction = correction.to(torch.float32)
        input_float = torch.as_tensor(item["input"], dtype=torch.float32)
        raw_candidate = input_float + correction
        atom_batch = torch.zeros(len(correction), dtype=torch.long)
        trusted_correction, clipping = trust_clip_with_diagnostics(
            correction, atom_batch,
            max_atom_norm=max_atom_norm, max_graph_rms=max_graph_rms,
        )
        trusted_candidate = input_float + trusted_correction
        accepted_candidate, decision = select_trajectory_candidate(
            input_float, [trusted_candidate], item["record"], validity,
            mode="final_step", uncertainties=[0.0],
        )
        raw.append(raw_candidate)
        trusted.append(trusted_candidate)
        accepted.append(accepted_candidate)
        metadata.append({
            "accepted": decision.accepted, "selected_step": decision.selected_step,
            "reject_reasons": ";".join(decision.reject_reasons),
            "uncertainty": 0.0,
        })
        diagnostics.append({
            "molecule_id": str(item["row"].molecule_id),
            "sample_id": str(item["row"].sample_id),
            "source": str(item["row"].generator_name),
            "severity": str(item["row"].source_severity),
            "target_residual_rms": float(torch.sqrt(torch.mean(residual.square()))) if residual.numel() else 0.0,
            "target_residual_max": float(residual.abs().max()) if residual.numel() else 0.0,
            "correction_rms": float(torch.sqrt(torch.mean(correction.square().sum(-1)))) if correction.numel() else 0.0,
            "correction_max_atom": float(torch.linalg.vector_norm(correction, dim=-1).max()) if correction.numel() else 0.0,
            "raw_atom_mean": clipping["raw"]["atom_mean"],
            "raw_atom_p95": clipping["raw"]["atom_p95"],
            "raw_atom_max": clipping["raw"]["atom_max"],
            "raw_graph_rms": clipping["raw"]["graph_rms"],
            "clipped_atom_mean": clipping["clipped"]["atom_mean"],
            "clipped_atom_p95": clipping["clipped"]["atom_p95"],
            "clipped_atom_max": clipping["clipped"]["atom_max"],
            "clipped_graph_rms": clipping["clipped"]["graph_rms"],
            "atom_clipped_fraction": clipping["atom_clipped_fraction"],
            "graph_clipped_fraction": clipping["graph_clipped_fraction"],
            "accepted": bool(decision.accepted),
            **solver,
        })
    return raw, trusted, accepted, metadata, pd.DataFrame(diagnostics)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    formal_checkpoint = Path("logs_ecir_mvr/medium/run_a_seed42_schedule_v4_10k/checkpoints/step001500.ckpt")
    protected = Path("reports/global4d_profile_bundle_verification.json")
    if _sha(formal_checkpoint) != SELECTED_SHA256:
        raise RuntimeError("formal V4 checkpoint changed")
    if _sha(protected) != PROTECTED_SHA256:
        raise RuntimeError("protected file changed")
    if config["frozen_identities"] != json.loads(
        Path("diagnostics/ecir_mvr/medium/run_a_seed42_schedule_v4/preflight.json").read_text(encoding="utf-8")
    )["identities"]:
        raise RuntimeError("Stage D frozen identities changed")

    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = build_items(config["data"]["val_sources"], config["data"]["val_targets"], validity)
    max_atom_norm = float(config["model"]["max_velocity_atom_norm"])
    max_graph_rms = float(config["model"]["max_velocity_graph_rms"])
    raw, trusted, accepted, metadata, diagnostics = solve_items(
        items, validity, max_atom_norm=max_atom_norm, max_graph_rms=max_graph_rms,
    )
    methods = {
        "upstream": [item["input"] for item in items],
        "minimal_target": [item["minimal_target"] for item in items],
        "bond_oracle_raw": raw,
        "bond_oracle_trusted": trusted,
        "bond_oracle_accepted": accepted,
    }
    rows = method_rows(
        items, methods, validity,
        {"bond_oracle_accepted": metadata},
    )
    summary, molecules = summarize_groups(rows, items, methods)
    delta_bootstrap = paired_bootstrap(
        molecules, candidate="bond_oracle_accepted", draws=BOOTSTRAP_DRAWS, seed=SEED,
    )
    all_molecules = molecules[molecules.group.eq("all")]
    pivot = all_molecules.pivot(
        index="molecule_id", columns="method", values="bond_outlier_rate"
    ).dropna()
    bootstrap_result = paired_relative_bootstrap(
        pivot.upstream, pivot.bond_oracle_accepted,
        draws=BOOTSTRAP_DRAWS, seed=SEED,
    )

    upstream = _row(summary, "all", "upstream")
    target = _row(summary, "all", "minimal_target")
    oracle_raw = _row(summary, "all", "bond_oracle_raw")
    oracle_trusted = _row(summary, "all", "bond_oracle_trusted")
    oracle = _row(summary, "all", "bond_oracle_accepted")
    high = _row(summary, "rotatable_ge_6", "bond_oracle_accepted")
    high_upstream = _row(summary, "rotatable_ge_6", "upstream")
    target_relative = relative_improvement(
        upstream.bond_outlier_rate, target.bond_outlier_rate
    )
    accepted_relative = relative_improvement(
        upstream.bond_outlier_rate, oracle.bond_outlier_rate
    )
    recovery = accepted_relative / max(target_relative, 1.0e-12)

    clean_items = build_clean_control_items(items, validity, limit=20)
    clean_raw, clean_trusted, clean_accepted, clean_metadata, clean_diagnostics = solve_items(
        [{**item, "minimal_target": item["input"]} for item in clean_items], validity,
        max_atom_norm=max_atom_norm, max_graph_rms=max_graph_rms,
    )
    clean_identity = float(np.mean([
        torch.equal(torch.as_tensor(candidate), torch.as_tensor(item["input"]))
        for candidate, item in zip(clean_accepted, clean_items)
    ]))
    failure_statuses = {"FALLBACK_ZERO", "CONDITION_FALLBACK", "NONFINITE_FALLBACK"}
    numerical_failure_fraction = float(diagnostics.status.isin(failure_statuses).mean())
    criteria = {
        "01_accepted_bond_relative_improvement_ge_25pct": accepted_relative >= 0.25,
        "02_target_recovery_ge_40pct": recovery >= 0.40,
        "03_rmsd_delta_le_0p003": oracle.aligned_RMSD - upstream.aligned_RMSD <= 0.003,
        "04_angle_not_clearly_worse": oracle.angle_outlier_rate <= upstream.angle_outlier_rate + CHEMICAL_RATE_NONINFERIORITY_MARGIN,
        "05_ring_not_clearly_worse": oracle.ring_bond_outlier_rate <= upstream.ring_bond_outlier_rate + CHEMICAL_RATE_NONINFERIORITY_MARGIN,
        "06_clash_not_worse": (
            oracle.clash_penetration <= upstream.clash_penetration + 1.0e-12
            and oracle.severe_clash_rate <= upstream.severe_clash_rate + 1.0e-12
        ),
        "07_chirality_not_worse": oracle.chirality_error <= upstream.chirality_error + 1.0e-12,
        "08_high_flex_validity_improves": high.total_thresholded_validity_score < high_upstream.total_thresholded_validity_score,
        "09_high_flex_torsion_controlled": high.high_flex_torsion_change <= 0.05,
        "10_clean_identity_ge_90pct": clean_identity >= 0.90,
        "11_numerical_failure_lt_1pct": numerical_failure_fraction < 0.01,
    }
    decision = "PASS" if all(criteria.values()) else "NO_GO"
    result = {
        "schema_version": "ecir-mvr-stage-d0-bond-oracle-v1",
        "stage": "MCVR_STAGE_D0_BOND_LOCAL_ORACLE",
        "decision": decision,
        "formal_v4_decision_unchanged": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "validation_only": True, "test_records_read": 0,
        "records": len(items), "molecules": len(pivot),
        "config_sha256": _sha(CONFIG_PATH),
        "formal_checkpoint_sha256": _sha(formal_checkpoint),
        "protected_file_sha256": _sha(protected),
        "frozen_identities": config["frozen_identities"],
        "solver": {
            "method": "direct_damped_dual", "damping": DAMPING,
            "max_condition": MAX_CONDITION,
            "max_atom_norm": max_atom_norm, "max_graph_rms": max_graph_rms,
        },
        "gate_preregistration": {
            "chemical_rate_noninferiority_margin": CHEMICAL_RATE_NONINFERIORITY_MARGIN,
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": SEED,
        },
        "metrics": {
            "upstream_bond_outlier_rate": float(upstream.bond_outlier_rate),
            "target_bond_outlier_rate": float(target.bond_outlier_rate),
            "oracle_raw_bond_outlier_rate": float(oracle_raw.bond_outlier_rate),
            "oracle_trusted_bond_outlier_rate": float(oracle_trusted.bond_outlier_rate),
            "oracle_accepted_bond_outlier_rate": float(oracle.bond_outlier_rate),
            "target_bond_relative_improvement": target_relative,
            "oracle_accepted_bond_relative_improvement": accepted_relative,
            "model_to_target_recovery_upper_bound": recovery,
            "bond_magnitude_delta": float(oracle.bond_outlier_magnitude - upstream.bond_outlier_magnitude),
            "total_validity_delta": float(oracle.total_thresholded_validity_score - upstream.total_thresholded_validity_score),
            "angle_rate_delta": float(oracle.angle_outlier_rate - upstream.angle_outlier_rate),
            "ring_rate_delta": float(oracle.ring_bond_outlier_rate - upstream.ring_bond_outlier_rate),
            "rmsd_delta": float(oracle.aligned_RMSD - upstream.aligned_RMSD),
            "rms_displacement": float(oracle.molecule_rms_displacement),
            "high_flex_torsion_change": float(high.high_flex_torsion_change),
            "clash_penetration_delta": float(oracle.clash_penetration - upstream.clash_penetration),
            "severe_clash_delta": float(oracle.severe_clash_rate - upstream.severe_clash_rate),
            "chirality_error_delta": float(oracle.chirality_error - upstream.chirality_error),
            "clean_identity_fraction": clean_identity,
            "acceptance_fraction": float(oracle.accepted_fraction),
            "numerical_failure_fraction": numerical_failure_fraction,
        },
        "bootstrap": {
            "paired_relative_bond": bootstrap_result,
            "paired_delta_metrics": delta_bootstrap,
        },
        "criteria": {name: bool(value) for name, value in criteria.items()},
        "stage_d1_permitted": decision == "PASS",
        "stage_d_20k_permitted": False, "stage_d_100k_permitted": False,
        "next_command": None,
    }
    atomic_json_save(result, OUTPUT_DIR / "result.json")
    summary.to_csv(OUTPUT_DIR / "subgroup_summary.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "solver_diagnostics.csv", index=False)

    lines = [
        "# MCVR Stage D Bond Oracle", "",
        f"Decision: **{decision}**", "",
        "The D0 oracle solves `J^T (J J^T + lambda I)^-1 r` globally over unique undirected bonds. Corrections are translation-free and then pass through the frozen trust and deterministic acceptance rules.", "",
        "| Metric | Value |", "|---|---:|",
        f"| Accepted bond relative improvement | {_fmt(accepted_relative)} |",
        f"| Minimal Target available improvement | {_fmt(target_relative)} |",
        f"| Target recovery upper bound | {_fmt(recovery)} |",
        f"| RMSD delta | {_fmt(oracle.aligned_RMSD-upstream.aligned_RMSD)} |",
        f"| Total validity delta | {_fmt(oracle.total_thresholded_validity_score-upstream.total_thresholded_validity_score)} |",
        f"| Angle rate delta | {_fmt(oracle.angle_outlier_rate-upstream.angle_outlier_rate)} |",
        f"| Ring rate delta | {_fmt(oracle.ring_bond_outlier_rate-upstream.ring_bond_outlier_rate)} |",
        f"| High-flex torsion change | {_fmt(high.high_flex_torsion_change)} |",
        f"| Clean identity | {_fmt(clean_identity)} |",
        f"| Numerical failure fraction | {_fmt(numerical_failure_fraction)} |", "",
        "## Five-way comparison", "",
        "| Method | Bond rate | Bond magnitude | Total validity | Angle rate | Ring rate | RMSD | Displacement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        value = _row(summary, "all", method)
        lines.append(
            f"| {method} | {_fmt(value.bond_outlier_rate)} | {_fmt(value.bond_outlier_magnitude)} | "
            f"{_fmt(value.total_thresholded_validity_score)} | {_fmt(value.angle_outlier_rate)} | "
            f"{_fmt(value.ring_bond_outlier_rate)} | {_fmt(value.aligned_RMSD)} | "
            f"{_fmt(value.molecule_rms_displacement)} |"
        )
    lines += [
        "", "The chemical-rate noninferiority margin was fixed at `0.005` before D0 execution.", "",
        "## Gate", "", "| Condition | Result |", "|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in criteria.items())
    lines += [
        "", "No training, test evaluation, seed43/44, 20k, or 100k run was performed.",
        "Stage D1 is permitted only when this decision is PASS.",
    ]
    Path("docs/MCVR_STAGE_D_BOND_ORACLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    state_path = Path("reports/ecir_mvr/progressive_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "stage_d_status": "D1_PREPARATION" if decision == "PASS" else "ORACLE_NO_GO_COMPLETE",
        "stage_d_oracle_decision": decision,
        "stage_d_pilot_decision": None,
        "stage_d_selected_method": None,
        "stage_d_20k_permitted": False, "stage_d_100k_permitted": False,
        "medium_seed42_schedule_v4_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "100k_permitted": False, "100k_started": False,
        "seed43_44_permitted": False, "test_records_read": 0,
        "next_command": None, "next_commands": [],
    })
    atomic_json_save(state, state_path)
    print(json.dumps({
        "decision": decision, "accepted_relative_improvement": accepted_relative,
        "target_recovery": recovery, "criteria": criteria,
    }, indent=2))


if __name__ == "__main__":
    main()
