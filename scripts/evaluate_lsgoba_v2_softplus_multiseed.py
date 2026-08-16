#!/usr/bin/env python3
"""Unified non-xTB development evaluation for Softplus-v2 seeds 307/331/353."""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
SOURCE = ROOT / "scripts/evaluate_lsgoba_v2_softplus_training_plateau.py"
source_text = SOURCE.read_text(encoding="utf-8")
source_text = source_text.replace(
    "STEPS = (12500, 15000, 17500, 20000, 22500)",
    "STEPS = (307, 331, 353)",
)
source_text = source_text.replace("STEP{step}", "SEED{step}")
source_text = source_text.replace("len(diagnostics_frame) != 25000", "len(diagnostics_frame) != 15000")
evaluation = types.ModuleType("softplus_multiseed_evaluation_base")
evaluation.__file__ = str(SOURCE)
exec(compile(source_text, str(SOURCE), "exec"), evaluation.__dict__)

SEEDS = (307, 331, 353)
METHODS = tuple(f"SEED{seed}_{stage}" for seed in SEEDS for stage in ("PROPOSAL", "FINAL"))
REPORT = ROOT / "reports/ecir_mvr/lsgoba_v2_softplus_multiseed/final_development_evaluation"
ARTIFACT = ROOT / "artifacts/ecir_mvr/lsgoba_v2_softplus_multiseed/final_development_evaluation"
SDF = ARTIFACT / "sdf"
STATUS = REPORT / "STATUS.json"
PREFLIGHT = REPORT / "PREFLIGHT.json"
DIAGNOSTICS = ARTIFACT / "COORDINATE_DIAGNOSTICS.parquet"
METADATA = ARTIFACT / "DEVELOPMENT_RECORDS.pt"
FREEZE = ARTIFACT / "COORDINATE_FREEZE.json"
LOSS_SUMMARY = ARTIFACT / "LOSS_SUMMARY.csv"
BOND_MU_SUMMARY = ARTIFACT / "BOND_MU_SUMMARY.csv"
PB_PATH = ARTIFACT / "POSEBUSTERS.parquet"
V3D_PATH = ARTIFACT / "VALIDITY3D.parquet"
ENDPOINTS = ARTIFACT / "ENDPOINT_COMPLETION.json"
FIDELITY_RECORD = ARTIFACT / "FIDELITY_PER_RECORD.parquet"
FIDELITY_MOLECULE = ARTIFACT / "FIDELITY_PER_MOLECULE.parquet"
FIDELITY_SUMMARY = ARTIFACT / "FIDELITY_SUMMARY.csv"
FIDELITY_COMPLETION = ARTIFACT / "FIDELITY_COMPLETION.json"
SUMMARY = REPORT / "SOFTPLUS_MULTISEED_SUMMARY.csv"
RESULT = REPORT / "SOFTPLUS_MULTISEED_RESULT.json"
PLAN = ROOT / "configs/ecir_mvr_lsgoba_v2_softplus_multiseed.json"
SEED307_ROOT = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-v2-softplus-seed307")


def checkpoint_path(seed: int) -> Path:
    if seed == 307:
        return SEED307_ROOT / "artifacts/ecir_mvr/lsgoba_v2_softplus_seed307/checkpoints/step17500.ckpt"
    return ROOT / f"artifacts/ecir_mvr/lsgoba_v2_softplus_multiseed/seed{seed}/checkpoints/step17500.ckpt"


def train_report(seed: int) -> Path:
    if seed == 307:
        return SEED307_ROOT / "reports/ecir_mvr/lsgoba_v2_softplus_seed307"
    return ROOT / f"reports/ecir_mvr/lsgoba_v2_softplus_multiseed/seed{seed}"


def configure() -> None:
    assignments = {
        "STEPS": SEEDS, "METHODS": METHODS, "REPORT": REPORT, "ARTIFACT": ARTIFACT,
        "SDF": SDF, "STATUS": STATUS, "PREFLIGHT": PREFLIGHT, "DIAGNOSTICS": DIAGNOSTICS,
        "METADATA": METADATA, "FREEZE": FREEZE, "LOSS_SUMMARY": LOSS_SUMMARY,
        "BOND_MU_SUMMARY": BOND_MU_SUMMARY, "PB_PATH": PB_PATH, "V3D_PATH": V3D_PATH,
        "ENDPOINTS": ENDPOINTS, "FIDELITY_RECORD": FIDELITY_RECORD,
        "FIDELITY_MOLECULE": FIDELITY_MOLECULE, "FIDELITY_SUMMARY": FIDELITY_SUMMARY,
        "FIDELITY_COMPLETION": FIDELITY_COMPLETION, "SUMMARY": SUMMARY, "RESULT": RESULT,
        "TRAIN_REPORT": train_report(331), "checkpoint_path": checkpoint_path,
    }
    evaluation.__dict__.update(assignments)


def preflight() -> None:
    configure()
    protocol = json.loads(PLAN.read_text(encoding="utf-8"))
    checkpoints = {}
    manifests = {}
    for seed in SEEDS:
        path = checkpoint_path(seed)
        if not path.is_file():
            raise RuntimeError(f"seed{seed} checkpoint missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("seed") != seed or payload.get("global_step") != 17500:
            raise RuntimeError(f"seed{seed} checkpoint seed/step identity mismatch")
        if payload.get("warm_start") is not False or payload.get("initialization") != "FROM_SCRATCH_FULL_TRAINABLE_MODEL":
            raise RuntimeError(f"seed{seed} is not a from-scratch replicate")
        if payload.get("bond_mean_parameterization") != "softplus(z_B)":
            raise RuntimeError(f"seed{seed} Bond parameterization changed")
        scheduler = payload["scheduler_state"]
        if scheduler.get("T_max") != 22500 or scheduler.get("last_epoch") != 17500:
            raise RuntimeError(f"seed{seed} scheduler trajectory changed")
        if seed != 307:
            integrity = json.loads((train_report(seed) / "INTEGRITY.json").read_text(encoding="utf-8"))
            if integrity.get("status") != "PASS":
                raise RuntimeError(f"seed{seed} training integrity is not PASS")
        manifest_path = train_report(seed) / "V2_DEV_TEST_MANIFEST.json"
        manifest_sha = (train_report(seed) / "V2_DEV_TEST_MANIFEST.sha256").read_text(encoding="utf-8").strip()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if evaluation.file_sha(manifest_path) != manifest_sha or manifest.get("source_records") != 5000 or manifest.get("molecules") != 2500:
            raise RuntimeError(f"seed{seed} development manifest changed")
        manifests[str(seed)] = {"path": str(manifest_path), "sha256": manifest_sha}
        checkpoints[str(seed)] = {"path": str(path), "sha256": evaluation.file_sha(path)}
    if len({entry["sha256"] for entry in manifests.values()}) != 1:
        raise RuntimeError("three seeds do not use the identical V2_DEV_TEST manifest")
    checks = {
        "schema_version": "lsgoba-v2-softplus-multiseed-evaluation-preflight-v1",
        "status": "PASS", "development_only": True, "seeds": list(SEEDS),
        "methods": list(METHODS), "cohort": "V2_DEV_TEST", "records_per_method": 5000,
        "molecules_per_method": 2500, "checkpoints": checkpoints, "manifests": manifests,
        "scheduler_T_max": 22500, "checkpoint_step": 17500,
        "decision_thresholds": protocol["replication_decision_thresholds"],
        "proposal_definition": "atom-capped scaled proposal before safety_accept",
        "final_definition": "current safety_accept with exact Raw rollback",
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
        "xtb_stage": "PROHIBITED", "orca_stage": "PROHIBITED", "docking_stage": "PROHIBITED",
    }
    evaluation.atomic_json(PREFLIGHT, checks)
    evaluation.update_status("PREFLIGHT", "PASS", preflight_sha256=evaluation.file_sha(PREFLIGHT))


def summarize() -> None:
    configure()
    endpoints = json.loads(ENDPOINTS.read_text(encoding="utf-8"))
    fidelity_completion = json.loads(FIDELITY_COMPLETION.read_text(encoding="utf-8"))
    if endpoints.get("status") != "COMPLETE" or fidelity_completion.get("status") != "COMPLETE":
        raise RuntimeError("multiseed external/fidelity evaluation incomplete")
    diagnostics = pd.read_parquet(DIAGNOSTICS)
    pb = pd.read_parquet(PB_PATH)[["method", "record_id", "PB"]]
    v3d = pd.read_parquet(V3D_PATH)[["method", "record_id", "validity3d"]].rename(columns={"validity3d": "V3D"})
    fidelity = pd.read_csv(FIDELITY_SUMMARY).set_index("method")
    bonds = pd.read_csv(BOND_MU_SUMMARY).set_index("step")
    losses = pd.read_csv(LOSS_SUMMARY).set_index("step")
    rows = []
    for seed in SEEDS:
        diag = diagnostics[diagnostics.step == seed]
        tau = diag.tau.to_numpy(dtype=np.float64)
        for stage in ("PROPOSAL", "FINAL"):
            method = f"SEED{seed}_{stage}"
            f = fidelity.loc[method]
            rows.append({
                "seed": seed, "stage": stage, "method": method, "V3D": float(v3d[v3d.method == method].V3D.mean()),
                "PB": float(pb[pb.method == method].PB.mean()),
                "proposal_rms_mean": float(diag.proposal_graph_rms.mean()), "final_rms_mean": float(diag.final_graph_rms.mean()),
                "source_rmsd_mean": float(f.source_rmsd_mean), "reference_rmsd_mean": float(f.reference_rmsd_mean),
                "COV_P": float(f.COV_P), "COV_R": float(f.COV_R), "AMR_P": float(f.AMR_P), "AMR_R": float(f.AMR_R),
                "tau_mean": float(tau.mean()), "tau_median": float(np.median(tau)),
                "tau_p90": float(np.quantile(tau, .90)), "tau_p95": float(np.quantile(tau, .95)),
                "tau_p99": float(np.quantile(tau, .99)), "tau_ge0099_fraction": float((tau >= .0099).mean()),
                "atom_cap_fraction": float(diag.atom_cap_active.mean()), "rollback_fraction": float(diag.rollback.mean()),
                "ring_trigger_fraction": float((~diag.ring.astype(bool)).mean()),
                "penetration_trigger_fraction": float((~diag.penetration.astype(bool)).mean()),
                "chirality_trigger_fraction": float((~diag.chirality.astype(bool)).mean()),
                "bond_mu_mean": float(bonds.loc[seed]["mean"]), "bond_mu_median": float(bonds.loc[seed]["median"]),
                "bond_mu_p99": float(bonds.loc[seed].p99), "bond_mu_p999": float(bonds.loc[seed].p999),
                "bond_mu_max": float(bonds.loc[seed]["max"]), "bond_mu_nan": int(bonds.loc[seed].nan_count),
                "bond_mu_inf": int(bonds.loc[seed].inf_count),
                "L_prior": float(losses.loc[seed].prior), "L_post": float(losses.loc[seed].post), "L_move": float(losses.loc[seed].move),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(SUMMARY, index=False)
    proposal = frame[frame.stage == "PROPOSAL"].set_index("seed")
    final = frame[frame.stage == "FINAL"].set_index("seed")
    aggregate_metrics = {
        "Proposal_V3D": proposal.V3D, "Proposal_PB": proposal.PB,
        "Proposal_RMS": proposal.proposal_rms_mean, "Final_V3D": final.V3D,
        "Final_PB": final.PB, "Source_RMSD": proposal.source_rmsd_mean,
        "Reference_RMSD": proposal.reference_rmsd_mean, "tau_mean": proposal.tau_mean,
        "Bond_mu_mean": proposal.bond_mu_mean, "Bond_mu_P99": proposal.bond_mu_p99,
        "Bond_mu_max": proposal.bond_mu_max,
    }
    mean_std = {name: {"mean": float(values.mean()), "std_sample": float(values.std(ddof=1))} for name, values in aggregate_metrics.items()}
    thresholds = json.loads(PREFLIGHT.read_text(encoding="utf-8"))["decision_thresholds"]
    ranges = {name: float(values.max() - values.min()) for name, values in aggregate_metrics.items()}
    decisions = {
        "MULTISEED_V3D_STABILITY": "PASS" if ranges["Proposal_V3D"] <= thresholds["proposal_v3d_max_minus_min"] else "CONCERN",
        "MULTISEED_PB_STABILITY": "PASS" if ranges["Proposal_PB"] <= thresholds["proposal_pb_max_minus_min"] else "CONCERN",
        "MULTISEED_TAU_STABILITY": "PASS" if ranges["tau_mean"] <= thresholds["tau_mean_max_minus_min_angstrom"] else "CONCERN",
        "MULTISEED_BOND_MU_STABILITY": "PASS" if (
            ranges["Bond_mu_mean"] <= thresholds["bond_mu_mean_max_minus_min_angstrom"]
            and ranges["Bond_mu_P99"] <= thresholds["bond_mu_p99_max_minus_min_angstrom"]
            and ranges["Bond_mu_max"] <= thresholds["bond_mu_max_max_minus_min_angstrom"]
            and int(proposal.bond_mu_nan.sum()) == 0 and int(proposal.bond_mu_inf.sum()) == 0
        ) else "CONCERN",
    }
    replicated = all(value == "PASS" for value in decisions.values())
    decisions["SOFTPLUS_ARCHITECTURE_REPLICATED"] = "YES" if replicated else "PARTIAL"
    decisions["FINAL_RECIPE_READY_TO_FREEZE"] = "YES" if replicated else "PARTIAL"
    result = {
        "schema_version": "lsgoba-v2-softplus-multiseed-result-v1", "status": "COMPLETE",
        "development_only": True, "seeds": list(SEEDS), "checkpoint_step": 17500,
        "per_seed": frame.to_dict("records"), "mean_std": mean_std, "ranges": ranges,
        "thresholds_preregistered_before_seed331_completed": thresholds, "decisions": decisions,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
        "xtb_started": False, "orca_started": False, "docking_started": False,
        "seed307_retrained": False, "model_architecture_changed": False, "loss_changed": False,
        "lambda_changed": False, "scheduler_T_max_changed": False,
    }
    evaluation.atomic_json(RESULT, result)
    evaluation.update_status("COMPLETE", "COMPLETE", result_sha256=evaluation.file_sha(RESULT))
    print(json.dumps(result, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "generate", "fidelity", "summarize", "status"))
    args = parser.parse_args()
    configure()
    if args.phase == "status":
        print(STATUS.read_text(encoding="utf-8") if STATUS.is_file() else json.dumps({"status": "NOT_STARTED"}, indent=2))
        return 0
    try:
        if args.phase == "preflight":
            preflight()
        elif args.phase == "summarize":
            summarize()
        else:
            evaluation.__dict__[args.phase]()
    except Exception as error:
        evaluation.update_status(args.phase.upper(), "FAILED", error_type=type(error).__name__, error=str(error))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
