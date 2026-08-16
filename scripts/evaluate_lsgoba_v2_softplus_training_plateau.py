#!/usr/bin/env python3
"""Existing-checkpoint training-horizon evaluation for Softplus seed307."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.formal_rdkit_adapter import adapt_formal_cache_record
from etflow.ecir.learned_geometry import safety_accept
from etflow.ecir.lsgoba_v2_joint_magnitude import JointMagnitudeLSGO
from scripts.run_mcvr_lsgo import collate_graphs
from scripts.run_lsgoba_v2_softplus_seed307 import (
    ARTIFACT as TRAIN_ARTIFACT,
    REPORT as TRAIN_REPORT,
    SOURCE_PAYLOAD,
    base,
    config,
    loss_terms,
)
from scripts import evaluate_lsgoba_v2_softplus_seed307 as shared


STEPS = (12500, 15000, 17500, 20000, 22500)
METHODS = tuple(f"STEP{step}_{stage}" for step in STEPS for stage in ("PROPOSAL", "FINAL"))
REPORT = TRAIN_REPORT / "training_plateau_evaluation"
ARTIFACT = TRAIN_ARTIFACT / "training_plateau_evaluation"
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
SUMMARY = REPORT / "SOFTPLUS_TRAINING_PLATEAU_SUMMARY.csv"
RESULT = REPORT / "SOFTPLUS_TRAINING_PLATEAU_RESULT.json"


file_sha = shared.file_sha
atomic_json = shared.atomic_json
atomic_frame = shared.atomic_frame
atomic_torch = shared.atomic_torch


def update_status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    current = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    if state in {"RUNNING", "PASS", "COMPLETE"}:
        current.pop("error", None)
        current.pop("error_type", None)
    current.update({
        "schema_version": "lsgoba-v2-softplus-training-plateau-status-v1",
        "status": state,
        "stage": stage,
        "worker_pid": os.getpid(),
        "heartbeat": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
        "xtb_started": False,
        "other_seed_started": False,
        "training_started": False,
        **extra,
    })
    atomic_json(STATUS, current)


def checkpoint_path(step: int) -> Path:
    return TRAIN_ARTIFACT / f"checkpoints/step{step:05d}.ckpt"


def preflight() -> None:
    checkpoints = {}
    for step in STEPS:
        path = checkpoint_path(step)
        if not path.is_file():
            raise RuntimeError(f"checkpoint missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("global_step") != step or payload.get("warm_start") is not False:
            raise RuntimeError(f"checkpoint identity mismatch: {step}")
        if payload.get("bond_mean_parameterization") != "softplus(z_B)":
            raise RuntimeError(f"checkpoint parameterization mismatch: {step}")
        checkpoints[str(step)] = {"path": str(path), "sha256": file_sha(path)}
    manifest_path = TRAIN_REPORT / "V2_DEV_TEST_MANIFEST.json"
    manifest_sha = (TRAIN_REPORT / "V2_DEV_TEST_MANIFEST.sha256").read_text(encoding="utf-8").strip()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_sha(manifest_path) != manifest_sha or manifest.get("source_records") != 5000 or manifest.get("molecules") != 2500:
        raise RuntimeError("development manifest identity/denominator mismatch")
    payload = {
        "schema_version": "lsgoba-v2-softplus-training-plateau-preflight-v1",
        "status": "PASS",
        "steps": list(STEPS),
        "methods": list(METHODS),
        "checkpoints": checkpoints,
        "cohort": "V2_DEV_TEST",
        "records_per_method": 5000,
        "molecules_per_method": 2500,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "loss_reference_rule": "first frozen Reference conformer per molecule for every step and Source record",
        "proposal_definition": "atom-capped scaled proposal before safety_accept",
        "final_definition": "current safety_accept with exact Raw rollback",
        "scheduler_horizon": config()["training"]["scheduler_horizon"],
        "scheduler_review_deferred_until_plateau_decision": True,
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
        "xtb_stage": "PROHIBITED",
    }
    atomic_json(PREFLIGHT, payload)
    update_status("PREFLIGHT", "PASS", preflight_sha256=file_sha(PREFLIGHT))


def load_model(step: int, device: torch.device) -> JointMagnitudeLSGO:
    model = JointMagnitudeLSGO(hidden_dim=128, layers=3, initial_tau=.003, tau_max=.010).to(device)
    payload = torch.load(checkpoint_path(step), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def write_mol(writer: Chem.SDWriter, record: dict[str, Any], xyz: torch.Tensor,
              sample_id: str, molecule_id: str, method: str) -> None:
    adapted = adapt_formal_cache_record(record)
    molecule = Chem.Mol(adapted["_formal_rdkit_mol"])
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, point in enumerate(xyz.detach().cpu().double().tolist()):
        conformer.SetAtomPosition(index, point)
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    molecule.SetProp("_Name", sample_id)
    molecule.SetProp("sample_id", sample_id)
    molecule.SetProp("molecule_id", molecule_id)
    molecule.SetProp("method", method)
    writer.write(molecule)


def generate() -> None:
    if not PREFLIGHT.is_file() or json.loads(PREFLIGHT.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("plateau preflight must pass first")
    cfg = config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {step: load_model(step, device) for step in STEPS}
    prepared = base.load_prepared(cfg)
    by_item = {str(item["molecule_id"]): item for item in prepared["val"]}
    manifest = json.loads((TRAIN_REPORT / "V2_DEV_TEST_MANIFEST.json").read_text(encoding="utf-8"))
    manifest_sha = (TRAIN_REPORT / "V2_DEV_TEST_MANIFEST.sha256").read_text(encoding="utf-8").strip()
    source_payload = torch.load(SOURCE_PAYLOAD, map_location="cpu", weights_only=False)
    by_sample = {str(row["sample_id"]): row for row in source_payload["val"]}
    source_manifest = pd.read_parquet(cfg["sources"]["val_manifest"])
    source_paths = {
        str(row.sample_id): Path(cfg["sources"]["val_cache"]) / Path(str(row.source_path)).name
        for row in source_manifest.itertuples(index=False)
    }
    ids = [sample_id for row in manifest["rows"] for sample_id in row["sample_ids"]]
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise RuntimeError("development record identity mismatch")
    SDF.mkdir(parents=True, exist_ok=True)
    temps = {method: Path(str(SDF / f"{method}.sdf") + f".tmp.{os.getpid()}") for method in METHODS}
    writers = {method: Chem.SDWriter(str(path)) for method, path in temps.items()}
    diagnostics: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    losses = {step: {"weight": 0, "prior": 0.0, "post": 0.0, "move": 0.0} for step in STEPS}
    bond_values: dict[int, list[torch.Tensor]] = {step: [] for step in STEPS}
    update_status("GENERATE", completed_records=0, expected_records=5000, device=str(device))
    try:
        for start in range(0, len(ids), 64):
            batch_ids = ids[start:start + 64]
            source_rows = [by_sample[sample_id] for sample_id in batch_ids]
            items = [by_item[str(row["molecule_id"])] for row in source_rows]
            cpu_graphs = [item["graph"] for item in items]
            graphs = [graph.to(device) for graph in cpu_graphs]
            batch_graph = collate_graphs(cpu_graphs).to(device)
            source_tensor = torch.cat([row["source"].to(device=device, dtype=torch.float64) for row in source_rows])
            reference_tensor = torch.cat([item["references"][0] for item in items]).to(device)
            records = []
            for sample_id, row in zip(batch_ids, source_rows, strict=True):
                record = torch.load(source_paths[sample_id], map_location="cpu", weights_only=False)
                if base.coordinate_sha(torch.as_tensor(record["x_init"], dtype=torch.float32)) != row["coordinate_sha256"]:
                    raise RuntimeError("development Source rebinding failed")
                records.append(record)
            outputs = {}
            for step, model in models.items():
                with torch.enable_grad():
                    terms = loss_terms(model, graphs, batch_graph, source_tensor, reference_tensor)
                weight = len(batch_ids)
                losses[step]["weight"] += weight
                for name in ("prior", "post", "move"):
                    losses[step][name] += weight * float(terms[name].detach())
                bond_values[step].append(terms["bond_mu"].detach().cpu().double())
                outputs[step] = (terms["proposal"].detach(), terms["tau"].detach(), terms["cap_active"].detach())
            offset = 0
            for local_index, (sample_id, source_row, graph, record) in enumerate(
                zip(batch_ids, source_rows, graphs, records, strict=True)
            ):
                atom_count = int(graph.atom_categorical.size(0))
                local_source = source_tensor[offset:offset + atom_count].detach()
                molecule_id = str(source_row["molecule_id"])
                if start == 0 or STEPS[0] == STEPS[0]:
                    metadata.append({
                        "sample_id": sample_id, "molecule_id": molecule_id,
                        "source": local_source.cpu().double(), "source_path": str(source_paths[sample_id]),
                    })
                for step in STEPS:
                    proposal_batch, tau_batch, cap_batch = outputs[step]
                    proposal = proposal_batch[offset:offset + atom_count]
                    tau = float(tau_batch[local_index])
                    final, safety = safety_accept(local_source, proposal, graph)
                    if safety["fallback"] and not torch.equal(final, local_source):
                        raise RuntimeError("rollback is not exact Raw")
                    proposal_rms = float((proposal - local_source).square().sum(-1).mean().sqrt())
                    final_rms = float((final - local_source).square().sum(-1).mean().sqrt())
                    proposal_max = float(torch.linalg.vector_norm(proposal - local_source, dim=-1).max())
                    if proposal_rms > tau + 1e-12 or proposal_max > .03 + 1e-12:
                        raise RuntimeError("deployment budget/cap changed")
                    proposal_method = f"STEP{step}_PROPOSAL"
                    final_method = f"STEP{step}_FINAL"
                    write_mol(writers[proposal_method], record, proposal, sample_id, molecule_id, proposal_method)
                    write_mol(writers[final_method], record, final, sample_id, molecule_id, final_method)
                    diagnostics.append({
                        "step": step, "record_id": sample_id, "molecule_id": molecule_id,
                        "tau": tau, "proposal_graph_rms": proposal_rms, "final_graph_rms": final_rms,
                        "proposal_max_atom_displacement": proposal_max,
                        "atom_cap_active": bool(cap_batch[local_index]),
                        "accepted": bool(safety["accepted"]), "rollback": bool(safety["fallback"]),
                        "finite": bool(safety["finite"]),
                        "chirality": bool(safety["chirality_preserved"]),
                        "ring": bool(safety["ring_nonregression"]),
                        "penetration": bool(safety["catastrophic_clash_nonregression"]),
                    })
                offset += atom_count
            completed = min(start + len(batch_ids), len(ids))
            if completed % 128 == 0 or completed == 5000:
                update_status("GENERATE", completed_records=completed, expected_records=5000, device=str(device))
            if completed % 256 == 0 or completed == 5000:
                print(json.dumps({"stage": "GENERATE", "records": completed, "expected": 5000}), flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    for method in METHODS:
        os.replace(temps[method], SDF / f"{method}.sdf")
    diagnostics_frame = pd.DataFrame(diagnostics)
    if len(diagnostics_frame) != 25000 or diagnostics_frame.duplicated(["step", "record_id"]).any():
        raise RuntimeError("diagnostics denominator changed")
    atomic_frame(DIAGNOSTICS, diagnostics_frame)
    atomic_torch(METADATA, {
        "schema_version": "lsgoba-v2-softplus-training-plateau-records-v1",
        "records": metadata, "manifest_sha256": manifest_sha,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    })
    loss_rows = []
    bond_rows = []
    for step in STEPS:
        row = losses[step]
        loss_rows.append({"step": step, **{name: row[name] / row["weight"] for name in ("prior", "post", "move")}})
        values = torch.cat(bond_values[step]).numpy()
        bond_rows.append({
            "step": step, "primitive_predictions": int(values.size),
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p99": float(np.quantile(values, .99)), "p999": float(np.quantile(values, .999)),
            "max": float(np.max(values)), "nan_count": int(np.isnan(values).sum()),
            "inf_count": int(np.isinf(values).sum()),
        })
    pd.DataFrame(loss_rows).to_csv(LOSS_SUMMARY, index=False)
    pd.DataFrame(bond_rows).to_csv(BOND_MU_SUMMARY, index=False)
    freeze = {
        "schema_version": "lsgoba-v2-softplus-training-plateau-coordinate-freeze-v1",
        "status": "FROZEN_BEFORE_ENDPOINTS", "cohort": "V2_DEV_TEST",
        "records_per_method": 5000, "molecules": 2500, "methods": list(METHODS),
        "steps": list(STEPS), "manifest_sha256": manifest_sha,
        "checkpoints": json.loads(PREFLIGHT.read_text(encoding="utf-8"))["checkpoints"],
        "diagnostics_sha256": file_sha(DIAGNOSTICS), "metadata_sha256": file_sha(METADATA),
        "loss_summary_sha256": file_sha(LOSS_SUMMARY), "bond_mu_summary_sha256": file_sha(BOND_MU_SUMMARY),
        "sdfs": {method: {"path": str(SDF / f"{method}.sdf"), "sha256": file_sha(SDF / f"{method}.sdf")} for method in METHODS},
        "xtb_stage": "NOT_STARTED_AND_PROHIBITED",
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    }
    atomic_json(FREEZE, freeze)
    update_status("COORDINATES_FROZEN", "PASS", coordinate_freeze_sha256=file_sha(FREEZE))


def configure_shared() -> None:
    shared.METHODS = METHODS
    shared.REPORT = REPORT
    shared.ARTIFACT = ARTIFACT
    shared.SDF = SDF
    shared.STATUS = STATUS
    shared.DIAGNOSTICS = DIAGNOSTICS
    shared.BOND_MU_SUMMARY = BOND_MU_SUMMARY
    shared.METADATA = METADATA
    shared.FREEZE = FREEZE
    shared.PB_PATH = PB_PATH
    shared.V3D_PATH = V3D_PATH
    shared.ENDPOINTS = ENDPOINTS
    shared.FIDELITY_RECORD = FIDELITY_RECORD
    shared.FIDELITY_MOLECULE = FIDELITY_MOLECULE
    shared.FIDELITY_SUMMARY = FIDELITY_SUMMARY
    shared.FIDELITY_COMPLETION = FIDELITY_COMPLETION
    shared.update_status = update_status


def fidelity() -> None:
    configure_shared()
    shared.fidelity()


def summarize() -> None:
    endpoints = json.loads(ENDPOINTS.read_text(encoding="utf-8"))
    fidelity_completion = json.loads(FIDELITY_COMPLETION.read_text(encoding="utf-8"))
    if endpoints.get("status") != "COMPLETE" or fidelity_completion.get("status") != "COMPLETE":
        raise RuntimeError("plateau evaluation incomplete")
    diagnostics = pd.read_parquet(DIAGNOSTICS)
    pb = pd.read_parquet(PB_PATH)[["method", "record_id", "PB"]]
    v3d = pd.read_parquet(V3D_PATH)[["method", "record_id", "validity3d"]].rename(columns={"validity3d": "V3D"})
    fidelity_summary = pd.read_csv(FIDELITY_SUMMARY).set_index("method")
    losses = pd.read_csv(LOSS_SUMMARY).set_index("step")
    bonds = pd.read_csv(BOND_MU_SUMMARY).set_index("step")
    rows = []
    for step in STEPS:
        diag = diagnostics[diagnostics.step == step]
        tau = diag.tau.to_numpy(dtype=np.float64)
        for stage in ("PROPOSAL", "FINAL"):
            method = f"STEP{step}_{stage}"
            fidelity_row = fidelity_summary.loc[method]
            rows.append({
                "step": step, "stage": stage, "method": method,
                "V3D": float(v3d[v3d.method == method].V3D.mean()),
                "PB": float(pb[pb.method == method].PB.mean()),
                "proposal_rms_mean": float(diag.proposal_graph_rms.mean()),
                "final_rms_mean": float(diag.final_graph_rms.mean()),
                "rollback_fraction": float(diag.rollback.mean()),
                "ring_trigger_fraction": float((~diag.ring).mean()),
                "penetration_trigger_fraction": float((~diag.penetration).mean()),
                "chirality_trigger_fraction": float((~diag.chirality).mean()),
                "tau_mean": float(tau.mean()), "tau_median": float(np.median(tau)),
                "tau_p90": float(np.quantile(tau, .90)), "tau_p95": float(np.quantile(tau, .95)),
                "tau_p99": float(np.quantile(tau, .99)),
                "source_rmsd_mean": float(fidelity_row.source_rmsd_mean),
                "reference_rmsd_mean": float(fidelity_row.reference_rmsd_mean),
                "COV_P": float(fidelity_row.COV_P), "COV_R": float(fidelity_row.COV_R),
                "AMR_P": float(fidelity_row.AMR_P), "AMR_R": float(fidelity_row.AMR_R),
                "L_prior": float(losses.loc[step].prior), "L_post": float(losses.loc[step].post),
                "L_move": float(losses.loc[step].move),
                "bond_mu_mean": float(bonds.loc[step]["mean"]),
                "bond_mu_median": float(bonds.loc[step]["median"]),
                "bond_mu_p99": float(bonds.loc[step].p99),
                "bond_mu_p999": float(bonds.loc[step].p999),
                "bond_mu_max": float(bonds.loc[step]["max"]),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(SUMMARY, index=False)
    proposal = frame[frame.stage == "PROPOSAL"].sort_values("step")
    result = {
        "schema_version": "lsgoba-v2-softplus-training-plateau-result-v1",
        "status": "COMPLETE", "development_only": True,
        "steps": list(STEPS), "summary": frame.to_dict("records"),
        "proposal_curve": proposal.to_dict("records"),
        "scheduler": {
            "type": config()["training"]["scheduler"],
            "T_max": config()["training"]["scheduler_horizon"],
            "step22500_is_T_max": config()["training"]["scheduler_horizon"] == 22500,
            "continuation_beyond_T_max_started": False,
        },
        "plateau_decision": "PENDING_INTERPRETATION",
        "xtb_started": False, "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0, "other_seed_started": False,
        "new_training_started": False, "model_modified": False,
    }
    atomic_json(RESULT, result)
    update_status("COMPLETE", "COMPLETE", result_sha256=file_sha(RESULT))
    print(json.dumps(result, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "generate", "fidelity", "summarize", "status"))
    args = parser.parse_args()
    if args.phase == "status":
        print(STATUS.read_text(encoding="utf-8") if STATUS.is_file() else json.dumps({"status": "NOT_STARTED"}, indent=2))
        return 0
    try:
        globals()[args.phase]()
    except Exception as error:
        update_status(args.phase.upper(), "FAILED", error_type=type(error).__name__, error=str(error))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
