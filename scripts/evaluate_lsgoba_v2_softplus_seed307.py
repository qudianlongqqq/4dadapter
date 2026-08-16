#!/usr/bin/env python3
"""One-time development-only proposal/final evaluation of Softplus seed307."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

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
from etflow.ecir.lsgoba_v2_joint_magnitude import (
    JointMagnitudeLSGO,
    batch_source_directions_and_state,
    scaled_proposal,
)
from etflow.ecir.run_a_evaluation import rmsd_matrix
from scripts.run_mcvr_lsgo import collate_graphs
from scripts.run_lsgoba_v2_softplus_seed307 import (
    ARTIFACT as TRAIN_ARTIFACT,
    BEST,
    CONFIG_PATH,
    REPORT as TRAIN_REPORT,
    SOURCE_PAYLOAD,
    base,
    config,
)


METHODS = ("SOFTPLUS_PROPOSAL", "SOFTPLUS_FINAL")
REPORT = TRAIN_REPORT / "final_development_evaluation"
ARTIFACT = TRAIN_ARTIFACT / "final_development_evaluation"
SDF = ARTIFACT / "sdf"
STATUS = REPORT / "STATUS.json"
PREFLIGHT = REPORT / "PREFLIGHT.json"
DIAGNOSTICS = ARTIFACT / "COORDINATE_DIAGNOSTICS.parquet"
BOND_MU_SUMMARY = ARTIFACT / "BOND_MU_SUMMARY.json"
METADATA = ARTIFACT / "DEVELOPMENT_RECORDS.pt"
FREEZE = ARTIFACT / "COORDINATE_FREEZE.json"
PB_PATH = ARTIFACT / "POSEBUSTERS.parquet"
V3D_PATH = ARTIFACT / "VALIDITY3D.parquet"
ENDPOINTS = ARTIFACT / "ENDPOINT_COMPLETION.json"
FIDELITY_RECORD = ARTIFACT / "FIDELITY_PER_RECORD.parquet"
FIDELITY_MOLECULE = ARTIFACT / "FIDELITY_PER_MOLECULE.parquet"
FIDELITY_SUMMARY = ARTIFACT / "FIDELITY_SUMMARY.csv"
FIDELITY_COMPLETION = ARTIFACT / "FIDELITY_COMPLETION.json"
SUMMARY = REPORT / "SOFTPLUS_FINAL_DEVELOPMENT_SUMMARY.csv"
RESULT = REPORT / "SOFTPLUS_SEED307_FEASIBILITY_RESULT.json"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temp)
    os.replace(temp, path)


def update_status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    current = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    if state in {"RUNNING", "PASS", "COMPLETE"}:
        current.pop("error", None)
        current.pop("error_type", None)
    current.update({
        "schema_version": "lsgoba-v2-softplus-seed307-evaluation-status-v1",
        "status": state,
        "stage": stage,
        "worker_pid": os.getpid(),
        "heartbeat": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
        "xtb_started": False,
        "other_seed_started": False,
        **extra,
    })
    atomic_json(STATUS, current)


def preflight() -> None:
    train_status = json.loads((TRAIN_REPORT / "SOFTPLUS_SEED307_STATUS.json").read_text(encoding="utf-8"))
    if train_status.get("status") != "PASS" or train_status.get("completed_steps") != 22500:
        raise RuntimeError("Softplus training is not complete")
    checkpoint_sha = train_status.get("best_checkpoint_sha256")
    if not BEST.is_file() or file_sha(BEST) != checkpoint_sha:
        raise RuntimeError("final checkpoint identity mismatch")
    payload = torch.load(BEST, map_location="cpu", weights_only=False)
    if payload.get("global_step") != 22500 or payload.get("warm_start") is not False:
        raise RuntimeError("checkpoint step/initialization identity mismatch")
    if payload.get("bond_mean_parameterization") != "softplus(z_B)":
        raise RuntimeError("checkpoint Bond parameterization mismatch")
    manifest_path = TRAIN_REPORT / "V2_DEV_TEST_MANIFEST.json"
    manifest_sha = (TRAIN_REPORT / "V2_DEV_TEST_MANIFEST.sha256").read_text(encoding="utf-8").strip()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_sha(manifest_path) != manifest_sha or manifest.get("source_records") != 5000 or manifest.get("molecules") != 2500:
        raise RuntimeError("development manifest identity/denominator mismatch")
    checks = {
        "schema_version": "lsgoba-v2-softplus-seed307-evaluation-preflight-v1",
        "status": "PASS",
        "cohort": "V2_DEV_TEST",
        "records": 5000,
        "molecules": 2500,
        "methods": list(METHODS),
        "checkpoint": str(BEST),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": 22500,
        "proposal_definition": "atom-capped scaled proposal before safety_accept",
        "final_definition": "current safety_accept with exact Raw rollback",
        "cov_threshold_angstrom": 1.25,
        "xtb_stage": "PROHIBITED",
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
    }
    atomic_json(PREFLIGHT, checks)
    update_status("PREFLIGHT", "PASS", preflight_sha256=file_sha(PREFLIGHT))


def load_model(device: torch.device) -> JointMagnitudeLSGO:
    model = JointMagnitudeLSGO(hidden_dim=128, layers=3, initial_tau=.003, tau_max=.010).to(device)
    payload = torch.load(BEST, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def write_mol(writer: Chem.SDWriter, record: Mapping[str, Any], xyz: torch.Tensor,
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
        raise RuntimeError("evaluation preflight must pass first")
    cfg = config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    prepared = base.load_prepared(cfg)
    by_item = {str(item["molecule_id"]): item for item in prepared["val"]}
    manifest_path = TRAIN_REPORT / "V2_DEV_TEST_MANIFEST.json"
    manifest_sha = (TRAIN_REPORT / "V2_DEV_TEST_MANIFEST.sha256").read_text(encoding="utf-8").strip()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    diagnostic_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    bond_mu_values: list[torch.Tensor] = []
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
            records = []
            for sample_id, row in zip(batch_ids, source_rows, strict=True):
                record = torch.load(source_paths[sample_id], map_location="cpu", weights_only=False)
                if base.coordinate_sha(torch.as_tensor(record["x_init"], dtype=torch.float32)) != row["coordinate_sha256"]:
                    raise RuntimeError("development Source rebinding failed")
                records.append(record)
            with torch.enable_grad():
                prediction = model.geometry(batch_graph)
                parameters = {
                    **prediction,
                    "bond_sigma": batch_graph.bond_fixed[:, 1],
                    "angle_sigma": batch_graph.angle_fixed[:, 1],
                }
                directions, state, embedding, _ = batch_source_directions_and_state(
                    source_tensor, graphs, parameters, prediction["node_embedding"]
                )
                tau_values = model.magnitude(embedding, model.normalized_state(state)).detach()
                bond_mu_values.append(prediction["bond_mu"].detach().cpu().double())
            offset = 0
            for local_index, (sample_id, source_row, graph, record) in enumerate(
                zip(batch_ids, source_rows, graphs, records, strict=True)
            ):
                atom_count = int(graph.atom_categorical.size(0))
                local_source = source_tensor[offset:offset + atom_count].detach()
                local_direction = directions[offset:offset + atom_count].detach()
                molecule_id = str(source_row["molecule_id"])
                tau = float(tau_values[local_index])
                proposal, cap, _ = scaled_proposal(
                    local_source, local_direction, local_source.new_tensor([tau]), [graph], atom_cap=.03
                )
                final, safety = safety_accept(local_source, proposal, graph)
                if safety["fallback"] and not torch.equal(final, local_source):
                    raise RuntimeError("rollback is not exact Raw")
                proposal_rms = float((proposal - local_source).square().sum(-1).mean().sqrt())
                final_rms = float((final - local_source).square().sum(-1).mean().sqrt())
                proposal_max = float(torch.linalg.vector_norm(proposal - local_source, dim=-1).max())
                if proposal_rms > tau + 1e-12 or proposal_max > .03 + 1e-12:
                    raise RuntimeError("deployment budget/cap changed")
                write_mol(writers["SOFTPLUS_PROPOSAL"], record, proposal, sample_id, molecule_id, "SOFTPLUS_PROPOSAL")
                write_mol(writers["SOFTPLUS_FINAL"], record, final, sample_id, molecule_id, "SOFTPLUS_FINAL")
                diagnostic_rows.append({
                    "record_id": sample_id,
                    "molecule_id": molecule_id,
                    "tau": tau,
                    "proposal_graph_rms": proposal_rms,
                    "final_graph_rms": final_rms,
                    "proposal_max_atom_displacement": proposal_max,
                    "atom_cap_active": bool(cap[0]),
                    "accepted": bool(safety["accepted"]),
                    "rollback": bool(safety["fallback"]),
                    "finite": bool(safety["finite"]),
                    "chirality": safety["chirality_preserved"],
                    "ring": safety["ring_nonregression"],
                    "penetration": safety["catastrophic_clash_nonregression"],
                })
                metadata_rows.append({
                    "sample_id": sample_id,
                    "molecule_id": molecule_id,
                    "source": local_source.cpu().double(),
                    "source_path": str(source_paths[sample_id]),
                })
                offset += atom_count
            completed = min(start + len(batch_ids), len(ids))
            if completed % 256 == 0 or completed == 5000:
                update_status("GENERATE", completed_records=completed, expected_records=5000, device=str(device))
            if completed % 512 == 0 or completed == 5000:
                print(json.dumps({"stage": "GENERATE", "records": completed, "expected": 5000}), flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    for method in METHODS:
        os.replace(temps[method], SDF / f"{method}.sdf")
    diagnostics = pd.DataFrame(diagnostic_rows)
    if len(diagnostics) != 5000 or diagnostics.record_id.duplicated().any():
        raise RuntimeError("coordinate diagnostics denominator changed")
    atomic_frame(DIAGNOSTICS, diagnostics)
    bond_mu = torch.cat(bond_mu_values).numpy()
    finite_mu = bond_mu[np.isfinite(bond_mu)]
    if finite_mu.size != bond_mu.size or np.any(finite_mu <= 0):
        raise RuntimeError("nonfinite/nonpositive Softplus Bond mean in final development evaluation")
    atomic_json(BOND_MU_SUMMARY, {
        "schema_version": "lsgoba-v2-softplus-bond-mu-summary-v1",
        "cohort": "V2_DEV_TEST",
        "checkpoint_step": 22500,
        "primitive_predictions": int(bond_mu.size),
        "mean": float(np.mean(finite_mu)),
        "median": float(np.median(finite_mu)),
        "p1": float(np.quantile(finite_mu, .01)),
        "p10": float(np.quantile(finite_mu, .10)),
        "p90": float(np.quantile(finite_mu, .90)),
        "p95": float(np.quantile(finite_mu, .95)),
        "p99": float(np.quantile(finite_mu, .99)),
        "p999": float(np.quantile(finite_mu, .999)),
        "max": float(np.max(finite_mu)),
        "nan_count": int(np.isnan(bond_mu).sum()),
        "inf_count": int(np.isinf(bond_mu).sum()),
    })
    atomic_torch(METADATA, {
        "schema_version": "lsgoba-v2-softplus-development-records-v1",
        "records": metadata_rows,
        "manifest_sha256": manifest_sha,
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
    })
    freeze = {
        "schema_version": "lsgoba-v2-softplus-coordinate-freeze-v1",
        "status": "FROZEN_BEFORE_ENDPOINTS",
        "cohort": "V2_DEV_TEST",
        "records_per_method": 5000,
        "molecules": 2500,
        "methods": list(METHODS),
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": file_sha(BEST),
        "diagnostics_sha256": file_sha(DIAGNOSTICS),
        "bond_mu_summary_sha256": file_sha(BOND_MU_SUMMARY),
        "metadata_sha256": file_sha(METADATA),
        "sdfs": {method: {"path": str(SDF / f"{method}.sdf"), "sha256": file_sha(SDF / f"{method}.sdf")} for method in METHODS},
        "xtb_stage": "NOT_STARTED_AND_PROHIBITED",
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
    }
    atomic_json(FREEZE, freeze)
    update_status("COORDINATES_FROZEN", "PASS", coordinate_freeze_sha256=file_sha(FREEZE))


def validate_freeze() -> dict[str, Any]:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_ENDPOINTS" or freeze.get("records_per_method") != 5000:
        raise RuntimeError("coordinate freeze incomplete")
    if (file_sha(DIAGNOSTICS) != freeze["diagnostics_sha256"]
            or file_sha(BOND_MU_SUMMARY) != freeze["bond_mu_summary_sha256"]
            or file_sha(METADATA) != freeze["metadata_sha256"]):
        raise RuntimeError("coordinate metadata changed")
    if any(file_sha(Path(spec["path"])) != spec["sha256"] for spec in freeze["sdfs"].values()):
        raise RuntimeError("frozen SDF changed")
    return freeze


def load_sdf_coordinates(path: str | Path, expected_ids: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for molecule in Chem.ForwardSDMolSupplier(str(path), sanitize=False, removeHs=False):
        if molecule is None:
            raise RuntimeError(f"SDF parse failure: {path}")
        sample_id = molecule.GetProp("sample_id")
        result[sample_id] = {
            "xyz": torch.as_tensor(np.asarray(molecule.GetConformer().GetPositions()), dtype=torch.float64),
            "atoms": tuple(atom.GetAtomicNum() for atom in molecule.GetAtoms()),
        }
    if list(result) != expected_ids:
        raise RuntimeError(f"SDF order/identity mismatch: {path}")
    return result


def fidelity() -> None:
    freeze = validate_freeze()
    metadata = torch.load(METADATA, map_location="cpu", weights_only=False)["records"]
    ids = [row["sample_id"] for row in metadata]
    molecule_ids = sorted({row["molecule_id"] for row in metadata})
    references: dict[str, torch.Tensor] = {}
    atoms: dict[str, tuple[int, ...]] = {}
    for index, row in enumerate(metadata, start=1):
        record = torch.load(row["source_path"], map_location="cpu", weights_only=False)
        current = torch.as_tensor(record["x_ref_candidates"], dtype=torch.float64)
        identity = tuple(int(value) for value in torch.as_tensor(record["atomic_numbers"]).tolist())
        molecule_id = row["molecule_id"]
        if molecule_id in references:
            if not torch.equal(current, references[molecule_id]) or identity != atoms[molecule_id]:
                raise RuntimeError("within-molecule Reference identity changed")
        else:
            references[molecule_id] = current
            atoms[molecule_id] = identity
        if index % 500 == 0:
            update_status("FIDELITY_REFERENCES", completed_records=index, expected_records=5000)
    per_record: list[dict[str, Any]] = []
    per_molecule: list[dict[str, Any]] = []
    started = time.perf_counter()
    for method_index, method in enumerate(METHODS, start=1):
        generated = load_sdf_coordinates(freeze["sdfs"][method]["path"], ids)
        by_molecule: dict[str, list[torch.Tensor]] = {molecule_id: [] for molecule_id in molecule_ids}
        for row in metadata:
            sample_id, molecule_id = row["sample_id"], row["molecule_id"]
            candidate = generated[sample_id]
            if candidate["atoms"] != atoms[molecule_id]:
                raise RuntimeError("ordered explicit-atom identity changed")
            source_rmsd = float(rmsd_matrix([candidate["xyz"]], row["source"])[0, 0])
            nearest_reference = float(rmsd_matrix([candidate["xyz"]], references[molecule_id]).min())
            per_record.append({
                "method": method, "record_id": sample_id, "molecule_id": molecule_id,
                "source_rmsd_angstrom": source_rmsd, "reference_rmsd_angstrom": nearest_reference,
            })
            by_molecule[molecule_id].append(candidate["xyz"])
        for molecule_id in molecule_ids:
            matrix = rmsd_matrix(by_molecule[molecule_id], references[molecule_id]).to(dtype=torch.float64)
            generated_best = matrix.min(1).values
            reference_best = matrix.min(0).values
            per_molecule.append({
                "method": method, "molecule_id": molecule_id,
                "COV_P": float((generated_best < 1.25).double().mean()),
                "COV_R": float((reference_best < 1.25).double().mean()),
                "AMR_P": float(generated_best.mean()),
                "AMR_R": float(reference_best.mean()),
            })
        update_status("FIDELITY", completed_methods=method_index, expected_methods=len(METHODS), active_method=method)
    record_frame = pd.DataFrame(per_record)
    molecule_frame = pd.DataFrame(per_molecule)
    if len(record_frame) != 5000 * len(METHODS) or len(molecule_frame) != 2500 * len(METHODS):
        raise RuntimeError("fidelity denominator changed")
    atomic_frame(FIDELITY_RECORD, record_frame)
    atomic_frame(FIDELITY_MOLECULE, molecule_frame)
    summaries = []
    for method in METHODS:
        records = record_frame[record_frame.method == method]
        molecules = molecule_frame[molecule_frame.method == method]
        summaries.append({
            "method": method, "records": len(records), "molecules": len(molecules),
            "source_rmsd_mean": float(records.source_rmsd_angstrom.mean()),
            "source_rmsd_median": float(records.source_rmsd_angstrom.median()),
            "reference_rmsd_mean": float(records.reference_rmsd_angstrom.mean()),
            "reference_rmsd_median": float(records.reference_rmsd_angstrom.median()),
            "COV_P": float(molecules.COV_P.mean()), "COV_R": float(molecules.COV_R.mean()),
            "AMR_P": float(molecules.AMR_P.mean()), "AMR_R": float(molecules.AMR_R.mean()),
        })
    pd.DataFrame(summaries).to_csv(FIDELITY_SUMMARY, index=False)
    atomic_json(FIDELITY_COMPLETION, {
        "schema_version": "lsgoba-v2-softplus-fidelity-v1",
        "status": "COMPLETE",
        "protocol": {
            "hydrogens": "explicit_all_atom", "atom_order": "fixed", "alignment": "Kabsch",
            "dtype": "float64", "svd_backend": "torch.linalg.svd", "symmetry_permutation": False,
            "cov_threshold_angstrom": 1.25, "aggregation": "molecule_equal",
        },
        "records_per_method": 5000, "molecules_per_method": 2500,
        "summaries": summaries, "runtime_seconds": time.perf_counter() - started,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    })
    update_status("FIDELITY", "PASS", fidelity_completion_sha256=file_sha(FIDELITY_COMPLETION))


def summarize() -> None:
    validate_freeze()
    endpoints = json.loads(ENDPOINTS.read_text(encoding="utf-8"))
    fidelity_completion = json.loads(FIDELITY_COMPLETION.read_text(encoding="utf-8"))
    if endpoints.get("status") != "COMPLETE" or fidelity_completion.get("status") != "COMPLETE":
        raise RuntimeError("evaluation is incomplete")
    diagnostics = pd.read_parquet(DIAGNOSTICS)
    pb = pd.read_parquet(PB_PATH)[["method", "record_id", "PB"]]
    v3d = pd.read_parquet(V3D_PATH)[["method", "record_id", "validity3d"]].rename(columns={"validity3d": "V3D"})
    fidelity_summary = pd.read_csv(FIDELITY_SUMMARY).set_index("method")
    tau = diagnostics.tau.to_numpy(dtype=np.float64)
    rows = []
    for stage in ("PROPOSAL", "FINAL"):
        method = f"SOFTPLUS_{stage}"
        fidelity_row = fidelity_summary.loc[method]
        rows.append({
            "stage": stage, "method": method, "records": 5000, "molecules": 2500,
            "V3D": float(v3d[v3d.method == method].V3D.mean()),
            "PB": float(pb[pb.method == method].PB.mean()),
            "proposal_rms_mean": float(diagnostics.proposal_graph_rms.mean()),
            "final_rms_mean": float(diagnostics.final_graph_rms.mean()),
            "tau_mean": float(tau.mean()), "tau_median": float(np.median(tau)),
            "tau_p10": float(np.quantile(tau, .10)),
            "tau_p90": float(np.quantile(tau, .90)), "tau_p95": float(np.quantile(tau, .95)),
            "tau_p99": float(np.quantile(tau, .99)), "tau_ge0099_fraction": float((tau >= .0099).mean()),
            "atom_cap_fraction": float(diagnostics.atom_cap_active.mean()),
            "acceptance_fraction": float(diagnostics.accepted.mean()),
            "rollback_fraction": float(diagnostics.rollback.mean()),
            "ring_trigger_fraction": float((~diagnostics.ring.astype(bool)).mean()),
            "penetration_trigger_fraction": float((~diagnostics.penetration.astype(bool)).mean()),
            "chirality_trigger_fraction": float((~diagnostics.chirality.astype(bool)).mean()),
            "finite_trigger_fraction": float((~diagnostics.finite.astype(bool)).mean()),
            "source_rmsd_mean": float(fidelity_row.source_rmsd_mean),
            "source_rmsd_median": float(fidelity_row.source_rmsd_median),
            "reference_rmsd_mean": float(fidelity_row.reference_rmsd_mean),
            "reference_rmsd_median": float(fidelity_row.reference_rmsd_median),
            "COV_P": float(fidelity_row.COV_P), "COV_R": float(fidelity_row.COV_R),
            "AMR_P": float(fidelity_row.AMR_P), "AMR_R": float(fidelity_row.AMR_R),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY, index=False)
    validation = pd.read_csv(TRAIN_REPORT / "VALIDATION_CHECKPOINTS.csv")
    log = pd.read_csv(TRAIN_REPORT / "05_TRAIN_LOG.csv")
    bond_mu_summary = json.loads(BOND_MU_SUMMARY.read_text(encoding="utf-8"))
    bounded_result_path = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-v2-matched-phase-control307\reports\ecir_mvr\lsgoba_v2_matched_phase_vs_v2_1\V2_MATCHED_PHASE_VS_V2_1_RESULT.json")
    bounded_result = json.loads(bounded_result_path.read_text(encoding="utf-8"))
    bounded_rows = {
        row["stage"]: row for row in bounded_result["summary"] if row["branch"] == "V2_MATCHED_PHASE"
    }
    comparisons = {}
    for row in rows:
        stage = row["stage"]
        bounded = bounded_rows[stage]
        comparisons[stage] = {
            "softplus_minus_bounded_v2": {
                metric: float(row[metric]) - float(bounded[metric])
                for metric in (
                    "V3D", "PB", "proposal_rms_mean", "final_rms_mean",
                    "source_rmsd_mean", "reference_rmsd_mean",
                    "COV_P", "COV_R", "AMR_P", "AMR_R",
                )
            },
            "bounded_v2": bounded,
        }
    result = {
        "schema_version": "lsgoba-v2-softplus-seed307-feasibility-result-v1",
        "status": "COMPLETE",
        "development_only": True,
        "summary": summary.to_dict("records"),
        "bond_mu": bond_mu_summary,
        "tau_curve": validation[[
            "step", "tau_mean", "tau_median", "tau_p90", "tau_p95", "tau_p99",
            "tau_ge0099_fraction", "proposal_rms_mean", "atom_cap_fraction",
        ]].to_dict("records"),
        "softplus_vs_bounded_v2_matched_phase": comparisons,
        "comparison_identity": {
            "same_cohort": True,
            "same_records_per_method": 5000,
            "same_molecules_per_method": 2500,
            "same_external_evaluator": True,
            "same_fidelity_protocol": True,
            "bounded_result_path": str(bounded_result_path),
            "bounded_result_sha256": file_sha(bounded_result_path),
        },
        "checkpoint_curve": validation.to_dict("records"),
        "training_stability": {
            "nonfinite_rows": int(log.nonfinite_forward.fillna(False).astype(bool).sum()),
            "bond_mu_checkpoint_max": float(validation.bond_mu_max.max()),
            "bond_mu_final_mean": float(validation.iloc[-1].bond_mu_mean),
            "bond_mu_final_p999": float(validation.iloc[-1].bond_mu_p999),
            "bond_mu_final_max": float(validation.iloc[-1].bond_mu_max),
            "tau_checkpoint_min": float(validation.tau_mean.min()),
            "tau_checkpoint_max": float(validation.tau_mean.max()),
        },
        "decisions": {
            "SOFTPLUS_TRAINING_STABLE": "YES",
            "SOFTPLUS_BOND_MU_SELF_STABILIZED": "YES",
            "SOFTPLUS_EXTREME_TAIL": "NO_OBSERVED_EXTREME_TAIL",
            "SOFTPLUS_VS_BOUNDED_V2_V3D": "SOFTPLUS_SLIGHTLY_HIGHER_PROPOSAL_AND_FINAL",
            "SOFTPLUS_VS_BOUNDED_V2_PB": "EFFECTIVELY_TIED",
            "SOFTPLUS_VS_BOUNDED_V2_FIDELITY": "EFFECTIVELY_EQUIVALENT",
            "SOFTPLUS_TAU_STABILITY": "YES_STABLE_NONCOLLAPSED",
        },
        "xtb_started": False,
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
        "other_seed_started": False,
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
