#!/usr/bin/env python
"""Zero-shot AvgFlow/DiTMC evaluation for the frozen final SIXS-U checkpoints.

This is a thin adapter over the previously validated Softplus-v2 cross-upstream
cohort/reference infrastructure.  It deliberately replaces only the scientific
model/action with the current J1-R1 Full-Joint Unrestricted implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import time
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
from etflow.ecir.j1r1_full_joint_unrestricted import (
    UnrestrictedFullJointModel,
    unrestricted_full_joint_action,
)
from etflow.ecir.learned_geometry import prepare_graph
from scripts.run_mcvr_lsgo import collate_graphs

OLD_ROOT = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-v2-softplus-multiseed")
OLD_DRIVER = OLD_ROOT / "scripts/evaluate_final_softplus_v2_cross_upstream.py"
REPORT = ROOT / "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted"
ASSET = Path(r"E:\3dconformergenerationcode\dataset\sixs_final_cross_upstream_unrestricted")
STATUS = REPORT / "RUN_STATUS.json"
PREFLIGHT = REPORT / "01_PREFLIGHT.json"
CHECKPOINT_MANIFEST = REPORT / "00_FINAL_UNRESTRICTED_CHECKPOINT_MANIFEST.json"
SEEDS = (307, 331, 353)
CHECKPOINTS = {
    307: ROOT / "reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/FINAL_CHECKPOINT.pt",
    331: ROOT / "reports/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed/unrestricted_seed331/FINAL_CHECKPOINT.pt",
    353: ROOT / "reports/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed/unrestricted_seed353/FINAL_CHECKPOINT.pt",
}
EXPECTED_SHA = {
    307: "f63e9d796cc82297f2f2d5fd732c35aa80421ce2f604f2ac80d823a9f825b704",
    331: "b9d655d14c7bc97dc6f54d1dd00e8bf84e40187cdf5d1518ffaaa517b69886af",
    353: "e3be9e2ecf4d633cb2e5f7e3fff4ffd94b38fda13a1412013afd5f4f07450bc6",
}


def load_legacy():
    spec = importlib.util.spec_from_file_location("sixs_cross_upstream_legacy", OLD_DRIVER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = OLD_ROOT  # retained only for frozen reference reconstruction
    module.REPORT = REPORT
    module.ARTIFACT = ASSET
    module.STATUS = STATUS
    module.PREFLIGHT = PREFLIGHT
    module.branch_paths = branch_paths
    module.methods = methods
    return module


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def update_status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    current = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    current.update({
        "schema_version": "sixs-final-cross-upstream-unrestricted-status-v1",
        "status": state,
        "stage": stage,
        "pid": os.getpid(),
        "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "zero_shot": True,
        "model_training_performed": False,
        "model_changed": False,
        "checkpoint_selection_changed": False,
        **extra,
    })
    atomic_json(STATUS, current)


def branch_paths(upstream: str) -> dict[str, Path]:
    report = REPORT / upstream
    artifact = ASSET / upstream
    return {
        "report": report,
        "artifact": artifact,
        "status": report / "STATUS.json",
        "sdf": artifact / "sdf",
        "diagnostics": artifact / "COORDINATE_DIAGNOSTICS.parquet",
        "bond": artifact / "BOND_MU_SUMMARY.csv",
        "freeze": artifact / "COORDINATE_FREEZE.json",
        "pb": artifact / "POSEBUSTERS.parquet",
        "v3d": artifact / "VALIDITY3D.parquet",
        "endpoints": artifact / "ENDPOINT_COMPLETION.json",
        "fidelity_record": artifact / "FIDELITY_PER_RECORD.parquet",
        "fidelity_molecule": artifact / "FIDELITY_PER_MOLECULE.parquet",
        "fidelity_summary": artifact / "FIDELITY_SUMMARY.csv",
        "fidelity_completion": artifact / "FIDELITY_COMPLETION.json",
        "summary": report / "METHOD_SUMMARY.csv",
        "result": report / "RESULT.json",
    }


def methods(upstream: str) -> tuple[str, ...]:
    prefix = upstream.upper()
    return (f"{prefix}_RAW", *(f"{prefix}_SIXS_U_SEED{seed}" for seed in SEEDS))


def checkpoint_identity(seed: int) -> dict[str, Any]:
    path = CHECKPOINTS[seed]
    observed = sha256(path)
    if observed != EXPECTED_SHA[seed]:
        raise RuntimeError(f"seed{seed} checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "seed": seed,
        "step": 17500,
        "TAU_PARAMETERIZATION": "SOFTPLUS_RAW_NO_FINITE_UPPER_BOUND",
        "MOVEMENT_REGULARIZER": "NONE",
        "TAU_MAX": "NONE",
        "ATOM_CAP": "NONE",
        "ROLLBACK_USED": "NO",
        "TRAINING_DEVICE": "cuda",
        "FULL_JOINT_TRAINING": "YES",
        "CHECKPOINT_SELECTED_BY_DEV_OUTCOME": "NO",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"seed{seed} checkpoint identity mismatch: {key}={payload.get(key)!r}")
    probe = UnrestrictedFullJointModel(128, 3)
    probe.load_state_dict(payload["model_state"], strict=True)
    return {
        "seed": seed,
        "path": str(path.resolve()),
        "sha256": observed,
        "step": 17500,
        "formulation": "J1_R1_FULL_JOINT_ADAPTIVE_BA_UNRESTRICTED_MOVEMENT",
        "experiment_id": payload.get("experiment_id", payload.get("EXPERIMENT_ID")),
        "config_sha256": payload.get("config_sha256"),
        "model_state_keys": len(payload["model_state"]),
        "identity": "PASS",
    }


def preflight() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; zero-shot neural inference did not start")
    identities = [checkpoint_identity(seed) for seed in SEEDS]
    legacy = load_legacy()
    cohorts = {}
    for upstream, expected in (("avgflow", 10000), ("ditmc", 10000)):
        raw, raw_sdf, expected_sdf_sha = legacy.raw_spec(upstream)
        frame = pd.read_parquet(raw, columns=["sample_id"])
        if len(frame) != expected or frame.sample_id.nunique() != expected:
            raise RuntimeError(f"{upstream} frozen cohort denominator mismatch")
        if sha256(raw_sdf) != expected_sdf_sha:
            raise RuntimeError(f"{upstream} raw SDF identity mismatch")
        cohorts[upstream] = {
            "records": expected,
            "raw_parquet": str(raw),
            "raw_parquet_sha256": sha256(raw),
            "raw_sdf": str(raw_sdf),
            "raw_sdf_sha256": expected_sdf_sha,
            "identity": "PASS",
        }
    atomic_json(CHECKPOINT_MANIFEST, {
        "schema_version": "sixs-final-unrestricted-checkpoint-manifest-v1",
        "status": "PASS",
        "CHECKPOINT_IDENTITY": "PASS",
        "SEEDS": list(SEEDS),
        "FORMULATION": "J1_R1_FULL_JOINT_ADAPTIVE_BA_UNRESTRICTED_MOVEMENT",
        "checkpoints": identities,
        "zero_shot": True,
    })
    atomic_json(PREFLIGHT, {
        "schema_version": "sixs-final-cross-upstream-unrestricted-preflight-v1",
        "status": "PASS",
        "cuda_available": True,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "checkpoint_manifest": str(CHECKPOINT_MANIFEST),
        "checkpoint_manifest_sha256": sha256(CHECKPOINT_MANIFEST),
        "cohorts": cohorts,
        "zero_shot": True,
        "same_frozen_sixs_unrestricted": True,
        "no_best_seed_selection": True,
        "model_training_performed": False,
    })
    update_status("PREFLIGHT", "PASS", checkpoint_identity="PASS")


def load_models(device: torch.device) -> dict[int, UnrestrictedFullJointModel]:
    result = {}
    for seed in SEEDS:
        checkpoint_identity(seed)
        payload = torch.load(CHECKPOINTS[seed], map_location="cpu", weights_only=False)
        model = UnrestrictedFullJointModel(128, 3)
        model.load_state_dict(payload["model_state"], strict=True)
        model.to(device).eval()
        if not all(parameter.device.type == "cuda" for parameter in model.parameters()):
            raise RuntimeError(f"seed{seed} parameters not on CUDA")
        result[seed] = model
    return result


def smoke() -> None:
    legacy = load_legacy()
    calibration = json.loads(legacy.CALIBRATION.read_text(encoding="utf-8"))
    device = torch.device("cuda:0")
    models = load_models(device)
    checked = {}
    for upstream in ("avgflow", "ditmc"):
        raw_path, _, _ = legacy.raw_spec(upstream)
        raw = pd.read_parquet(raw_path).head(3)
        records = [legacy.record_from_row(upstream, row) for row in raw.itertuples(index=False)]
        graphs_cpu = [prepare_graph(record, calibration) for record, _, _ in records]
        graphs = [graph.to(device) for graph in graphs_cpu]
        collated = collate_graphs(graphs_cpu).to(device)
        source = torch.cat([torch.as_tensor(xyz, dtype=torch.float64, device=device) for _, _, xyz in records])
        for seed, model in models.items():
            with torch.inference_mode():
                prediction = model.belief(collated, detach_sigma_features=False)
                action = unrestricted_full_joint_action(model, source, graphs, prediction)
            if action.proposal.device.type != "cuda" or not bool(torch.isfinite(action.proposal).all()):
                raise RuntimeError(f"smoke failure: {upstream}/seed{seed}")
        checked[upstream] = {"records": 3, "seeds": list(SEEDS), "atom_order_and_finite_coordinates": "PASS"}
    atomic_json(REPORT / "02_ENGINEERING_SMOKE_STATUS.json", {"status": "PASS", "checked": checked, "scientific_outcome_read": False, "model_training_performed": False})
    update_status("SMOKE", "PASS", scientific_outcome_read=False)


def write_molecule(writer: Chem.SDWriter, template: Chem.Mol, xyz: torch.Tensor, row: Any, method: str) -> None:
    mol = Chem.Mol(template)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, point in enumerate(xyz.detach().cpu().double().tolist()):
        conformer.SetAtomPosition(index, point)
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)
    mol.SetProp("_Name", str(row.sample_id))
    mol.SetProp("sample_id", str(row.sample_id))
    mol.SetProp("record_id", str(row.sample_id))
    mol.SetProp("molecule_id", str(row.molecule_id))
    mol.SetProp("method", method)
    writer.write(mol)


def generate(upstream: str) -> None:
    if not PREFLIGHT.is_file() or json.loads(PREFLIGHT.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("preflight must PASS before generation")
    paths = branch_paths(upstream)
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["sdf"].mkdir(parents=True, exist_ok=True)
    if paths["freeze"].is_file():
        frozen = json.loads(paths["freeze"].read_text(encoding="utf-8"))
        if frozen.get("status") == "FROZEN_BEFORE_ENDPOINTS" and all(Path(v["path"]).is_file() and sha256(v["path"]) == v["sha256"] for v in frozen["sdfs"].values()):
            update_status(f"{upstream.upper()}_COORDINATES_REUSED", "PASS")
            return
    legacy = load_legacy()
    raw_path, raw_sdf, _ = legacy.raw_spec(upstream)
    raw = pd.read_parquet(raw_path)
    raw = raw.sort_values("record_index", kind="stable") if upstream == "avgflow" else raw.sort_values(["repeat_index", "input_index"], kind="stable")
    raw = raw.reset_index(drop=True)
    if len(raw) != 10000 or raw.sample_id.nunique() != 10000:
        raise RuntimeError(f"{upstream} raw denominator changed")
    calibration = json.loads(legacy.CALIBRATION.read_text(encoding="utf-8"))
    device = torch.device("cuda:0")
    models = load_models(device)
    prefix = upstream.upper()
    output_methods = methods(upstream)[1:]
    temporary = {method: paths["sdf"] / f".{method}.{os.getpid()}.tmp.sdf" for method in output_methods}
    writers = {method: Chem.SDWriter(str(path)) for method, path in temporary.items()}
    diagnostics: list[dict[str, Any]] = []
    graph_cache: dict[str, Any] = {}
    update_status(f"{prefix}_GENERATE", completed_records=0, expected_records=10000)
    try:
        for start in range(0, len(raw), 64):
            batch_rows = list(raw.iloc[start:start + 64].itertuples(index=False))
            templates, cpu_graphs, source_parts = [], [], []
            for row in batch_rows:
                record, molecule, xyz = legacy.record_from_row(upstream, row)
                key = str(row.molecule_id)
                if key not in graph_cache:
                    graph_cache[key] = prepare_graph(record, calibration)
                templates.append(molecule)
                cpu_graphs.append(graph_cache[key])
                source_parts.append(torch.as_tensor(xyz, dtype=torch.float64, device=device))
            graphs = [graph.to(device) for graph in cpu_graphs]
            collated = collate_graphs(cpu_graphs).to(device)
            source = torch.cat(source_parts)
            outputs = {}
            for seed, model in models.items():
                with torch.inference_mode():
                    prediction = model.belief(collated, detach_sigma_features=False)
                    action = unrestricted_full_joint_action(model, source, graphs, prediction)
                if not bool(torch.isfinite(action.proposal).all()):
                    raise RuntimeError(f"nonfinite unrestricted proposal: {upstream}/seed{seed}")
                outputs[seed] = action
            offset = 0
            for local, (row, template, graph) in enumerate(zip(batch_rows, templates, graphs, strict=True)):
                count = int(graph.atom_categorical.size(0))
                source_local = source[offset:offset + count]
                for seed, action in outputs.items():
                    proposal = action.proposal[offset:offset + count]
                    delta = proposal - source_local
                    method = f"{prefix}_SIXS_U_SEED{seed}"
                    write_molecule(writers[method], template, proposal, row, method)
                    diagnostics.append({
                        "record_id": str(row.sample_id), "molecule_id": str(row.molecule_id), "seed": seed,
                        "tau": float(action.tau[local]),
                        "source_rmsd_raw": float(delta.square().sum(-1).mean().sqrt()),
                        "max_atom_displacement": float(torch.linalg.vector_norm(delta, dim=-1).max()),
                        "finite": bool(torch.isfinite(proposal).all()),
                        "w_B": float(action.family_weights[local, 0]), "w_A": float(action.family_weights[local, 1]),
                    })
                offset += count
            completed = min(start + len(batch_rows), len(raw))
            if completed % 256 == 0 or completed == len(raw):
                update_status(f"{prefix}_GENERATE", completed_records=completed, expected_records=10000)
                print(json.dumps({"stage": f"{prefix}_GENERATE", "records": completed}), flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    raw_target = paths["sdf"] / f"{prefix}_RAW.sdf"
    shutil.copy2(raw_sdf, raw_target)
    for method, temp in temporary.items():
        os.replace(temp, paths["sdf"] / f"{method}.sdf")
    frame = pd.DataFrame(diagnostics)
    if len(frame) != 30000 or frame.duplicated(["seed", "record_id"]).any():
        raise RuntimeError("diagnostic denominator mismatch")
    atomic_frame(paths["diagnostics"], frame)
    atomic_frame(
        paths["artifact"] / "SOURCE_RECORDS.parquet",
        raw[["sample_id", "molecule_id"]].rename(columns={"sample_id": "record_id"}),
    )
    sdfs = {method: {"path": str(paths["sdf"] / f"{method}.sdf"), "sha256": sha256(paths["sdf"] / f"{method}.sdf")} for method in methods(upstream)}
    atomic_json(paths["freeze"], {
        "schema_version": "sixs-final-unrestricted-cross-upstream-coordinate-freeze-v1",
        "status": "FROZEN_BEFORE_ENDPOINTS", "upstream": upstream,
        "records_per_method": 10000, "methods": list(methods(upstream)),
        "checkpoints": {str(seed): EXPECTED_SHA[seed] for seed in SEEDS},
        "sdfs": sdfs, "diagnostics_sha256": sha256(paths["diagnostics"]),
        "zero_shot": True, "reference_used_for_inference": False,
        "movement_regularizer": "NONE", "tau_upper_bound": "NONE", "atom_cap": "NONE", "rollback": "NONE",
    })
    update_status(f"{prefix}_COORDINATES_FROZEN", "PASS")


def fidelity(upstream: str) -> None:
    from scripts.evaluate_sixs_final_cross_upstream_reference_rmsd import evaluate

    evaluate(upstream)
    update_status(f"{upstream.upper()}_FIDELITY", "PASS")


def summarize(upstream: str) -> None:
    paths = branch_paths(upstream)
    pb = pd.read_parquet(paths["pb"])
    v3d = pd.read_parquet(paths["v3d"])
    pb_method_column = "method" if "method" in pb.columns else "arm"
    v3d_method_column = "method" if "method" in v3d.columns else "arm"
    fidelity_rows = pd.read_parquet(paths["fidelity_record"])
    diagnostics = pd.read_parquet(paths["diagnostics"])
    xtb_dir = paths["report"] / "xtb_singlepoint"
    rows = []
    for method in methods(upstream):
        is_raw = method.endswith("_RAW")
        seed = None if is_raw else int(method.rsplit("SEED", 1)[1])
        fr = fidelity_rows[fidelity_rows.method == method]
        if is_raw:
            source_summary = {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0}
        else:
            source_values = diagnostics.loc[diagnostics.seed == seed, "source_rmsd_raw"]
            if len(source_values) != 10000:
                raise RuntimeError(f"{upstream}/seed{seed} source-RMSD denominator changed")
            source_summary = {
                "mean": float(source_values.mean()),
                "median": float(source_values.median()),
                "p95": float(source_values.quantile(.95)),
                "p99": float(source_values.quantile(.99)),
            }
        row = {
            "upstream": upstream, "method": method, "seed": seed,
            "records": 10000,
            "reference_records": int(len(fr)),
            "V3D": float(v3d[v3d[v3d_method_column] == method].validity3d.mean()),
            "PB": float(pb[pb[pb_method_column] == method].PB.mean()),
            "source_rmsd_mean": source_summary["mean"],
            "source_rmsd_median": source_summary["median"],
            "source_rmsd_p95": source_summary["p95"],
            "source_rmsd_p99": source_summary["p99"],
            "reference_rmsd": float(fr.reference_rmsd_angstrom.mean()),
        }
        if seed is not None:
            part = diagnostics[diagnostics.seed == seed]
            tau = part.tau.to_numpy(float)
            row.update({
                "finite_coordinate_rate": float(part.finite.mean()),
                "tau_median": float(np.median(tau)), "tau_p95": float(np.quantile(tau, .95)),
                "tau_p99": float(np.quantile(tau, .99)), "tau_max": float(np.max(tau)),
            })
            delta_path = xtb_dir / f"{method.upper()}_DELTA_VS_SOURCE.csv"
            if delta_path.is_file():
                delta_frame = pd.read_csv(delta_path)
                values = delta_frame.loc[delta_frame.matched_success.astype(bool), "delta_e_kcal_mol"].to_numpy(float)
                trim = np.sort(values)[int(.05 * len(values)):len(values) - int(.05 * len(values))]
                row.update({
                    "xtb_median_delta_e": float(np.median(values)), "xtb_trimmed_mean_delta_e": float(np.mean(trim)),
                    "xtb_lower_fraction": float(np.mean(values < 0)),
                    "xtb_p90": float(np.quantile(values, .90)), "xtb_p95": float(np.quantile(values, .95)),
                    "xtb_p99": float(np.quantile(values, .99)), "xtb_gt25": int(np.sum(values > 25)),
                    "xtb_gt50": int(np.sum(values > 50)), "xtb_gt100": int(np.sum(values > 100)),
                })
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(paths["summary"], index=False)
    raw = table[table.seed.isna()].iloc[0]
    candidates = table[table.seed.notna()].sort_values("seed")
    # JSON has no NaN representation.  Fields such as raw tau and raw delta-E
    # are scientifically not applicable, so serialize them as JSON null.
    raw_json = json.loads(raw.to_json())
    candidates_json = json.loads(candidates.to_json(orient="records"))
    result = {
        "schema_version": "sixs-final-unrestricted-cross-upstream-result-v1",
        "status": "PASS", "upstream": upstream, "zero_shot": True,
        "raw": raw_json, "per_seed": candidates_json,
        "mean_sd": {column: {"mean": float(candidates[column].mean()), "sd": float(candidates[column].std(ddof=1))} for column in candidates.select_dtypes(include=[np.number]).columns if column not in {"seed", "records"}},
        "delta_v3d": float(candidates.V3D.mean() - raw.V3D),
        "delta_pb": float(candidates.PB.mean() - raw.PB),
        "no_best_seed_selection": True,
        "ensemble_cov_amr": "NOT_APPLICABLE" if upstream == "ditmc" else "REPORTED_IN_FIDELITY_ARTIFACT",
    }
    atomic_json(paths["result"], result)
    update_status(f"{upstream.upper()}_COMPLETE", "PASS")


def combined() -> None:
    payloads = {name: json.loads(branch_paths(name)["result"].read_text(encoding="utf-8")) for name in ("avgflow", "ditmc")}
    rows = []
    for upstream, value in payloads.items():
        candidates = pd.DataFrame(value["per_seed"])
        raw = value["raw"]
        rows.append({
            "Upstream": upstream, "Raw_V3D": raw["V3D"],
            **{f"SIXS_U_V3D_seed{int(row.seed)}": row.V3D for row in candidates.itertuples()},
            "SIXS_U_V3D_mean": candidates.V3D.mean(), "SIXS_U_V3D_sd": candidates.V3D.std(ddof=1),
            "Delta_V3D": candidates.V3D.mean() - raw["V3D"], "Raw_PB": raw["PB"],
            "SIXS_U_PB_mean": candidates.PB.mean(), "Delta_PB": candidates.PB.mean() - raw["PB"],
            "Source_RMSD_mean": candidates.source_rmsd_mean.mean(), "Reference_RMSD_Raw": raw["reference_rmsd"],
            "Reference_RMSD_SIXS": candidates.reference_rmsd.mean(),
            "xTB_median_DeltaE": candidates.xtb_median_delta_e.mean(),
            "xTB_trimmed_DeltaE": candidates.xtb_trimmed_mean_delta_e.mean(),
            "xTB_lower_fraction": candidates.xtb_lower_fraction.mean(),
            "tau_median": candidates.tau_median.mean(), "tau_p99": candidates.tau_p99.mean(), "tau_max": candidates.tau_max.max(),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT / "CROSS_UPSTREAM_SUMMARY.csv", index=False)
    transfer = []
    for row in frame.itertuples():
        transfer.append("SUPPORTED" if row.Delta_V3D >= 0 and row.Delta_PB >= 0 and (row.Delta_V3D > 0 or row.Delta_PB > 0) else "NOT_SUPPORTED" if row.Delta_V3D < 0 and row.Delta_PB < 0 else "MIXED")
    classification = "TRANSFER_SUPPORTED" if all(v == "SUPPORTED" for v in transfer) else "TRANSFER_NOT_SUPPORTED" if all(v == "NOT_SUPPORTED" for v in transfer) else "TRANSFER_MIXED"
    # Keep finalization self-contained in the frozen CUDA environment.  Pandas'
    # ``to_markdown`` has an optional tabulate dependency that is deliberately
    # not part of the scientific environment.
    columns = list(frame.columns)
    markdown = "| " + " | ".join(columns) + " |\n"
    markdown += "| " + " | ".join("---" for _ in columns) + " |\n"
    for values in frame.itertuples(index=False, name=None):
        markdown += "| " + " | ".join(str(value) for value in values) + " |\n"
    (REPORT / "CROSS_UPSTREAM_CONCLUSION.md").write_text("# Cross-upstream conclusion\n\n" + markdown + f"\n`CROSS_UPSTREAM_GENERALIZATION = {classification}`\n", encoding="utf-8")
    atomic_json(REPORT / "FINAL_STATUS.json", {"status": "PASS", "FINAL_PRIMARY_METHOD": "UNRESTRICTED", "CROSS_UPSTREAM_GENERALIZATION": classification, "branches": {k: str(branch_paths(k)["result"]) for k in payloads}, "model_training_performed": False})
    update_status("COMPLETE", "PASS", cross_upstream_generalization=classification)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "smoke", "generate", "fidelity", "summarize", "combined"))
    parser.add_argument("--upstream", choices=("avgflow", "ditmc"))
    args = parser.parse_args()
    if args.phase in {"generate", "fidelity", "summarize"} and not args.upstream:
        parser.error("--upstream is required")
    try:
        if args.phase == "preflight": preflight()
        elif args.phase == "smoke": smoke()
        elif args.phase == "generate": generate(args.upstream)
        elif args.phase == "fidelity": fidelity(args.upstream)
        elif args.phase == "summarize": summarize(args.upstream)
        else: combined()
    except Exception as error:
        update_status(f"{(args.upstream or 'combined').upper()}_{args.phase.upper()}", "FAIL", error_type=type(error).__name__, error=str(error))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
