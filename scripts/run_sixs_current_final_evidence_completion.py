#!/usr/bin/env python3
"""Complete frozen step-17500 matched DEV evidence without model changes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
import scripts.run_sixs_j1r1_full_joint_adaptive_ba_movement as full
import scripts.run_sixs_j1r1_full_joint_xtb_dev as xtb_sp
from etflow.commons.kabsch_utils import kabsch_rmsd
from etflow.ecir.external_refinement_baselines import (
    derive_total_charge,
    derive_unpaired_electrons,
    mol_from_frozen_record,
    refine_with_gfn2_xtb,
    refine_with_mmff94s,
)

RUN = "sixs_reference_xtb_and_unrestricted_movement_seed307"
BASE = ROOT / "reports/ecir_mvr" / RUN
OUT = BASE / "01_CURRENT_FINAL_EVIDENCE"
CACHE = ROOT / "artifacts/ecir_mvr" / RUN / "01_CURRENT_FINAL_EVIDENCE"
STATUS = OUT / "PHASE_I_STATUS.json"
CURRENT_REPORT = ROOT / "reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307"
CURRENT_ART = ROOT / "artifacts/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/dev_evaluation"
CURRENT_XTB = CURRENT_REPORT / "xtb_single_point_dev"
CONFIG = ROOT / "configs/sixs_j1r1_full_joint_adaptive_ba_movement.json"
EXT_CONFIG = ROOT / "configs/ecir_external_refinement_baselines.json"
METHODS = ("SOURCE", "SIXS_FULL_JOINT_STEP17500", "MMFF94S_OPTIMIZED", "GFN2_XTB_OPTIMIZED")
HARTREE_TO_KCAL = 627.509474


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if path.suffix == ".parquet":
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    atomic_json(STATUS, {
        "schema_version": "sixs-current-final-evidence-phase-i-v1",
        "status": state,
        "stage": stage,
        "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "worker_pid": os.getpid(),
        "formal_read": False,
        "large_holdout_read": False,
        "current_final_model_modified": False,
        **extra,
    })


def markdown(frame: pd.DataFrame) -> str:
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.10g}")
            else:
                cells.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def config_and_inputs() -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    ext = json.loads(EXT_CONFIG.read_text(encoding="utf-8"))
    if sha256(config["data"]["source_payload"]) != config["data"]["source_payload_sha256"]:
        raise RuntimeError("source payload hash changed")
    if sha256(config["data"]["prepared_payload"]) != config["data"]["prepared_sha256"]:
        raise RuntimeError("prepared payload hash changed")
    manifest = json.loads((ROOT / config["data"]["dev_manifest"]).read_text(encoding="utf-8"))
    ids = [str(sample) for row in manifest["rows"] for sample in row["sample_ids"]]
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise RuntimeError("DEV identity changed")
    prepared = torch.load(config["data"]["prepared_payload"], map_location="cpu", weights_only=False)
    source_payload = torch.load(config["data"]["source_payload"], map_location="cpu", weights_only=False)
    items = {str(row["molecule_id"]): row for row in prepared["val"]}
    sources = {str(row["sample_id"]): row for row in source_payload["val"]}
    if any(record_id not in sources for record_id in ids):
        raise RuntimeError("DEV source binding incomplete")
    return config, ext, ids, items, sources, manifest


def load_current_sdf(ids: list[str]) -> dict[str, tuple[tuple[int, ...], torch.Tensor]]:
    path = CURRENT_ART / "PROPOSAL.sdf"
    mols = [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False) if mol is not None]
    if len(mols) != len(ids):
        raise RuntimeError("current SDF denominator changed")
    result = {}
    for expected, mol in zip(ids, mols, strict=True):
        observed = mol.GetProp("_Name")
        if observed != expected:
            raise RuntimeError("current SDF ordering changed")
        atoms = tuple(atom.GetAtomicNum() for atom in mol.GetAtoms())
        xyz = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
        result[expected] = (atoms, xyz)
    return result


def source_record(config: dict[str, Any], metadata: Any) -> dict[str, Any]:
    path = Path(config["data"]["val_cache"]) / Path(str(metadata.source_path)).name
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != str(metadata.source_file_sha256):
        raise RuntimeError("DEV source cache hash changed")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


def audit_correspondence() -> None:
    if (OUT / "CORRESPONDENCE_AUDIT.json").is_file() and (CACHE / "SOURCE.sdf").is_file():
        return
    status("REFERENCE_RMSD_CORRESPONDENCE_AUDIT")
    config, ext, ids, items, sources, _ = config_and_inputs()
    current = load_current_sdf(ids)
    val = pd.read_parquet(config["data"]["val_manifest"])
    meta = {str(row.sample_id): row for row in val.itertuples(index=False)}
    source_sdf = CACHE / "SOURCE.sdf"
    source_sdf.parent.mkdir(parents=True, exist_ok=True)
    tmp = source_sdf.with_name(source_sdf.name + f".tmp.{os.getpid()}")
    writer = Chem.SDWriter(str(tmp))
    rows = []
    try:
        for index, record_id in enumerate(ids):
            source_row = sources[record_id]
            item = items[str(source_row["molecule_id"])]
            record = source_record(config, meta[record_id])
            source = torch.as_tensor(source_row["source"], dtype=torch.float64)
            references = torch.as_tensor(item["references"], dtype=torch.float64)
            atomic_numbers = tuple(int(v) for v in torch.as_tensor(record["atomic_numbers"]).tolist())
            current_atoms, current_xyz = current[record_id]
            passed = (
                source.ndim == 2 and source.shape[1] == 3 and references.ndim == 3
                and references.shape[1:] == source.shape and current_xyz.shape == source.shape
                and current_atoms == atomic_numbers and len(atomic_numbers) == source.shape[0]
                and bool(torch.isfinite(source).all()) and bool(torch.isfinite(references).all())
                and bool(torch.isfinite(current_xyz).all())
            )
            if not passed:
                raise RuntimeError(f"atom/reference correspondence failed: {record_id}")
            full.frozen.write_molecule(writer, record, source, record_id, "SOURCE")
            rows.append({"record_id": record_id, "molecule_id": str(source_row["molecule_id"]), "atom_count": len(atomic_numbers), "atomic_numbers_sha256": hashlib.sha256(bytes(atomic_numbers)).hexdigest(), "status": "PASS"})
            if (index + 1) % 500 == 0:
                status("REFERENCE_RMSD_CORRESPONDENCE_AUDIT", completed=index + 1, total=5000)
    finally:
        writer.close()
    os.replace(tmp, source_sdf)
    atomic_frame(CACHE / "SOURCE_PER_RECORD.parquet", pd.DataFrame(rows))
    audit = {
        "schema_version": "sixs-reference-rmsd-correspondence-audit-v1",
        "status": "PASS",
        "records": 5000,
        "molecules": 2500,
        "same_molecule_identity": True,
        "same_atom_count": True,
        "same_atomic_numbers": True,
        "same_atom_order": True,
        "hydrogens": "explicit_all_atom",
        "alignment": "fixed-order Kabsch proper rotation",
        "reference_matching": "nearest over frozen Reference ensemble",
        "dtype": "float64",
        "symmetry_permutation": False,
        "symmetry_note": "Inherited frozen project fidelity protocol; fixed atom order is hash/audit protected, no unregistered permutation search.",
        "source_sdf_sha256": sha256(source_sdf),
        "rdkit_version": rdBase.rdkitVersion,
        "mmff_version": f"RDKit-{rdBase.rdkitVersion}/MMFF94s",
        "xtb_version": ext["gfn2_xtb"]["xtb_version"],
        "gfn_level": 2,
        "formal_read": False,
        "large_holdout_read": False,
    }
    atomic_json(OUT / "CORRESPONDENCE_AUDIT.json", audit)
    atomic_text(OUT / "01_REFERENCE_RMSD_INTEGRITY_AUDIT.md", "# Reference RMSD integrity audit\n\n" + markdown(pd.DataFrame([audit])) + "\n\nREFERENCE_RMSD_CORRESPONDENCE_AUDIT = PASS")


def optimization_chunks(method: str) -> Path:
    return CACHE / "optimization" / method / "chunks"


def optimize_method(method: str) -> None:
    complete = CACHE / "optimization" / method / "COMPLETE.json"
    if complete.is_file():
        return
    config, ext, ids, items, sources, _ = config_and_inputs()
    del items
    val = pd.read_parquet(config["data"]["val_manifest"])
    meta = {str(row.sample_id): row for row in val.itertuples(index=False)}
    method_config = ext["mmff94s"] if method == "MMFF94S" else ext["gfn2_xtb"]
    chunks = optimization_chunks(method)
    chunks.mkdir(parents=True, exist_ok=True)
    chunk_size = 25
    workers = 1 if method == "MMFF94S" else int(method_config["worker_count"])
    status(f"{method}_OPTIMIZATION", completed=0, total=5000, workers=workers)

    def run_one(position: int, record_id: str) -> dict[str, Any]:
        source_row = sources[record_id]
        record = full.frozen.adapt_formal_cache_record(source_record(config, meta[record_id]))
        source = torch.as_tensor(source_row["source"], dtype=torch.float32)
        if method == "MMFF94S":
            result = refine_with_mmff94s(record, source, method_config)
        else:
            work = CACHE / "optimization" / method / "work" / f"record_{position:05d}_{hashlib.sha256(record_id.encode()).hexdigest()[:12]}"
            result = refine_with_gfn2_xtb(record, source, method_config, work)
        return {
            "record_index": position,
            "record_id": record_id,
            "molecule_id": str(source_row["molecule_id"]),
            "coordinates": result.refined_coordinates.detach().cpu().float(),
            "success": bool(result.success),
            "converged": bool(result.converged),
            "fallback_to_source": bool(result.fallback_to_source),
            "failure_reason": result.failure_reason,
            "runtime_seconds": float(result.runtime_seconds),
            "method_version": result.method_version,
            "initial_native_energy": result.initial_native_energy,
            "final_native_energy": result.final_native_energy,
            "native_energy_delta": result.native_energy_delta,
            "iteration_count": result.iteration_count,
            "cycle_count": result.cycle_count,
            "atom_order_verified": bool(result.atom_order_verified),
            "topology_verified": bool(result.topology_verified),
            "chirality_verified": bool(result.chirality_verified),
        }

    for start in range(0, 5000, chunk_size):
        end = min(start + chunk_size, 5000)
        path = chunks / f"chunk_{start:05d}_{end:05d}.pt"
        if path.is_file():
            continue
        entries = [(i, ids[i]) for i in range(start, end)]
        if workers == 1:
            rows = [run_one(*entry) for entry in entries]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(lambda entry: run_one(*entry), entries))
        rows.sort(key=lambda row: row["record_index"])
        atomic_torch(path, rows)
        status(f"{method}_OPTIMIZATION", completed=end, total=5000, workers=workers)
        print(json.dumps({"stage": f"{method}_OPTIMIZATION", "completed": end, "total": 5000}), flush=True)

    all_rows = []
    for start in range(0, 5000, chunk_size):
        end = min(start + chunk_size, 5000)
        all_rows.extend(torch.load(chunks / f"chunk_{start:05d}_{end:05d}.pt", map_location="cpu", weights_only=False))
    if [row["record_id"] for row in all_rows] != ids:
        raise RuntimeError(f"{method} optimization identity changed")
    summary_rows = [{key: value for key, value in row.items() if key != "coordinates"} for row in all_rows]
    atomic_frame(CACHE / "optimization" / method / "PER_RECORD.parquet", pd.DataFrame(summary_rows))
    atomic_torch(CACHE / "optimization" / method / "COORDINATES.pt", {"record_ids": ids, "coordinates": [row["coordinates"] for row in all_rows]})
    write_method_sdf(method, all_rows, config, meta)
    atomic_json(complete, {
        "status": "COMPLETE",
        "method": method,
        "records": 5000,
        "successes": sum(row["success"] for row in all_rows),
        "fallbacks": sum(row["fallback_to_source"] for row in all_rows),
        "method_config": method_config,
        "per_record_sha256": sha256(CACHE / "optimization" / method / "PER_RECORD.parquet"),
        "coordinates_sha256": sha256(CACHE / "optimization" / method / "COORDINATES.pt"),
        "sdf_sha256": sha256(CACHE / "optimization" / method / "PROPOSAL.sdf"),
        "formal_read": False,
        "large_holdout_read": False,
    })


def write_method_sdf(method: str, rows: list[dict[str, Any]], config: dict[str, Any], meta: dict[str, Any]) -> None:
    path = CACHE / "optimization" / method / "PROPOSAL.sdf"
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    writer = Chem.SDWriter(str(tmp))
    try:
        for row in rows:
            record = source_record(config, meta[row["record_id"]])
            full.frozen.write_molecule(writer, record, row["coordinates"], row["record_id"], method)
    finally:
        writer.close()
    os.replace(tmp, path)


def evaluate_validity() -> None:
    config, ext, ids, items, sources, manifest = config_and_inputs()
    del config, ext, items, sources, manifest
    paths = {
        "SOURCE": CACHE / "SOURCE.sdf",
        "MMFF94S_OPTIMIZED": CACHE / "optimization/MMFF94S/PROPOSAL.sdf",
        "GFN2_XTB_OPTIMIZED": CACHE / "optimization/GFN2_XTB/PROPOSAL.sdf",
    }
    source_records = pd.read_parquet(CACHE / "SOURCE_PER_RECORD.parquet")
    for method, sdf in paths.items():
        out = CACHE / "validity" / method
        out.mkdir(parents=True, exist_ok=True)
        per = out / "PER_RECORD.parquet"
        pb = out / "POSEBUSTERS.parquet"
        v3d = out / "VALIDITY3D.parquet"
        if not per.is_file():
            atomic_frame(per, source_records[["record_id", "molecule_id"]].assign(method=method))
        if not pb.is_file() or not v3d.is_file():
            status(f"{method}_VALIDITY")
            full.frozen.run_external_evaluators(method, {"sdf": sdf, "per_record": per, "pb": pb, "v3d": v3d}, ids)


def optimized_coordinates(method: str) -> dict[str, torch.Tensor]:
    payload = torch.load(CACHE / "optimization" / method / "COORDINATES.pt", map_location="cpu", weights_only=False)
    return {str(i): torch.as_tensor(x, dtype=torch.float64) for i, x in zip(payload["record_ids"], payload["coordinates"], strict=True)}


def rmsd_evidence() -> pd.DataFrame:
    path = OUT / "02_MATCHED_SOURCE_REFERENCE_RMSD.csv"
    if path.is_file():
        return pd.read_csv(path)
    status("MATCHED_SOURCE_REFERENCE_RMSD")
    config, ext, ids, items, sources, manifest = config_and_inputs()
    del ext, manifest
    current = {key: value[1] for key, value in load_current_sdf(ids).items()}
    mmff = optimized_coordinates("MMFF94S")
    xtb = optimized_coordinates("GFN2_XTB")
    method_coords = {
        "SOURCE": {record_id: torch.as_tensor(sources[record_id]["source"], dtype=torch.float64) for record_id in ids},
        "SIXS_FULL_JOINT_STEP17500": current,
        "MMFF94S_OPTIMIZED": mmff,
        "GFN2_XTB_OPTIMIZED": xtb,
    }
    rows = []
    for position, record_id in enumerate(ids):
        source_row = sources[record_id]
        references = torch.as_tensor(items[str(source_row["molecule_id"])]["references"], dtype=torch.float64)
        source = method_coords["SOURCE"][record_id]
        source_ref = min(float(kabsch_rmsd(source, ref)) for ref in references)
        for method in METHODS:
            xyz = method_coords[method][record_id]
            src_rmsd = float(kabsch_rmsd(xyz, source))
            ref_rmsd = min(float(kabsch_rmsd(xyz, ref)) for ref in references)
            rows.append({
                "record_index": position,
                "record_id": record_id,
                "molecule_id": str(source_row["molecule_id"]),
                "method": method,
                "source_rmsd": src_rmsd,
                "reference_rmsd": ref_rmsd,
                "source_reference_rmsd": source_ref,
                "delta_reference_rmsd_vs_source": ref_rmsd - source_ref,
            })
        if (position + 1) % 500 == 0:
            status("MATCHED_SOURCE_REFERENCE_RMSD", completed=position + 1, total=5000)
    frame = pd.DataFrame(rows)
    atomic_frame(path, frame)
    return frame


def make_xtb_task(record_id: str, molecule_id: str, position: int, xyz: torch.Tensor, record: dict[str, Any]) -> dict[str, Any]:
    record = full.frozen.adapt_formal_cache_record(record)
    mol = mol_from_frozen_record(record, xyz)
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    value = np.ascontiguousarray(xyz.double().numpy(), dtype=np.float64)
    coordinate_hash = hashlib.sha256(f"float64|{value.shape}|".encode() + value.tobytes()).hexdigest()
    identity = {
        "elements": elements,
        "charge": derive_total_charge(mol),
        "uhf": derive_unpaired_electrons(mol),
        "coordinate_sha256": coordinate_hash,
        "settings": {key: xtb_sp.SETTINGS[key] for key in xtb_sp.SETTINGS if key != "workers"},
    }
    return {
        "method": "MMFF94S_OPTIMIZED",
        "record_index": position,
        "record_id": record_id,
        "molecule_id": molecule_id,
        "coordinates": value,
        "identity": identity,
        "identity_sha256": xtb_sp.canonical_sha(identity),
    }


def mmff_xtb_singlepoints() -> None:
    output = CACHE / "xtb_singlepoint_mmff"
    result_path = output / "MMFF94S_OPTIMIZED_XTB.csv"
    if result_path.is_file():
        result = pd.read_csv(result_path)
        if len(result) == 5000 and result.record_id.nunique() == 5000:
            return
    config, ext, ids, items, sources, manifest = config_and_inputs()
    del ext, items, manifest
    mmff = optimized_coordinates("MMFF94S")
    val = pd.read_parquet(config["data"]["val_manifest"])
    meta = {str(row.sample_id): row for row in val.itertuples(index=False)}
    # The evaluator uses a 64-character identity as the work-directory name.
    # Keep its scratch/cache root short enough for Windows MAX_PATH while the
    # frozen scientific CSV remains in the run-specific evidence directory.
    xtb_sp.OUT = ROOT / ".phase1_xtb_mmff_sp"
    xtb_sp.OUT.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for position, record_id in enumerate(ids):
        record = source_record(config, meta[record_id])
        tasks.append(make_xtb_task(record_id, str(sources[record_id]["molecule_id"]), position, mmff[record_id], record))
    rows = []
    status("MMFF94S_XTB_SINGLEPOINT", completed=0, total=5000, workers=xtb_sp.SETTINGS["workers"])
    with ThreadPoolExecutor(max_workers=xtb_sp.SETTINGS["workers"]) as pool:
        futures = [pool.submit(xtb_sp.execute, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if count % 100 == 0 or count == 5000:
                status("MMFF94S_XTB_SINGLEPOINT", completed=count, total=5000, workers=xtb_sp.SETTINGS["workers"])
                print(json.dumps({"stage": "MMFF94S_XTB_SINGLEPOINT", "completed": count, "total": 5000}), flush=True)
    frame = pd.DataFrame(rows).sort_values("record_index", kind="stable").reset_index(drop=True)
    atomic_frame(result_path, frame)


def distribution(values: np.ndarray) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std(ddof=1)),
        "p05": float(np.quantile(a, .05)), "p25": float(np.quantile(a, .25)),
        "p75": float(np.quantile(a, .75)), "p90": float(np.quantile(a, .90)),
        "p95": float(np.quantile(a, .95)), "p99": float(np.quantile(a, .99)), "max": float(a.max()),
    }


def trimmed_mean(values: np.ndarray, fraction: float = .05) -> float:
    a = np.sort(np.asarray(values, dtype=np.float64)); n = int(math.floor(len(a) * fraction))
    return float(a[n:len(a)-n].mean()) if n else float(a.mean())


def cluster_bootstrap(frame: pd.DataFrame, value: str, seed: int) -> dict[str, float]:
    grouped = [group[value].to_numpy(dtype=np.float64) for _, group in frame.groupby("molecule_id", sort=False)]
    rng = np.random.default_rng(seed); means = np.empty(10000); medians = np.empty(10000)
    for draw in range(10000):
        selected = rng.integers(0, len(grouped), len(grouped))
        values = np.concatenate([grouped[i] for i in selected])
        means[draw] = values.mean(); medians[draw] = np.median(values)
    return {
        "mean_ci95_low": float(np.quantile(means, .025)), "mean_ci95_high": float(np.quantile(means, .975)),
        "median_ci95_low": float(np.quantile(medians, .025)), "median_ci95_high": float(np.quantile(medians, .975)),
    }


def summarize() -> None:
    status("PHASE_I_SUMMARIZATION")
    rmsd = rmsd_evidence()
    summary_rows = []
    for index, method in enumerate(METHODS):
        part = rmsd[rmsd.method == method]
        summary_rows.append({
            "method": method,
            **{f"source_rmsd_{k}": v for k, v in distribution(part.source_rmsd.to_numpy()).items()},
            **{f"reference_rmsd_{k}": v for k, v in distribution(part.reference_rmsd.to_numpy()).items()},
            "delta_reference_rmsd_mean_vs_source": float(part.delta_reference_rmsd_vs_source.mean()),
            "delta_reference_rmsd_median_vs_source": float(part.delta_reference_rmsd_vs_source.median()),
            **{f"source_rmsd_{k}": v for k, v in cluster_bootstrap(part, "source_rmsd", 20260831 + index).items()},
            **{f"delta_reference_{k}": v for k, v in cluster_bootstrap(part, "delta_reference_rmsd_vs_source", 20260841 + index).items()},
        })
    rmsd_summary = pd.DataFrame(summary_rows)

    source_energy = pd.read_csv(CURRENT_XTB / "SOURCE_XTB.csv")
    current_energy = pd.read_csv(CURRENT_XTB / "FULL_JOINT_XTB.csv")
    mmff_energy = pd.read_csv(CACHE / "xtb_singlepoint_mmff/MMFF94S_OPTIMIZED_XTB.csv")
    xtb_opt = pd.read_parquet(CACHE / "optimization/GFN2_XTB/PER_RECORD.parquet")
    base = rmsd[["record_id", "molecule_id"]].drop_duplicates("record_id").sort_values("record_id")
    energy_frames = []
    source_ok = source_energy[source_energy.success.astype(bool)][["record_id", "energy_hartree"]].rename(columns={"energy_hartree": "source_energy_hartree"})
    for method, frame, column in (
        ("SOURCE", source_energy, "energy_hartree"),
        ("SIXS_FULL_JOINT_STEP17500", current_energy, "energy_hartree"),
        ("MMFF94S_OPTIMIZED", mmff_energy, "energy_hartree"),
    ):
        temp = frame[["record_id", "success", column]].rename(columns={column: "energy_hartree"}).merge(source_ok, on="record_id", how="left", validate="one_to_one")
        temp["method"] = method
        temp["deltaE_kcal_mol"] = (temp.energy_hartree - temp.source_energy_hartree) * HARTREE_TO_KCAL
        energy_frames.append(temp)
    xt = xtb_opt[["record_id", "success", "fallback_to_source", "final_native_energy"]].merge(source_ok, on="record_id", how="left", validate="one_to_one")
    # Preserve optimization convergence separately from single-point energy
    # availability.  A failed optimization is explicitly represented by its
    # frozen Source fallback, whose already-computed GFN2-xTB energy remains a
    # valid matched energy observation.
    xt = xt.rename(columns={"success": "optimization_success"})
    xt["energy_hartree"] = np.where(xt.optimization_success.astype(bool), xt.final_native_energy, xt.source_energy_hartree)
    xt["success"] = xt.energy_hartree.notna()
    xt["method"] = "GFN2_XTB_OPTIMIZED"; xt["deltaE_kcal_mol"] = (xt.energy_hartree - xt.source_energy_hartree) * HARTREE_TO_KCAL
    energy_frames.append(xt)
    energy = pd.concat(energy_frames, ignore_index=True).merge(base, on="record_id", how="left", validate="many_to_one")
    atomic_frame(OUT / "04_XTB_ENERGY_COMPARISON.csv", energy)
    energy_summary = []
    for method in METHODS:
        part = energy[(energy.method == method) & energy.success.astype(bool) & energy.deltaE_kcal_mol.notna()]
        values = part.deltaE_kcal_mol.to_numpy(dtype=np.float64)
        energy_summary.append({
            "method": method, "attempted": 5000, "successful": len(part), "failure_rate": 1 - len(part) / 5000,
            "absolute_energy_mean_hartree": float(part.energy_hartree.mean()), "deltaE_mean": float(values.mean()),
            "deltaE_median": float(np.median(values)), "deltaE_trimmed_mean_5pct": trimmed_mean(values),
            "deltaE_p90": float(np.quantile(values, .9)), "deltaE_p95": float(np.quantile(values, .95)),
            "deltaE_p99": float(np.quantile(values, .99)), "deltaE_max": float(values.max()),
            "fraction_lt_0": float(np.mean(values < 0)), "number_gt_25": int(np.sum(values > 25)),
            "number_gt_50": int(np.sum(values > 50)), "number_gt_100": int(np.sum(values > 100)),
        })
    energy_summary = pd.DataFrame(energy_summary)

    validity_rows = []
    for method in METHODS:
        if method == "SIXS_FULL_JOINT_STEP17500":
            pb = pd.read_parquet(CURRENT_ART / "POSEBUSTERS.parquet")
            v3 = pd.read_parquet(CURRENT_ART / "VALIDITY3D.parquet")
        else:
            pb = pd.read_parquet(CACHE / "validity" / method / "POSEBUSTERS.parquet")
            v3 = pd.read_parquet(CACHE / "validity" / method / "VALIDITY3D.parquet")
        validity_rows.append({
            "method": method, "V3D": float(v3.validity3d.mean()), "PB": float(pb.PB.mean()),
            "V3D_Bond": float(v3.bond_geometry_valid.mean()), "V3D_Angle": float(v3.angle_geometry_valid.mean()),
            "V3D_Aromatic": float(v3.aromatic_ring_valid.mean()), "V3D_Clash": float(v3.intramolecular_steric_clash_valid.mean()),
            "PB_Bond": float(pb.bond_lengths.mean()), "PB_Angle": float(pb.bond_angles.mean()),
            "PB_RingAromatic": float(pb.aromatic_ring_flatness.mean()), "PB_RingNonAromatic": float(pb["non-aromatic_ring_non-flatness"].mean()),
            "PB_Clash": float(pb.internal_steric_clash.mean()),
        })
    validity = pd.DataFrame(validity_rows)

    matched = rmsd_summary.merge(validity, on="method", validate="one_to_one").merge(energy_summary, on="method", validate="one_to_one")
    atomic_frame(OUT / "03_MMFF_XTB_MATCHED_COMPARISON.csv", matched)
    current = matched[matched.method == "SIXS_FULL_JOINT_STEP17500"].iloc[0]
    mmff = matched[matched.method == "MMFF94S_OPTIMIZED"].iloc[0]
    xtb = matched[matched.method == "GFN2_XTB_OPTIMIZED"].iloc[0]
    less_mmff = "YES" if current.source_rmsd_mean < mmff.source_rmsd_mean else "NO"
    less_xtb = "YES" if current.source_rmsd_mean < xtb.source_rmsd_mean else "NO"
    atomic_text(OUT / "05_CURRENT_FINAL_EVIDENCE_SUMMARY.md", f"""# Current final evidence summary

{markdown(matched[["method","source_rmsd_mean","source_rmsd_median","reference_rmsd_mean","reference_rmsd_median","delta_reference_rmsd_mean_vs_source","V3D","PB","deltaE_mean","deltaE_median","fraction_lt_0"]])}

`Source RMSD` is intervention magnitude. `Reference RMSD` is accuracy to the nearest frozen Reference conformer under fixed-order all-atom Kabsch; they are not interchangeable.

```text
REFERENCE_RMSD_CORRESPONDENCE_AUDIT = PASS
SIXS_USES_LESS_DISPLACEMENT_THAN_MMFF = {less_mmff}
SIXS_USES_LESS_DISPLACEMENT_THAN_XTB_OPT = {less_xtb}
FORMAL_READ = NO
LARGE_HOLDOUT_READ = NO
```
""")
    artifacts = {}
    for name in ("01_REFERENCE_RMSD_INTEGRITY_AUDIT.md", "02_MATCHED_SOURCE_REFERENCE_RMSD.csv", "03_MMFF_XTB_MATCHED_COMPARISON.csv", "04_XTB_ENERGY_COMPARISON.csv", "05_CURRENT_FINAL_EVIDENCE_SUMMARY.md", "CORRESPONDENCE_AUDIT.json"):
        artifacts[name] = sha256(OUT / name)
    atomic_json(OUT / "PHASE_I_COMPLETE.json", {
        "status": "COMPLETE",
        "current_final_model": "J1_R1_FULL_JOINT_ADAPTIVE_BA_MOVEMENT_STEP17500",
        "reference_rmsd_correspondence_audit": "PASS",
        "matched": matched.to_dict("records"),
        "artifacts": artifacts,
        "formal_read": False, "large_holdout_read": False,
    })
    status("PHASE_I_COMPLETE", "PASS", phase_i_complete_sha256=sha256(OUT / "PHASE_I_COMPLETE.json"))


def pipeline() -> None:
    audit_correspondence()
    optimize_method("MMFF94S")
    optimize_method("GFN2_XTB")
    evaluate_validity()
    rmsd_evidence()
    mmff_xtb_singlepoints()
    summarize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "mmff", "xtb-opt", "validity", "rmsd", "xtb-sp", "summarize", "pipeline"), default="pipeline")
    args = parser.parse_args()
    stages = {
        "audit": audit_correspondence, "mmff": lambda: optimize_method("MMFF94S"),
        "xtb-opt": lambda: optimize_method("GFN2_XTB"), "validity": evaluate_validity,
        "rmsd": rmsd_evidence, "xtb-sp": mmff_xtb_singlepoints, "summarize": summarize,
        "pipeline": pipeline,
    }
    try:
        stages[args.stage]()
        return 0
    except BaseException as error:
        status("FAILED", "FAIL", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
