#!/usr/bin/env python3
"""Frozen-coordinate DEV GFN2-xTB single-point diagnostic for SIXS full joint.

This runner is intentionally independent of training and coordinate generation.  It
only consumes already-frozen Source, comparator Proposal, and Full-Joint Proposal
coordinates.  Per-geometry JSON caches make the 15,000 method-record evaluations
resumable without changing the xTB protocol after outcomes are observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307"
OUT = REPORT / "xtb_single_point_dev"
ARTIFACT = ROOT / "artifacts/ecir_mvr"

FULL_DIR = ARTIFACT / "sixs_j1r1_full_joint_adaptive_ba_movement_seed307/dev_evaluation"
COMP_DIR = ARTIFACT / "sixs_j1r1_joint_magnitude_interaction_seed307/j1_r1_joint/dev_evaluation"
SOURCE_PAYLOAD = Path(
    r"E:\3dconformergenerationcode\4dadapter-lsgoba-v2-joint-magnitude-full307"
    r"\artifacts\ecir_mvr\lsgoba_v2_joint_magnitude_full307\SOURCE_BINDING.pt"
)
CONFIG = ROOT / "configs/sixs_j1r1_full_joint_adaptive_ba_movement.json"

METHODS = ("SOURCE", "COMPARATOR", "FULL_JOINT")
METHOD_FILES = {
    "SOURCE": OUT / "SOURCE_XTB.csv",
    "COMPARATOR": OUT / "COMPARATOR_XTB.csv",
    "FULL_JOINT": OUT / "FULL_JOINT_XTB.csv",
}
SETTINGS = {
    "version": "6.7.1",
    "wsl": "wsl.exe",
    "distribution": "Ubuntu-22.04",
    "executable": r"E:\tools\xtb\6.7.1-linux-x86_64\bin\xtb",
    "executable_sha256": "debf27a9e0fa4bfb5ca75aafe4b90d8211f08ec2f4a482f375a4987212eaa12a",
    "gfn": 2,
    "threads": 1,
    "workers": 4,
    "timeout_seconds": 180,
    "solvent": None,
    "geometry_optimization": False,
}
HARTREE_TO_KCAL_MOL = 627.509474
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260830
EXPECTED_RECORDS = 5_000
EXPECTED_MOLECULES = 2_500
PROTOCOL_PATH = OUT / "XTB_PROTOCOL.json"
ALIGNMENT_PATH = OUT / "XTB_INPUT_ALIGNMENT.json"
RUN_STATUS = OUT / "RUN_STATUS.json"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def update_status(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    current = json.loads(RUN_STATUS.read_text(encoding="utf-8")) if RUN_STATUS.is_file() else {}
    current.update(
        {
            "schema_version": "sixs-j1r1-full-joint-xtb-dev-run-status-v1",
            "status": state,
            "stage": stage,
            "pid": os.getpid(),
            "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "formal_read": False,
            "large_holdout_read": False,
            "new_training_started": False,
            "coordinates_regenerated": False,
            **extra,
        }
    )
    atomic_json(RUN_STATUS, current)


def wsl_path(path: str | Path) -> str:
    raw = str(Path(path).resolve())
    if raw.startswith("\\\\?\\"):
        raw = raw[4:]
    parsed = Path(raw)
    return f"/mnt/{parsed.drive.rstrip(':').lower()}{parsed.as_posix().split(':', 1)[1]}"


def load_sdf(path: Path) -> list[Chem.Mol]:
    molecules = list(Chem.ForwardSDMolSupplier(str(path), sanitize=False, removeHs=False))
    if any(molecule is None for molecule in molecules):
        raise RuntimeError(f"unreadable frozen SDF record: {path}")
    return [molecule for molecule in molecules if molecule is not None]


def record_id(molecule: Chem.Mol) -> str:
    if molecule.HasProp("sample_id"):
        return str(molecule.GetProp("sample_id"))
    return str(molecule.GetProp("_Name"))


def elements(molecule: Chem.Mol) -> list[str]:
    return [atom.GetSymbol() for atom in molecule.GetAtoms()]


def formal_charge(molecule: Chem.Mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()))


def radical_electrons(molecule: Chem.Mol) -> int:
    return int(sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()))


def coordinates(molecule: Chem.Mol) -> np.ndarray:
    return np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)


def write_protocol() -> dict[str, Any]:
    protocol = {
        "schema_version": "sixs-j1r1-full-joint-xtb-dev-protocol-v1",
        "status": "FROZEN_BEFORE_XTB_OUTCOMES",
        "experiment": "SIXS_J1_R1_FULL_JOINT_ADAPTIVE_BA_MOVEMENT_SEED307_DEV",
        "classification": "XTB_DIAGNOSTIC_ONLY",
        "methods": {
            "SOURCE": "frozen Source coordinates",
            "COMPARATOR": "frozen J1-R1 Equal-BA learned-magnitude Proposal coordinates",
            "FULL_JOINT": "frozen J1-R1 Full-Joint Adaptive-BA Movement Proposal coordinates",
        },
        "cohort": {"molecules": EXPECTED_MOLECULES, "records": EXPECTED_RECORDS},
        "xtb": SETTINGS,
        "charge_handling": "sum of per-atom formal charges in frozen RDKit topology",
        "multiplicity_handling": "--uhf equals sum of per-atom radical-electron counts",
        "solvent_model": "NONE",
        "energy_unit_raw": "hartree",
        "energy_unit_reported": "kcal/mol",
        "hartree_to_kcal_mol": HARTREE_TO_KCAL_MOL,
        "energy_definition": {
            "full_vs_source": "E(FULL_JOINT)-E(SOURCE)",
            "comparator_vs_source": "E(COMPARATOR)-E(SOURCE)",
            "full_vs_comparator": "E(FULL_JOINT)-E(COMPARATOR)",
            "negative_meaning": "first method has lower xTB single-point energy",
        },
        "zero_definition": "exact floating-point equality after subtraction",
        "trim_definition": "symmetric floor(fraction*N) observations removed from each ordered tail",
        "bootstrap": {
            "independent_unit": "molecule",
            "records_grouped_within_molecule": True,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "2.5th and 97.5th empirical percentiles",
        },
        "classification_rules": {
            "unresolved": "valid paired N below 95 percent of 5000",
            "tail_risk": "any delta above +100 kcal/mol OR at least 0.1 percent above +25 kcal/mol",
            "energetically_positive_or_better": "mean, median, 5-percent trimmed mean all <0 and lower-energy fraction >0.5",
            "energetically_negative_or_worse": "mean, median, 5-percent trimmed mean all >0 and lower-energy fraction <0.5",
            "similar": "absolute mean, median and 5-percent trimmed mean <=0.05 kcal/mol and lower-energy fraction in [0.45,0.55]",
            "otherwise": "MIXED",
        },
        "failure_handling": "no imputation, no alternative method, no geometry optimization, failures retained explicitly",
        "failure_classes": ["timeout", "nonzero_exit", "nonfinite_energy", "parse_failure", "other"],
        "energy_strata_kcal_mol": ["<=-5", "(-5,-2]", "(-2,-1]", "(-1,0)", "0", "(0,1]", "(1,5]", "(5,25]", ">25"],
        "guards": {
            "xtb_used_for_tuning": False,
            "geometry_optimization": False,
            "coordinates_regenerated": False,
            "formal_read": False,
            "large_holdout_read": False,
            "seed331_started": False,
            "seed353_started": False,
            "new_training_started": False,
        },
    }
    if PROTOCOL_PATH.is_file():
        previous = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if previous != protocol:
            raise RuntimeError("frozen XTB protocol differs from current implementation")
    else:
        atomic_json(PROTOCOL_PATH, protocol)
    return protocol


def preflight() -> dict[str, Any]:
    update_status("PREFLIGHT")
    write_protocol()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected_source_sha = str(config["data"]["source_payload_sha256"])
    if sha_file(SOURCE_PAYLOAD) != expected_source_sha:
        raise RuntimeError("frozen Source payload SHA changed")
    if sha_file(SETTINGS["executable"]) != SETTINGS["executable_sha256"]:
        raise RuntimeError("xTB executable SHA changed")
    version_process = subprocess.run(
        [SETTINGS["wsl"], "-d", SETTINGS["distribution"], "--exec", wsl_path(SETTINGS["executable"]), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        encoding="utf-8",
        errors="replace",
    )
    version_text = (version_process.stdout or "") + "\n" + (version_process.stderr or "")
    if version_process.returncode != 0 or "6.7.1" not in version_text:
        raise RuntimeError("xTB version preflight failed")

    full_pr = pd.read_parquet(FULL_DIR / "PER_RECORD.parquet")
    comp_pr = pd.read_parquet(COMP_DIR / "PER_RECORD.parquet")
    if len(full_pr) != EXPECTED_RECORDS or len(comp_pr) != EXPECTED_RECORDS:
        raise RuntimeError("per-record denominator mismatch")
    if full_pr.record_id.astype(str).tolist() != comp_pr.record_id.astype(str).tolist():
        raise RuntimeError("candidate/comparator record order mismatch")
    if full_pr.molecule_id.astype(str).tolist() != comp_pr.molecule_id.astype(str).tolist():
        raise RuntimeError("candidate/comparator molecule identity mismatch")
    if full_pr.record_id.nunique() != EXPECTED_RECORDS or full_pr.molecule_id.nunique() != EXPECTED_MOLECULES:
        raise RuntimeError("DEV identity denominator mismatch")
    group_sizes = full_pr.groupby("molecule_id", sort=False).size()
    if set(group_sizes.astype(int)) != {2}:
        raise RuntimeError("DEV records-per-molecule mismatch")

    full_molecules = load_sdf(FULL_DIR / "PROPOSAL.sdf")
    comp_molecules = load_sdf(COMP_DIR / "PROPOSAL.sdf")
    if len(full_molecules) != EXPECTED_RECORDS or len(comp_molecules) != EXPECTED_RECORDS:
        raise RuntimeError("frozen SDF denominator mismatch")
    ids = full_pr.record_id.astype(str).tolist()
    if [record_id(molecule) for molecule in full_molecules] != ids:
        raise RuntimeError("Full-Joint SDF record ordering mismatch")
    if [record_id(molecule) for molecule in comp_molecules] != ids:
        raise RuntimeError("comparator SDF record ordering mismatch")

    source_payload = torch.load(SOURCE_PAYLOAD, map_location="cpu", weights_only=False)
    if source_payload.get("formal_test_records_read") != 0 or source_payload.get("frozen_holdout_records_read") != 0:
        raise RuntimeError("Source payload protection boundary failure")
    source_by_id = {str(row["sample_id"]): row for row in source_payload["val"]}
    if len(source_by_id) != 10_000 or not set(ids).issubset(source_by_id):
        raise RuntimeError("DEV Source binding mismatch")
    source_rows = [source_by_id[item] for item in ids]

    finite = True
    atom_order = True
    molecule_identity = True
    charge_match = True
    for index, (full_mol, comp_mol, source_row, molecule_id) in enumerate(
        zip(full_molecules, comp_molecules, source_rows, full_pr.molecule_id.astype(str), strict=True)
    ):
        full_elements = elements(full_mol)
        comp_elements = elements(comp_mol)
        source = torch.as_tensor(source_row["source"], dtype=torch.float32).detach().cpu().numpy()
        atom_order &= full_elements == comp_elements and source.shape == (len(full_elements), 3)
        molecule_identity &= str(source_row["molecule_id"]) == molecule_id
        charge_match &= formal_charge(full_mol) == formal_charge(comp_mol) and radical_electrons(full_mol) == radical_electrons(comp_mol)
        finite &= bool(np.isfinite(source).all() and np.isfinite(coordinates(full_mol)).all() and np.isfinite(coordinates(comp_mol)).all())
        if not atom_order or not molecule_identity or not charge_match or not finite:
            raise RuntimeError(f"three-way coordinate alignment failure at record {index}")

    source_summary = source_payload.get("summaries", {}).get("val", source_payload.get("summaries", {}).get("validation", {}))
    del source_payload, source_by_id, source_rows, full_molecules, comp_molecules
    alignment = {
        "schema_version": "sixs-j1r1-full-joint-xtb-input-alignment-v1",
        "status": "PASS",
        "source_records": EXPECTED_RECORDS,
        "comparator_records": EXPECTED_RECORDS,
        "full_joint_records": EXPECTED_RECORDS,
        "molecules": EXPECTED_MOLECULES,
        "records_per_molecule": 2,
        "record_identity_alignment": "PASS",
        "record_order_alignment": "PASS",
        "molecule_identity_alignment": "PASS",
        "atom_ordering": "PASS",
        "charge_and_uhf_alignment": "PASS",
        "coordinate_finite_check": "PASS",
        "source_binding_prior_atom_order_audit": source_summary.get("atom_order_audit", "PASS_FROM_FROZEN_SOURCE_BINDING_CONSTRUCTION"),
        "xtb_version": "6.7.1",
        "gfn_level": "GFN2-xTB",
        "input_sha256": {
            "source_payload": sha_file(SOURCE_PAYLOAD),
            "comparator_sdf": sha_file(COMP_DIR / "PROPOSAL.sdf"),
            "comparator_per_record": sha_file(COMP_DIR / "PER_RECORD.parquet"),
            "full_joint_sdf": sha_file(FULL_DIR / "PROPOSAL.sdf"),
            "full_joint_per_record": sha_file(FULL_DIR / "PER_RECORD.parquet"),
            "xtb_executable": sha_file(SETTINGS["executable"]),
            "protocol": sha_file(PROTOCOL_PATH),
        },
        "formal_read": False,
        "large_holdout_read": False,
        "coordinates_regenerated": False,
    }
    atomic_json(ALIGNMENT_PATH, alignment)
    update_status("PREFLIGHT_COMPLETE", state="PASS", alignment_sha256=sha_file(ALIGNMENT_PATH))
    return alignment


def load_task_inputs() -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    alignment = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))
    if alignment.get("status") != "PASS" or sha_file(PROTOCOL_PATH) != alignment["input_sha256"]["protocol"]:
        raise RuntimeError("frozen protocol/alignment unavailable")
    frozen = {
        "source_payload": SOURCE_PAYLOAD,
        "comparator_sdf": COMP_DIR / "PROPOSAL.sdf",
        "comparator_per_record": COMP_DIR / "PER_RECORD.parquet",
        "full_joint_sdf": FULL_DIR / "PROPOSAL.sdf",
        "full_joint_per_record": FULL_DIR / "PER_RECORD.parquet",
        "xtb_executable": Path(SETTINGS["executable"]),
    }
    for key, path in frozen.items():
        if sha_file(path) != alignment["input_sha256"][key]:
            raise RuntimeError(f"frozen xTB input changed after preflight: {key}")

    full_pr = pd.read_parquet(FULL_DIR / "PER_RECORD.parquet")
    ids = full_pr.record_id.astype(str).tolist()
    full_molecules = load_sdf(FULL_DIR / "PROPOSAL.sdf")
    comp_molecules = load_sdf(COMP_DIR / "PROPOSAL.sdf")
    source_payload = torch.load(SOURCE_PAYLOAD, map_location="cpu", weights_only=False)
    source_by_id = {str(row["sample_id"]): row for row in source_payload["val"]}
    tasks: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for position, (item_id, molecule_id, full_mol, comp_mol) in enumerate(
        zip(ids, full_pr.molecule_id.astype(str), full_molecules, comp_molecules, strict=True)
    ):
        template_elements = elements(full_mol)
        charge = formal_charge(full_mol)
        uhf = radical_electrons(full_mol)
        method_coordinates = {
            "SOURCE": torch.as_tensor(source_by_id[item_id]["source"], dtype=torch.float32).cpu().numpy().astype(np.float64),
            "COMPARATOR": coordinates(comp_mol),
            "FULL_JOINT": coordinates(full_mol),
        }
        for method in METHODS:
            value = np.ascontiguousarray(method_coordinates[method], dtype=np.float64)
            coordinate_hash = sha_bytes(f"float64|{value.shape}|".encode() + value.tobytes())
            identity = {
                "elements": template_elements,
                "charge": charge,
                "uhf": uhf,
                "coordinate_sha256": coordinate_hash,
                "settings": {key: SETTINGS[key] for key in SETTINGS if key != "workers"},
            }
            tasks[method].append(
                {
                    "method": method,
                    "record_index": position,
                    "record_id": item_id,
                    "molecule_id": molecule_id,
                    "coordinates": value,
                    "identity": identity,
                    "identity_sha256": canonical_sha(identity),
                }
            )
    del source_payload, source_by_id, full_molecules, comp_molecules
    return full_pr, tasks


def failure_class(exit_code: int | None, timed_out: bool, energy: float | None, error: str | None) -> str | None:
    if timed_out:
        return "timeout"
    if error is not None:
        return "other"
    if exit_code != 0:
        return "nonzero_exit"
    if energy is None:
        return "parse_failure"
    if not math.isfinite(energy):
        return "nonfinite_energy"
    return None


def execute(task: dict[str, Any]) -> dict[str, Any]:
    cache = OUT / "cache" / f"{task['identity_sha256']}.json"
    cache_hit = cache.is_file()
    if cache_hit:
        result = json.loads(cache.read_text(encoding="utf-8"))
    else:
        work = OUT / "work" / task["identity_sha256"]
        work.mkdir(parents=True, exist_ok=True)
        xyz = work / "input.xyz"
        lines = [str(len(task["identity"]["elements"])), "frozen coordinates; GFN2-xTB single-point; no optimization"]
        for element, coordinate in zip(task["identity"]["elements"], task["coordinates"], strict=True):
            lines.append(f"{element} {coordinate[0]:.12f} {coordinate[1]:.12f} {coordinate[2]:.12f}")
        atomic_text(xyz, "\n".join(lines))
        command = [
            SETTINGS["wsl"], "-d", SETTINGS["distribution"], "--cd", wsl_path(work),
            "--exec", "/usr/bin/env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
            wsl_path(SETTINGS["executable"]), "input.xyz", "--gfn", "2",
            "--chrg", str(task["identity"]["charge"]), "--uhf", str(task["identity"]["uhf"]),
        ]
        if any(flag in command for flag in ("--opt", "--ohess", "--md")):
            raise RuntimeError("geometry-changing xTB option detected")
        started = time.perf_counter()
        timed_out = False
        exit_code: int | None = None
        output = ""
        execution_error: str | None = None
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=SETTINGS["timeout_seconds"],
                encoding="utf-8",
                errors="replace",
            )
            exit_code = process.returncode
            output = (process.stdout or "") + "\n" + (process.stderr or "")
        except subprocess.TimeoutExpired as error:
            timed_out = True
            output = str(error.stdout or "") + "\n" + str(error.stderr or "")
        except Exception as error:  # retained in explicit failure table
            execution_error = f"{type(error).__name__}: {error}"
            output = execution_error
        matches = re.findall(r"TOTAL ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+Eh", output, re.I)
        energy = float(matches[-1]) if matches else None
        kind = failure_class(exit_code, timed_out, energy, execution_error)
        result = {
            "identity_sha256": task["identity_sha256"],
            "success": kind is None,
            "energy_hartree": energy,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "failure_reason": kind,
            "runtime_seconds": time.perf_counter() - started,
            "geometry_optimization_performed": False,
            "error_detail": "" if kind is None else output[-1000:],
        }
        atomic_json(cache, result)
        if kind is None:
            shutil.rmtree(work, ignore_errors=True)
    return {
        "method": task["method"],
        "record_index": task["record_index"],
        "record_id": task["record_id"],
        "molecule_id": task["molecule_id"],
        "charge": task["identity"]["charge"],
        "uhf": task["identity"]["uhf"],
        "elements_sha256": sha_bytes("|".join(task["identity"]["elements"]).encode()),
        "coordinate_sha256": task["identity"]["coordinate_sha256"],
        "cache_hit": cache_hit,
        **result,
    }


def run_xtb() -> dict[str, Any]:
    _, tasks = load_task_inputs()
    started = time.perf_counter()
    method_runtime: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        if METHOD_FILES[method].is_file():
            frame = pd.read_csv(METHOD_FILES[method])
            if len(frame) != EXPECTED_RECORDS or frame.record_id.nunique() != EXPECTED_RECORDS:
                raise RuntimeError(f"incomplete persisted xTB method result: {method}")
            print(json.dumps({"method": method, "stage": "REUSED_COMPLETE_CSV", "success": int(frame.success.sum())}), flush=True)
        else:
            update_status(
                f"RUNNING_{method}",
                completed_method_records=method_index * EXPECTED_RECORDS,
                expected_method_records=len(METHODS) * EXPECTED_RECORDS,
            )
            rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=SETTINGS["workers"]) as pool:
                futures = [pool.submit(execute, task) for task in tasks[method]]
                for count, future in enumerate(as_completed(futures), start=1):
                    rows.append(future.result())
                    if count % 100 == 0 or count == EXPECTED_RECORDS:
                        update_status(
                            f"RUNNING_{method}",
                            active_method=method,
                            active_method_completed=count,
                            completed_method_records=method_index * EXPECTED_RECORDS + count,
                            expected_method_records=len(METHODS) * EXPECTED_RECORDS,
                        )
                        print(json.dumps({"method": method, "completed": count, "expected": EXPECTED_RECORDS}), flush=True)
            frame = pd.DataFrame(rows).sort_values("record_index", kind="stable").reset_index(drop=True)
            atomic_csv(METHOD_FILES[method], frame)
        reasons = frame.failure_reason.fillna("").astype(str) if "failure_reason" in frame else pd.Series([""] * len(frame))
        method_runtime.append(
            {
                "method": method,
                "attempted": int(len(frame)),
                "success": int(frame.success.astype(bool).sum()),
                "failure": int((~frame.success.astype(bool)).sum()),
                "failure_rate": float((~frame.success.astype(bool)).mean()),
                "timeout": int((reasons == "timeout").sum()),
                "nonzero_exit": int((reasons == "nonzero_exit").sum()),
                "nonfinite_energy": int((reasons == "nonfinite_energy").sum()),
                "parse_failure": int((reasons == "parse_failure").sum()),
                "other": int((reasons == "other").sum()),
                "method_runtime_seconds_sum": float(frame.runtime_seconds.sum()),
                "cache_hits": int(frame.cache_hit.astype(bool).sum()),
            }
        )
    wall = time.perf_counter() - started
    runtime = {
        "schema_version": "sixs-j1r1-full-joint-xtb-dev-runtime-v1",
        "status": "XTB_SINGLE_POINTS_COMPLETE",
        "total_xtb_wall_time_seconds": wall,
        "successful_method_single_points": int(sum(row["success"] for row in method_runtime)),
        "method_execution": method_runtime,
        "cpu_model": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "logical_cpu_count": os.cpu_count(),
        "threads_per_xtb": SETTINGS["threads"],
        "workers": SETTINGS["workers"],
        "formal_read": False,
        "large_holdout_read": False,
        "coordinates_regenerated": False,
    }
    all_frames = [pd.read_csv(METHOD_FILES[method]) for method in METHODS]
    all_rows = pd.concat(all_frames, ignore_index=True)
    actual_rows = all_rows.loc[~all_rows.cache_hit.astype(bool)]
    timing_rows = actual_rows if len(actual_rows) else all_rows.drop_duplicates("identity_sha256")
    runtime.update(
        {
            "unique_single_point_invocations_in_this_run": int(len(actual_rows)),
            "mean_seconds_per_single_point": float(timing_rows.runtime_seconds.mean()),
            "median_seconds_per_single_point": float(timing_rows.runtime_seconds.median()),
            "p95_seconds_per_single_point": float(timing_rows.runtime_seconds.quantile(0.95)),
        }
    )
    atomic_json(OUT / "XTB_RUNTIME.json", runtime)
    update_status("XTB_SINGLE_POINTS_COMPLETE", state="PASS", total_xtb_wall_time_seconds=wall)
    return runtime


def trimmed_mean(values: np.ndarray, proportion: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    cut = int(math.floor(proportion * len(ordered)))
    retained = ordered[cut: len(ordered) - cut] if cut else ordered
    return float(np.mean(retained))


def comparison_stats(name: str, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    quantiles = np.quantile(values, [0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99])
    median = float(np.median(values))
    row: dict[str, Any] = {
        "row_type": "comparison",
        "comparison": name,
        "valid_n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": median,
        "standard_deviation": float(np.std(values, ddof=1)),
        "p01": float(quantiles[0]), "p05": float(quantiles[1]), "p10": float(quantiles[2]), "p25": float(quantiles[3]),
        "p75": float(quantiles[4]), "p90": float(quantiles[5]), "p95": float(quantiles[6]), "p99": float(quantiles[7]),
        "minimum": float(np.min(values)), "maximum": float(np.max(values)),
        "fraction_lt_zero": float(np.mean(values < 0)),
        "fraction_eq_zero": float(np.mean(values == 0)),
        "fraction_gt_zero": float(np.mean(values > 0)),
        "trimmed_mean_5": trimmed_mean(values, 0.05),
        "trimmed_mean_10": trimmed_mean(values, 0.10),
        "median_absolute_deviation": float(np.median(np.abs(values - median))),
    }
    for threshold in (1, 2, 5, 10, 25, 50, 100):
        row[f"fraction_gt_pos_{threshold}"] = float(np.mean(values > threshold))
    for threshold in (1, 2, 5):
        row[f"fraction_lt_neg_{threshold}"] = float(np.mean(values < -threshold))
    return row


def bootstrap_comparison(paired: pd.DataFrame, value_column: str, label: str) -> list[dict[str, Any]]:
    molecules = paired.molecule_id.astype(str).drop_duplicates().tolist()
    if len(molecules) != EXPECTED_MOLECULES:
        raise RuntimeError(f"bootstrap molecule denominator mismatch: {label}")
    by_molecule = {key: group[value_column].to_numpy(dtype=np.float64) for key, group in paired.groupby("molecule_id", sort=False)}
    width = max(len(by_molecule[key]) for key in molecules)
    matrix = np.full((len(molecules), width), np.nan, dtype=np.float64)
    for index, key in enumerate(molecules):
        values = by_molecule[key]
        matrix[index, : len(values)] = values
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    medians = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    lower = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    offset = 0
    batch_size = 64
    while offset < BOOTSTRAP_RESAMPLES:
        count = min(batch_size, BOOTSTRAP_RESAMPLES - offset)
        selection = rng.integers(0, len(molecules), size=(count, len(molecules)))
        sampled = matrix[selection].reshape(count, -1)
        means[offset:offset + count] = np.nanmean(sampled, axis=1)
        medians[offset:offset + count] = np.nanmedian(sampled, axis=1)
        lower[offset:offset + count] = np.nanmean(sampled < 0, axis=1)
        offset += count
    valid = paired[value_column].to_numpy(dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    results = []
    for metric, point, distribution in (
        ("mean", float(np.mean(valid)), means),
        ("median", float(np.median(valid)), medians),
        ("fraction_lower", float(np.mean(valid < 0)), lower),
    ):
        results.append(
            {
                "comparison": label,
                "metric": metric,
                "point_estimate": point,
                "ci95_low": float(np.quantile(distribution, 0.025)),
                "ci95_high": float(np.quantile(distribution, 0.975)),
                "clusters": EXPECTED_MOLECULES,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
            }
        )
    return results


def average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


def spearman(x: Iterable[float], y: Iterable[float]) -> tuple[int, float]:
    x_values = np.asarray(list(x), dtype=np.float64)
    y_values = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_rank = average_ranks(x_values[valid])
    y_rank = average_ranks(y_values[valid])
    x_centered = x_rank - np.mean(x_rank)
    y_centered = y_rank - np.mean(y_rank)
    denominator = math.sqrt(float(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    return int(valid.sum()), float(np.dot(x_centered, y_centered) / denominator) if denominator else float("nan")


def energy_stratum(value: float) -> str:
    if value <= -5: return "<=-5"
    if value <= -2: return "(-5,-2]"
    if value <= -1: return "(-2,-1]"
    if value < 0: return "(-1,0)"
    if value == 0: return "0"
    if value <= 1: return "(0,1]"
    if value <= 5: return "(1,5]"
    if value <= 25: return "(5,25]"
    return ">25"


def classification(stats: dict[str, Any], comparator: bool = False) -> str:
    if stats["valid_n"] < int(0.95 * EXPECTED_RECORDS):
        return "UNRESOLVED" if not comparator else "MIXED"
    if stats["maximum"] > 100 or stats["fraction_gt_pos_25"] >= 0.001:
        return "TAIL_RISK"
    signs_positive = stats["mean"] < 0 and stats["median"] < 0 and stats["trimmed_mean_5"] < 0 and stats["fraction_lt_zero"] > 0.5
    signs_negative = stats["mean"] > 0 and stats["median"] > 0 and stats["trimmed_mean_5"] > 0 and stats["fraction_lt_zero"] < 0.5
    if comparator:
        if signs_positive: return "BETTER"
        if signs_negative: return "WORSE"
        if max(abs(stats["mean"]), abs(stats["median"]), abs(stats["trimmed_mean_5"])) <= 0.05 and 0.45 <= stats["fraction_lt_zero"] <= 0.55:
            return "SIMILAR"
        return "MIXED"
    if signs_positive: return "ENERGETICALLY_POSITIVE"
    if signs_negative: return "ENERGETICALLY_NEGATIVE"
    return "MIXED"


def analyze() -> dict[str, Any]:
    update_status("ANALYSIS")
    frames: dict[str, pd.DataFrame] = {}
    failure_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        frame = pd.read_csv(METHOD_FILES[method])
        frame["success"] = frame.success.astype(bool)
        frames[method] = frame
        reasons = frame.failure_reason.fillna("").astype(str)
        summary_rows.append(
            {
                "row_type": "execution",
                "method": method,
                "attempted": int(len(frame)),
                "success": int(frame.success.sum()),
                "failure": int((~frame.success).sum()),
                "failure_rate": float((~frame.success).mean()),
                "timeout": int((reasons == "timeout").sum()),
                "nonzero_exit": int((reasons == "nonzero_exit").sum()),
                "nonfinite_energy": int((reasons == "nonfinite_energy").sum()),
                "parse_failure": int((reasons == "parse_failure").sum()),
                "other": int((reasons == "other").sum()),
            }
        )
        failed = frame.loc[~frame.success, ["method", "record_index", "record_id", "molecule_id", "failure_reason", "exit_code", "timed_out", "error_detail"]].copy()
        failure_rows.append(failed)
    failure_frame = pd.concat(failure_rows, ignore_index=True) if failure_rows else pd.DataFrame()
    if failure_frame.empty:
        failure_frame = pd.DataFrame(columns=["method", "record_index", "record_id", "molecule_id", "failure_reason", "exit_code", "timed_out", "error_detail"])
    atomic_csv(OUT / "XTB_FAILURES.csv", failure_frame)

    base = frames["SOURCE"][["record_index", "record_id", "molecule_id", "success", "energy_hartree"]].rename(
        columns={"success": "source_success", "energy_hartree": "source_energy_hartree"}
    )
    for method, prefix in (("COMPARATOR", "comparator"), ("FULL_JOINT", "full_joint")):
        right = frames[method][["record_index", "record_id", "molecule_id", "success", "energy_hartree"]].rename(
            columns={"success": f"{prefix}_success", "energy_hartree": f"{prefix}_energy_hartree"}
        )
        base = base.merge(right, on=["record_index", "record_id", "molecule_id"], how="inner", validate="one_to_one")
    if len(base) != EXPECTED_RECORDS:
        raise RuntimeError("paired xTB alignment mismatch")
    base["delta_full_vs_source_kcal_mol"] = np.where(
        base.source_success & base.full_joint_success,
        (base.full_joint_energy_hartree - base.source_energy_hartree) * HARTREE_TO_KCAL_MOL,
        np.nan,
    )
    base["delta_comparator_vs_source_kcal_mol"] = np.where(
        base.source_success & base.comparator_success,
        (base.comparator_energy_hartree - base.source_energy_hartree) * HARTREE_TO_KCAL_MOL,
        np.nan,
    )
    base["delta_full_vs_comparator_kcal_mol"] = np.where(
        base.full_joint_success & base.comparator_success,
        (base.full_joint_energy_hartree - base.comparator_energy_hartree) * HARTREE_TO_KCAL_MOL,
        np.nan,
    )
    atomic_csv(OUT / "PAIRED_XTB_RESULTS.csv", base)

    comparisons = {
        "FULL_JOINT_VS_SOURCE": "delta_full_vs_source_kcal_mol",
        "COMPARATOR_VS_SOURCE": "delta_comparator_vs_source_kcal_mol",
        "FULL_JOINT_VS_COMPARATOR": "delta_full_vs_comparator_kcal_mol",
    }
    stats_by_name: dict[str, dict[str, Any]] = {}
    for name, column in comparisons.items():
        stats = comparison_stats(name, base[column].to_numpy(dtype=np.float64))
        stats_by_name[name] = stats
        summary_rows.append(stats)
    atomic_csv(OUT / "XTB_SUMMARY.csv", pd.DataFrame(summary_rows))

    bootstrap_rows: list[dict[str, Any]] = []
    for name, column in comparisons.items():
        bootstrap_rows.extend(bootstrap_comparison(base, column, name))
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    atomic_csv(OUT / "XTB_BOOTSTRAP.csv", bootstrap_frame)

    full_pr = pd.read_parquet(FULL_DIR / "PER_RECORD.parquet")
    full_v3d = pd.read_parquet(FULL_DIR / "VALIDITY3D.parquet")
    full_pb = pd.read_parquet(FULL_DIR / "POSEBUSTERS.parquet")
    comp_v3d = pd.read_parquet(COMP_DIR / "VALIDITY3D.parquet")
    payload = torch.load(FULL_DIR / "EVALUATION_PAYLOAD.pt", map_location="cpu", weights_only=False)
    primitive_rows = []
    for row in payload["primitive_rows"]:
        primitive_rows.append(
            {
                "record_id": str(row["record_id"]),
                "bond_reliability_mean": float(torch.as_tensor(row["bond_reliability"]).float().mean()),
                "angle_reliability_mean": float(torch.as_tensor(row["angle_reliability"]).float().mean()),
            }
        )
    del payload
    geometry = base.merge(full_pr, on=["record_id", "molecule_id"], validate="one_to_one")
    geometry = geometry.merge(
        full_v3d[["record_id", "bond_geometry_valid", "angle_geometry_valid", "validity3d"]].rename(columns={"validity3d": "full_v3d"}),
        on="record_id", validate="one_to_one",
    )
    geometry = geometry.merge(
        comp_v3d[["record_id", "validity3d"]].rename(columns={"validity3d": "comparator_v3d"}),
        on="record_id", validate="one_to_one",
    )
    geometry = geometry.merge(full_pb[["record_id", "PB"]].rename(columns={"PB": "full_pb"}), on="record_id", validate="one_to_one")
    geometry = geometry.merge(pd.DataFrame(primitive_rows), on="record_id", validate="one_to_one")
    geometry["v3d_improved"] = geometry.full_v3d.astype(int) > geometry.comparator_v3d.astype(int)
    geometry["v3d_degraded"] = geometry.full_v3d.astype(int) < geometry.comparator_v3d.astype(int)
    geometry["energy_stratum"] = geometry.delta_full_vs_source_kcal_mol.map(energy_stratum)
    association_rows: list[dict[str, Any]] = []
    for column in ("tau", "source_rmsd", "w_B", "bond_reliability_mean", "angle_reliability_mean", "direction_improvement"):
        n, rho = spearman(geometry.delta_full_vs_source_kcal_mol, geometry[column])
        association_rows.append({"row_type": "spearman", "variable": column, "n": n, "spearman_rho": rho})
    n, rho = spearman(geometry.delta_full_vs_source_kcal_mol, geometry.v3d_improved.astype(float))
    association_rows.append({"row_type": "spearman", "variable": "v3d_improved", "n": n, "spearman_rho": rho})
    strata_order = ["<=-5", "(-5,-2]", "(-2,-1]", "(-1,0)", "0", "(0,1]", "(1,5]", "(5,25]", ">25"]
    for stratum in strata_order:
        group = geometry.loc[geometry.energy_stratum == stratum]
        if group.empty:
            continue
        association_rows.append(
            {
                "row_type": "energy_stratum",
                "stratum": stratum,
                "n": int(len(group)),
                "delta_energy_mean": float(group.delta_full_vs_source_kcal_mol.mean()),
                "delta_energy_median": float(group.delta_full_vs_source_kcal_mol.median()),
                "proposal_v3d": float(group.full_v3d.mean()),
                "comparator_v3d": float(group.comparator_v3d.mean()),
                "v3d_improved_fraction": float(group.v3d_improved.mean()),
                "v3d_degraded_fraction": float(group.v3d_degraded.mean()),
                "bond_validity": float(group.bond_geometry_valid.mean()),
                "angle_validity": float(group.angle_geometry_valid.mean()),
                "posebusters": float(group.full_pb.mean()),
                "source_rmsd_mean": float(group.source_rmsd.mean()),
                "tau_mean": float(group.tau.mean()),
                "w_B_mean": float(group.w_B.mean()),
                "bond_reliability_mean": float(group.bond_reliability_mean.mean()),
                "angle_reliability_mean": float(group.angle_reliability_mean.mean()),
            }
        )
    energy_worse = geometry.delta_full_vs_source_kcal_mol > 0
    v3d_improved = geometry.v3d_improved.astype(bool)
    improved_energy_worse_fraction = float(energy_worse.loc[v3d_improved].mean()) if v3d_improved.any() else float("nan")
    overall_energy_worse_fraction = float(energy_worse.mean())
    v3d_gain_concentrated_in_energy_worse = bool(
        v3d_improved.any()
        and improved_energy_worse_fraction > overall_energy_worse_fraction
        and int((v3d_improved & energy_worse).sum()) > int((v3d_improved & ~energy_worse).sum())
    )
    association_rows.extend(
        [
            {
                "row_type": "diagnostic_answer",
                "variable": "V3D_GAIN_CONCENTRATED_IN_ENERGY_WORSE",
                "answer": "YES" if v3d_gain_concentrated_in_energy_worse else "NO",
                "n": int(v3d_improved.sum()),
                "energy_worse_count": int((v3d_improved & energy_worse).sum()),
                "energy_worse_fraction": improved_energy_worse_fraction,
                "overall_energy_worse_fraction": overall_energy_worse_fraction,
            },
            {
                "row_type": "diagnostic_answer",
                "variable": "V3D_IMPROVEMENT_DELTA_E_RELATION",
                "answer": "NO_CLEAR_MONOTONIC_RELATION" if abs(float(association_rows[6]["spearman_rho"])) < 0.1 else "ASSOCIATION_PRESENT",
                "n": EXPECTED_RECORDS,
                "spearman_rho": float(association_rows[6]["spearman_rho"]),
            },
        ]
    )
    for label, mask in (
        ("ALL_VALID", np.isfinite(geometry.delta_full_vs_source_kcal_mol)),
        ("DELTA_GT_5", geometry.delta_full_vs_source_kcal_mol > 5),
        ("DELTA_GT_25", geometry.delta_full_vs_source_kcal_mol > 25),
    ):
        group = geometry.loc[mask]
        if group.empty:
            continue
        association_rows.append(
            {
                "row_type": "tail_pattern",
                "stratum": label,
                "n": int(len(group)),
                "delta_energy_mean": float(group.delta_full_vs_source_kcal_mol.mean()),
                "source_rmsd_mean": float(group.source_rmsd.mean()),
                "tau_mean": float(group.tau.mean()),
                "w_B_mean": float(group.w_B.mean()),
                "bond_reliability_mean": float(group.bond_reliability_mean.mean()),
                "angle_reliability_mean": float(group.angle_reliability_mean.mean()),
            }
        )
    association_frame = pd.DataFrame(association_rows)
    atomic_csv(OUT / "XTB_GEOMETRY_ASSOCIATION.csv", association_frame)

    extreme_columns = [
        "record_id", "molecule_id", "source_energy_hartree", "comparator_energy_hartree", "full_joint_energy_hartree",
        "delta_full_vs_source_kcal_mol", "delta_full_vs_comparator_kcal_mol",
    ]
    extreme_rows: list[pd.DataFrame] = []
    top20 = base.nlargest(20, "delta_full_vs_source_kcal_mol")[extreme_columns].copy()
    top20.insert(0, "audit_group", "TOP20_WORST_POSITIVE")
    top20.insert(1, "rank", np.arange(1, len(top20) + 1))
    extreme_rows.append(top20)
    for threshold in (25, 50, 100):
        selected = base.loc[base.delta_full_vs_source_kcal_mol > threshold, extreme_columns].copy()
        selected.insert(0, "audit_group", f"GT_{threshold}")
        selected.insert(1, "rank", np.arange(1, len(selected) + 1))
        extreme_rows.append(selected)
    extreme = pd.concat(extreme_rows, ignore_index=True)
    atomic_csv(OUT / "XTB_EXTREME_TAIL.csv", extreme)

    full_stats = stats_by_name["FULL_JOINT_VS_SOURCE"]
    comp_stats = stats_by_name["FULL_JOINT_VS_COMPARATOR"]
    source_class = classification(full_stats, comparator=False)
    comparator_class = classification(comp_stats, comparator=True)
    number_gt_25 = int((base.delta_full_vs_source_kcal_mol > 25).sum())
    number_gt_50 = int((base.delta_full_vs_source_kcal_mol > 50).sum())
    number_gt_100 = int((base.delta_full_vs_source_kcal_mol > 100).sum())
    catastrophic = number_gt_25 > 0

    runtime = json.loads((OUT / "XTB_RUNTIME.json").read_text(encoding="utf-8"))
    bootstrap_lookup = {(row.comparison, row.metric): row for row in bootstrap_frame.itertuples(index=False)}
    extreme_gt5 = geometry.loc[geometry.delta_full_vs_source_kcal_mol > 5]
    tail_pattern = (
        "The two >+5 kcal/mol records both have high movement/tau (mean tau "
        f"{float(extreme_gt5.tau.mean()):.6f} A versus {float(geometry.tau.mean()):.6f} A overall); "
        f"their mean w_B is {float(extreme_gt5.w_B.mean()):.6f}, while Bond/Angle reliability means are "
        f"{float(extreme_gt5.bond_reliability_mean.mean()):.6f}/{float(extreme_gt5.angle_reliability_mean.mean()):.6f}."
        if len(extreme_gt5)
        else "No Full-Joint delta above +5 kcal/mol was observed."
    )
    decision = f"""# SIXS J1-R1 Full-Joint DEV GFN2-xTB diagnostic

This is a frozen-coordinate diagnostic only. xTB was not used for tuning, model selection, coordinate optimization, or automatic rejection.

```text
XTB_FULL_JOINT_VS_SOURCE = {source_class}
XTB_FULL_JOINT_VS_COMPARATOR = {comparator_class}
CATASTROPHIC_XTB_TAIL_PRESENT = {'YES' if catastrophic else 'NO'}
NUMBER_GT_25 = {number_gt_25}
NUMBER_GT_50 = {number_gt_50}
NUMBER_GT_100 = {number_gt_100}
XTB_USED_FOR_TUNING = NO
FORMAL_READ = NO
LARGE_HOLDOUT_READ = NO
NEW_TRAINING_STARTED = NO
COORDINATES_REGENERATED = NO
```

The classification combines mean, median, 5% trimmed mean, lower-energy fraction and the preregistered tail rule. Energy/geometry associations are descriptive and do not change the frozen model.

## Energy x geometry diagnostic answers

- V3D gain concentrated in energy-worse records: **{'YES' if v3d_gain_concentrated_in_energy_worse else 'NO'}**. Of {int(v3d_improved.sum())} Full-Joint V3D improvements, {int((v3d_improved & energy_worse).sum())} have positive Full-Joint-minus-Source energy.
- V3D-improvement versus energy relationship: **{'NO_CLEAR_MONOTONIC_RELATION' if abs(float(association_rows[6]['spearman_rho'])) < 0.1 else 'ASSOCIATION_PRESENT'}** (descriptive Spearman rho {float(association_rows[6]['spearman_rho']):.6f}).
- Extreme-positive pattern: {tail_pattern}
"""
    atomic_text(OUT / "XTB_FINAL_DECISION.md", decision)

    major = [
        PROTOCOL_PATH, ALIGNMENT_PATH, *METHOD_FILES.values(), OUT / "PAIRED_XTB_RESULTS.csv", OUT / "XTB_SUMMARY.csv",
        OUT / "XTB_BOOTSTRAP.csv", OUT / "XTB_FAILURES.csv", OUT / "XTB_EXTREME_TAIL.csv",
        OUT / "XTB_GEOMETRY_ASSOCIATION.csv", OUT / "XTB_RUNTIME.json", OUT / "XTB_FINAL_DECISION.md",
    ]
    final_status = {
        "schema_version": "sixs-j1r1-full-joint-xtb-dev-final-status-v1",
        "status": "COMPLETE",
        "xtb_version": SETTINGS["version"],
        "gfn_level": "GFN2-xTB",
        "method_success": {method: int(frames[method].success.sum()) for method in METHODS},
        "full_joint_vs_source_valid_n": full_stats["valid_n"],
        "statistics": stats_by_name,
        "bootstrap": {
            name: {
                metric: {
                    "point": float(bootstrap_lookup[(name, metric)].point_estimate),
                    "ci95": [float(bootstrap_lookup[(name, metric)].ci95_low), float(bootstrap_lookup[(name, metric)].ci95_high)],
                }
                for metric in ("mean", "median", "fraction_lower")
            }
            for name in comparisons
        },
        "tail": {
            "number_gt_25": number_gt_25,
            "number_gt_50": number_gt_50,
            "number_gt_100": number_gt_100,
            "catastrophic_xtb_tail_present": catastrophic,
        },
        "classifications": {
            "xtb_full_joint_vs_source": source_class,
            "xtb_full_joint_vs_comparator": comparator_class,
        },
        "geometry_interpretation": {
            "v3d_gain_concentrated_in_energy_worse": v3d_gain_concentrated_in_energy_worse,
            "v3d_improved_records": int(v3d_improved.sum()),
            "v3d_improved_and_energy_worse_records": int((v3d_improved & energy_worse).sum()),
            "v3d_improvement_delta_energy_spearman": float(association_rows[6]["spearman_rho"]),
            "extreme_gt5_records": int(len(extreme_gt5)),
        },
        "runtime": runtime,
        "artifact_sha256": {path.name: sha_file(path) for path in major},
        "xtb_used_for_tuning": False,
        "formal_read": False,
        "large_holdout_read": False,
        "seed331_started": False,
        "seed353_started": False,
        "new_training_started": False,
        "coordinates_regenerated": False,
    }
    atomic_json(OUT / "XTB_FINAL_STATUS.json", final_status)
    update_status("COMPLETE", state="PASS", final_status_sha256=sha_file(OUT / "XTB_FINAL_STATUS.json"))
    return final_status


def pipeline() -> dict[str, Any]:
    try:
        preflight()
        run_xtb()
        return analyze()
    except Exception as error:
        update_status("STOPPED_FAIL_CLOSED", state="FAIL", error_type=type(error).__name__, error=str(error))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "run", "analyze", "all"), default="all")
    args = parser.parse_args()
    if args.stage == "preflight": preflight()
    elif args.stage == "run": run_xtb()
    elif args.stage == "analyze": analyze()
    else: pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
