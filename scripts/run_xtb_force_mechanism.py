#!/usr/bin/env python3
"""Audit xTB --grad and evaluate frozen Source/BA force records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.lsgo_io import atomic_json, file_sha256
from scripts.run_xtb_singlepoint_lsgo import SETTINGS, sha_bytes, wsl_path

OUT = ROOT / "reports/ecir_mvr/lsgo_mechanism"
CONFIG = yaml.safe_load((ROOT / "configs/ecir_mvr_lsgo_mechanism.yaml").read_text(encoding="utf-8"))
BOHR_ANGSTROM = 0.529177210903


def atomic_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False); os.replace(temporary, path)


def command(work: Path, task: dict, gradient: bool):
    xyz = work / "input.xyz"; lines = [str(len(task["elements"])), "frozen mechanism coordinate; no optimization"]
    for element, coordinate in zip(task["elements"], task["coordinates"], strict=True): lines.append(f"{element} {coordinate[0]:.12f} {coordinate[1]:.12f} {coordinate[2]:.12f}")
    xyz.write_text("\n".join(lines) + "\n", encoding="ascii")
    result = [SETTINGS["wsl"], "-d", SETTINGS["distribution"], "--cd", wsl_path(work), "--exec", "/usr/bin/env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", wsl_path(SETTINGS["executable"]), "input.xyz", "--gfn", "2", "--chrg", str(task["charge"]), "--uhf", str(task["uhf"])]
    if gradient: result.append("--grad")
    if any(flag in result for flag in ("--opt", "--ohess", "--md")): raise RuntimeError("geometry changing option detected")
    return result


def parse_gradient(path: Path, coordinates: np.ndarray):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(); numeric3, coordinate4 = [], []
    for line in lines:
        fields = line.split()
        if len(fields) == 3:
            try: numeric3.append([float(value.replace("D", "E")) for value in fields])
            except ValueError: pass
        elif len(fields) == 4:
            try:
                values = [float(value.replace("D", "E")) for value in fields[:3]]
                if fields[3].isalpha(): coordinate4.append(values)
            except ValueError: pass
    atom_count = len(coordinates)
    if len(numeric3) < atom_count: raise RuntimeError(f"gradient rows missing: {len(numeric3)}/{atom_count}")
    gradient = np.asarray(numeric3[-atom_count:], dtype=np.float64)
    order_error = np.nan
    if len(coordinate4) >= atom_count:
        stored = np.asarray(coordinate4[:atom_count], dtype=np.float64)
        order_error = float(np.max(np.abs(stored - coordinates / BOHR_ANGSTROM)))
    return gradient, order_error


def execute_gradient(task: dict, suffix="main"):
    identity = hashlib.sha256((task["coordinate_sha256"] + "|grad|" + suffix).encode()).hexdigest(); work = OUT / "logs/xtb_gradient" / identity; work.mkdir(parents=True, exist_ok=True)
    started = time.time(); process = subprocess.run(command(work, task, True), capture_output=True, text=True, timeout=180, encoding="utf-8", errors="replace")
    text = (process.stdout or "") + "\n" + (process.stderr or ""); matches = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", text, re.I)
    gradient_path = work / "gradient"; energy = float(matches[-1]) if matches else None
    if process.returncode or energy is None or not gradient_path.is_file(): raise RuntimeError(f"xTB gradient failed: {task['sample_id']} {text[-1000:]}")
    gradient, order_error = parse_gradient(gradient_path, np.asarray(task["coordinates"], dtype=np.float64))
    return {"energy_hartree": energy, "gradient_hartree_per_bohr": gradient.tolist(), "finite": bool(np.isfinite(gradient).all()), "coordinate_order_max_error_bohr": order_error, "runtime_seconds": time.time() - started, "exit_code": process.returncode}


def execute_energy(task: dict, coordinates: np.ndarray, label: str):
    local = dict(task); local["coordinates"] = np.asarray(coordinates).tolist(); local["coordinate_sha256"] = sha_bytes(np.asarray(coordinates, dtype=np.float64).tobytes())
    identity = hashlib.sha256((local["coordinate_sha256"] + "|fd|" + label).encode()).hexdigest(); work = OUT / "logs/xtb_gradient_fd" / identity; work.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(command(work, local, False), capture_output=True, text=True, timeout=180, encoding="utf-8", errors="replace")
    text = (process.stdout or "") + "\n" + (process.stderr or ""); matches = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", text, re.I)
    if process.returncode or not matches: raise RuntimeError("xTB FD singlepoint failed")
    return float(matches[-1])


def main():
    freeze = json.loads((OUT / "COORDINATE_FREEZE_MANIFEST.json").read_text(encoding="utf-8")); coordinate_path = Path(freeze["coordinate_path"])
    if freeze["status"] != "FROZEN_BEFORE_XTB" or file_sha256(coordinate_path) != freeze["coordinate_sha256"]: raise RuntimeError("coordinate freeze invalid")
    if file_sha256(SETTINGS["executable"]) != SETTINGS["executable_sha256"]: raise RuntimeError("xTB SHA mismatch")
    tasks = torch.load(coordinate_path, map_location="cpu", weights_only=False)["tasks"]
    sources = [row for row in tasks if row["condition"] == "Source" and int(row["source_index"]) == int(CONFIG["force"]["source_index"])]
    selected = []
    for flex_bin in CONFIG["dataset"]["flex_bins"]:
        local = sorted((row for row in sources if row["flex_bin"] == flex_bin), key=lambda row: hashlib.sha256(row["molecule_id"].encode()).hexdigest())
        selected.extend(local[:int(CONFIG["force"]["diagnostic_records_per_flex_bin"])])
    subset = {"schema_version": "mcvr-lsgo-mechanism-force-subset-v1", "status": "FROZEN_BEFORE_FORCE_ACCESS", "selection": "SHA-ranked molecule within preregistered flex bin; Source index 0", "records": [{"molecule_id": row["molecule_id"], "sample_id": row["sample_id"], "flex_bin": row["flex_bin"], "coordinate_sha256": row["coordinate_sha256"]} for row in selected], "count": len(selected), "formal_test_records_read": 0, "frozen_holdout_records_read": 0}
    atomic_json(OUT / "manifests/FORCE_SUBSET_FREEZE.json", subset)
    ba_lookup = {row["sample_id"]: row for row in tasks if row["condition"] == f"BA_seed{CONFIG['representative_ba_seed']}"}
    force_tasks = selected + [ba_lookup[row["sample_id"]] for row in selected]
    # Interface feasibility on the first pre-frozen Source.
    first = selected[0]; first_result = execute_gradient(first, "repeat_a"); repeat = execute_gradient(first, "repeat_b")
    single = pd.read_parquet(OUT / "per_record/XTB_SINGLE_POINT.parquet"); expected_energy = float(single[(single.condition == "Source") & (single.sample_id == first["sample_id"])].energy_hartree.iloc[0])
    energy_error = abs(first_result["energy_hartree"] - expected_energy); repeat_error = float(np.max(np.abs(np.asarray(first_result["gradient_hartree_per_bohr"]) - np.asarray(repeat["gradient_hartree_per_bohr"]))))
    xyz = np.asarray(first["coordinates"], dtype=np.float64); analytic = np.asarray(first_result["gradient_hartree_per_bohr"]) / BOHR_ANGSTROM; fd_rows = []
    step = float(CONFIG["force"]["finite_difference_step_angstrom"])
    for flat_index in range(3):
        plus, minus = xyz.copy().reshape(-1), xyz.copy().reshape(-1); plus[flat_index] += step; minus[flat_index] -= step
        numeric = (execute_energy(first, plus.reshape(xyz.shape), f"plus{flat_index}") - execute_energy(first, minus.reshape(xyz.shape), f"minus{flat_index}")) / (2 * step)
        predicted = float(analytic.reshape(-1)[flat_index]); relative = abs(numeric - predicted) / max(abs(numeric), abs(predicted), 1e-10)
        fd_rows.append({"flat_index": flat_index, "numeric_gradient_hartree_per_angstrom": numeric, "cli_gradient_hartree_per_angstrom": predicted, "relative_error": relative})
    tolerances = CONFIG["force"]
    feasible = energy_error <= float(tolerances["energy_identity_tolerance_hartree"]) and repeat_error <= float(tolerances["repeatability_tolerance_hartree_per_bohr"]) and first_result["finite"] and (np.isnan(first_result["coordinate_order_max_error_bohr"]) or first_result["coordinate_order_max_error_bohr"] <= 1e-8) and max(row["relative_error"] for row in fd_rows) <= float(tolerances["finite_difference_relative_tolerance"])
    protocol = {"schema_version": "mcvr-lsgo-mechanism-xtb-force-protocol-v1", "status": "AVAILABLE" if feasible else "XTB_FORCE_DIAGNOSTIC_UNAVAILABLE", "xTB_version": SETTINGS["version"], "xTB_sha256": SETTINGS["executable_sha256"], "command_mode": "--grad; no optimization", "gradient_units": "Hartree/Bohr (energy gradient); force = -gradient", "atom_order": "input XYZ cache order", "energy_identity_error_hartree": energy_error, "gradient_repeat_max_error_hartree_per_bohr": repeat_error, "coordinate_order_max_error_bohr": first_result["coordinate_order_max_error_bohr"], "finite_difference": fd_rows, "formal_test_records_read": 0, "frozen_holdout_records_read": 0}
    atomic_json(OUT / "manifests/XTB_FORCE_PROTOCOL.json", protocol)
    if not feasible:
        print("XTB_FORCE_DIAGNOSTIC_UNAVAILABLE"); return 0
    with ThreadPoolExecutor(max_workers=4) as pool: results = list(pool.map(execute_gradient, force_tasks))
    rows = []
    for task, result in zip(force_tasks, results, strict=True):
        rows.append({"condition": task["condition"], "molecule_id": task["molecule_id"], "sample_id": task["sample_id"], "flex_bin": task["flex_bin"], "rotatable_bond_count": task["rotatable_bond_count"], "coordinate_sha256": task["coordinate_sha256"], **result})
    frame = pd.DataFrame(rows); path = OUT / "per_record/XTB_FORCE.parquet"; atomic_frame(path, frame)
    atomic_json(OUT / "manifests/XTB_FORCE_COMPLETE.json", {"schema_version": "mcvr-lsgo-mechanism-xtb-force-v1", "status": "COMPLETED", "records": len(frame), "result_path": str(path), "result_sha256": file_sha256(path), "force_subset_sha256": file_sha256(OUT / "manifests/FORCE_SUBSET_FREEZE.json"), "formal_test_records_read": 0, "frozen_holdout_records_read": 0})
    print("LSGO_MECHANISM_XTB_FORCE_COMPLETED")
    return 0


if __name__ == "__main__": raise SystemExit(main())
