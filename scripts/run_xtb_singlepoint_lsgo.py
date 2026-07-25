#!/usr/bin/env python3
"""Resumable frozen-coordinate GFN2-xTB single-point evaluation for LSGO."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ecir_mvr/learned_geometry"
SETTINGS = {
    "version": "6.7.1", "wsl": "wsl.exe", "distribution": "Ubuntu-22.04",
    "executable": r"E:\tools\xtb\6.7.1-linux-x86_64\bin\xtb",
    "executable_sha256": "debf27a9e0fa4bfb5ca75aafe4b90d8211f08ec2f4a482f375a4987212eaa12a",
    "gfn": 2, "threads": 1, "timeout_seconds": 180,
    "solvent": None, "geometry_optimization": False,
}
CONVERSION = 627.509474


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


def wsl_path(path: str | Path) -> str:
    raw = str(Path(path).resolve())
    if raw.startswith("\\\\?\\"):
        raw = raw[4:]
    parsed = Path(raw)
    return f"/mnt/{parsed.drive.rstrip(':').lower()}{parsed.as_posix().split(':', 1)[1]}"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def make_task(method: str, position: int, molecule: Chem.Mol) -> dict:
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)
    elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    uhf = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    identity = {
        "elements": elements, "charge": charge, "uhf": uhf,
        "coordinate_sha256": sha_bytes(f"float64|{coordinates.shape}|".encode() + coordinates.tobytes()),
        "settings": SETTINGS,
    }
    return {
        "method": method, "position": position, "sample_id": molecule.GetProp("sample_id"),
        "coordinates": coordinates, "identity": identity, "identity_sha256": canonical(identity),
    }


def execute(task: dict) -> dict:
    cache = OUT / f"per_record/xtb/cache/{task['identity_sha256']}.json"
    if cache.is_file():
        result = json.loads(cache.read_text(encoding="utf-8"))
    else:
        work = OUT / f"logs/xtb_work/{task['identity_sha256']}"
        work.mkdir(parents=True, exist_ok=True)
        xyz = work / "input.xyz"
        lines = [str(len(task["identity"]["elements"])), "frozen coordinates; GFN2-xTB single-point; no optimization"]
        for element, coordinate in zip(task["identity"]["elements"], task["coordinates"], strict=True):
            lines.append(f"{element} {coordinate[0]:.12f} {coordinate[1]:.12f} {coordinate[2]:.12f}")
        xyz.write_text("\n".join(lines) + "\n", encoding="ascii")
        command = [
            SETTINGS["wsl"], "-d", SETTINGS["distribution"], "--cd", wsl_path(work),
            "--exec", "/usr/bin/env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
            wsl_path(SETTINGS["executable"]), "input.xyz", "--gfn", "2",
            "--chrg", str(task["identity"]["charge"]), "--uhf", str(task["identity"]["uhf"]),
        ]
        if any(flag in command for flag in ("--opt", "--ohess", "--md")):
            raise RuntimeError("geometry-changing xTB option detected")
        started = time.time()
        timed_out = False
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=SETTINGS["timeout_seconds"],
                encoding="utf-8", errors="replace",
            )
            code = process.returncode
            text = (process.stdout or "") + "\n" + (process.stderr or "")
        except subprocess.TimeoutExpired as error:
            timed_out, code = True, None
            text = str(error.stdout or "") + "\n" + str(error.stderr or "")
        matches = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", text, re.I)
        energy = float(matches[-1]) if matches else None
        success = code == 0 and energy is not None and math.isfinite(energy)
        result = {
            "identity_sha256": task["identity_sha256"], "success": success,
            "energy_hartree": energy, "exit_code": code, "timeout": timed_out,
            "runtime_seconds": time.time() - started,
            "nonfinite": energy is not None and not math.isfinite(energy),
            "geometry_optimization_performed": False,
            "stderr_tail": "" if success else text[-1000:],
        }
        atomic_json(cache, result)
        if success:
            shutil.rmtree(work, ignore_errors=True)
    return {
        "method": task["method"], "candidate": "primary", "record_index": task["position"],
        "sample_id": task["sample_id"], "charge": task["identity"]["charge"],
        "uhf": task["identity"]["uhf"],
        "elements_sha256": sha_bytes("|".join(task["identity"]["elements"]).encode()),
        "coordinate_sha256": task["identity"]["coordinate_sha256"], **result,
    }


def energy_summary(method: str, frame: pd.DataFrame, source: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    paired = frame.set_index("sample_id").loc[source.sample_id].reset_index()
    source_values = source.set_index("sample_id").loc[paired.sample_id].energy_hartree.to_numpy()
    valid = paired.success.to_numpy(dtype=bool) & np.isfinite(source_values)
    delta = np.full(len(paired), np.nan)
    delta[valid] = (paired.energy_hartree.to_numpy()[valid] - source_values[valid]) * CONVERSION
    paired["delta_energy_kcal_mol"] = delta
    finite = delta[np.isfinite(delta)]
    positive = finite[finite > 0]
    summary = {
        "method": method, "records": len(paired), "paired_success": int(len(finite)),
        "failure": int(len(paired) - len(finite)), "timeout": int(paired.timeout.sum()),
        "nonfinite": int(paired.nonfinite.sum()), "mean_delta_energy_kcal_mol": float(np.mean(finite)),
        "median_delta_energy_kcal_mol": float(np.median(finite)),
        "improved_fraction": float(np.mean(finite < 0)),
        "p75": float(np.quantile(finite, .75)), "p90": float(np.quantile(finite, .90)),
        "p95": float(np.quantile(finite, .95)), "p99": float(np.quantile(finite, .99)),
        "maximum": float(np.max(finite)),
        "positive_tail_mean": float(np.mean(positive)) if len(positive) else 0.0,
    }
    return summary, paired


def main() -> int:
    freeze_path = OUT / "COORDINATE_FREEZE_MANIFEST.json"
    pb_path = OUT / "manifests/POSEBUSTERS_COMPLETE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    pb = json.loads(pb_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN" or pb.get("status") != "COMPLETED":
        raise RuntimeError("xTB locked until coordinate freeze and PoseBusters completion")
    if SETTINGS["geometry_optimization"] is not False:
        raise RuntimeError("xTB geometry optimization must remain disabled")
    if sha_file(SETTINGS["executable"]) != SETTINGS["executable_sha256"]:
        raise RuntimeError("xTB executable SHA changed")
    started = time.time()
    frames: dict[str, pd.DataFrame] = {}
    execution: list[dict] = []
    for export in freeze["exports"]:
        method = str(export["method"])
        if sha_file(export["sdf_path"]) != export["sdf_sha256"]:
            raise RuntimeError(f"frozen SDF changed: {method}")
        target = OUT / f"per_record/xtb/{method}__primary.parquet"
        if target.is_file():
            frame = pd.read_parquet(target)
        else:
            molecules = [
                molecule for molecule in Chem.ForwardSDMolSupplier(export["sdf_path"], sanitize=False, removeHs=False)
                if molecule is not None
            ]
            if len(molecules) != 600:
                raise RuntimeError(f"SDF denominator changed: {method}")
            tasks = [make_task(method, index, molecule) for index, molecule in enumerate(molecules)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                frame = pd.DataFrame(pool.map(execute, tasks)).sort_values("record_index")
            atomic_frame(target, frame)
        required = {"sample_id", "method", "success", "energy_hartree", "identity_sha256", "coordinate_sha256"}
        if len(frame) != 600 or frame.sample_id.nunique() != 600 or not required.issubset(frame.columns):
            raise RuntimeError(f"incomplete xTB result: {method}")
        frames[method] = frame
        row = {
            "method": method, "records": 600, "success": int(frame.success.sum()),
            "failure": int((~frame.success).sum()), "timeout": int(frame.timeout.sum()),
            "nonfinite": int(frame.nonfinite.sum()), "runtime_seconds": float(frame.runtime_seconds.sum()),
        }
        execution.append(row)
        print(json.dumps(row), flush=True)

    source = frames["Source"]
    summaries: list[dict] = []
    bindings: list[dict] = []
    for method, frame in frames.items():
        if method == "Source":
            continue
        summary, paired = energy_summary(method, frame, source)
        paired_path = OUT / f"per_record/xtb/PAIRED_DELTA_{method}__primary.parquet"
        atomic_frame(paired_path, paired)
        summaries.append(summary)
        bindings.append({
            "method": method, "result_path": str(paired_path), "result_sha256": sha_file(paired_path),
        })
    pd.DataFrame(execution).to_csv(OUT / "tables/XTB_EXECUTION_SUMMARY.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "tables/XTB_PAIRED_SUMMARY.csv", index=False)
    payload = {
        "schema_version": "mcvr-lsgo-xtb-singlepoint-v1", "status": "COMPLETED",
        "settings": SETTINGS, "energy_conversion_kcal_per_mol": CONVERSION,
        "coordinate_freeze_sha256": sha_file(freeze_path), "posebusters_manifest_sha256": sha_file(pb_path),
        "execution": execution, "summaries": summaries, "bindings": bindings,
        "runtime_seconds": time.time() - started,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
        "full10k_used_for_tuning": False, "used_for_model_selection": False,
        "xtb_geometry_optimization_run": False,
    }
    atomic_json(OUT / "manifests/XTB_SINGLE_POINT_COMPLETE.json", payload)
    print("LSGO_XTB_SINGLE_POINT_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
