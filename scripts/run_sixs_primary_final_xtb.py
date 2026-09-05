#!/usr/bin/env python
"""Resumable frozen-coordinate GFN2-xTB single-point evaluation for final SIXS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()
import scripts.run_sixs_j1r1_full_joint_xtb_dev as engine


EXPECTED_RECORDS = 5000


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_sdf(path: Path) -> list[Chem.Mol]:
    values = list(Chem.ForwardSDMolSupplier(str(path), sanitize=False, removeHs=False))
    if len(values) != EXPECTED_RECORDS or any(value is None for value in values):
        raise RuntimeError(f"invalid xTB SDF denominator: {path}")
    return [value for value in values if value is not None]


def update_status(path: Path, stage: str, **extra: Any) -> None:
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update({"schema_version": "sixs-primary-final-xtb-status-v1", "status": "RUNNING", "stage": stage, "pid": os.getpid(), "geometry_optimization_performed": False, **extra})
    atomic_json(path, current)


def main(args: argparse.Namespace) -> int:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    settings = dict(protocol["xtb_single_point"])
    engine.OUT = args.output_dir
    engine.SETTINGS = {
        "version": settings["version"],
        "wsl": settings["wsl_executable"],
        "distribution": settings["wsl_distribution"],
        "executable": settings["executable"],
        "executable_sha256": settings["executable_sha256"],
        "gfn": 2,
        "threads": 1,
        "workers": int(settings["workers"]),
        "timeout_seconds": int(settings["timeout_seconds"]),
        "solvent": None,
        "geometry_optimization": False,
    }
    if engine.sha_file(engine.SETTINGS["executable"]) != engine.SETTINGS["executable_sha256"]:
        raise RuntimeError("frozen xTB executable SHA256 mismatch")
    method_ids = ["source"] + [method["id"] for method in protocol["model_methods"]] + ["mmff94s"]
    source_records = pd.read_parquet(args.coordinate_dir / "methods/source/PER_RECORD.parquet")
    ids = source_records.record_id.astype(str).tolist()
    if len(ids) != EXPECTED_RECORDS or len(set(ids)) != EXPECTED_RECORDS:
        raise RuntimeError("xTB record identity denominator mismatch")
    source_molecules = load_sdf(args.coordinate_dir / "methods/source/COORDINATES.sdf")
    if [engine.record_id(molecule) for molecule in source_molecules] != ids:
        raise RuntimeError("Source SDF record order mismatch")
    elements = [engine.elements(molecule) for molecule in source_molecules]
    charges = [engine.formal_charge(molecule) for molecule in source_molecules]
    radicals = [engine.radical_electrons(molecule) for molecule in source_molecules]
    status_path = args.report_dir / "XTB_RUN_STATUS.json"
    summaries = []
    for method_index, method in enumerate(method_ids):
        result_path = args.output_dir / f"{method.upper()}_XTB.csv"
        if result_path.is_file():
            frame = pd.read_csv(result_path)
            if len(frame) != EXPECTED_RECORDS or frame.record_id.astype(str).tolist() != ids:
                raise RuntimeError(f"stale/incomplete xTB result: {method}")
        else:
            molecules = load_sdf(args.coordinate_dir / f"methods/{method}/COORDINATES.sdf")
            if [engine.record_id(molecule) for molecule in molecules] != ids:
                raise RuntimeError(f"SDF identity mismatch: {method}")
            tasks = []
            for index, (record_id, molecule_id, molecule) in enumerate(zip(ids, source_records.molecule_id.astype(str), molecules, strict=True)):
                if engine.elements(molecule) != elements[index] or engine.formal_charge(molecule) != charges[index] or engine.radical_electrons(molecule) != radicals[index]:
                    raise RuntimeError(f"xTB topology/charge mismatch: {method}/{record_id}")
                coordinates = np.ascontiguousarray(engine.coordinates(molecule), dtype=np.float64)
                coordinate_hash = engine.sha_bytes(f"float64|{coordinates.shape}|".encode() + coordinates.tobytes())
                identity = {
                    "elements": elements[index],
                    "charge": charges[index],
                    "uhf": radicals[index],
                    "coordinate_sha256": coordinate_hash,
                    "settings": {key: engine.SETTINGS[key] for key in engine.SETTINGS if key != "workers"},
                }
                tasks.append({
                    "method": method,
                    "record_index": index,
                    "record_id": record_id,
                    "molecule_id": molecule_id,
                    "coordinates": coordinates,
                    "identity": identity,
                    "identity_sha256": engine.canonical_sha(identity),
                })
            rows = []
            update_status(status_path, f"RUNNING_{method}", method_index=method_index + 1, method_total=len(method_ids), completed=0, total=EXPECTED_RECORDS)
            with ThreadPoolExecutor(max_workers=engine.SETTINGS["workers"]) as pool:
                futures = [pool.submit(engine.execute, task) for task in tasks]
                for count, future in enumerate(as_completed(futures), start=1):
                    rows.append(future.result())
                    if count % 100 == 0 or count == EXPECTED_RECORDS:
                        update_status(status_path, f"RUNNING_{method}", method_index=method_index + 1, method_total=len(method_ids), completed=count, total=EXPECTED_RECORDS)
                        print(f"PRIMARY_FINAL_XTB {method} {count}/{EXPECTED_RECORDS}", flush=True)
            frame = pd.DataFrame(rows).sort_values("record_index", kind="stable").reset_index(drop=True)
            atomic_csv(result_path, frame)
        summaries.append({
            "method": method,
            "attempted": len(frame),
            "success": int(frame.success.sum()),
            "failures": int((~frame.success.astype(bool)).sum()),
        })
    source_energy = pd.read_csv(args.output_dir / "SOURCE_XTB.csv")[["record_id", "energy_hartree", "success"]].rename(columns={"energy_hartree": "source_energy_hartree", "success": "source_success"})
    for method in method_ids[1:]:
        frame = pd.read_csv(args.output_dir / f"{method.upper()}_XTB.csv").merge(source_energy, on="record_id", validate="one_to_one")
        frame["delta_e_kcal_mol"] = (frame.energy_hartree - frame.source_energy_hartree) * 627.509474
        frame["matched_success"] = frame.success.astype(bool) & frame.source_success.astype(bool) & np.isfinite(frame.delta_e_kcal_mol)
        atomic_csv(args.output_dir / f"{method.upper()}_DELTA_VS_SOURCE.csv", frame)
    atomic_json(args.report_dir / "XTB_FINAL_STATUS.json", {"status": "PASS", "methods": summaries, "records_per_method": EXPECTED_RECORDS, "geometry_optimization_performed": False, "protocol_sha256": engine.sha_file(args.protocol)})
    update_status(status_path, "COMPLETE", status="PASS", methods=len(method_ids), records_per_method=EXPECTED_RECORDS)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--coordinate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("protocol", "coordinate_dir", "output_dir", "report_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
