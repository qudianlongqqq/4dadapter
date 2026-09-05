#!/usr/bin/env python
"""Frozen GFN2-xTB single points for one closure-only coordinate method."""

from __future__ import annotations

import argparse
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

EXPECTED = 5000


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


def molecules(path: Path) -> list[Chem.Mol]:
    values = list(Chem.ForwardSDMolSupplier(str(path), sanitize=False, removeHs=False))
    if len(values) != EXPECTED or any(value is None for value in values):
        raise RuntimeError("closure xTB SDF denominator mismatch")
    return [value for value in values if value is not None]


def main(args: argparse.Namespace) -> int:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    settings = protocol["xtb_single_point"]
    engine.OUT = args.cache_dir
    engine.SETTINGS = {
        "version": settings["version"], "wsl": settings["wsl_executable"],
        "distribution": settings["wsl_distribution"], "executable": settings["executable"],
        "executable_sha256": settings["executable_sha256"], "gfn": 2, "threads": 1,
        "workers": int(settings["workers"]), "timeout_seconds": int(settings["timeout_seconds"]),
        "solvent": None, "geometry_optimization": False,
    }
    if engine.sha_file(engine.SETTINGS["executable"]) != engine.SETTINGS["executable_sha256"]:
        raise RuntimeError("frozen xTB executable hash mismatch")
    records = pd.read_parquet(args.records)
    ids = records.record_id.astype(str).tolist()
    if len(ids) != EXPECTED or len(set(ids)) != EXPECTED:
        raise RuntimeError("closure xTB record denominator mismatch")
    mols = molecules(args.sdf)
    if [engine.record_id(mol) for mol in mols] != ids:
        raise RuntimeError("closure xTB record order mismatch")
    result_path = args.output_dir / f"{args.method.upper()}_XTB.csv"
    if result_path.is_file():
        result = pd.read_csv(result_path)
        if len(result) != EXPECTED or result.record_id.astype(str).tolist() != ids:
            raise RuntimeError("stale closure xTB result")
    else:
        tasks = []
        molecule_ids = records.molecule_id.astype(str).tolist()
        for index, (record_id, molecule_id, mol) in enumerate(zip(ids, molecule_ids, mols, strict=True)):
            xyz = np.ascontiguousarray(engine.coordinates(mol), dtype=np.float64)
            coordinate_hash = engine.sha_bytes(f"float64|{xyz.shape}|".encode() + xyz.tobytes())
            identity = {
                "elements": engine.elements(mol), "charge": engine.formal_charge(mol),
                "uhf": engine.radical_electrons(mol), "coordinate_sha256": coordinate_hash,
                "settings": {key: engine.SETTINGS[key] for key in engine.SETTINGS if key != "workers"},
            }
            tasks.append({
                "method": args.method, "record_index": index, "record_id": record_id,
                "molecule_id": molecule_id, "coordinates": xyz, "identity": identity,
                "identity_sha256": engine.canonical_sha(identity),
            })
        # Reference-context materialization intentionally repeats reference0 for
        # the two frozen records of each molecule.  Those rows have the same
        # scientific xTB identity and therefore share one cache/work directory.
        # Execute each identity once so concurrent workers cannot race on that
        # directory, then fan the immutable result back out to every record.
        unique_tasks: dict[str, dict[str, Any]] = {}
        for task in tasks:
            unique_tasks.setdefault(str(task["identity_sha256"]), task)
        results_by_identity: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=engine.SETTINGS["workers"]) as pool:
            futures = {
                pool.submit(engine.execute, task): identity_sha256
                for identity_sha256, task in unique_tasks.items()
            }
            for count, future in enumerate(as_completed(futures), start=1):
                results_by_identity[futures[future]] = future.result()
                if count % 100 == 0 or count == len(unique_tasks):
                    atomic_json(args.output_dir / f"{args.method.upper()}_XTB_STATUS.json", {
                        "status": "RUNNING", "stage": "GFN2_XTB_SINGLE_POINT", "pid": os.getpid(),
                        "method": args.method, "completed_unique_identities": count,
                        "total_unique_identities": len(unique_tasks), "total_records": EXPECTED,
                        "geometry_optimization": False,
                    })
        rows = []
        for task in tasks:
            row = dict(results_by_identity[str(task["identity_sha256"])])
            row.update({
                "method": task["method"], "record_index": task["record_index"],
                "record_id": task["record_id"], "molecule_id": task["molecule_id"],
            })
            rows.append(row)
        result = pd.DataFrame(rows).sort_values("record_index", kind="stable").reset_index(drop=True)
        atomic_csv(result_path, result)
    source = pd.read_csv(args.source_xtb)[["record_id", "energy_hartree", "success"]].rename(
        columns={"energy_hartree": "source_energy_hartree", "success": "source_success"}
    )
    merged = result.merge(source, on="record_id", validate="one_to_one")
    merged["delta_e_kcal_mol"] = (merged.energy_hartree - merged.source_energy_hartree) * 627.509474
    merged["matched_success"] = (
        merged.success.astype(bool) & merged.source_success.astype(bool) & np.isfinite(merged.delta_e_kcal_mol)
    )
    delta_path = args.output_dir / f"{args.method.upper()}_DELTA_VS_SOURCE.csv"
    atomic_csv(delta_path, merged)
    atomic_json(args.output_dir / f"{args.method.upper()}_XTB_STATUS.json", {
        "status": "PASS", "method": args.method, "attempted": EXPECTED,
        "success": int(result.success.astype(bool).sum()),
        "matched_success": int(merged.matched_success.sum()), "geometry_optimization": False,
        "unique_scientific_identities": int(result.identity_sha256.astype(str).nunique()),
        "duplicate_identity_execution_policy": "EXECUTE_ONCE_THEN_FAN_OUT_TO_FROZEN_RECORDS",
        "result_sha256": engine.sha_file(result_path), "delta_sha256": engine.sha_file(delta_path),
    })
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--source-xtb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("protocol", "sdf", "records", "source_xtb", "output_dir", "cache_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
