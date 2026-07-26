#!/usr/bin/env python3
"""Run GFN2-xTB single points on the frozen mechanism coordinate set."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.lsgo_io import atomic_json, file_sha256
from scripts import run_xtb_singlepoint_lsgo as xtb

OUT = ROOT / "reports/ecir_mvr/lsgo_mechanism"


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False); os.replace(temporary, path)


def prepare(task: dict, position: int) -> dict:
    identity = {"elements": task["elements"], "charge": task["charge"], "uhf": task["uhf"], "coordinate_sha256": task["coordinate_sha256"], "settings": xtb.SETTINGS}
    return {"method": task["condition"], "position": position, "sample_id": task["sample_id"], "coordinates": np.asarray(task["coordinates"], dtype=np.float64), "identity": identity, "identity_sha256": xtb.canonical(identity)}


def main() -> int:
    started = time.time(); freeze_path = OUT / "COORDINATE_FREEZE_MANIFEST.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_XTB" or freeze["xtb_used_for_coordinates"]: raise RuntimeError("coordinate freeze invalid")
    coordinate_path = Path(freeze["coordinate_path"])
    if file_sha256(coordinate_path) != freeze["coordinate_sha256"]: raise RuntimeError("coordinate SHA mismatch")
    if xtb.sha_file(xtb.SETTINGS["executable"]) != xtb.SETTINGS["executable_sha256"]: raise RuntimeError("xTB executable SHA mismatch")
    if xtb.SETTINGS["geometry_optimization"] is not False: raise RuntimeError("xTB optimization must be disabled")
    frozen = torch.load(coordinate_path, map_location="cpu", weights_only=False); raw_tasks = list(frozen["tasks"])
    if frozen["formal_test_records_read"] or frozen["frozen_holdout_records_read"] or len(raw_tasks) != 2441: raise RuntimeError("coordinate denominator/protection changed")
    xtb.OUT = OUT
    prepared = [prepare(task, index) for index, task in enumerate(raw_tasks)]
    # Multiple masked/safety-guarded conditions may be exact coordinate no-ops.
    # Execute each immutable molecular identity once to avoid concurrent cache writes.
    unique = {}
    for task in prepared: unique.setdefault(task["identity_sha256"], task)
    with ThreadPoolExecutor(max_workers=4) as pool:
        unique_results = list(pool.map(xtb.execute, unique.values()))
    lookup = {row["identity_sha256"]: row for row in unique_results}
    results = [lookup[task["identity_sha256"]] | {"method": task["method"], "sample_id": task["sample_id"], "record_index": task["position"]} for task in prepared]
    rows = []
    for task, result in zip(raw_tasks, results, strict=True):
        rows.append({key: task.get(key) for key in ("condition", "seed", "molecule_id", "sample_id", "source_sample_id", "source_index", "reference_index", "flex_bin", "rotatable_bond_count", "heavy_atom_count", "aromatic", "ring", "amide_like", "coordinate_sha256")} | {key: result[key] for key in ("success", "energy_hartree", "exit_code", "timeout", "nonfinite", "runtime_seconds", "identity_sha256", "charge", "uhf")})
    frame = pd.DataFrame(rows)
    if len(frame) != 2441 or int(frame.success.sum()) != 2441: raise RuntimeError(f"incomplete xTB results: {int(frame.success.sum())}/2441")
    result_path = OUT / "per_record/XTB_SINGLE_POINT.parquet"; atomic_frame(result_path, frame)
    execution = frame.groupby("condition", dropna=False).agg(records=("success", "size"), success=("success", "sum"), runtime_seconds=("runtime_seconds", "sum")).reset_index().to_dict("records")
    manifest = {"schema_version": "mcvr-lsgo-mechanism-xtb-singlepoint-v1", "status": "COMPLETED", "settings": xtb.SETTINGS, "records": len(frame), "success": int(frame.success.sum()), "failure": int((~frame.success).sum()), "conditions": execution,
        "result_path": str(result_path), "result_sha256": file_sha256(result_path), "coordinate_freeze_sha256": file_sha256(freeze_path), "runtime_seconds": time.time() - started,
        "geometry_optimization_performed": False, "used_for_training": False, "used_for_checkpoint_or_coordinate_selection": False,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0}
    atomic_json(OUT / "manifests/XTB_SINGLE_POINT_COMPLETE.json", manifest)
    print("LSGO_MECHANISM_XTB_SINGLE_POINT_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
