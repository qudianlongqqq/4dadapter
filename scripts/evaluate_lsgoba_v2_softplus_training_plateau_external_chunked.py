#!/usr/bin/env python3
"""Crash-isolated per-method PB/V3D evaluation and strict merge."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import types
from pathlib import Path

import pandas as pd
from rdkit import rdBase


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-v2-matched-phase-control307\scripts\evaluate_lsgoba_v2_matched_phase_vs_v2_1_external.py")
STEPS = (12500, 15000, 17500, 20000, 22500)
METHODS = tuple(f"STEP{step}_{stage}" for step in STEPS for stage in ("PROPOSAL", "FINAL"))
REPORT = ROOT / "reports/ecir_mvr/lsgoba_v2_softplus_seed307/training_plateau_evaluation"
ARTIFACT = ROOT / "artifacts/ecir_mvr/lsgoba_v2_softplus_seed307/training_plateau_evaluation"
STATUS = REPORT / "STATUS.json"
FREEZE = ARTIFACT / "COORDINATE_FREEZE.json"
PB_PATH = ARTIFACT / "POSEBUSTERS.parquet"
V3D_PATH = ARTIFACT / "VALIDITY3D.parquet"
ENDPOINTS = ARTIFACT / "ENDPOINT_COMPLETION.json"
CHUNKS = ARTIFACT / "external_chunks"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def load_base() -> types.ModuleType:
    text = SOURCE.read_text(encoding="utf-8")
    text = text.replace("len(pb) != 20000", "len(pb) != 5000 * len(METHODS)")
    text = text.replace("len(v3d) != 20000", "len(v3d) != 5000 * len(METHODS)")
    module = types.ModuleType("softplus_plateau_external_chunk_base")
    module.__file__ = str(SOURCE)
    exec(compile(text, str(SOURCE), "exec"), module.__dict__)
    return module


def run_chunk(method: str) -> int:
    if method not in METHODS:
        raise ValueError(f"unknown frozen method: {method}")
    chunk = CHUNKS / method
    module = load_base()
    module.ROOT = ROOT
    module.METHODS = (method,)
    module.REPORT = chunk
    module.ARTIFACT = chunk
    module.STATUS = chunk / "STATUS.json"
    module.FREEZE = FREEZE
    module.PB_PATH = chunk / "POSEBUSTERS.parquet"
    module.V3D_PATH = chunk / "VALIDITY3D.parquet"
    module.ENDPOINTS = chunk / "ENDPOINT_COMPLETION.json"
    return int(module.main())


def merge() -> int:
    pb_frames = []
    v3d_frames = []
    completions = {}
    for method in METHODS:
        chunk = CHUNKS / method
        completion_path = chunk / "ENDPOINT_COMPLETION.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        pb_path = chunk / "POSEBUSTERS.parquet"
        v3d_path = chunk / "VALIDITY3D.parquet"
        if completion.get("status") != "COMPLETE" or completion.get("methods") != [method]:
            raise RuntimeError(f"external chunk incomplete: {method}")
        if file_sha(pb_path) != completion["posebusters_sha256"] or file_sha(v3d_path) != completion["validity3d_sha256"]:
            raise RuntimeError(f"external chunk hash mismatch: {method}")
        pb_frames.append(pd.read_parquet(pb_path))
        v3d_frames.append(pd.read_parquet(v3d_path))
        completions[method] = {"path": str(completion_path), "sha256": file_sha(completion_path)}
    pb = pd.concat(pb_frames, ignore_index=True, sort=False)
    v3d = pd.concat(v3d_frames, ignore_index=True, sort=False)
    if len(pb) != 50000 or pb.duplicated(["method", "record_id"]).any() or set(pb.method) != set(METHODS):
        raise RuntimeError("merged PoseBusters denominator/identity mismatch")
    if len(v3d) != 50000 or v3d.duplicated(["method", "record_id"]).any() or set(v3d.method) != set(METHODS):
        raise RuntimeError("merged Validity3D denominator/identity mismatch")
    atomic_frame(PB_PATH, pb)
    atomic_frame(V3D_PATH, v3d)
    completion = {
        "schema_version": "lsgoba-v2-softplus-training-plateau-chunked-endpoints-v1",
        "status": "COMPLETE", "records_per_method": 5000, "methods": list(METHODS),
        "posebusters_sha256": file_sha(PB_PATH), "validity3d_sha256": file_sha(V3D_PATH),
        "posebusters_version": importlib.metadata.version("posebusters"),
        "rdkit_version": rdBase.rdkitVersion, "chunks": completions,
        "xtb_stage": "NOT_STARTED_AND_PROHIBITED",
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    }
    atomic_json(ENDPOINTS, completion)
    current = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    current.update({
        "status": "PASS", "stage": "EXTERNAL_ENDPOINTS", "worker_pid": os.getpid(),
        "heartbeat": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "completed_methods": len(METHODS), "expected_methods": len(METHODS),
        "endpoint_completion_sha256": file_sha(ENDPOINTS),
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
        "xtb_started": False, "training_started": False,
    })
    atomic_json(STATUS, current)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("chunk")
    one.add_argument("method", choices=METHODS)
    sub.add_parser("merge")
    args = parser.parse_args()
    return run_chunk(args.method) if args.command == "chunk" else merge()


if __name__ == "__main__":
    raise SystemExit(main())
