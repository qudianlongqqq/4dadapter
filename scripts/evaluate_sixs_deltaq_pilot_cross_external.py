#!/usr/bin/env python
"""External-validity endpoint adapter for one DeltaQ diagnostic method."""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-v2-softplus-multiseed\scripts\evaluate_final_softplus_v2_cross_upstream_external.py")
ASSET = Path(r"E:\3dconformergenerationcode\dataset\sixs_deltaq_single_seed_pilot")
REPORT = ROOT / "reports/ecir_mvr/sixs_deltaq_single_seed_pilot/cross_internal"

def methods(upstream: str):
    prefix=upstream.upper()
    return (f"{prefix}_RAW", f"{prefix}_SIXS_U_SEED307")

def paths(upstream: str):
    artifact=ASSET/upstream; report=REPORT/upstream
    return {"report":report,"artifact":artifact,"status":report/"STATUS.json","freeze":artifact/"COORDINATE_FREEZE.json",
            "pb":artifact/"POSEBUSTERS.parquet","v3d":artifact/"VALIDITY3D.parquet","endpoints":artifact/"ENDPOINT_COMPLETION.json"}

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--upstream",required=True,choices=("avgflow","ditmc")); args=p.parse_args()
    spec=importlib.util.spec_from_file_location("deltaq_cross_external_legacy",OLD); module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None; spec.loader.exec_module(module)
    module.ROOT=ROOT; module.paths=paths; module.methods=methods
    sys.argv=[str(OLD),"--upstream",args.upstream]
    return int(module.main())

if __name__ == "__main__": raise SystemExit(main())
