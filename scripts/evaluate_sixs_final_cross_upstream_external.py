#!/usr/bin/env python
"""10K denominator adapter for the frozen SIXS-U cross-upstream SDFs."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-v2-softplus-multiseed\scripts\evaluate_final_softplus_v2_cross_upstream_external.py")
ASSET = Path(r"E:\3dconformergenerationcode\dataset\sixs_final_cross_upstream_unrestricted")
REPORT = ROOT / "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted"


def methods(upstream: str) -> tuple[str, ...]:
    prefix = upstream.upper()
    return (f"{prefix}_RAW", *(f"{prefix}_SIXS_U_SEED{seed}" for seed in (307, 331, 353)))


def paths(upstream: str) -> dict[str, Path]:
    report = REPORT / upstream
    artifact = ASSET / upstream
    return {
        "report": report,
        "artifact": artifact,
        "status": report / "STATUS.json",
        "freeze": artifact / "COORDINATE_FREEZE.json",
        "pb": artifact / "POSEBUSTERS.parquet",
        "v3d": artifact / "VALIDITY3D.parquet",
        "endpoints": artifact / "ENDPOINT_COMPLETION.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, choices=("avgflow", "ditmc"))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("legacy_cross_external", OLD)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.paths = paths
    module.methods = methods
    import sys
    sys.argv = [str(OLD), "--upstream", args.upstream]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
