#!/usr/bin/env python3
"""Run frozen multiseed proposal/final PoseBusters and Validity3D endpoints."""

from __future__ import annotations

import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-v2-matched-phase-control307\scripts\evaluate_lsgoba_v2_matched_phase_vs_v2_1_external.py")
source_text = SOURCE.read_text(encoding="utf-8")
source_text = source_text.replace("len(pb) != 20000", "len(pb) != 5000 * len(METHODS)")
source_text = source_text.replace("len(v3d) != 20000", "len(v3d) != 5000 * len(METHODS)")
external = types.ModuleType("softplus_multiseed_external_base")
external.__file__ = str(SOURCE)
exec(compile(source_text, str(SOURCE), "exec"), external.__dict__)

external.ROOT = ROOT
external.METHODS = tuple(f"SEED{seed}_{stage}" for seed in (307, 331, 353) for stage in ("PROPOSAL", "FINAL"))
external.REPORT = ROOT / "reports/ecir_mvr/lsgoba_v2_softplus_multiseed/final_development_evaluation"
external.ARTIFACT = ROOT / "artifacts/ecir_mvr/lsgoba_v2_softplus_multiseed/final_development_evaluation"
external.STATUS = external.REPORT / "STATUS.json"
external.FREEZE = external.ARTIFACT / "COORDINATE_FREEZE.json"
external.PB_PATH = external.ARTIFACT / "POSEBUSTERS.parquet"
external.V3D_PATH = external.ARTIFACT / "VALIDITY3D.parquet"
external.ENDPOINTS = external.ARTIFACT / "ENDPOINT_COMPLETION.json"

if __name__ == "__main__":
    raise SystemExit(external.main())
