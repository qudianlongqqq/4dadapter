#!/usr/bin/env python3
"""Run frozen Softplus proposal/final PB and V3D in external-validity."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-v2-matched-phase-control307\scripts\evaluate_lsgoba_v2_matched_phase_vs_v2_1_external.py")
source_text = SOURCE.read_text(encoding="utf-8")
source_text = source_text.replace("len(pb) != 20000", "len(pb) != 5000 * len(METHODS)")
source_text = source_text.replace("len(v3d) != 20000", "len(v3d) != 5000 * len(METHODS)")
module = types.ModuleType("softplus_external_base")
module.__file__ = str(SOURCE)
exec(compile(source_text, str(SOURCE), "exec"), module.__dict__)

module.ROOT = ROOT
module.METHODS = ("SOFTPLUS_PROPOSAL", "SOFTPLUS_FINAL")
module.REPORT = ROOT / "reports/ecir_mvr/lsgoba_v2_softplus_seed307/final_development_evaluation"
module.ARTIFACT = ROOT / "artifacts/ecir_mvr/lsgoba_v2_softplus_seed307/final_development_evaluation"
module.STATUS = module.REPORT / "STATUS.json"
module.FREEZE = module.ARTIFACT / "COORDINATE_FREEZE.json"
module.PB_PATH = module.ARTIFACT / "POSEBUSTERS.parquet"
module.V3D_PATH = module.ARTIFACT / "VALIDITY3D.parquet"
module.ENDPOINTS = module.ARTIFACT / "ENDPOINT_COMPLETION.json"


if __name__ == "__main__":
    raise SystemExit(module.main())
