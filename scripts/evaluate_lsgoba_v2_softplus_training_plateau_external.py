#!/usr/bin/env python3
"""PB/V3D driver for the frozen Softplus training-horizon coordinates."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-v2-matched-phase-control307\scripts\evaluate_lsgoba_v2_matched_phase_vs_v2_1_external.py")
text = SOURCE.read_text(encoding="utf-8")
text = text.replace("len(pb) != 20000", "len(pb) != 5000 * len(METHODS)")
text = text.replace("len(v3d) != 20000", "len(v3d) != 5000 * len(METHODS)")
module = types.ModuleType("softplus_plateau_external_base")
module.__file__ = str(SOURCE)
exec(compile(text, str(SOURCE), "exec"), module.__dict__)

steps = (12500, 15000, 17500, 20000, 22500)
module.ROOT = ROOT
module.METHODS = tuple(f"STEP{step}_{stage}" for step in steps for stage in ("PROPOSAL", "FINAL"))
module.REPORT = ROOT / "reports/ecir_mvr/lsgoba_v2_softplus_seed307/training_plateau_evaluation"
module.ARTIFACT = ROOT / "artifacts/ecir_mvr/lsgoba_v2_softplus_seed307/training_plateau_evaluation"
module.STATUS = module.REPORT / "STATUS.json"
module.FREEZE = module.ARTIFACT / "COORDINATE_FREEZE.json"
module.PB_PATH = module.ARTIFACT / "POSEBUSTERS.parquet"
module.V3D_PATH = module.ARTIFACT / "VALIDITY3D.parquet"
module.ENDPOINTS = module.ARTIFACT / "ENDPOINT_COMPLETION.json"


if __name__ == "__main__":
    raise SystemExit(module.main())
