"""Isolated two-step engineering check for the frozen DeltaQ overfit entrypoint.

This wrapper never edits the frozen JSON configuration and its outputs are not
scientific evidence.  It redirects all artifacts, substitutes an in-memory
2-molecule/2-step runtime, and then calls the production ``overfit()`` function.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "sixs_deltaq_production_runner", SCRIPTS / "run_sixs_deltaq_prototype.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load production DeltaQ runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

SMOKE_REPORT = (
    ROOT / "reports" / "ecir_mvr" / "sixs_deltaq_prototype" / "runtime" / "engineering_smoke"
)
SMOKE_ARTIFACT = (
    ROOT / "artifacts" / "ecir_mvr" / "sixs_deltaq_prototype" / "engineering_smoke"
)
SMOKE_REPORT.mkdir(parents=True, exist_ok=True)
SMOKE_ARTIFACT.mkdir(parents=True, exist_ok=True)

production_config = runner.cfg()
smoke_config = copy.deepcopy(production_config)
smoke_config["small_overfit"]["molecules"] = 2
smoke_config["small_overfit"]["optimizer_steps"] = 2
# The smoke checks execution, serialization and exit propagation only.
smoke_config["small_overfit"]["pass_relative_mae"] = 1.0e30

runner.REPORT = SMOKE_REPORT
runner.ARTIFACT = SMOKE_ARTIFACT
runner.STATUS = SMOKE_REPORT / "FINAL_STATUS.json"
runner.OVERFIT_CHECKPOINT = SMOKE_ARTIFACT / "SMALL_OVERFIT_CHECKPOINT.pt"
runner.cfg = lambda: copy.deepcopy(smoke_config)

runner.overfit()

result_path = SMOKE_REPORT / "07_SMALL_OVERFIT_RESULTS.csv"
checkpoint_path = SMOKE_ARTIFACT / "SMALL_OVERFIT_CHECKPOINT.pt"
if not result_path.is_file() or not checkpoint_path.is_file():
    raise RuntimeError("production overfit entrypoint returned without required smoke artifacts")

(SMOKE_REPORT / "ENGINEERING_SMOKE_RESULT.json").write_text(
    json.dumps(
        {
            "status": "PASS",
            "scientific_evidence": "NO",
            "production_config_changed": "NO",
            "molecules": 2,
            "optimizer_steps": 2,
            "train_loop_entered": "YES",
            "first_backward_completed": "YES",
            "final_step_completed": "YES",
            "checkpoint_save_called": "YES",
            "result_write_called": "YES",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
