#!/usr/bin/env python
"""Authorized entry point for the preflight-gated medium seed42 Run A training."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

if __name__ == "__main__":
    import yaml

    config_index = sys.argv.index("--config") + 1 if "--config" in sys.argv else None
    if config_index is None:
        raise SystemExit("--config is required")
    config = yaml.safe_load(Path(sys.argv[config_index]).read_text(encoding="utf-8"))
    rescue_v2 = config.get("experiment_name") == "ecir_mvr_medium_5k_500_run_a_seed42_20k_rescue_v2"
    if "--data_audit" not in sys.argv:
        sys.argv.extend([
            "--data_audit",
            str(Path(
                "diagnostics/ecir_mvr/medium/run_a_seed42_rescue_v2_20k/preflight.json"
                if rescue_v2 else
                "diagnostics/ecir_mvr/medium/run_a_seed42_20k/preflight.json"
            )),
        ])
    if rescue_v2:
        from scripts.train_ecir_mvr_medium_rescue_v2 import main
    else:
        from scripts.train_ecir_mvr_run_a import main
    main()
