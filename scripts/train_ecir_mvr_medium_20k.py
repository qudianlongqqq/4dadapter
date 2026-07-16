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

from scripts.train_ecir_mvr_run_a import main


if __name__ == "__main__":
    if "--data_audit" not in sys.argv:
        sys.argv.extend([
            "--data_audit",
            str(Path("diagnostics/ecir_mvr/medium/run_a_seed42_20k/preflight.json")),
        ])
    main()
