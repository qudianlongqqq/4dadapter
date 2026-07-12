#!/usr/bin/env python
"""Verify frozen validation selections before any formal test access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.formal_large import canonical_sha256, verify_frozen_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", required=True, type=Path)
    parser.add_argument("--validation_manifest", required=True, type=Path)
    parser.add_argument("--test_manifest", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.best.read_text(encoding="utf-8"))
    validation = json.loads(args.validation_manifest.read_text(encoding="utf-8"))
    test = json.loads(args.test_manifest.read_text(encoding="utf-8"))
    for config in payload["configs"].values():
        verify_frozen_config(
            config,
            checkpoint_path=config["checkpoint_path"],
            resolved_config_path=config["config_path"],
            manifest=validation,
        )
        if config.get("selection_split") != "validation" or config.get("test_used_for_selection"):
            raise ValueError("Frozen config was not selected exclusively on validation")
    lock = {
        "best_configs_sha256": canonical_sha256(payload),
        "test_manifest_sha256": canonical_sha256(test),
        "methods": {
            method: {
                "checkpoint_file_sha256": config["checkpoint_file_sha256"],
                "config_file_sha256": config["config_file_sha256"],
                "alpha": config["alpha"],
                "refinement_steps": config["refinement_steps"],
            }
            for method, config in payload["configs"].items()
        },
    }
    if args.lock.is_file():
        previous = json.loads(args.lock.read_text(encoding="utf-8"))
        if previous != lock:
            raise ValueError("Formal test lock changed after it was frozen")
    else:
        atomic_json_save(lock, args.lock)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
