#!/usr/bin/env python
"""Materialize the final operational config after the DataLoader benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if benchmark["status"] != "PASS" or not benchmark["sample_order_preserved"]:
        raise RuntimeError("cannot resolve config from failed DataLoader benchmark")
    selected = benchmark["selected"]
    resolved = {
        **config,
        "training": {
            **config["training"],
            "num_workers": int(selected["num_workers"]),
            "pin_memory": bool(selected["pin_memory"]),
            "persistent_workers": bool(selected["persistent_workers"]),
            "prefetch_factor": int(selected["prefetch_factor"] or 2),
        },
        "resolved": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "dataloader_benchmark": str(args.benchmark.resolve()),
            "dataloader_selection_basis": benchmark["selection_basis"],
            "sample_order_preserved": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(resolved["training"], sort_keys=False))


if __name__ == "__main__":
    main()
