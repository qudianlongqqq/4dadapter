#!/usr/bin/env python
"""Create an explicit Gated Global4D V2 initialization checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch
import yaml

from etflow.commons.global_coupled_4d_sampling import atomic_torch_save
from etflow.models.global4d_checkpoint import warm_start_global4d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initialize_missing_gate", action="store_true")
    args = parser.parse_args()
    for destination in (args.output, args.report):
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(args.seed)
    model, report = warm_start_global4d(
        args.checkpoint,
        config,
        initialize_missing_gate=args.initialize_missing_gate,
        map_location="cpu",
    )
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    output = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": source.get("pytorch-lightning_version"),
        "global_step": 0,
        "epoch": 0,
        "warm_start": {
            "source_checkpoint": str(args.checkpoint.expanduser().resolve()),
            "seed": args.seed,
            "report": report,
        },
    }
    atomic_torch_save(output, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({**report, "seed": args.seed, "output": str(args.output.resolve())}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
