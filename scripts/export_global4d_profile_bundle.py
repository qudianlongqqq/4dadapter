#!/usr/bin/env python
"""Export a deterministic minimal real-data bundle for Global 4D profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.commons.global4d_profile_bundle import create_profile_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_molecules", type=int, default=3)
    parser.add_argument("--max_records", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include_sampling_state", action="store_true")
    parser.add_argument("--sampling_state", type=Path)
    parser.add_argument("--include_partial_samples", action="store_true")
    parser.add_argument("--partial_samples", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.max_molecules < 1 or args.max_records < 1:
        parser.error("--max_molecules and --max_records must be positive")
    result = create_profile_bundle(
        checkpoint=args.checkpoint,
        config=args.config,
        cache_dir=args.cache_dir,
        manifest=args.manifest,
        split=args.split,
        output=args.output,
        max_molecules=args.max_molecules,
        max_records=args.max_records,
        seed=args.seed,
        include_sampling_state=args.include_sampling_state,
        sampling_state=args.sampling_state,
        include_partial_samples=args.include_partial_samples,
        partial_samples=args.partial_samples,
        force=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
