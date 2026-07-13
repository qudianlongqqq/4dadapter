#!/usr/bin/env python
"""Print bounded Global 4D profile commands without executing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.commons.global4d_profile_bundle import resolve_inside


def _quote(path: Path) -> str:
    return f'"{path.as_posix()}"'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.bundle_dir
    metadata_path = root / "metadata/bundle_metadata.json"
    if not metadata_path.is_file():
        parser.error(f"Missing bundle metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    paths = metadata["paths"]
    checkpoint = resolve_inside(root, paths["checkpoint"])
    config = resolve_inside(root, paths["config"])
    manifest = resolve_inside(root, paths["manifest"])
    cache = resolve_inside(root, paths["cache_dir"])
    split = str(metadata["split"])

    base = (
        f"python scripts/profile_global4d_sampling.py --checkpoint {_quote(checkpoint)} "
        f"--config {_quote(config)} --cache_dir {_quote(cache)} "
        f"--manifest {_quote(manifest)} --split {split} --refinement_steps 10"
    )
    commands = {
        "Linux RTX 5090 legacy, at most 20 records": (
            f"{base} --max_molecules 2 --max_records 20 --warmup_records 2 "
            "--profile_records 18 --device cuda --cuda_sync_timing "
            "--partial_format legacy --save_every_records 1 "
            "--skip_batch_benchmark "
            "--output_dir reports/profile_linux_rtx5090_legacy_max20"
        ),
        "Linux RTX 5090 chunked, at most 20 records": (
            f"{base} --max_molecules 2 --max_records 20 --warmup_records 2 "
            "--profile_records 18 --device cuda --cuda_sync_timing "
            "--partial_format chunked --save_every_records 10 "
            "--skip_batch_benchmark "
            "--output_dir reports/profile_linux_rtx5090_chunked_max20"
        ),
        "Windows, at most 30 records": (
            f"{base} --max_molecules 3 --max_records 30 --warmup_records 2 "
            "--device cuda --output_dir reports/profile_windows"
        ),
        "Pure compute, no partial/final writes": (
            f"{base} --max_molecules 3 --max_records 20 --warmup_records 2 "
            "--profile_records 18 --device cuda --disable_partial_save "
            "--output_dir reports/profile_compute_only"
        ),
        "Current save protocol simulation": (
            f"{base} --max_molecules 3 --max_records 20 --warmup_records 2 "
            "--profile_records 18 --device cuda --partial_format legacy "
            "--save_every_records 1 "
            "--output_dir reports/profile_current_save"
        ),
        "I/O benchmark only": (
            "python scripts/benchmark_global4d_sampling_io.py --records 500 "
            "--atoms 40 --save_every_records 1 10 50 100 "
            "--output_dir reports/global4d_sampling_io"
        ),
    }
    for title, command in commands.items():
        print(f"# {title}\n{command}\n")


if __name__ == "__main__":
    main()
