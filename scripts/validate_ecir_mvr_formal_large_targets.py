#!/usr/bin/env python
"""Strictly validate formal-large D1-B source/target pairing assets."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import pandas as pd  # noqa: E402

from etflow.ecir.formal_target_assets import (  # noqa: E402
    atomic_json,
    load_config,
    require_parquet_engine,
    validate_formal_assets,
    verify_stage_d_identities,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ecir_mvr_formal_large_minimal_targets.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--strict-sample-count", type=int, default=100)
    args = parser.parse_args()
    require_parquet_engine()
    config = load_config(args.config, output_root=args.output_root)
    output_root = Path(config["output_root"]).expanduser().resolve()
    identities = verify_stage_d_identities(config)
    identities["target_builder_config"] = dict(config["target_builder"])
    identities["config_file_sha256"] = config["config_file_sha256"]
    source_frames = {
        split: pd.read_parquet(output_root / "real_sources" / f"{split}.parquet")
        for split in ("train", "val")
    }
    result = validate_formal_assets(
        output_root=output_root,
        source_frames=source_frames,
        identities=identities,
        require_complete=True,
        strict_sample_count=args.strict_sample_count,
    )
    atomic_json(result, output_root / "statistics" / "validation.json")
    print(result["decision"])
    if result["decision"] != "D1B_FORMAL_TARGETS_READY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
