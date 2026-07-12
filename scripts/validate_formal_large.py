#!/usr/bin/env python
"""Fail-closed validation for formal-large data and matched training budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import yaml

from etflow.formal_large import (
    TEST_MOLECULES,
    TRAIN_MOLECULES,
    VAL_MOLECULES,
    assert_disjoint_splits,
    assert_matched_training_budgets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/flexbond_cache_formal_large"))
    parser.add_argument(
        "--inference",
        type=Path,
        default=Path("data/flexbond_inference_formal_large"),
    )
    parser.add_argument("--manifest_dir", type=Path, default=Path("manifests"))
    args = parser.parse_args()
    manifests = {
        split: json.loads(
            (args.manifest_dir / f"formal_large_{split}.json").read_text(encoding="utf-8")
        )
        for split in ("train", "val", "test")
    }
    assert_disjoint_splits(manifests)
    targets = {"train": TRAIN_MOLECULES, "val": VAL_MOLECULES, "test": TEST_MOLECULES}
    for split, manifest in manifests.items():
        molecules = {str(row["mol_id"]) for row in manifest["records"]}
        if len(molecules) != targets[split]:
            raise ValueError(
                f"{split} molecule count {len(molecules)} != {targets[split]}"
            )
        expected = len(manifest["records"])
        cache_count = len(list((args.cache / split).glob("*.pt")))
        inference_count = len(list((args.inference / split).glob("*.pt")))
        if cache_count != expected or inference_count != expected:
            raise ValueError(
                f"{split} pair count mismatch: manifest={expected}, "
                f"cache={cache_count}, inference={inference_count}"
            )
    configs = {
        "cartesian": yaml.safe_load(
            Path("configs/formal_large_cartesian_seed42_200k.yaml").read_text(
                encoding="utf-8"
            )
        ),
        "global4d": yaml.safe_load(
            Path("configs/formal_large_global4d_seed42_200k.yaml").read_text(
                encoding="utf-8"
            )
        ),
    }
    cache_paths = {
        str(Path(config["data"]["cache_dir"]).resolve()) for config in configs.values()
    }
    if cache_paths != {str(args.cache.resolve())}:
        raise ValueError(f"Methods do not share the formal-large pair cache: {cache_paths}")
    budget = assert_matched_training_budgets(configs)
    print(json.dumps({"status": "ready", "budget": budget}, indent=2))


if __name__ == "__main__":
    main()
