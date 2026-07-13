#!/usr/bin/env python
"""Fail-closed validation for formal-large data and matched training budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

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
from etflow.commons.record_identity import source_record_identity


def validate_manifest_data(
    manifests: Mapping[str, Mapping],
    *,
    cache: Path,
    inference: Path,
    targets: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    missing_splits = sorted(set(targets).difference(manifests))
    unexpected_splits = sorted(set(manifests).difference(targets))
    if missing_splits or unexpected_splits:
        raise ValueError(
            "Manifest split mismatch: "
            f"missing={missing_splits}, unexpected={unexpected_splits}."
        )
    split_records = {}
    for split, manifest in manifests.items():
        if not isinstance(manifest, Mapping) or "records" not in manifest:
            raise ValueError(
                f"Split {split!r} manifest is missing required 'records' field."
            )
        split_records[split] = manifest["records"]
    assert_disjoint_splits(split_records)

    summary = {}
    for split, records in split_records.items():
        if split not in targets:
            raise ValueError(f"No molecule target configured for split {split!r}.")
        molecules = {source_record_identity(row) for row in records}
        if len(molecules) != int(targets[split]):
            raise ValueError(
                f"{split} molecule count {len(molecules)} != {targets[split]}"
            )
        expected = len(records)
        cache_count = len(list((cache / split).glob("*.pt")))
        inference_count = len(list((inference / split).glob("*.pt")))
        if cache_count != expected or inference_count != expected:
            raise ValueError(
                f"{split} pair count mismatch: manifest={expected}, "
                f"cache={cache_count}, inference={inference_count}"
            )
        summary[split] = {
            "molecules": len(molecules),
            "records": expected,
        }
    return summary


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
    targets = {"train": TRAIN_MOLECULES, "val": VAL_MOLECULES, "test": TEST_MOLECULES}
    validate_manifest_data(
        manifests,
        cache=args.cache,
        inference=args.inference,
        targets=targets,
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
