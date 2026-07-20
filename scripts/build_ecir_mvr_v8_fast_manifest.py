#!/usr/bin/env python
"""Freeze a representative, source-ordered FAST1000 validation manifest."""

# ruff: noqa: E402

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

from etflow.ecir.v8_validation_cache import (
    ISOLATION,
    atomic_json,
    canonical_sha256,
    file_sha256,
    iter_prediction_records,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    source_manifest = json.loads(args.source_cache_manifest.read_text(encoding="utf-8"))
    records = list(iter_prediction_records(args.source_cache_manifest))
    if args.count < 1 or args.count > len(records):
        raise ValueError("FAST record count is outside the source cache")
    generator = torch.Generator().manual_seed(args.seed)
    selected = sorted(torch.randperm(len(records), generator=generator)[: args.count].tolist())
    chosen = [records[index] for index in selected]

    def cohorts(row):
        item = row["item"]
        active = torch.as_tensor(item.active_mode_mask).reshape(-1)
        displacement = torch.linalg.vector_norm(item.x_target - item.x_input, dim=-1).mean()
        return {
            "active_angle": bool(active.numel() > 1 and active[1] > 0),
            "active_clash": bool(active.numel() > 3 and active[3] > 0),
            "ring_risk": bool(active.numel() > 2 and active[2] > 0),
            "high_flexibility": int(torch.as_tensor(item.num_rotatable_bonds).max()) >= 6,
            "low_error_minimal_movement": float(displacement) <= 0.0025,
        }

    memberships = [cohorts(row) for row in chosen]
    payload = {
        "schema_version": "mcvr-v8-formal-large-fast1000-manifest-v1",
        "validation_identity_sha256": source_manifest["identity"]["identity_sha256"],
        "source_cache_manifest_sha256": file_sha256(args.source_cache_manifest),
        "selection_seed": args.seed,
        "selection_rule": "torch_randperm_seed43_then_restore_original_record_order",
        "record_count": len(chosen),
        "record_indices": selected,
        "record_ids": [row["sample_id"] for row in chosen],
        "molecule_ids": [row["molecule_id"] for row in chosen],
        "cohort_counts": {
            name: sum(int(row[name]) for row in memberships)
            for name in memberships[0]
        },
        "cohort_memberships": memberships,
        **ISOLATION,
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
