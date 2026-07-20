#!/usr/bin/env python
"""Estimate fixed robust V8 residual scales from real train records only."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import pandas as pd
import torch

from etflow.ecir.bac_constraints import sparse_clash_edges
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.geometry import bond_angles, bond_lengths
from etflow.ecir.mvr_dataset import MCVRMixedDataset


def _positive_median(values: list[torch.Tensor], fallback: float) -> float:
    populated = [value.detach().cpu().reshape(-1) for value in values if value.numel()]
    if not populated:
        return float(fallback)
    flat = torch.cat(populated)
    flat = flat[torch.isfinite(flat) & (flat > 0)]
    return float(flat.median()) if flat.numel() else float(fallback)


def _interval(values: torch.Tensor, ranges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lower, upper = ranges[:, 0], ranges[:, 1]
    violation = torch.maximum(lower - values, values - upper).clamp_min(0.0)
    target = torch.where(values < lower, lower, torch.where(values > upper, upper, values))
    return violation, target


def _volumes(coordinates: torch.Tensor, quads: torch.Tensor) -> torch.Tensor:
    if not quads.numel():
        return coordinates.new_empty(0)
    center, first, second, third = quads
    return torch.linalg.det(
        torch.stack(
            (
                coordinates[first] - coordinates[center],
                coordinates[second] - coordinates[center],
                coordinates[third] - coordinates[center],
            ),
            dim=1,
        )
    ).abs()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sources", type=Path, required=True)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path)
    parser.add_argument("--target-cache-root", type=Path)
    parser.add_argument("--validity-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    sources = pd.read_parquet(args.train_sources)
    targets = pd.read_parquet(args.train_targets)
    if set(sources.split.astype(str)) != {"train"} or set(targets.split.astype(str)) != {"train"}:
        raise RuntimeError("V8 scales accept train manifests only")
    count = min(len(sources), int(args.max_records or len(sources)))
    validity = ChemicalValidity(args.validity_statistics)
    dataset = MCVRMixedDataset(
        args.train_sources,
        args.train_targets,
        validity,
        length=count,
        ratios={"real_error": 1.0, "synthetic_error": 0.0, "clean_identity": 0.0},
        seed=43,
        source_cache_root=args.source_cache_root,
        target_cache_root=args.target_cache_root,
        canonical_constraints=True,
        constraint_source_identity_sha256=hashlib.sha256(
            args.train_sources.read_bytes()
        ).hexdigest(),
    )
    bond_values: list[torch.Tensor] = []
    angle_values: list[torch.Tensor] = []
    clash_values: list[torch.Tensor] = []
    ring_values: list[torch.Tensor] = []
    chirality_values: list[torch.Tensor] = []
    for index in range(count):
        item = dataset[index]
        coordinates = item.x_input.to(torch.float64)
        bonds = item.active_bond_constraint_index.reshape(2, -1)
        bond_ranges = item.bond_allowed_range.to(torch.float64).reshape(-1, 3)
        bond_violation, _ = _interval(bond_lengths(coordinates, bonds), bond_ranges)
        bond_values.append(bond_violation)
        angles = item.active_angle_constraint_index.t().reshape(-1, 3)
        angle_ranges = item.angle_allowed_range.to(torch.float64).reshape(-1, 3)
        angle_violation, target = _interval(bond_angles(coordinates, angles), angle_ranges)
        angle_values.append(
            (torch.cos(bond_angles(coordinates, angles)) - torch.cos(target)).abs()[
                angle_violation > 0
            ]
        )
        clash = sparse_clash_edges(coordinates, bonds, allowed_contact=1.0)
        clash_values.append(clash["penetration"])
        ring_bonds = item.protected_ring_bond_index.reshape(2, -1)
        if ring_bonds.numel():
            ring_values.append(
                (
                    bond_lengths(item.x_target.to(torch.float64), ring_bonds)
                    - bond_lengths(coordinates, ring_bonds)
                ).abs()
            )
        quads = item.protected_chirality_constraint_index.reshape(4, -1)
        chirality_values.append(_volumes(coordinates, quads))
        if (index + 1) % 250 == 0 or index + 1 == count:
            print(f"train_scale_progress={index + 1}/{count}", flush=True)
    scales = {
        "bond": max(_positive_median(bond_values, 0.01), 1.0e-6),
        "angle": max(_positive_median(angle_values, 0.05), 1.0e-6),
        "clash": max(_positive_median(clash_values, 0.05), 1.0e-6),
        "ring": max(_positive_median(ring_values, 0.01), 1.0e-6),
        "chirality": max(_positive_median(chirality_values, 1.0), 1.0e-6),
    }
    payload = {
        "schema_version": "mcvr-v8-train-residual-scales-v1",
        "split": "train",
        "estimator": "median_absolute_positive_residual",
        "record_count": count,
        "train_source_manifest_sha256": hashlib.sha256(args.train_sources.read_bytes()).hexdigest(),
        "train_target_manifest_sha256": hashlib.sha256(args.train_targets.read_bytes()).hexdigest(),
        "scales": scales,
        "validation_used": False,
        "test_used": False,
        "frozen_holdout_used": False,
    }
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
