#!/usr/bin/env python
"""Estimate fixed robust V8 residual scales from real train records only."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timezone
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
from etflow.ecir.formal_rdkit_adapter import adapt_formal_cache_record
from etflow.ecir.mvr_dataset import MCVRMixedDataset


class _ScaleTopologyCache:
    """Reuse exact static formal-adapter fields across conformers of one topology."""

    def __init__(self, max_size: int = 32768) -> None:
        self.max_size = int(max_size)
        self.values: OrderedDict[str, dict] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def adapt(self, row, record: dict) -> dict:
        atomic_numbers = torch.as_tensor(record.get("atomic_numbers", []), dtype=torch.long)
        key_payload = {
            "topology_signature": str(record.get("topology_signature", "")),
            "x_init_topology_signature": str(record.get("x_init_topology_signature", "")),
            "ordered_smiles": str(record.get("ordered_smiles", record.get("smiles", ""))),
            "atomic_numbers": atomic_numbers.tolist(),
        }
        key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        cached = self.values.get(key)
        if cached is None:
            self.misses += 1
            adapted = adapt_formal_cache_record(record)
            cached = {
                name: value for name, value in adapted.items() if str(name).startswith("_formal_")
            }
            self.values[key] = cached
            if len(self.values) > self.max_size:
                self.values.popitem(last=False)
            return adapted
        self.hits += 1
        self.values.move_to_end(key)
        result = dict(record)
        result.update(cached)
        return result


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


def _positive_values(values: list[torch.Tensor]) -> torch.Tensor:
    populated = [value.detach().cpu().reshape(-1) for value in values if value.numel()]
    if not populated:
        return torch.empty(0, dtype=torch.float64)
    flat = torch.cat(populated).to(torch.float64)
    return flat[torch.isfinite(flat) & (flat > 0)]


def _write_scales(
    output: Path,
    *,
    values: dict[str, list[torch.Tensor]],
    record_count: int,
    source_sha256: str,
    target_sha256: str,
) -> None:
    fallbacks = {"bond": 0.01, "angle": 0.05, "clash": 0.05, "ring": 0.01, "chirality": 1.0}
    scales = {
        name: max(_positive_median(type_values, fallbacks[name]), 1.0e-6)
        for name, type_values in values.items()
    }
    payload = {
        "schema_version": "mcvr-v8-train-residual-scales-v1",
        "split": "train",
        "estimator": "median_absolute_positive_residual",
        "record_count": int(record_count),
        "records_scanned": int(record_count),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_source_manifest_sha256": source_sha256,
        "train_target_manifest_sha256": target_sha256,
        "scales": scales,
        "validation_used": False,
        "validation_records_read": 0,
        "test_used": False,
        "formal_test_records_read": 0,
        "formal_test_assets_opened": False,
        "minimal_validity_target_test_used": False,
        "frozen_holdout_used": False,
        "frozen_holdout_records_read": 0,
        "parameter_selection_from_formal_test": False,
    }
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                **payload,
                "output": str(output.resolve()),
                "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sources", type=Path, required=True)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path)
    parser.add_argument("--target-cache-root", type=Path)
    parser.add_argument("--validity-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--merge-raw", type=Path, nargs="+")
    args = parser.parse_args()
    sources = pd.read_parquet(args.train_sources)
    targets = pd.read_parquet(args.train_targets)
    if set(sources.split.astype(str)) != {"train"} or set(targets.split.astype(str)) != {"train"}:
        raise RuntimeError("V8 scales accept train manifests only")
    source_sha256 = hashlib.sha256(args.train_sources.read_bytes()).hexdigest()
    target_sha256 = hashlib.sha256(args.train_targets.read_bytes()).hexdigest()
    if args.merge_raw:
        if args.output is None:
            raise ValueError("--merge-raw requires --output")
        shards = [
            torch.load(path, map_location="cpu", weights_only=False) for path in args.merge_raw
        ]
        if any(shard.get("schema_version") != "mcvr-v8-raw-scale-shard-v1" for shard in shards):
            raise RuntimeError("V8 raw scale shard schema changed")
        if any(
            shard.get("train_source_manifest_sha256") != source_sha256
            or shard.get("train_target_manifest_sha256") != target_sha256
            for shard in shards
        ):
            raise RuntimeError("V8 raw scale shard manifest identity changed")
        ordered = sorted(shards, key=lambda shard: int(shard["start_index"]))
        cursor = 0
        merged: dict[str, list[torch.Tensor]] = {
            name: [] for name in ("bond", "angle", "clash", "ring", "chirality")
        }
        for shard in ordered:
            if int(shard["start_index"]) != cursor:
                raise RuntimeError("V8 raw scale shards have a gap or overlap")
            cursor = int(shard["end_index"])
            for name in merged:
                merged[name].append(torch.as_tensor(shard["values"][name], dtype=torch.float64))
        if cursor != len(sources):
            raise RuntimeError("V8 raw scale shards do not cover the full train manifest")
        _write_scales(
            args.output,
            values=merged,
            record_count=cursor,
            source_sha256=source_sha256,
            target_sha256=target_sha256,
        )
        return
    upper_bound = min(len(sources), int(args.max_records or len(sources)))
    start = int(args.start_index)
    end = min(int(args.end_index or upper_bound), upper_bound)
    if not 0 <= start < end <= len(sources):
        raise ValueError("invalid V8 train-scale shard bounds")
    count = end - start
    validity = ChemicalValidity(args.validity_statistics)
    dataset = MCVRMixedDataset(
        args.train_sources,
        args.train_targets,
        validity,
        length=len(sources),
        ratios={"real_error": 1.0, "synthetic_error": 0.0, "clean_identity": 0.0},
        seed=43,
        source_cache_root=args.source_cache_root,
        target_cache_root=args.target_cache_root,
        canonical_constraints=True,
        constraint_source_identity_sha256=hashlib.sha256(
            args.train_sources.read_bytes()
        ).hexdigest(),
    )
    dataset.formal_adapter_cache = _ScaleTopologyCache()
    bond_values: list[torch.Tensor] = []
    angle_values: list[torch.Tensor] = []
    clash_values: list[torch.Tensor] = []
    ring_values: list[torch.Tensor] = []
    chirality_values: list[torch.Tensor] = []
    for index in range(start, end):
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
        processed = index - start + 1
        if processed % 250 == 0 or processed == count:
            print(f"train_scale_progress={start + processed}/{end}", flush=True)
    values = {
        "bond": bond_values,
        "angle": angle_values,
        "clash": clash_values,
        "ring": ring_values,
        "chirality": chirality_values,
    }
    if args.raw_output is not None:
        shard = {
            "schema_version": "mcvr-v8-raw-scale-shard-v1",
            "start_index": start,
            "end_index": end,
            "record_count": count,
            "train_source_manifest_sha256": source_sha256,
            "train_target_manifest_sha256": target_sha256,
            "values": {name: _positive_values(type_values) for name, type_values in values.items()},
            "validation_records_read": 0,
            "formal_test_records_read": 0,
            "formal_test_assets_opened": False,
            "frozen_holdout_records_read": 0,
        }
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.raw_output.with_suffix(args.raw_output.suffix + ".tmp")
        torch.save(shard, temporary)
        temporary.replace(args.raw_output)
        print(json.dumps({key: value for key, value in shard.items() if key != "values"}, indent=2))
        return
    if args.output is None:
        raise ValueError("scale estimation requires --output or --raw-output")
    _write_scales(
        args.output,
        values=values,
        record_count=count,
        source_sha256=source_sha256,
        target_sha256=target_sha256,
    )


if __name__ == "__main__":
    main()
