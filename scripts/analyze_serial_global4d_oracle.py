#!/usr/bin/env python
"""Validation-only Oracle ceiling for Serial Global4D residual refinement."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.global_coupled_4d_jacobian import (
    build_global_coupled_4d_jacobian,
    joint_geometry,
)
from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.commons.global_coupled_4d_topology import GlobalCoupled4DTopologyCache
from etflow.commons.kabsch_utils import kabsch_align
from etflow.serial_global4d.cache import SerialGlobal4DResidualDataset
from etflow.serial_global4d.oracle import solve_serial_residual_oracle


def _rmsd(candidate, reference):
    aligned = kabsch_align(reference, candidate)
    return torch.sqrt((aligned - candidate).square().sum(-1).mean())


def _tier(rotatable: int) -> str:
    return "low" if rotatable <= 2 else ("medium" if rotatable <= 5 else "high")


def _mean(rows, key):
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_cache", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--ridge", type=float, default=1.0e-5)
    parser.add_argument("--rank_tol", type=float, default=1.0e-6)
    parser.add_argument("--times", default="0,0.125,0.25")
    parser.add_argument("--scales", default="0.1,0.25,0.5,0.75,1.0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite Oracle report: {args.output}")
    times = [float(value) for value in args.times.split(",")]
    scales = [float(value) for value in args.scales.split(",")]
    dataset = SerialGlobal4DResidualDataset(args.stage2_cache, "val")
    count = len(dataset) if args.max_records is None else min(len(dataset), args.max_records)
    topology_cache = GlobalCoupled4DTopologyCache()
    rows = []
    for index in range(count):
        data = dataset[index]
        x_cart = data.x_cart
        x_ref = data.x_ref_aligned
        residual = x_ref - x_cart
        prepared = topology_cache.get_prepared(
            int(data.num_nodes), data.edge_index, data.rotatable_bond_index
        )
        rotatable = int(data.rotatable_bond_index.size(1))
        for time_value in times:
            x_t = (1.0 - time_value) * x_cart + time_value * x_ref
            topology = prepared.topology
            jacobian, _ = build_global_coupled_4d_jacobian(
                x_t, topology, flat_index=prepared.jacobian_flat_index
            )
            axes = joint_geometry(x_t, topology).axis
            oracle = solve_serial_residual_oracle(
                jacobian, residual, axes, ridge=args.ridge, rank_tol=args.rank_tol
            )
            baseline = _rmsd(x_cart, x_ref)
            component_energy = (
                oracle.stretch_energy + oracle.bending_energy + oracle.torsion_energy
            ).clamp_min(1.0e-20)
            row = {
                "index": index,
                "sample_id": str(data.sample_id),
                "mol_id": str(getattr(data, "source_mol_id", data.mol_id)),
                "time": time_value,
                "num_rotatable_bonds": rotatable,
                "flexibility_tier": _tier(rotatable),
                "residual_norm": float(oracle.residual_norm),
                "projected_residual_norm": float(oracle.projected_residual_norm),
                "projection_energy_ratio": float(oracle.projection_energy_ratio),
                "oracle_residual_error": float(oracle.oracle_residual_error),
                "stretch_energy": float(oracle.stretch_energy),
                "bending_energy": float(oracle.bending_energy),
                "torsion_energy": float(oracle.torsion_energy),
                "stretch_energy_fraction": float(
                    oracle.stretch_energy / component_energy
                ),
                "bending_energy_fraction": float(
                    oracle.bending_energy / component_energy
                ),
                "torsion_energy_fraction": float(
                    oracle.torsion_energy / component_energy
                ),
                "cartesian_rmsd": float(baseline),
                "effective_rank": oracle.projection.effective_rank,
            }
            for scale in scales:
                corrected = x_cart + scale * oracle.r_j_star
                value = _rmsd(corrected, x_ref)
                row[f"oracle_rmsd_scale_{scale:g}"] = float(value)
                row[f"improved_scale_{scale:g}"] = bool(value < baseline)
            rows.append(row)
    by_tier = defaultdict(list)
    for row in rows:
        by_tier[row["flexibility_tier"]].append(row)
    summary = {
        "analysis_split": "validation",
        "test_used_for_selection": False,
        "record_count": count,
        "state_count": len(rows),
        "ridge": args.ridge,
        "rank_tol": args.rank_tol,
        "times": times,
        "scales": scales,
        "mean_residual_norm": _mean(rows, "residual_norm"),
        "mean_projected_residual_norm": _mean(rows, "projected_residual_norm"),
        "mean_projection_energy_ratio": _mean(rows, "projection_energy_ratio"),
        "mean_stretch_energy_fraction": _mean(rows, "stretch_energy_fraction"),
        "mean_bending_energy_fraction": _mean(rows, "bending_energy_fraction"),
        "mean_torsion_energy_fraction": _mean(rows, "torsion_energy_fraction"),
        "tiers": {},
        "scale_scan": {},
        "rows": rows,
    }
    for tier, tier_rows in by_tier.items():
        summary["tiers"][tier] = {
            "states": len(tier_rows),
            "projection_energy_ratio": _mean(tier_rows, "projection_energy_ratio"),
            "stretch_energy_fraction": _mean(tier_rows, "stretch_energy_fraction"),
            "bending_energy_fraction": _mean(tier_rows, "bending_energy_fraction"),
            "torsion_energy_fraction": _mean(tier_rows, "torsion_energy_fraction"),
        }
    for scale in scales:
        key = f"oracle_rmsd_scale_{scale:g}"
        improved_key = f"improved_scale_{scale:g}"
        summary["scale_scan"][str(scale)] = {
            "cartesian_rmsd": _mean(rows, "cartesian_rmsd"),
            "oracle_global4d_rmsd": _mean(rows, key),
            "improvement_rate": sum(bool(row[improved_key]) for row in rows) / len(rows),
            "degraded_rate": sum(not bool(row[improved_key]) for row in rows) / len(rows),
            "high_flex_improvement_rate": (
                sum(bool(row[improved_key]) for row in by_tier["high"]) / len(by_tier["high"])
                if by_tier["high"] else None
            ),
            "tiers": {
                tier: {
                    "cartesian_rmsd": _mean(tier_rows, "cartesian_rmsd"),
                    "oracle_global4d_rmsd": _mean(tier_rows, key),
                    "improvement_rate": (
                        sum(bool(row[improved_key]) for row in tier_rows)
                        / len(tier_rows)
                    ),
                    "degraded_rate": (
                        sum(not bool(row[improved_key]) for row in tier_rows)
                        / len(tier_rows)
                    ),
                }
                for tier, tier_rows in by_tier.items()
            },
        }
    optimal_scale, optimal = min(
        summary["scale_scan"].items(),
        key=lambda item: item[1]["oracle_global4d_rmsd"],
    )
    summary["optimal_scale"] = float(optimal_scale)
    summary["optimal_result"] = optimal
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_save(summary, args.output)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
