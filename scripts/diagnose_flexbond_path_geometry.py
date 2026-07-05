#!/usr/bin/env python
"""Measure local-geometry distortion along the Cartesian adapter path."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from etflow.commons.geometry_diagnostics import (
    angle_triplets,
    bond_angles,
    bond_lengths,
    dihedral_angles,
    unique_undirected_edges,
    wrapped_angle_delta,
)
from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset


METRICS = (
    "bond_length_mean_abs_change",
    "bond_length_max_abs_change",
    "bond_angle_mean_abs_change",
    "torsion_mean_abs_change",
    "rotatable_torsion_mean_abs_change",
    "nonrotatable_torsion_mean_abs_change",
    "too_close_atom_pair_count",
    "ring_bond_distortion",
    "invalid_geometry_count",
)


def _neighbors(edge_index: torch.Tensor, num_atoms: int) -> list[set[int]]:
    result = [set() for _ in range(num_atoms)]
    for atom_a, atom_b in unique_undirected_edges(edge_index).t().tolist():
        result[atom_a].add(atom_b)
        result[atom_b].add(atom_a)
    return result


def _bond_dihedrals(
    edge_index: torch.Tensor,
    rotatable_bond_index: torch.Tensor,
    num_atoms: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one deterministic dihedral per bond and label rotatable bonds."""

    adjacent = _neighbors(edge_index, num_atoms)
    rotatable = {
        tuple(sorted((int(atom_a), int(atom_b))))
        for atom_a, atom_b in rotatable_bond_index.t().tolist()
    }
    quads, masks = [], []
    for center_a, center_b in unique_undirected_edges(edge_index).t().tolist():
        left = sorted(adjacent[center_a].difference({center_b}))
        right = sorted(adjacent[center_b].difference({center_a}))
        if left and right:
            quads.append((left[0], center_a, center_b, right[0]))
            masks.append(tuple(sorted((center_a, center_b))) in rotatable)
    device = edge_index.device
    if not quads:
        return (
            torch.empty((4, 0), dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.bool, device=device),
        )
    return (
        torch.tensor(quads, dtype=torch.long, device=device).t(),
        torch.tensor(masks, dtype=torch.bool, device=device),
    )


def _unique_edge_flags(edge_index: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    mapping: dict[tuple[int, int], bool] = {}
    for index, (atom_a, atom_b) in enumerate(edge_index.t().tolist()):
        pair = tuple(sorted((int(atom_a), int(atom_b))))
        mapping[pair] = mapping.get(pair, False) or bool(flags[index])
    ordered = [tuple(pair) for pair in unique_undirected_edges(edge_index).t().tolist()]
    return torch.tensor(
        [mapping.get(pair, False) for pair in ordered], dtype=torch.bool
    )


def _mean(values: torch.Tensor) -> float:
    return float(values.mean()) if values.numel() else float("nan")


def _max(values: torch.Tensor) -> float:
    return float(values.max()) if values.numel() else float("nan")


def _metrics(data, pos: torch.Tensor, collision_distance: float) -> dict[str, float]:
    x_init = data.x_init.cpu()
    edge_index = data.edge_index.cpu()
    edges = unique_undirected_edges(edge_index)
    triplets = angle_triplets(edge_index, x_init.size(0))
    quads, rotatable_mask = _bond_dihedrals(
        edge_index, data.rotatable_bond_index.cpu(), x_init.size(0)
    )

    length_change = (bond_lengths(pos, edges) - bond_lengths(x_init, edges)).abs()
    angle_change = wrapped_angle_delta(
        bond_angles(pos, triplets), bond_angles(x_init, triplets)
    ).abs()
    torsion_change = wrapped_angle_delta(
        dihedral_angles(pos, quads), dihedral_angles(x_init, quads)
    ).abs()

    bonded = {tuple(sorted(pair)) for pair in edges.t().tolist()}
    too_close = 0
    for atom_a in range(pos.size(0)):
        for atom_b in range(atom_a + 1, pos.size(0)):
            if (atom_a, atom_b) not in bonded:
                too_close += int(
                    torch.linalg.norm(pos[atom_a] - pos[atom_b]) < collision_distance
                )

    ring_flags = _unique_edge_flags(
        edge_index, torch.as_tensor(data.bond_is_in_ring).cpu().view(-1)
    )
    ring_change = length_change[ring_flags]
    lengths = bond_lengths(pos, edges)
    invalid = int((~torch.isfinite(pos)).any())
    invalid += int((~torch.isfinite(lengths) | (lengths < 1.0e-6)).sum())
    invalid += int((~torch.isfinite(bond_angles(pos, triplets))).sum())
    invalid += int((~torch.isfinite(dihedral_angles(pos, quads))).sum())
    return {
        "bond_length_mean_abs_change": _mean(length_change),
        "bond_length_max_abs_change": _max(length_change),
        "bond_angle_mean_abs_change": math.degrees(_mean(angle_change)),
        "torsion_mean_abs_change": math.degrees(_mean(torsion_change)),
        "rotatable_torsion_mean_abs_change": math.degrees(
            _mean(torsion_change[rotatable_mask])
        ),
        "nonrotatable_torsion_mean_abs_change": math.degrees(
            _mean(torsion_change[~rotatable_mask])
        ),
        "too_close_atom_pair_count": too_close,
        "ring_bond_distortion": _mean(ring_change),
        "invalid_geometry_count": invalid,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--num_records", type=int, default=500)
    parser.add_argument(
        "--t_values", nargs="+", type=float, default=(0, 0.25, 0.5, 0.75, 1.0)
    )
    parser.add_argument("--collision_distance", type=float, default=0.7)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    if args.num_records < 1:
        raise ValueError("num_records must be positive")
    if any(t < 0 or t > 1 for t in args.t_values):
        raise ValueError("All t_values must be in [0, 1]")

    dataset = FlexBondOptimizerDataset(args.cache_dir, args.split, validate=True)
    rows = []
    for index in range(min(args.num_records, len(dataset))):
        data = dataset[index]
        for t in args.t_values:
            pos = (1.0 - t) * data.x_init.cpu() + t * data.x_ref_aligned.cpu()
            rows.append(
                {
                    "record_index": index,
                    "sample_id": str(data.mol_id),
                    "source_mol_id": str(data.source_mol_id),
                    "t": t,
                    "angle_unit": "degree",
                    **_metrics(data, pos, args.collision_distance),
                }
            )

    summary = []
    for t in args.t_values:
        selected = [row for row in rows if row["t"] == t]
        record = {"t": t, "num_records": len(selected)}
        for name in METRICS:
            values = np.asarray([row[name] for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            record[f"{name}_mean"] = float(finite.mean()) if finite.size else float("nan")
            record[f"{name}_median"] = (
                float(np.median(finite)) if finite.size else float("nan")
            )
            record[f"{name}_max"] = float(finite.max()) if finite.size else float("nan")
        summary.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "path_geometry_per_record.csv", rows)
    _write_csv(args.output_dir / "path_geometry_summary.csv", summary)
    print(f"Wrote {len(rows)} path rows for {len(dataset)} available records.")


if __name__ == "__main__":
    main()
