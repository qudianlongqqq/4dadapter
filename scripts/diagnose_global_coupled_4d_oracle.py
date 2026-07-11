#!/usr/bin/env python
"""Compare global 1D torsion and Global Coupled 4D oracle subspaces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.global_coupled_4d_jacobian import (
    apply_global_coupled_4d_jacobian,
    build_global_coupled_4d_jacobian,
    decompose_joint_rates,
)
from etflow.commons.global_coupled_4d_projection import gram_solve, svd_oracle
from etflow.commons.global_coupled_4d_topology import build_global_coupled_4d_topology
from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset
from etflow.data.flexbond_eval_manifest import (
    limit_manifest_molecules,
    load_eval_manifest,
    validate_dataset_against_manifest,
)
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset


def _one_dimensional_jacobian(jacobian, axis):
    columns = []
    for joint in range(axis.size(0)):
        columns.append(jacobian[:, 4 * joint + 1 : 4 * joint + 4] @ axis[joint])
    return torch.stack(columns, dim=1) if columns else jacobian[:, :0]


def _energy_fractions(pos, topology, q, axis):
    parts = decompose_joint_rates(q, axis)
    zeros = torch.zeros_like(parts["stretch"][:, None])
    modes = {
        "stretch": torch.cat((parts["stretch"][:, None], torch.zeros_like(parts["omega"])), -1),
        "bending": torch.cat((zeros, parts["bending_vector"]), -1),
        "torsion": torch.cat((zeros, parts["torsion_vector"]), -1),
    }
    energy = {}
    for name, values in modes.items():
        velocity, _ = apply_global_coupled_4d_jacobian(pos, values, topology)
        energy[name] = float(velocity.square().sum())
    total = max(sum(energy.values()), 1.0e-20)
    return {f"{name}_energy_fraction": value / total for name, value in energy.items()}


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["subset"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _mean(rows, key):
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inference_cache", required=True, type=Path)
    parser.add_argument("--reference_cache", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_molecules", type=int, default=200)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.manifest, args.inference_cache, args.reference_cache):
        if not path.exists():
            raise FileNotFoundError(f"required oracle input is missing: {path}")
    manifest = load_eval_manifest(args.manifest)
    manifest = limit_manifest_molecules(manifest, args.max_molecules)
    inference_dataset = FlexBondInferenceDataset(args.inference_cache, args.split)
    validate_dataset_against_manifest(inference_dataset, manifest)
    allowed = {str(row["sample_id"]) for row in manifest.get("records", [])}
    dataset = FlexBondOptimizerDataset(args.reference_cache, args.split, validate=True)
    molecule_rows, joint_rows = [], []
    seen_sources = set()
    for data in dataset:
        sample_id = str(getattr(data, "sample_id", data.mol_id))
        source_id = str(getattr(data, "source_mol_id", data.mol_id))
        if allowed and sample_id not in allowed and str(data.mol_id) not in allowed:
            continue
        seen_sources.add(source_id)
        pos = data.x_init
        target = data.x_ref_aligned - data.x_init
        topology = build_global_coupled_4d_topology(
            pos.size(0), data.edge_index, data.rotatable_bond_index
        ).to(pos.device)
        jacobian, geometry = build_global_coupled_4d_jacobian(pos, topology)
        one_d = _one_dimensional_jacobian(jacobian, geometry.axis)
        result_1d = svd_oracle(one_d, target)
        result_4d = svd_oracle(jacobian, target)
        fast = gram_solve(jacobian, target)
        fractions = _energy_fractions(
            pos, topology, result_4d.coefficients.reshape(-1, 4), geometry.axis
        ) if topology.num_joints else {
            "stretch_energy_fraction": 0.0,
            "bending_energy_fraction": 0.0,
            "torsion_energy_fraction": 0.0,
        }
        row = {
            "sample_id": sample_id,
            "source_mol_id": source_id,
            "num_atoms": pos.size(0),
            "num_joints": topology.num_joints,
            "topology_status": topology.status,
            "1d_num_columns": one_d.size(1),
            "4d_num_columns": jacobian.size(1),
            "1d_effective_rank": result_1d.effective_rank,
            "4d_effective_rank": result_4d.effective_rank,
            "1d_explained_ratio": float(result_1d.explained_ratio),
            "4d_explained_ratio": float(result_4d.explained_ratio),
            "4d_incremental_explained_ratio": float(result_4d.explained_ratio - result_1d.explained_ratio),
            "1d_explained_per_rank": float(result_1d.explained_ratio) / max(result_1d.effective_rank, 1),
            "4d_explained_per_rank": float(result_4d.explained_ratio) / max(result_4d.effective_rank, 1),
            "1d_condition_number": result_1d.condition_number,
            "4d_condition_number": result_4d.condition_number,
            "1d_reconstruction_error": float(result_1d.reconstruction_error),
            "4d_reconstruction_error": float(result_4d.reconstruction_error),
            **fractions,
            "orthogonality_error": float(result_4d.orthogonality_error),
            "solver_fallback_rate": float(fast.solver_fallback_count > 0),
        }
        molecule_rows.append(row)
        q = result_4d.coefficients.reshape(-1, 4)
        parts = decompose_joint_rates(q, geometry.axis)
        for joint in range(topology.num_joints):
            joint_rows.append({
                "sample_id": sample_id, "joint_index": joint,
                "parent_atom": int(topology.parent_atom[joint]),
                "child_atom": int(topology.child_atom[joint]),
                "downstream_atoms": int((topology.affected_joint_index == joint).sum()),
                "stretch": float(parts["stretch"][joint]),
                "torsion": float(parts["torsion"][joint]),
                "bending_norm": float(parts["bending_norm"][joint]),
                "axis_valid": bool(geometry.valid[joint]),
            })
    if not molecule_rows:
        raise RuntimeError("oracle cohort is empty after manifest filtering")
    subsets = {
        "all": lambda row: True,
        "rotatable_ge_3": lambda row: int(row["num_joints"]) >= 3,
        "rotatable_ge_5": lambda row: int(row["num_joints"]) >= 5,
        "rotatable_ge_6": lambda row: int(row["num_joints"]) >= 6,
    }
    metric_fields = [key for key in molecule_rows[0] if key not in {"sample_id", "source_mol_id", "topology_status"}]
    summaries = []
    for name, predicate in subsets.items():
        selected = [row for row in molecule_rows if predicate(row)]
        summary = {"subset": name, "num_molecules": len(selected)}
        summary.update({key: _mean(selected, key) for key in metric_fields})
        summaries.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_molecule.csv", molecule_rows)
    _write_csv(args.output_dir / "per_joint.csv", joint_rows)
    _write_csv(args.output_dir / "summary.csv", summaries)
    all_row = summaries[0]
    report = ["# Global Coupled 4D oracle", "", f"Molecules: {len(molecule_rows)}", "",
              f"- Global 1D explained ratio: {all_row['1d_explained_ratio']:.6f}",
              f"- Global 4D explained ratio: {all_row['4d_explained_ratio']:.6f}",
              f"- Incremental explained ratio: {all_row['4d_incremental_explained_ratio']:.6f}",
              f"- 4D mean effective rank: {all_row['4d_effective_rank']:.3f}",
              f"- 4D mean condition number: {all_row['4d_condition_number']:.6g}",
              f"- Orthogonality error: {all_row['orthogonality_error']:.6g}", ""]
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
