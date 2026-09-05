#!/usr/bin/env python
"""Run only the repaired current-final MMFF94s baseline.

This runner consumes the already-frozen primary topology/reference cache and
ETFlow Source record assets.  It never invokes a neural model and writes into
an independent closure asset directory, leaving the superseded MMFF outputs
and every Source/SIXS artifact untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rdkit import Chem, rdBase

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

from etflow.ecir.external_refinement_baselines import refine_with_mmff94s
from scripts.run_sixs_primary_final_coordinates import (
    TARGET_MOLECULES,
    TARGET_RECORDS,
    atomic_json,
    atomic_parquet,
    atomic_torch,
    formal_record,
    metric_row,
    sha256_file,
    write_sdf,
)


def topology_paths(cache_dir: Path, protocol: Path) -> list[Path]:
    complete = json.loads((cache_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    paths = [Path(value) for value in complete["chunks"]]
    if complete.get("status") != "PASS" or complete.get("protocol_sha256") != sha256_file(protocol):
        raise RuntimeError("frozen topology cache binding mismatch")
    if not paths or not all(path.is_file() for path in paths):
        raise RuntimeError("frozen topology cache incomplete")
    return paths


def repair_record(mol: Chem.Mol) -> dict[str, Any]:
    record = formal_record(mol)
    record["atomic_numbers"] = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long
    )
    if int(record["num_atoms"]) != mol.GetNumAtoms():
        raise RuntimeError("authoritative num_atoms mismatch")
    if tuple(record["_formal_cache_to_rdkit"]) != tuple(range(mol.GetNumAtoms())):
        raise RuntimeError("frozen atom order mismatch")
    return record


def evaluate_item(item: dict[str, Any], config: dict[str, Any], protocol_hash: str) -> list[dict[str, Any]]:
    mol = Chem.Mol(item["mol_binary"])
    record = repair_record(mol)
    rows: list[dict[str, Any]] = []
    for source_manifest_row in item["source_records"]:
        payload = torch.load(source_manifest_row["record_asset"], map_location="cpu", weights_only=False)
        source = torch.as_tensor(payload["source_coordinates"], dtype=torch.float32)
        if source.shape != (mol.GetNumAtoms(), 3):
            raise RuntimeError("source/topology atom-count mismatch")
        result = refine_with_mmff94s(record, source, config)
        candidate = result.refined_coordinates.detach().cpu().float()
        rows.append(
            {
                "record_id": source_manifest_row["record_id"],
                "final_molecule_index": int(item["final_molecule_index"]),
                "molecule_id": item["molecule_id"],
                "molecule_identity_sha256": item["molecule_identity_sha256"],
                "etflow_record_index": int(source_manifest_row["etflow_record_index"]),
                "method": "mmff94s",
                "formulation": "MMFF94s",
                "seed": None,
                "source_record_asset": source_manifest_row["record_asset"],
                "proposal_coordinates": candidate,
                "num_atoms": int(record["num_atoms"]),
                "mmff_parameterizable": result.failure_reason not in {
                    "mmff_parameters_unavailable", "mmff94s_properties_unavailable"
                },
                "mmff_success": bool(result.success),
                "mmff_converged": bool(result.converged),
                "mmff_fallback_to_source": bool(result.fallback_to_source),
                "mmff_failure_reason": result.failure_reason,
                "mmff_runtime_seconds": float(result.runtime_seconds),
                "mmff_method_version": result.method_version,
                "mmff_initial_native_energy": result.initial_native_energy,
                "mmff_final_native_energy": result.final_native_energy,
                **metric_row(source, candidate, item["references"], item["graph"]),
                "protocol_sha256": protocol_hash,
                "reference_used_at_optimization": False,
            }
        )
    return rows


def smoke(args: argparse.Namespace, paths: list[Path], config: dict[str, Any], protocol_hash: str) -> None:
    rows: list[dict[str, Any]] = []
    molecules = 0
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for item in payload["items"]:
            rows.extend(evaluate_item(item, config, protocol_hash))
            molecules += 1
            if molecules >= args.smoke_molecules:
                break
        if molecules >= args.smoke_molecules:
            break
    invoked = len(rows) == 2 * args.smoke_molecules
    atom_order = all(int(row["num_atoms"]) > 0 for row in rows)
    parsed = all(torch.isfinite(row["proposal_coordinates"]).all().item() for row in rows)
    audit = {
        "schema_version": "sixs-final-mmff94s-repair-smoke-v1",
        "status": "PASS" if invoked and atom_order and parsed else "FAIL",
        "molecules": molecules,
        "records": len(rows),
        "record_reconstruction": invoked,
        "num_atoms_present": atom_order,
        "atom_identity_and_order_preserved": atom_order,
        "mmff94s_parameterization_invoked": invoked,
        "optimization_invoked": invoked,
        "output_parsed_and_finite": parsed,
        "successes": sum(bool(row["mmff_success"]) for row in rows),
        "failure_reasons": dict(Counter(str(row["mmff_failure_reason"]) for row in rows if row["mmff_failure_reason"])),
        "scientific_outcome_used_for_protocol_selection": False,
        "rdkit_version": rdBase.rdkitVersion,
        "repair": "add authoritative mol.GetNumAtoms() to reconstructed frozen record",
        "scientific_semantics_changed": False,
    }
    atomic_json(args.report_dir / "02_MMFF_REPAIR_AUDIT.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError("MMFF engineering smoke failed")


def full(args: argparse.Namespace, paths: list[Path], config: dict[str, Any], protocol_hash: str) -> None:
    chunk_dir = args.output_dir / "chunks"
    for index, topology_path in enumerate(paths, start=1):
        output_path = chunk_dir / topology_path.name
        if output_path.is_file():
            continue
        payload = torch.load(topology_path, map_location="cpu", weights_only=False)
        rows: list[dict[str, Any]] = []
        for item in payload["items"]:
            rows.extend(evaluate_item(item, config, protocol_hash))
        atomic_torch(output_path, rows)
        atomic_json(args.output_dir / "RUN_STATUS.json", {
            "status": "RUNNING", "stage": "MMFF94S_FULL", "pid": os.getpid(),
            "topology_chunks_complete": index, "topology_chunks_total": len(paths),
        })

    all_rows: list[dict[str, Any]] = []
    topology_by_rank: dict[int, dict[str, Any]] = {}
    for topology_path in paths:
        payload = torch.load(topology_path, map_location="cpu", weights_only=False)
        for item in payload["items"]:
            topology_by_rank[int(item["final_molecule_index"])] = {"mol_binary": item["mol_binary"]}
        all_rows.extend(torch.load(chunk_dir / topology_path.name, map_location="cpu", weights_only=False))
    all_rows.sort(key=lambda row: (int(row["final_molecule_index"]), int(row["etflow_record_index"])))
    if len(all_rows) != TARGET_RECORDS or len({row["record_id"] for row in all_rows}) != TARGET_RECORDS:
        raise RuntimeError("repaired MMFF denominator/identity mismatch")
    write_sdf(args.output_dir / "COORDINATES.sdf", all_rows, topology_by_rank, "mmff94s")
    scalar = [{key: value for key, value in row.items() if key != "proposal_coordinates"} for row in all_rows]
    atomic_parquet(args.output_dir / "PER_RECORD.parquet", pd.DataFrame(scalar))
    reasons = Counter(str(row["mmff_failure_reason"]) for row in all_rows if row["mmff_failure_reason"])
    summary = {
        "status": "PASS", "attempted": TARGET_RECORDS,
        "parameterizable": sum(bool(row["mmff_parameterizable"]) for row in all_rows),
        "optimization_success": sum(bool(row["mmff_success"]) for row in all_rows),
        "optimization_failure": sum(not bool(row["mmff_success"]) for row in all_rows),
        "failure_reasons": dict(reasons), "no_replacement": True,
        "source_fallback_rows_for_fixed_denominator_evaluation": sum(bool(row["mmff_fallback_to_source"]) for row in all_rows),
        "fallback_rows_counted_as_mmff_success": False,
        "molecules": TARGET_MOLECULES, "records": TARGET_RECORDS,
        "sdf_sha256": sha256_file(args.output_dir / "COORDINATES.sdf"),
        "per_record_sha256": sha256_file(args.output_dir / "PER_RECORD.parquet"),
        "protocol_sha256": protocol_hash,
    }
    atomic_json(args.output_dir / "COMPLETE.json", summary)
    atomic_json(args.output_dir / "RUN_STATUS.json", {**summary, "stage": "MMFF94S_COMPLETE", "pid": os.getpid()})


def main(args: argparse.Namespace) -> int:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(args.protocol)
    paths = topology_paths(args.topology_cache, args.protocol)
    config = dict(protocol["mmff94s"])
    if args.mode == "smoke":
        smoke(args, paths, config, protocol_hash)
    else:
        full(args, paths, config, protocol_hash)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--topology-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--smoke-molecules", type=int, default=3)
    args = parser.parse_args()
    for name in ("protocol", "topology_cache", "output_dir", "report_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
