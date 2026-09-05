#!/usr/bin/env python
"""Generate frozen SIXS primary-final coordinates without changing any model.

The runner evaluates all six predeclared step-17,500 checkpoints separately,
plus Source and the frozen matched MMFF94s baseline.  It is resumable at the
official GEOM-DRUGS standardized-pickle shard boundary.  Reference coordinates
are used only for post-inference metrics, never as model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

from etflow.ecir.external_refinement_baselines import refine_with_mmff94s
from etflow.ecir.j1r1_full_joint import FullJointModel, full_joint_action
from etflow.ecir.j1r1_full_joint_unrestricted import (
    UnrestrictedFullJointModel,
    unrestricted_full_joint_action,
)
from etflow.commons.kabsch_utils import kabsch_rmsd
from etflow.ecir.learned_geometry import geometry_values, prepare_graph
from scripts.run_mcvr_lsgo import collate_graphs


TARGET_MOLECULES = 2500
TARGET_RECORDS = 5000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def update_status(report_dir: Path, stage: str, state: str = "RUNNING", **extra: Any) -> None:
    path = report_dir / "RUN_STATUS.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update(
        {
            "schema_version": "sixs-primary-final-evaluation-run-status-v1",
            "status": state,
            "stage": stage,
            "pid": os.getpid(),
            "protected_final_outcome_opened": stage not in {"PREFLIGHT", "SMOKE"},
            "model_training_performed": False,
            "primary_membership_changed": False,
            **extra,
        }
    )
    atomic_json(path, current)


def formal_record(mol: Chem.Mol) -> dict[str, Any]:
    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((left, right), (right, left)))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return {
        "_formal_rdkit_mol": Chem.Mol(mol),
        "_formal_cache_to_rdkit": tuple(range(mol.GetNumAtoms())),
        # Required by mol_from_frozen_record() when it validates the frozen
        # cache-to-RDKit permutation.  The authoritative count comes from the
        # already-frozen topology molecule; no identity or geometry changes.
        "num_atoms": int(mol.GetNumAtoms()),
        "edge_index": edge_index,
    }


def proper_kabsch_rmsd(left: torch.Tensor, right: torch.Tensor) -> float:
    # Reuse the repository's previously validated fixed-order, proper-rotation
    # Kabsch implementation.  NumPy's LAPACK loader can terminate natively in
    # the CUDA environment on Windows (0xc06d007f) before Python can raise.
    return float(kabsch_rmsd(left.detach().cpu().double(), right.detach().cpu().double()))


def reference_rmsd(coordinates: torch.Tensor, references: torch.Tensor) -> float:
    return min(proper_kabsch_rmsd(coordinates, reference) for reference in references)


def metric_row(
    source: torch.Tensor,
    candidate: torch.Tensor,
    references: torch.Tensor,
    graph: Any,
) -> dict[str, float]:
    source64 = source.detach().cpu().double()
    candidate64 = candidate.detach().cpu().double()
    reference0 = references[0].double()
    source_bond, source_angle = geometry_values(source64, graph)
    candidate_bond, candidate_angle = geometry_values(candidate64, graph)
    reference_bond, reference_angle = geometry_values(reference0, graph)
    bond_scale = graph.bond_fixed[:, 1].double()
    angle_scale = graph.angle_fixed[:, 1].double()
    source_objective = 0.5 * (
        ((source_bond - reference_bond) / bond_scale).square().mean()
        + ((source_angle - reference_angle) / angle_scale).square().mean()
    )
    candidate_objective = 0.5 * (
        ((candidate_bond - reference_bond) / bond_scale).square().mean()
        + ((candidate_angle - reference_angle) / angle_scale).square().mean()
    )
    displacement = candidate64 - source64
    atom_displacement = torch.linalg.vector_norm(displacement, dim=-1)
    return {
        "bond_raw_mae": float((candidate_bond - reference_bond).abs().mean()),
        "angle_cosine_raw_mae": float((candidate_angle - reference_angle).abs().mean()),
        "raw_source_displacement_rms": float(displacement.square().sum(-1).mean().sqrt()),
        "kabsch_source_rmsd": proper_kabsch_rmsd(candidate64, source64),
        "reference_rmsd": reference_rmsd(candidate64, references),
        "internal_source_objective": float(source_objective),
        "internal_post_objective": float(candidate_objective),
        "direction_improvement": float(source_objective - candidate_objective),
        "max_atom_displacement": float(atom_displacement.max()),
    }


def load_protocol(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    bindings = protocol["bindings"]
    checks = {
        "primary_manifest": args.primary,
        "source_record_manifest": args.source_manifest,
        "source_asset_freeze": args.source_asset_freeze,
        "calibration": Path(protocol["calibration"]["path"]),
    }
    for key, path in checks.items():
        observed = sha256_file(path)
        if observed != bindings[f"{key}_sha256"]:
            raise RuntimeError(f"frozen input SHA256 mismatch: {key}")
    if protocol["cohort"] != {"molecules": TARGET_MOLECULES, "records": TARGET_RECORDS, "records_per_molecule": 2}:
        raise RuntimeError("protocol cohort cardinality changed")
    for method in protocol["model_methods"]:
        checkpoint = Path(method["checkpoint"])
        if sha256_file(checkpoint) != method["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint SHA256 mismatch: {method['id']}")
    return protocol


def source_rows(path: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_rank: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_rank.setdefault(int(row["final_molecule_index"]), []).append(row)
    if len(rows) != TARGET_RECORDS or len(by_rank) != TARGET_MOLECULES:
        raise RuntimeError("source record denominator mismatch")
    if any(sorted(int(v["etflow_record_index"]) for v in values) != [0, 1] for values in by_rank.values()):
        raise RuntimeError("source records per molecule mismatch")
    return rows, by_rank


def build_topology_cache(args: argparse.Namespace, protocol: Mapping[str, Any], *, smoke: bool = False) -> list[Path]:
    cache_dir = args.output_dir / ("smoke_topology_reference_cache" if smoke else "topology_reference_cache")
    complete = cache_dir / "COMPLETE.json"
    if complete.is_file():
        result = json.loads(complete.read_text(encoding="utf-8"))
        paths = [Path(value) for value in result["chunks"]]
        if result.get("protocol_sha256") != sha256_file(args.protocol) or not all(path.is_file() for path in paths):
            raise RuntimeError("stale topology/reference cache")
        return paths
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    _, by_rank = source_rows(args.source_manifest)
    calibration = json.loads(Path(protocol["calibration"]["path"]).read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    selected_primary_rows = primary["rows"][: args.smoke_molecules] if smoke else primary["rows"]
    for row in selected_primary_rows:
        relative = row["reference_identity"].partition("::")[0]
        grouped.setdefault(relative, []).append(row)
    chunks: list[Path] = []
    source_root = Path(protocol["reference_source_root"])
    for shard_index, (relative, selected) in enumerate(sorted(grouped.items()), start=1):
        chunk = cache_dir / f"shard_{shard_index:03d}.pt"
        chunks.append(chunk.resolve())
        if chunk.is_file():
            continue
        with (source_root / relative).open("rb") as stream:
            payload = pickle.load(stream)
        items = []
        for row in sorted(selected, key=lambda value: int(value["selection_rank"])):
            _, _, raw_key = row["reference_identity"].partition("::")
            entry = payload[raw_key]
            conformers = entry["conformers"]
            mol = Chem.Mol(conformers[0]["rd_mol"])
            atomic_numbers = tuple(atom.GetAtomicNum() for atom in mol.GetAtoms())
            references = []
            for conformer in conformers:
                ref_mol = conformer["rd_mol"]
                if tuple(atom.GetAtomicNum() for atom in ref_mol.GetAtoms()) != atomic_numbers:
                    raise RuntimeError("Reference ensemble atom ordering changed")
                references.append(np.asarray(ref_mol.GetConformer().GetPositions(), dtype=np.float32))
            graph = prepare_graph(formal_record(mol), calibration)
            items.append(
                {
                    "final_molecule_index": int(row["selection_rank"]),
                    "molecule_id": row["molecule_id"],
                    "molecule_identity_sha256": row["molecule_identity_sha256"],
                    "mol_binary": mol.ToBinary(),
                    "graph": graph,
                    "references": torch.from_numpy(np.stack(references)),
                    "source_records": sorted(by_rank[int(row["selection_rank"])], key=lambda value: int(value["etflow_record_index"])),
                }
            )
        atomic_torch(chunk, {"relative_source_shard": relative, "items": items})
        del payload, items
    atomic_json(
        complete,
        {
            "status": "PASS",
            "chunks": [str(path) for path in chunks],
            "molecules": len(selected_primary_rows),
            "records": 2 * len(selected_primary_rows),
            "protocol_sha256": sha256_file(args.protocol),
            "reference_used_at_model_inference": False,
        },
    )
    return chunks


def load_model(method: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    if method["formulation"] == "Restricted":
        model: torch.nn.Module = FullJointModel(128, 3)
    elif method["formulation"] == "Unrestricted":
        model = UnrestrictedFullJointModel(128, 3)
    else:
        raise ValueError(f"unknown formulation: {method['formulation']}")
    payload = torch.load(method["checkpoint"], map_location="cpu", weights_only=False)
    if int(payload.get("step", payload.get("optimizer_steps", -1))) != 17500:
        raise RuntimeError(f"checkpoint is not step 17500: {method['id']}")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    if not all(parameter.device.type == "cuda" for parameter in model.parameters()):
        raise RuntimeError("model parameters are not on CUDA")
    return model


def infer_method(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    method: Mapping[str, Any],
    chunks: list[Path],
    *,
    smoke: bool,
) -> None:
    target = args.output_dir / ("smoke" if smoke else "methods") / method["id"]
    output_chunks = target / "chunks"
    device = torch.device("cuda:0")
    model = load_model(method, device)
    processed = 0
    for chunk_index, topology_path in enumerate(chunks):
        if smoke and processed >= args.smoke_molecules:
            break
        output_path = output_chunks / topology_path.name
        if output_path.is_file() and not smoke:
            continue
        topology = torch.load(topology_path, map_location="cpu", weights_only=False)
        flat: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for item in topology["items"]:
            if smoke and processed >= args.smoke_molecules:
                break
            for source_manifest_row in item["source_records"]:
                source_payload = torch.load(source_manifest_row["record_asset"], map_location="cpu", weights_only=False)
                flat.append((item, source_manifest_row, source_payload))
            processed += 1
        rows = []
        for start in range(0, len(flat), 64):
            batch = flat[start : start + 64]
            graphs = [item[0]["graph"] for item in batch]
            collated = collate_graphs(graphs).to(device)
            source = torch.cat([torch.as_tensor(item[2]["source_coordinates"], dtype=torch.float64) for item in batch]).to(device)
            with torch.inference_mode():
                prediction = model.belief(collated, detach_sigma_features=False)
                if method["formulation"] == "Restricted":
                    action = full_joint_action(model, source, graphs, prediction, atom_cap=0.03)
                else:
                    action = unrestricted_full_joint_action(model, source, graphs, prediction)
            if not bool(torch.isfinite(action.proposal).all()):
                raise RuntimeError(f"nonfinite proposal: {method['id']}")
            atom_offset = bond_offset = angle_offset = 0
            for local, (item, source_manifest_row, source_payload) in enumerate(batch):
                graph = item["graph"]
                atom_count = int(graph.atom_categorical.size(0))
                bond_count = int(graph.bonds.size(1))
                angle_count = int(graph.angles.size(0))
                aa = slice(atom_offset, atom_offset + atom_count)
                bb = slice(bond_offset, bond_offset + bond_count)
                cc = slice(angle_offset, angle_offset + angle_count)
                atom_offset += atom_count
                bond_offset += bond_count
                angle_offset += angle_count
                source_local = source[aa].detach().cpu()
                proposal = action.proposal[aa].detach().cpu()
                row = {
                    "record_id": source_manifest_row["record_id"],
                    "final_molecule_index": item["final_molecule_index"],
                    "molecule_id": item["molecule_id"],
                    "molecule_identity_sha256": item["molecule_identity_sha256"],
                    "etflow_record_index": int(source_manifest_row["etflow_record_index"]),
                    "method": method["id"],
                    "formulation": method["formulation"],
                    "seed": int(method["seed"]),
                    "source_record_asset": source_manifest_row["record_asset"],
                    "proposal_coordinates": proposal.float(),
                    "tau": float(action.tau[local]),
                    "w_B": float(action.family_weights[local, 0]),
                    "w_A": float(action.family_weights[local, 1]),
                    "atom_cap_active": bool(action.cap_active[local]),
                    "bond_sigma": prediction["bond_sigma"][bb].detach().cpu().float(),
                    "angle_sigma": prediction["angle_sigma"][cc].detach().cpu().float(),
                    "bond_reliability": action.bond_reliability[bb].detach().cpu().float(),
                    "angle_reliability": action.angle_reliability[cc].detach().cpu().float(),
                    **metric_row(source_local, proposal, item["references"], graph),
                    "checkpoint_sha256": method["checkpoint_sha256"],
                    "protocol_sha256": sha256_file(args.protocol),
                    "reference_used_at_inference": False,
                }
                rows.append(row)
        atomic_torch(output_path, rows)
        if not smoke:
            update_status(args.report_dir, "MODEL_INFERENCE", method=method["id"], topology_chunks_complete=chunk_index + 1, topology_chunks_total=len(chunks))
    if smoke:
        atomic_json(target / "SMOKE_STATUS.json", {"status": "PASS", "method": method["id"], "molecules": processed, "scientific_metrics_reported": False})


def run_mmff(args: argparse.Namespace, protocol: Mapping[str, Any], chunks: list[Path]) -> None:
    method_id = "mmff94s"
    output_chunks = args.output_dir / "methods" / method_id / "chunks"
    config = protocol["mmff94s"]
    for chunk_index, topology_path in enumerate(chunks):
        output_path = output_chunks / topology_path.name
        if output_path.is_file():
            continue
        topology = torch.load(topology_path, map_location="cpu", weights_only=False)
        rows = []
        for item in topology["items"]:
            mol = Chem.Mol(item["mol_binary"])
            record = formal_record(mol)
            record["atomic_numbers"] = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])
            for source_manifest_row in item["source_records"]:
                source_payload = torch.load(source_manifest_row["record_asset"], map_location="cpu", weights_only=False)
                source = torch.as_tensor(source_payload["source_coordinates"], dtype=torch.float32)
                result = refine_with_mmff94s(record, source, config)
                candidate = result.refined_coordinates.detach().cpu().float()
                rows.append(
                    {
                        "record_id": source_manifest_row["record_id"],
                        "final_molecule_index": item["final_molecule_index"],
                        "molecule_id": item["molecule_id"],
                        "molecule_identity_sha256": item["molecule_identity_sha256"],
                        "etflow_record_index": int(source_manifest_row["etflow_record_index"]),
                        "method": method_id,
                        "formulation": "MMFF94s",
                        "seed": None,
                        "source_record_asset": source_manifest_row["record_asset"],
                        "proposal_coordinates": candidate,
                        "mmff_success": bool(result.success),
                        "mmff_converged": bool(result.converged),
                        "mmff_fallback_to_source": bool(result.fallback_to_source),
                        "mmff_failure_reason": result.failure_reason,
                        "mmff_runtime_seconds": float(result.runtime_seconds),
                        "mmff_method_version": result.method_version,
                        "mmff_initial_native_energy": result.initial_native_energy,
                        "mmff_final_native_energy": result.final_native_energy,
                        **metric_row(source, candidate, item["references"], item["graph"]),
                        "protocol_sha256": sha256_file(args.protocol),
                        "reference_used_at_inference": False,
                    }
                )
        atomic_torch(output_path, rows)
        update_status(args.report_dir, "MMFF94S", topology_chunks_complete=chunk_index + 1, topology_chunks_total=len(chunks))


def write_sdf(path: Path, ordered_rows: list[dict[str, Any]], topology_by_rank: dict[int, dict[str, Any]], method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    writer = Chem.SDWriter(str(temporary))
    try:
        for row in ordered_rows:
            mol = Chem.Mol(topology_by_rank[int(row["final_molecule_index"])]["mol_binary"])
            coordinates = torch.as_tensor(row["proposal_coordinates"], dtype=torch.float64)
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for index, point in enumerate(coordinates.tolist()):
                conformer.SetAtomPosition(index, point)
            mol.RemoveAllConformers()
            mol.AddConformer(conformer, assignId=True)
            mol.SetProp("_Name", str(row["record_id"]))
            mol.SetProp("sample_id", str(row["record_id"]))
            mol.SetProp("method", method)
            writer.write(mol)
    finally:
        writer.close()
    os.replace(temporary, path)


def materialize(args: argparse.Namespace, protocol: Mapping[str, Any], chunks: list[Path]) -> None:
    topology_by_rank: dict[int, dict[str, Any]] = {}
    source_rows_out: list[dict[str, Any]] = []
    for path in chunks:
        for item in torch.load(path, map_location="cpu", weights_only=False)["items"]:
            rank = int(item["final_molecule_index"])
            topology_by_rank[rank] = {"mol_binary": item["mol_binary"]}
            for source_manifest_row in item["source_records"]:
                payload = torch.load(source_manifest_row["record_asset"], map_location="cpu", weights_only=False)
                source = torch.as_tensor(payload["source_coordinates"], dtype=torch.float32)
                source_rows_out.append(
                    {
                        "record_id": source_manifest_row["record_id"],
                        "final_molecule_index": rank,
                        "molecule_id": item["molecule_id"],
                        "molecule_identity_sha256": item["molecule_identity_sha256"],
                        "etflow_record_index": int(source_manifest_row["etflow_record_index"]),
                        "method": "source",
                        "formulation": "Source",
                        "seed": None,
                        "proposal_coordinates": source,
                        **metric_row(source, source, item["references"], item["graph"]),
                    }
                )
    methods = [method["id"] for method in protocol["model_methods"]] + ["mmff94s"]
    source_rows_out.sort(key=lambda row: (row["final_molecule_index"], row["etflow_record_index"]))
    all_methods = [("source", source_rows_out)]
    for method in methods:
        values = []
        for path in chunks:
            chunk_path = args.output_dir / "methods" / method / "chunks" / path.name
            if not chunk_path.is_file():
                raise RuntimeError(f"method chunk missing: {chunk_path}")
            values.extend(torch.load(chunk_path, map_location="cpu", weights_only=False))
        values.sort(key=lambda row: (row["final_molecule_index"], row["etflow_record_index"]))
        all_methods.append((method, values))
    expected_ids = [row["record_id"] for row in source_rows_out]
    for method, rows in all_methods:
        if len(rows) != TARGET_RECORDS or [row["record_id"] for row in rows] != expected_ids:
            raise RuntimeError(f"method record alignment failure: {method}")
        target = args.output_dir / "methods" / method
        write_sdf(target / "COORDINATES.sdf", rows, topology_by_rank, method)
        scalar_rows = []
        for row in rows:
            scalar_rows.append({key: value for key, value in row.items() if key not in {"proposal_coordinates", "bond_sigma", "angle_sigma", "bond_reliability", "angle_reliability"}})
        atomic_parquet(target / "PER_RECORD.parquet", pd.DataFrame(scalar_rows))
        atomic_json(
            target / "COORDINATES_READY.json",
            {
                "status": "PASS",
                "method": method,
                "molecules": TARGET_MOLECULES,
                "records": TARGET_RECORDS,
                "record_ids_sha256": hashlib.sha256("\n".join(expected_ids).encode()).hexdigest(),
                "sdf_sha256": sha256_file(target / "COORDINATES.sdf"),
                "per_record_sha256": sha256_file(target / "PER_RECORD.parquet"),
                "protocol_sha256": sha256_file(args.protocol),
                "model_training_performed": False,
            },
        )
    update_status(args.report_dir, "COORDINATES_READY", state="PASS", methods=len(all_methods), records_per_method=TARGET_RECORDS)


def preflight(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    cuda = torch.cuda.is_available()
    if not cuda:
        raise RuntimeError("CUDA unavailable; frozen neural inference did not start")
    snapshot = {
        "status": "PASS",
        "cuda_available": cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "rdkit": rdBase.rdkitVersion,
        "protocol_sha256": sha256_file(args.protocol),
        "model_methods": len(protocol["model_methods"]),
        "checkpoint_steps": sorted({int(method["step"]) for method in protocol["model_methods"]}),
        "model_training_performed": False,
    }
    atomic_json(args.report_dir / "01_EVALUATION_PREFLIGHT.json", snapshot)
    update_status(args.report_dir, "PREFLIGHT", state="PASS", **snapshot)


def main(args: argparse.Namespace) -> int:
    started = time.time()
    protocol = load_protocol(args)
    preflight(args, protocol)
    chunks = build_topology_cache(args, protocol, smoke=args.stage == "smoke")
    if args.stage == "smoke":
        for method in protocol["model_methods"]:
            infer_method(args, protocol, method, chunks, smoke=True)
        atomic_json(args.report_dir / "02_ENGINEERING_SMOKE_STATUS.json", {"status": "PASS", "molecules": args.smoke_molecules, "methods": len(protocol["model_methods"]), "scientific_metrics_reported": False})
        return 0
    if args.stage in {"coordinates", "pipeline"}:
        for index, method in enumerate(protocol["model_methods"], start=1):
            update_status(args.report_dir, "MODEL_INFERENCE", method=method["id"], method_index=index, method_total=len(protocol["model_methods"]))
            infer_method(args, protocol, method, chunks, smoke=False)
    if args.stage in {"mmff", "pipeline"}:
        run_mmff(args, protocol, chunks)
    if args.stage in {"materialize", "pipeline"}:
        materialize(args, protocol, chunks)
    update_status(args.report_dir, "COORDINATE_PIPELINE_COMPLETE", state="PASS", elapsed_seconds=time.time() - started)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "coordinates", "mmff", "materialize", "pipeline"), default="pipeline")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-asset-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--smoke-molecules", type=int, default=3)
    args = parser.parse_args()
    for name in ("protocol", "primary", "source_manifest", "source_asset_freeze", "output_dir", "report_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
