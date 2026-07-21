#!/usr/bin/env python
"""Run a resumable RAW, MMFF94s, or GFN2-xTB refinement cache."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import torch
from rdkit import rdBase

from etflow.ecir.external_refinement_baselines import (
    ISOLATION,
    ExternalRefinementResult,
    canonical_sha256,
    coordinate_sha256,
    derive_total_charge,
    derive_unpaired_electrons,
    mol_from_frozen_record,
    ordered_atom_identity_sha256,
    refine_with_gfn2_xtb,
    refine_with_mmff94s,
    serialize_refinement_record,
    topology_identity_sha256,
)
from etflow.ecir.v8_validation_cache import (
    append_prediction_chunk,
    atomic_json,
    completed_chunk_ranges,
    file_sha256,
    finish_prediction_manifest,
    iter_prediction_records,
    new_prediction_manifest,
    prediction_identity,
    utc_now,
    validate_manifest_identity,
)


def _status(path: Path, *, phase: str, method: str, total: int, completed: int,
            started: float, worker_count: int, threads: int, records: list[dict[str, Any]],
            state: str | None = None, error: str | None = None) -> None:
    elapsed = time.perf_counter() - started
    successes = sum(int(row["success"]) for row in records)
    fallbacks = sum(int(row["fallback_to_source"]) for row in records)
    payload = {
        "schema_version": "mcvr-external-refinement-live-status-v1",
        "status": state or f"{phase}_RUNNING",
        "phase": phase,
        "method": method,
        "total_records": total,
        "completed_records": completed,
        "successful_records": successes,
        "fallback_records": fallbacks,
        "timeout_records": sum(int(row.get("timeout", False)) for row in records),
        "unsupported_records": sum(int(row.get("unsupported", False)) for row in records),
        "convergence_failed_records": sum(int(not row["converged"] and not row.get("timeout", False) and not row.get("unsupported", False)) for row in records),
        "records_per_second": completed / max(elapsed, 1.0e-9),
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": (total - completed) * elapsed / completed if completed else None,
        "worker_count": worker_count,
        "threads_per_worker": threads,
        "current_record": records[-1]["record_index"] if records else None,
        "last_update_time": utc_now(),
        "error": error,
        **ISOLATION,
    }
    atomic_json(path, payload)


def _raw_result(source: torch.Tensor, record: dict[str, Any]) -> ExternalRefinementResult:
    mol = mol_from_frozen_record(record, source)
    return ExternalRefinementResult(
        method="RAW", method_version="frozen-source-cache-v1", source_coordinates=source.clone(),
        refined_coordinates=source.clone(), success=True, converged=True,
        fallback_to_source=False, failure_reason=None, runtime_seconds=0.0,
        atom_count_before=int(source.shape[0]), atom_count_after=int(source.shape[0]),
        atom_order_verified=True, topology_verified=True, chirality_verified=True,
        total_charge=derive_total_charge(mol), unpaired_electrons=derive_unpaired_electrons(mol),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("RAW", "MMFF94S", "GFN2_XTB"), required=True)
    parser.add_argument("--phase", choices=("SMOKE100", "FAST1000", "FULL10K"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ecir_external_refinement_baselines.json")
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--chunk-size", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config.get("isolation") != ISOLATION:
        raise RuntimeError("external refinement isolation contract changed")
    source_manifest = (ROOT / config["source_cache_manifest"]).resolve()
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    all_rows = list(iter_prediction_records(source_manifest))
    selection_path = args.selection_manifest
    if selection_path is None and args.phase == "SMOKE100":
        selection_path = ROOT / config["smoke100_manifest"]
    if selection_path is None and args.phase == "FAST1000":
        selection_path = ROOT / config["fast1000_manifest"]
    if selection_path is not None:
        selection = json.loads(selection_path.resolve().read_text(encoding="utf-8"))
        indices = [int(value) for value in selection["record_indices"]]
        if any(index < 0 or index >= len(all_rows) for index in indices):
            raise RuntimeError("selection manifest index is out of range")
        rows = [all_rows[index] for index in indices]
        if [str(row["sample_id"]) for row in rows] != [str(value) for value in selection["record_ids"]]:
            raise RuntimeError("selection manifest sample identity changed")
    else:
        rows = all_rows
    method_key = args.method.lower()
    output = (ROOT / config["output_root"] / method_key / args.phase.lower()).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output.parent / "status.json"
    method_config = config["mmff94s"] if args.method == "MMFF94S" else config["gfn2_xtb"] if args.method == "GFN2_XTB" else {"method": "RAW"}
    method_config_sha = canonical_sha256(method_config)
    evaluator_hash = source_payload["identity"]["evaluator_semantics_sha256"]
    safety_hash = canonical_sha256(config["safety"])
    identity = prediction_identity(
        checkpoint_sha256="EVALUATION_ONLY_NO_CHECKPOINT", config_sha256=method_config_sha,
        validation_sources_sha256=source_payload["identity"]["validation_sources_sha256"],
        validation_targets_sha256=source_payload["identity"]["validation_targets_sha256"],
        record_ids=[str(row["sample_id"]) for row in rows], evaluator_semantics_sha256=evaluator_hash,
        safety_semantics_sha256=safety_hash, method=args.method,
    )
    manifest_path = output / "prediction_manifest.json"
    chunk_size = int(args.chunk_size or config["chunk_size"])
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest_identity(manifest, identity)
        if manifest.get("status") == "COMPLETED":
            print(json.dumps({"status": f"{args.phase}_COMPLETED", "resumed": True, "records": len(rows)}))
            return
    else:
        manifest = new_prediction_manifest(identity, chunk_size=chunk_size, output_dir=output)
        manifest.update({"external_refinement_schema": "v1", "method_config": method_config, "phase": args.phase})
        atomic_json(manifest_path, manifest)
    completed = completed_chunk_ranges(manifest)
    prior_records = list(iter_prediction_records(manifest_path, require_completed=False))
    started = time.perf_counter()
    workers = int(args.worker_count or method_config.get("worker_count", 1))
    threads = int(method_config.get("omp_threads_per_process", method_config.get("num_threads", 1)))
    _status(status_path, phase=args.phase, method=args.method, total=len(rows), completed=len(prior_records), started=started, worker_count=workers, threads=threads, records=prior_records)

    def process(entry: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        offset, row = entry
        record = row["record"]
        source = row["item"].x_input.detach().cpu().to(torch.float32)
        before_mol = mol_from_frozen_record(record, source)
        if args.method == "RAW":
            result = _raw_result(source, record)
        elif args.method == "MMFF94S":
            result = refine_with_mmff94s(record, source, method_config)
        else:
            work = ROOT / method_config["scratch_root"] / f"record_{int(row['record_index']):06d}_{canonical_sha256(str(row['sample_id']))[:12]}"
            result = refine_with_gfn2_xtb(record, source, method_config, work)
        return serialize_refinement_record(
            result, record_index=int(row["record_index"]), sample_id=str(row["sample_id"]),
            molecule_id=str(row["molecule_id"]),
            source_record_sha256=canonical_sha256({"record_index": int(row["record_index"]), "sample_id": str(row["sample_id"]), "source_coordinate_sha256": coordinate_sha256(source), "topology_signature": str(record["topology_signature"])}),
            ordered_atom_identity_sha256=ordered_atom_identity_sha256(before_mol, record),
            topology_identity_sha256=topology_identity_sha256(before_mol, record),
            method_config_sha256=method_config_sha,
        )

    try:
        for start in range(0, len(rows), chunk_size):
            end = min(start + chunk_size, len(rows))
            if (start, end) in completed:
                continue
            entries = list(enumerate(rows[start:end], start=start))
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    chunk_records = list(executor.map(process, entries))
            else:
                chunk_records = [process(entry) for entry in entries]
            append_prediction_chunk(manifest_path, manifest, record_start=start, record_end=end, records=chunk_records)
            prior_records.extend(chunk_records)
            _status(status_path, phase=args.phase, method=args.method, total=len(rows), completed=end, started=started, worker_count=workers, threads=threads, records=prior_records)
        finish_prediction_manifest(manifest_path, manifest)
        _status(status_path, phase=args.phase, method=args.method, total=len(rows), completed=len(rows), started=started, worker_count=workers, threads=threads, records=prior_records, state=f"{args.phase}_COMPLETED")
        print(json.dumps({"status": f"{args.phase}_COMPLETED", "records": len(rows), "successes": sum(int(row["success"]) for row in prior_records), "fallbacks": sum(int(row["fallback_to_source"]) for row in prior_records), "elapsed_seconds": time.perf_counter() - started}, indent=2))
    except BaseException as error:
        _status(status_path, phase=args.phase, method=args.method, total=len(rows), completed=len(prior_records), started=started, worker_count=workers, threads=threads, records=prior_records, state="FAILED_CLOSED", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
