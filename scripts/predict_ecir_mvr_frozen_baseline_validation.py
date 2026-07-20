#!/usr/bin/env python
"""Materialize frozen D1, V5-B, or V7 validation prediction caches."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch
import yaml

from etflow.ecir.bac_evaluation import infer_bac
from etflow.ecir.bac_safety import BACSafetyConfig
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.mvr_v7_formal import load_v7_formal_config
from etflow.ecir.v8_validation_cache import (
    ValidationStatus,
    append_prediction_chunk,
    atomic_json,
    canonical_sha256,
    completed_chunk_ranges,
    file_sha256,
    finish_prediction_manifest,
    iter_prediction_records,
    new_prediction_manifest,
    prediction_identity,
    tensor_sha256,
    validate_manifest_identity,
)
from scripts.run_ecir_mvr_v7_formal_validation import _load_models


EVALUATOR_SHA = "47f123b59933d9f856bb65133082594451273b587010a25328756735d5e73bc1"
SAFETY_SHA = "1664147678a65086c23cd8df57af79a2a573da1df25ba9f762c5edf4b7b6614a"


def _items(rows):
    return [
        {
            "data": row["item"],
            "input": row["item"].x_input,
            "minimal_target": row["item"].x_target,
            "references": torch.as_tensor(
                row["record"].get("x_ref_candidates", row["record"].get("x_ref_aligned"))
            ),
            "record": row["record"],
            "row": SimpleNamespace(
                sample_id=row["sample_id"],
                molecule_id=row["molecule_id"],
                source_severity=str(row["item"].severity),
                generator_name=str(row["item"].source),
            ),
            "rotatable": int(torch.as_tensor(row["item"].num_rotatable_bonds).max()),
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("D1", "V5-B", "V7"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v7-config", type=Path, required=True)
    parser.add_argument("--v5-config", type=Path, required=True)
    parser.add_argument("--source-cache-manifest", type=Path, required=True)
    parser.add_argument("--validity-statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    v7_path = args.v7_config.resolve()
    v5_path = args.v5_config.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    wrapper = load_v7_formal_config(v7_path)
    v5_config = yaml.safe_load(v5_path.read_text(encoding="utf-8"))
    source_manifest_path = args.source_cache_manifest.resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_rows = list(iter_prediction_records(source_manifest_path))
    record_ids = [str(row["sample_id"]) for row in source_rows]
    config_sha = {
        "D1": canonical_sha256(checkpoint["config"]),
        "V5-B": file_sha256(v5_path),
        "V7": file_sha256(v7_path),
    }[args.method]
    identity = prediction_identity(
        method=args.method,
        checkpoint_sha256=file_sha256(checkpoint_path),
        config_sha256=config_sha,
        validation_sources_sha256=source_manifest["identity"]["validation_sources_sha256"],
        validation_targets_sha256=source_manifest["identity"]["validation_targets_sha256"],
        record_ids=record_ids,
        evaluator_semantics_sha256=EVALUATOR_SHA,
        safety_semantics_sha256=SAFETY_SHA,
    )
    output = args.output_dir.resolve()
    manifest_path = output / "prediction_manifest.json"
    status = ValidationStatus.start(
        output / "status.json",
        phase="FULL_PREDICTING",
        training_step=25000,
        validation_mode=f"BASELINE_{args.method}",
    )
    started = time.perf_counter()
    try:
        if manifest_path.exists():
            if not args.resume:
                raise RuntimeError("baseline prediction cache exists; use --resume")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_manifest_identity(manifest, identity)
            list(iter_prediction_records(manifest_path, require_completed=False))
        else:
            manifest = new_prediction_manifest(identity, chunk_size=args.chunk_size, output_dir=output)
            manifest["source_cache_identity_sha256"] = source_manifest["identity"]["identity_sha256"]
            atomic_json(manifest_path, manifest)
        completed = completed_chunk_ranges(manifest)
        device = torch.device(args.device)
        models = _load_models(checkpoint, wrapper, v5_config, device)
        model = models[args.method]
        validity = ChemicalValidity(args.validity_statistics.resolve())
        inference = wrapper["inference"]
        safety_settings = dict(inference.get("safety", {}))
        safety_settings["objective_mode"] = "weighted_thresholded_validity"
        count = len(source_rows)
        for start in range(0, count, args.chunk_size):
            end = min(start + args.chunk_size, count)
            if (start, end) in completed:
                continue
            selected_rows = source_rows[start:end]
            selected_items = _items(selected_rows)
            accepted, metadata = infer_bac(
                model,
                selected_items,
                validity,
                device=device,
                steps=int(inference.get("teacher_steps", 4)),
                step_size=float(inference.get("step_size", 0.25)),
                batch_size=int(inference.get("batch_size", 64)),
                safety_config=BACSafetyConfig(**safety_settings),
                trajectory_semantics="formal_d1b",
            )
            cache_rows = []
            for source_row, safe, meta in zip(selected_rows, accepted, metadata, strict=True):
                source = source_row["item"].x_input.detach().cpu()
                raw = torch.as_tensor(meta.pop("raw_prediction"), dtype=source.dtype)
                decision = {
                    "accepted": bool(meta["accepted"]),
                    "selected_scale": float(meta["selected_scale"]),
                    "rolled_back": bool(meta["rolled_back"]),
                    "reasons": [value for value in str(meta["reject_reasons"]).split(";") if value],
                    "selected_step": int(meta["selected_step"]),
                }
                cache_rows.append(
                    {
                        "record_index": int(source_row["record_index"]),
                        "sample_id": str(source_row["sample_id"]),
                        "molecule_id": str(source_row["molecule_id"]),
                        "ordered_atom_identity": torch.as_tensor(
                            source_row["record"].get(
                                "atom_map_ids", source_row["record"].get("atomic_numbers", [])
                            )
                        ).tolist(),
                        "source_coordinate_sha256": tensor_sha256(source),
                        "checkpoint_sha256": identity["checkpoint_sha256"],
                        "raw_coordinates": raw,
                        "safe_coordinates": torch.as_tensor(safe).detach().cpu(),
                        "accepted": bool(meta["accepted"]),
                        "rollback": bool(meta["rolled_back"]),
                        "backtracking_decision": decision,
                        "displacement": torch.linalg.vector_norm(safe - source, dim=-1),
                        "solver_diagnostics": {
                            "failure_count": 0.0,
                            "bond_contribution": float(meta.get("neural_delta", {}).get("rms", 0.0)),
                            "angle_contribution": float(meta.get("angle_delta", {}).get("rms", 0.0)),
                        },
                        "method_diagnostics": {"baseline_metadata": meta},
                    }
                )
            append_prediction_chunk(
                manifest_path,
                manifest,
                record_start=start,
                record_end=end,
                records=cache_rows,
            )
            elapsed = time.perf_counter() - started
            status.update(
                current_validation_record=end,
                prediction_chunks_completed=len(manifest["chunks"]),
                records_per_second=end / max(elapsed, 1.0e-9),
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=(count - end) * elapsed / max(end, 1),
            )
        finish_prediction_manifest(manifest_path, manifest)
        elapsed = time.perf_counter() - started
        status.update(
            status="COMPLETED",
            phase="COMPLETED",
            current_validation_record=count,
            prediction_chunks_completed=len(manifest["chunks"]),
            records_per_second=count / max(elapsed, 1.0e-9),
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=0.0,
        )
    except BaseException as error:
        status.fail(error, elapsed_seconds=time.perf_counter() - started)
        raise


if __name__ == "__main__":
    main()
