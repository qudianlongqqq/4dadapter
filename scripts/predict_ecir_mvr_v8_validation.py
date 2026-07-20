#!/usr/bin/env python
"""Run V8 once and write an immutable, resumable validation prediction cache."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch
from torch_geometric.data import Batch

from etflow.ecir.bac_safety import BACSafetyConfig, select_safe_bac_proposal
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.mcvr_v8_full import MCVRV8FullRefiner
from etflow.ecir.v8_constraint_normalization import FrozenResidualScales
from etflow.ecir.v8_validation_cache import (
    ISOLATION,
    ValidationStatus,
    append_prediction_chunk,
    atomic_json,
    completed_chunk_ranges,
    file_sha256,
    finish_prediction_manifest,
    iter_prediction_records,
    new_prediction_manifest,
    prediction_identity,
    tensor_sha256,
    validate_manifest_identity,
)


EVALUATOR_SHA = "47f123b59933d9f856bb65133082594451273b587010a25328756735d5e73bc1"
SAFETY_SHA = "1664147678a65086c23cd8df57af79a2a573da1df25ba9f762c5edf4b7b6614a"


def _model(checkpoint: dict, device: torch.device) -> MCVRV8FullRefiner:
    config = checkpoint["resolved_config"]
    constraint = dict(config["constraint_layer"])
    for key in ("frozen_scales", "frozen_scales_sha256", "use_frozen_scales"):
        constraint.pop(key, None)
    unroll = int(constraint.pop("unroll_steps"))
    model = MCVRV8FullRefiner.from_d1_checkpoint(
        config["model"]["d1_checkpoint"],
        expected_sha256=config["model"]["d1_checkpoint_sha256"],
        error_state=config["error_state"],
        constraint_layer=constraint,
        residual_scales=FrozenResidualScales.from_mapping(checkpoint["residual_scales"]),
        unroll_steps=unroll,
        step_embedding_enabled=config["model"]["step_embedding_enabled"],
        error_state_enabled=config["error_state"]["enabled"],
        train_d1_backbone=config["model"]["train_d1_backbone"],
        train_d1_head=config["model"]["train_d1_head"],
        max_cumulative_atom_displacement=config["safety"]["max_atom_displacement"],
        max_cumulative_graph_rms=config["safety"]["graph_rms_limit"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def _selected_source_records(source_manifest: Path, fast_manifest: Path | None):
    rows = iter_prediction_records(source_manifest)
    if fast_manifest is None:
        return list(rows)
    fast = json.loads(fast_manifest.read_text(encoding="utf-8"))
    selected = set(int(index) for index in fast["record_indices"])
    result = [row for row in rows if int(row["record_index"]) in selected]
    if [row["sample_id"] for row in result] != list(fast["record_ids"]):
        raise RuntimeError("FAST manifest identity/order changed")
    return result


def _compact_decision(decision: dict) -> dict:
    return {
        "accepted": bool(decision["accepted"]),
        "selected_scale": float(decision["selected_scale"]),
        "rolled_back": bool(decision["rolled_back"]),
        "reasons": [str(value) for value in decision.get("reasons", ())],
        "attempts": [
            {
                "scale": float(attempt["scale"]),
                "accepted": bool(attempt["accepted"]),
                "reasons": [str(value) for value in attempt.get("reasons", ())],
            }
            for attempt in decision.get("attempts", ())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-cache-manifest", type=Path, required=True)
    parser.add_argument("--validity-statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--fast-manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--inference-mode",
        action="store_true",
        help="Audit candidate; legacy-compatible no_grad remains the parity default.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.chunk_size < 1:
        raise ValueError("prediction batch and chunk sizes must be positive")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "mcvr-v8-full-v1-checkpoint-v1":
        raise RuntimeError("V8 validation checkpoint schema changed")
    for key, value in ISOLATION.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"V8 checkpoint isolation changed: {key}")
    source_manifest_path = args.source_cache_manifest.resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "COMPLETED":
        raise RuntimeError("source cache is incomplete")
    source_rows = _selected_source_records(source_manifest_path, args.fast_manifest)
    record_ids = [str(row["sample_id"]) for row in source_rows]
    source_identity = source_manifest["identity"]
    identity = prediction_identity(
        checkpoint_sha256=file_sha256(checkpoint_path),
        config_sha256=str(checkpoint["resolved_config_sha256"]),
        validation_sources_sha256=source_identity["validation_sources_sha256"],
        validation_targets_sha256=source_identity["validation_targets_sha256"],
        record_ids=record_ids,
        evaluator_semantics_sha256=EVALUATOR_SHA,
        safety_semantics_sha256=SAFETY_SHA,
    )
    output = args.output_dir.resolve()
    manifest_path = output / "prediction_manifest.json"
    mode = "FAST" if args.fast_manifest else "FULL"
    status = ValidationStatus.start(
        args.status_file.resolve() if args.status_file else output / "status.json",
        phase="FAST_VALIDATING" if mode == "FAST" else "FULL_PREDICTING",
        training_step=int(checkpoint["step"]),
        validation_mode=mode,
    )
    started = time.perf_counter()
    try:
        if manifest_path.exists():
            if not args.resume:
                raise RuntimeError("prediction output exists; use --resume")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_manifest_identity(manifest, identity)
            list(iter_prediction_records(manifest_path, require_completed=False))
        else:
            manifest = new_prediction_manifest(identity, chunk_size=args.chunk_size, output_dir=output)
            manifest["prediction_batch_size"] = args.batch_size
            manifest["inference_context"] = "inference_mode" if args.inference_mode else "no_grad"
            manifest["source_cache_identity_sha256"] = source_identity["identity_sha256"]
            manifest["fast_manifest_sha256"] = (
                file_sha256(args.fast_manifest) if args.fast_manifest else None
            )
            atomic_json(manifest_path, manifest)
        completed = completed_chunk_ranges(manifest)
        device = torch.device(args.device)
        model = _model(checkpoint, device)
        validity = ChemicalValidity(args.validity_statistics.resolve())
        safety = BACSafetyConfig(
            max_atom_displacement=0.12,
            max_molecule_rms_displacement=0.06,
            enable_backtracking=True,
            objective_mode="weighted_thresholded_validity",
        )
        count = len(source_rows)
        context = torch.inference_mode() if args.inference_mode else torch.no_grad()
        with context:
            for start in range(0, count, args.chunk_size):
                end = min(start + args.chunk_size, count)
                if (start, end) in completed:
                    continue
                cache_rows = []
                for batch_start in range(start, end, args.batch_size):
                    selected = source_rows[batch_start : min(batch_start + args.batch_size, end)]
                    items = [row["item"] for row in selected]
                    batch = Batch.from_data_list(items).to(device)
                    output_values = model(batch, batch.x_input, batch.x_input.new_full((batch.num_graphs,), 0.5))
                    raw_all = output_values["x_final"].detach().cpu()
                    confidence = output_values["bounded_prior_confidence"].detach().cpu()
                    ptr = batch.ptr.detach().cpu().tolist()
                    for local, (source_row, item) in enumerate(zip(selected, items, strict=True)):
                        left, right = ptr[local], ptr[local + 1]
                        raw = raw_all[left:right]
                        source = item.x_input.detach().cpu()
                        safe, decision = select_safe_bac_proposal(
                            source,
                            raw - source,
                            source_row["record"],
                            validity,
                            safety,
                        )
                        last = output_values["step_outputs"][-1]
                        solver_failures = sum(
                            float(step["solver_failure"][local].detach().cpu())
                            for step in output_values["step_outputs"]
                        )
                        atom_identity = source_row["record"].get(
                            "atom_map_ids",
                            source_row["record"].get("atomic_numbers", []),
                        )
                        cache_rows.append(
                            {
                                "record_index": int(source_row["record_index"]),
                                "sample_id": str(source_row["sample_id"]),
                                "molecule_id": str(source_row["molecule_id"]),
                                "ordered_atom_identity": torch.as_tensor(atom_identity).tolist(),
                                "source_coordinate_sha256": tensor_sha256(source),
                                "checkpoint_sha256": identity["checkpoint_sha256"],
                                "raw_coordinates": raw,
                                "safe_coordinates": safe.detach().cpu(),
                                "accepted": bool(decision["accepted"]),
                                "rollback": bool(decision["rolled_back"]),
                                "backtracking_decision": _compact_decision(decision),
                                "displacement": torch.linalg.vector_norm(safe - source, dim=-1),
                                "solver_diagnostics": {
                                    "failure_count": solver_failures,
                                    "bond_contribution": float(
                                        last["solver_bond_contribution"][local].detach().cpu()
                                    ),
                                    "angle_contribution": float(
                                        last["solver_angle_contribution"][local].detach().cpu()
                                    ),
                                },
                                "method_diagnostics": {
                                    "confidence": confidence[left:right],
                                    "unroll_steps": int(output_values["unroll_steps"]),
                                },
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
