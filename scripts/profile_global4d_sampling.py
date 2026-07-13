#!/usr/bin/env python
"""Reproducible, bounded profiler for Global Coupled 4D sampling.

Unlike the formal sampler this command never writes a final evaluator payload.
It processes an explicitly bounded manifest prefix and writes compact timing
reports.  Resume-write simulation is optional and isolated under output_dir.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch
import yaml

from etflow.commons.global4d_performance import (
    CudaEventTimer,
    PROFILE_SCHEMA_VERSION,
    StageAccumulator,
    benchmark_manifest_order_lookup,
    compact_json,
    numeric_summary,
    pearson_correlation,
    recover_record_chunks,
    run_current_full_rewrite_benchmark,
    run_save_policy_benchmark,
    write_csv,
)
from etflow.commons.global_coupled_4d_sampling import (
    atomic_json_save,
    atomic_torch_save,
    configure_cpu_threads,
    resolve_device,
)


INTERNAL_TIMING_MAP = {
    "egnn_forward_time": "backbone_forward",
    "joint_head_time": "joint_head_forward",
    "topology_time": "topology_cache_lookup",
    "local_frame_time": "local_frame",
    "fragment_pool_time": "fragment_pool",
    "jacobian_construction_time": "jacobian_assembly",
    "gram_matrix_time": "gram_matrix",
    "svd_time": "rank_check_and_svd",
    "cholesky_time": "cholesky",
    "solve_time": "dense_solve",
    "lstsq_time": "least_squares",
    "cartesian_projection_time": "cartesian_projection",
    "internal_mapping_time": "internal_velocity_mapping",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_molecules", type=int, default=1)
    parser.add_argument("--max_records", type=int, default=20)
    parser.add_argument("--warmup_records", type=int, default=2)
    parser.add_argument("--profile_records", type=int)
    parser.add_argument("--refinement_steps", type=int, default=10)
    parser.add_argument("--update_scale", type=float, default=0.2)
    parser.add_argument("--max_displacement", type=float, default=0.1)
    parser.add_argument("--max_coordinate_norm", type=float, default=1000.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--disable_partial_save", action="store_true")
    parser.add_argument(
        "--partial_format",
        choices=("legacy", "chunked", "disabled"),
        default="legacy",
    )
    parser.add_argument("--save_every_records", type=int, default=1)
    parser.add_argument("--torch_profiler", action="store_true")
    parser.add_argument("--cuda_sync_timing", action="store_true")
    parser.add_argument("--skip_batch_benchmark", action="store_true")
    parser.add_argument(
        "--io_protocol_matrix",
        action="store_true",
        help=(
            "Replay at most 30 measured real records through bounded persistence "
            "protocol simulations after computing them once."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    for name in (
        "max_molecules",
        "max_records",
        "refinement_steps",
        "cpu_threads",
        "save_every_records",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name} must be positive")
    if args.warmup_records < 0:
        parser.error("--warmup_records must be non-negative")
    if args.profile_records is not None and args.profile_records < 1:
        parser.error("--profile_records must be positive")
    return args


def _prefix_by_molecules(manifest: dict, maximum: int) -> dict:
    selected_ids = []
    selected = []
    for row in manifest["records"]:
        molecule = str(row["mol_id"])
        if molecule not in selected_ids:
            if len(selected_ids) >= maximum:
                continue
            selected_ids.append(molecule)
        selected.append(row)
    return {**manifest, "records": selected}


def _record_metadata(data: Any, row: dict, diagnostics: dict) -> dict[str, Any]:
    atoms = int(data.atomic_numbers.numel())
    joints = int(data.num_rotatable_bonds.item())
    algebra = diagnostics.get("linear_algebra", [])
    ranks = [int(item["effective_rank"]) for item in algebra]
    conditions = [float(item["condition_number"]) for item in algebra]
    columns = [int(item["jacobian_columns"]) for item in algebra]
    return {
        "sample_id": str(row["sample_id"]),
        "mol_id": str(row["mol_id"]),
        "num_atoms": atoms,
        "num_rotatable_bonds": joints,
        "jacobian_columns": max(columns, default=4 * joints),
        "effective_rank_mean": sum(ranks) / len(ranks) if ranks else 0.0,
        "condition_number_mean": sum(conditions) / len(conditions) if conditions else 0.0,
        "solver_backend": next(iter(diagnostics.get("solver_backend_counts", {})), "none"),
    }


def _profile_context(enabled: bool, trace_dir: Path):
    if not enabled:
        return nullcontext(None)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_dir.mkdir(parents=True, exist_ok=True)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
    )


def should_save_partial(disabled: bool, every: int, index: int, total: int) -> bool:
    """Return the diagnostic save decision without changing formal defaults."""

    if every < 1:
        raise ValueError("save frequency must be positive")
    return not disabled and (index % every == 0 or index == total)


def _io_protocol_matrix(
    records: list[dict[str, Any]],
    *,
    pure_compute_seconds: float,
    base_processing_seconds: float,
    output_dir: Path,
    manifest: dict[str, Any],
    selected_manifest: dict[str, Any],
    manifest_path: Path,
    split: str,
    cache_dir: Path,
    inference_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replay one real computed cohort without invoking the model again."""

    if not records:
        return []
    if len(records) > 30:
        raise ValueError("--io_protocol_matrix is limited to 30 measured records")

    from etflow.data.flexbond_eval_manifest import build_manifest_aware_sample_payload

    work = Path(tempfile.mkdtemp(prefix="global4d_real_io_", dir=output_dir))

    def decorate(label: str, result: dict[str, Any]) -> dict[str, Any]:
        result = dict(result)
        result.pop("save_events", None)
        result.setdefault("payload_build_seconds", 0.0)
        result.setdefault("final_merge_seconds", 0.0)
        result.setdefault("resume_scan_seconds", 0.0)
        result["protocol"] = label
        result["pure_compute_seconds"] = pure_compute_seconds
        result["base_processing_seconds"] = base_processing_seconds
        result["other_protocol_overhead_seconds"] = max(
            0.0,
            float(result["total_seconds"])
            - float(result["payload_build_seconds"])
            - float(result["save_seconds"])
            - float(result["state_seconds"]),
        )
        result["combined_total_seconds"] = base_processing_seconds + float(
            result["total_seconds"]
        ) + float(result["final_merge_seconds"])
        durable_bytes = int(result.get("final_partial_bytes", 0)) or int(
            result.get("chunk_bytes", 0)
        )
        result["tensor_write_amplification"] = (
            float(result.get("total_serialized_bytes", 0)) / durable_bytes
            if durable_bytes
            else 0.0
        )
        result["records_per_second"] = (
            len(records) / result["combined_total_seconds"]
            if result["combined_total_seconds"]
            else 0.0
        )
        return result

    def formal_payload(end: int) -> dict[str, Any]:
        completed_manifest = {
            **selected_manifest,
            "records": selected_manifest["records"][:end],
        }
        return build_manifest_aware_sample_payload(
            records=records[:end],
            manifest=manifest,
            manifest_path=manifest_path,
            selected_manifest=completed_manifest,
            split=split,
            inference_cache_path=cache_dir,
            inference_by_id=inference_by_id,
            extra={
                "partial": True,
                "run_identity": {"diagnostic_replay": True},
                "trajectory": [],
                "profile_rows": [],
                "failed_molecules": [],
            },
        )

    try:
        matrix = [
            {
                "protocol": "partial_disabled_compute_only",
                "mode": "disabled",
                "record_count": len(records),
                "save_every_records": None,
                "save_count": 0,
                "state_writes_per_save": 0,
                "payload_build_seconds": 0.0,
                "save_seconds": 0.0,
                "state_seconds": 0.0,
                "total_seconds": 0.0,
                "pure_compute_seconds": pure_compute_seconds,
                "base_processing_seconds": base_processing_seconds,
                "other_protocol_overhead_seconds": 0.0,
                "combined_total_seconds": base_processing_seconds,
                "total_serialized_bytes": 0,
                "final_partial_bytes": 0,
                "chunk_bytes": 0,
                "total_state_bytes": 0,
                "total_bytes_written": 0,
                "peak_partial_disk_bytes": 0,
                "final_merge_seconds": 0.0,
                "resume_scan_seconds": 0.0,
                "tensor_write_amplification": 0.0,
                "records_per_second": (
                    len(records) / base_processing_seconds
                    if base_processing_seconds
                    else 0.0
                ),
            }
        ]

        for every in (1, 10):
            matrix.append(
                decorate(
                    f"legacy_full_rewrite_every_{every}",
                    run_current_full_rewrite_benchmark(
                        records,
                        work / f"legacy_{every}",
                        payload_factory=formal_payload,
                        save_every=every,
                    ),
                )
            )

        for every in (10, 50):
            chunk_root = work / f"chunk_{every}"
            result = run_save_policy_benchmark(
                records,
                chunk_root,
                save_every=every,
                mode="chunk",
                state_writes_per_save=1,
            )
            merge_started = time.perf_counter()
            recovered = recover_record_chunks(chunk_root)
            result["final_merge_seconds"] = time.perf_counter() - merge_started
            result["resume_scan_seconds"] = result["final_merge_seconds"]
            if [row["sample_id"] for row in recovered] != [
                row["sample_id"] for row in records
            ]:
                raise ValueError("Chunk benchmark changed record order")
            matrix.append(decorate(f"chunked_every_{every}", result))
        return matrix
    finally:
        shutil.rmtree(work, ignore_errors=True)


def profiled_refine(
    model: Any,
    data: Any,
    *,
    device: str,
    cuda_sync_timing: bool,
    refinement_steps: int,
    update_scale: float,
    max_displacement: float,
    max_coordinate_norm: float,
):
    """Time refine while returning its coordinates without transformation."""

    started = time.perf_counter()
    with CudaEventTimer(device, cuda_sync_timing) as cuda_timer:
        refined, diagnostics = model.refine(
            data,
            refinement_steps=refinement_steps,
            update_scale=update_scale,
            max_displacement=max_displacement,
            max_coordinate_norm=max_coordinate_norm,
            profile=cuda_sync_timing,
            collect_diagnostics=True,
        )
    return refined, diagnostics, {
        "cpu_wall_seconds": time.perf_counter() - started,
        "cuda_seconds": cuda_timer.elapsed_seconds,
        "synchronize_seconds": cuda_timer.synchronize_seconds,
    }


def _run_batch_benchmark(
    model: Any,
    by_id: dict[str, Any],
    rows: list[dict],
    args: argparse.Namespace,
    device: str,
) -> list[dict]:
    from torch_geometric.data import Batch

    groups = defaultdict(list)
    for row in rows:
        groups[str(row["mol_id"])].append(row)
    candidates = max(groups.values(), key=len, default=[])
    output = []
    for batch_size in (1, 2, 4, 8):
        if len(candidates) < batch_size:
            output.append({"batch_size": batch_size, "status": "insufficient_same_topology_records"})
            continue
        data_list = [by_id[str(row["sample_id"])] for row in candidates[:batch_size]]
        sequential_outputs = []
        for item in data_list:
            single = item.to(device)
            single_refined, _ = model.refine(
                single,
                refinement_steps=args.refinement_steps,
                update_scale=args.update_scale,
                max_displacement=args.max_displacement,
                max_coordinate_norm=args.max_coordinate_norm,
                profile=False,
            )
            sequential_outputs.append(single_refined)
        sequential = torch.cat(sequential_outputs, dim=0)
        batch = Batch.from_data_list(data_list).to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        started = time.perf_counter()
        refined, _ = model.refine(
            batch,
            refinement_steps=args.refinement_steps,
            update_scale=args.update_scale,
            max_displacement=args.max_displacement,
            max_coordinate_norm=args.max_coordinate_norm,
            profile=False,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        elapsed = time.perf_counter() - started
        output.append(
            {
                "batch_size": batch_size,
                "status": "measured",
                "seconds": elapsed,
                "records_per_second": batch_size / elapsed,
                "output_finite": bool(torch.isfinite(refined).all()),
                "max_abs_coordinate_difference_vs_sequential": float(
                    (refined - sequential).abs().max()
                ),
            }
        )
    return output


def _markdown(payload: dict) -> str:
    stage_rows = "\n".join(
        "| {stage} | {calls} | {cpu_wall_seconds:.6f} | {cuda_seconds:.6f} | "
        "{self_seconds:.6f} | {seconds_per_record:.6f} | {seconds_per_refinement_step:.6f} | {wall_time_fraction:.2%} |".format(**row)
        for row in payload["stages"]
    )
    batch_rows = "\n".join(
        f"| {row['batch_size']} | {row['status']} | {row.get('seconds', 0):.6f} | {row.get('records_per_second', 0):.4f} |"
        for row in payload["batch_benchmark"]
    )
    io_rows = "\n".join(
        f"| {row['protocol']} | {row['pure_compute_seconds']:.6f} | "
        f"{row['payload_build_seconds']:.6f} | {row['save_seconds']:.6f} | "
        f"{row['state_seconds']:.6f} | {row.get('final_merge_seconds', 0):.6f} | "
        f"{row['combined_total_seconds']:.6f} | {row.get('total_bytes_written', 0)} | "
        f"{row.get('tensor_write_amplification', 0):.3f} | "
        f"{row.get('records_per_second', 0):.3f} |"
        for row in payload.get("io_protocol_matrix", {}).get("policies", [])
    )
    io_section = ""
    if io_rows:
        io_section = f"""
## Real-record persistence protocol matrix

The same computed records are replayed for every row; the model is not rerun.

| Protocol | Pure compute s | Payload build s | Tensor serialization s | State JSON s | Final merge s | Combined total s | Bytes written | Tensor write amp | Records/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{io_rows}
"""
    return f"""# Global Coupled 4D sampling profile

- Device: `{payload['environment']['device']}`
- Warmup records: {payload['counts']['warmup_records']}
- Profiled records: {payload['counts']['profiled_records']}
- Refinement steps: {payload['counts']['refinement_steps_total']}
- Measured records/s: {payload['throughput']['records_per_second']:.6f}
- Pure rollout records/s: {payload['throughput']['pure_compute_records_per_second']:.6f}
- Partial saving: `{payload['save_policy']['enabled']}` format `{payload['save_policy']['partial_format']}` every {payload['save_policy']['save_every_records']} records
- CUDA timing: `{payload['timing_method']['cuda']}`

## Stage timing

| Stage | Calls | CPU wall s | CUDA s | Self s | s/record | s/step | Wall share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{stage_rows}

## Same-topology batch benchmark

| Batch | Status | Seconds | Records/s |
| ---: | --- | ---: | ---: |
{batch_rows}

{io_section}

Raw per-record rows are in the CSV only; the JSON/Markdown reports stay compact.
"""


def main() -> None:
    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    threads = configure_cpu_threads(args.cpu_threads)
    stages = StageAccumulator()
    initialization = {}

    started = time.perf_counter()
    phase = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    initialization["config_load_seconds"] = time.perf_counter() - phase

    from etflow.data.flexbond_eval_manifest import load_eval_manifest, validate_dataset_against_manifest
    from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset
    from etflow.models.global_coupled_4d_flow import GlobalCoupled4DFlowLightningModule

    phase = time.perf_counter()
    manifest = load_eval_manifest(args.manifest)
    selected = _prefix_by_molecules(manifest, args.max_molecules)
    selected["records"] = selected["records"][: args.max_records]
    initialization["manifest_load_and_select_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    dataset = FlexBondInferenceDataset(args.cache_dir, args.split)
    initialization["cache_scan_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    by_id = validate_dataset_against_manifest(dataset, selected)
    initialization["cache_record_load_and_validation_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    model = GlobalCoupled4DFlowLightningModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    initialization["checkpoint_load_cpu_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    model = model.to(device).eval()
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    initialization["model_to_device_seconds"] = time.perf_counter() - phase

    rows = selected["records"]
    warmup_count = min(args.warmup_records, len(rows))
    for row in rows[:warmup_count]:
        data = by_id[str(row["sample_id"])].to(device)
        model.refine(
            data,
            refinement_steps=args.refinement_steps,
            update_scale=args.update_scale,
            max_displacement=args.max_displacement,
            max_coordinate_norm=args.max_coordinate_norm,
            profile=False,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))

    measured_rows = rows[warmup_count:]
    if args.profile_records is not None:
        measured_rows = measured_rows[: args.profile_records]
    partial_records = []
    last_saved_count = 0
    profile_chunks = args.output_dir / "profile_partial_chunks"
    record_rows = []
    algebra_rows = []
    backend_counts = Counter()
    sync_overhead = 0.0
    measured_started = time.perf_counter()
    trace_dir = args.output_dir / "torch_profiler_trace"
    with _profile_context(args.torch_profiler, trace_dir) as torch_profile:
        for record_index, row in enumerate(measured_rows, 1):
            record_started = time.perf_counter()
            stage_started = time.perf_counter()
            with CudaEventTimer(device, args.cuda_sync_timing) as transfer_cuda:
                data = by_id[str(row["sample_id"])].to(device)
            transfer_seconds = time.perf_counter() - stage_started
            sync_overhead += transfer_cuda.synchronize_seconds
            stages.add(
                "cpu_to_device",
                cpu_wall_seconds=transfer_seconds,
                cuda_seconds=transfer_cuda.elapsed_seconds,
            )

            refined, diagnostics, rollout_timing = profiled_refine(
                model,
                data,
                device=device,
                cuda_sync_timing=args.cuda_sync_timing,
                refinement_steps=args.refinement_steps,
                update_scale=args.update_scale,
                max_displacement=args.max_displacement,
                max_coordinate_norm=args.max_coordinate_norm,
            )
            rollout_wall = rollout_timing["cpu_wall_seconds"]
            sync_overhead += rollout_timing["synchronize_seconds"]
            stages.add(
                "rollout_total",
                cpu_wall_seconds=rollout_wall,
                cuda_seconds=rollout_timing["cuda_seconds"],
            )
            for source, target in INTERNAL_TIMING_MAP.items():
                value = float(diagnostics.get("mean_timing", {}).get(source, 0.0))
                stages.add(
                    target,
                    calls=args.refinement_steps,
                    cpu_wall_seconds=value * args.refinement_steps,
                    cuda_seconds=(value * args.refinement_steps if device.startswith("cuda") else 0.0),
                )
            preparation = diagnostics.get("preparation_timing", {})
            stages.add(
                "static_topology_preparation",
                cpu_wall_seconds=float(preparation.get("total_preparation_time", 0.0)),
            )

            output_started = time.perf_counter()
            with CudaEventTimer(device, args.cuda_sync_timing) as output_cuda:
                x_init = data.x_init.detach().cpu()
                x_refined = refined.detach().cpu()
            sync_overhead += output_cuda.synchronize_seconds
            output_seconds = time.perf_counter() - output_started
            stages.add(
                "device_to_cpu",
                cpu_wall_seconds=output_seconds,
                cuda_seconds=output_cuda.elapsed_seconds,
            )

            object_started = time.perf_counter()
            metadata = _record_metadata(data, row, diagnostics)
            saved = {
                **metadata,
                "mol_id": str(getattr(data, "mol_id", row["mol_id"])),
                "source_mol_id": str(getattr(data, "source_mol_id", row["mol_id"])),
                "smiles": str(getattr(data, "smiles", "")),
                "atomic_numbers": data.atomic_numbers.detach().cpu(),
                "x_init": x_init,
                "x_refined": x_refined,
                "x_init_hash": str(row["x_init_hash"]),
                "method_name": "global_coupled_4d_adapter",
                "motion_mode": str(getattr(model, "motion_mode", "unknown")),
                "status": "success" if diagnostics["stable"] else "failed",
                "checkpoint_path": str(args.checkpoint.resolve()),
                "config_path": str(args.config.resolve()),
                "refinement_steps": args.refinement_steps,
                "update_scale": args.update_scale,
                "alpha": args.update_scale,
                "max_displacement": args.max_displacement,
                **{
                    key: value
                    for key, value in diagnostics.items()
                    if key not in ("trajectory", "linear_algebra")
                },
            }
            partial_records.append(saved)
            object_seconds = time.perf_counter() - object_started
            stages.add("python_record_object", cpu_wall_seconds=object_seconds)

            save_seconds = state_seconds = 0.0
            should_save = should_save_partial(
                args.disable_partial_save or args.partial_format == "disabled",
                args.save_every_records,
                record_index,
                len(measured_rows),
            )
            partial_bytes = 0
            if should_save:
                save_started = time.perf_counter()
                if args.partial_format == "chunked":
                    profile_chunks.mkdir(exist_ok=True)
                    chunk_index = len(list(profile_chunks.glob("chunk_*.pt")))
                    chunk_path = profile_chunks / f"chunk_{chunk_index:06d}.pt"
                    atomic_torch_save(
                        {
                            "start": last_saved_count,
                            "end": len(partial_records),
                            "records": partial_records[last_saved_count:],
                        },
                        chunk_path,
                    )
                    partial_bytes = sum(
                        path.stat().st_size for path in profile_chunks.glob("chunk_*.pt")
                    )
                else:
                    atomic_torch_save(
                        {"partial": True, "records": partial_records},
                        args.output_dir / "profile_partial_samples.pt",
                    )
                    partial_bytes = (args.output_dir / "profile_partial_samples.pt").stat().st_size
                save_seconds = time.perf_counter() - save_started
                stages.add("partial_save", cpu_wall_seconds=save_seconds)
                state_started = time.perf_counter()
                statuses = (
                    ("PARTIAL",)
                    if args.partial_format == "chunked"
                    else ("running", "partial")
                )
                for status in statuses:
                    state_payload = {
                        "status": status,
                        "completed_count": record_index,
                        "total_count": len(measured_rows),
                    }
                    if args.partial_format == "chunked":
                        state_payload.update(
                            {
                                "format_version": "global4d-sampling-state-v2",
                                "partial_format": "chunked",
                                "completed_chunk_count": chunk_index + 1,
                                "next_chunk_index": chunk_index + 1,
                            }
                        )
                    else:
                        state_payload["completed_ordered_sample_ids"] = [
                            str(item["sample_id"]) for item in partial_records
                        ]
                    atomic_json_save(
                        state_payload,
                        args.output_dir / "profile_sampling_state.json",
                    )
                state_seconds = time.perf_counter() - state_started
                stages.add("sampling_state_save", cpu_wall_seconds=state_seconds)
                last_saved_count = len(partial_records)

            record_seconds = time.perf_counter() - record_started
            backend_counts.update(diagnostics.get("solver_backend_counts", {}))
            record_rows.append(
                {
                    **metadata,
                    "record_index": record_index,
                    "record_seconds": record_seconds,
                    "rollout_seconds": rollout_wall,
                    "cpu_to_device_seconds": transfer_seconds,
                    "device_to_cpu_seconds": output_seconds,
                    "python_object_seconds": object_seconds,
                    "partial_save_seconds": save_seconds,
                    "sampling_state_save_seconds": state_seconds,
                    "partial_file_bytes": partial_bytes,
                }
            )
            for item in diagnostics.get("linear_algebra", []):
                algebra_rows.append({"sample_id": metadata["sample_id"], **item})
            if torch_profile is not None:
                torch_profile.step()

    final_profile_path = args.output_dir / "profile_final_samples.pt"
    if args.disable_partial_save or args.partial_format == "disabled":
        stages.add("final_samples_save", calls=0)
    else:
        final_started = time.perf_counter()
        atomic_torch_save(
            {"diagnostic_only": True, "records": partial_records},
            final_profile_path,
        )
        stages.add(
            "final_samples_save",
            cpu_wall_seconds=time.perf_counter() - final_started,
        )
    measured_seconds = time.perf_counter() - measured_started

    operator_rows = []
    if torch_profile is not None:
        for event in torch_profile.key_averages(group_by_input_shape=False):
            operator_rows.append(
                {
                    "operator": event.key,
                    "calls": int(event.count),
                    "cpu_total_seconds": float(event.cpu_time_total) / 1.0e6,
                    "cpu_self_seconds": float(event.self_cpu_time_total) / 1.0e6,
                    "cuda_total_seconds": float(
                        getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))
                    ) / 1.0e6,
                    "cuda_self_seconds": float(
                        getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
                    ) / 1.0e6,
                }
            )
        operator_rows.sort(
            key=lambda row: row["cpu_total_seconds"] + row["cuda_total_seconds"],
            reverse=True,
        )

    total_steps = len(record_rows) * args.refinement_steps
    stages.add("cuda_synchronize_overhead", calls=len(record_rows), cpu_wall_seconds=sync_overhead)
    stage_rows = stages.compact(len(record_rows), total_steps, measured_seconds)
    record_times = [float(row["record_seconds"]) for row in record_rows]
    rollout_times = [float(row["rollout_seconds"]) for row in record_rows]
    save_times = [float(row["partial_save_seconds"]) for row in record_rows]
    first_count = max(1, len(record_times) // 10) if record_times else 0
    middle_start = len(record_times) // 4
    middle_end = len(record_times) - middle_start
    tail = {
        "first_10_percent_mean_seconds": (
            sum(record_times[:first_count]) / first_count if first_count else 0.0
        ),
        "middle_50_percent_mean_seconds": (
            sum(record_times[middle_start:middle_end]) / max(middle_end - middle_start, 1)
            if record_times else 0.0
        ),
        "last_10_percent_mean_seconds": (
            sum(record_times[-first_count:]) / first_count if first_count else 0.0
        ),
    }
    batch_benchmark = [] if args.skip_batch_benchmark else _run_batch_benchmark(
        model, by_id, rows, args, device
    )
    pure_compute_seconds = sum(rollout_times)
    io_policies = (
        _io_protocol_matrix(
            partial_records,
            pure_compute_seconds=pure_compute_seconds,
            base_processing_seconds=measured_seconds,
            output_dir=args.output_dir,
            manifest=manifest,
            selected_manifest={
                **selected,
                "records": selected["records"][warmup_count : warmup_count + len(record_rows)],
            },
            manifest_path=args.manifest,
            split=args.split,
            cache_dir=args.cache_dir,
            inference_by_id=by_id,
        )
        if args.io_protocol_matrix
        else []
    )
    payload = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
            "torch_version": torch.__version__,
            "cpu_threads": threads,
        },
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()),
            "config": str(args.config.resolve()),
            "cache_dir": str(args.cache_dir.resolve()),
            "manifest": str(args.manifest.resolve()),
            "split": args.split,
            "max_molecules": args.max_molecules,
            "max_records": args.max_records,
        },
        "initialization": initialization,
        "counts": {
            "selected_records": len(rows),
            "warmup_records": warmup_count,
            "profiled_records": len(record_rows),
            "refinement_steps_per_record": args.refinement_steps,
            "refinement_steps_total": total_steps,
        },
        "save_policy": {
            "enabled": not (
                args.disable_partial_save or args.partial_format == "disabled"
            ),
            "partial_format": (
                "disabled" if args.disable_partial_save else args.partial_format
            ),
            "save_every_records": args.save_every_records,
        },
        "timing_method": {
            "cpu": "time.perf_counter wall time",
            "cuda": "torch.cuda.Event with explicit synchronize" if args.cuda_sync_timing else "not collected; no profiling synchronization",
            "internal_cuda_note": "Internal component values are synchronized region wall times when cuda_sync_timing is enabled, not independent kernel self time.",
            "stage_overlap_note": "rollout_total is an inclusive parent of backbone/Jacobian/solver/projection component rows, so stage fractions must not be summed.",
            "synchronize_overhead_seconds": sync_overhead,
        },
        "throughput": {
            "measured_seconds": measured_seconds,
            "records_per_second": len(record_rows) / measured_seconds if measured_seconds else 0.0,
            "pure_rollout_seconds": sum(rollout_times),
            "pure_compute_records_per_second": (
                len(record_rows) / sum(rollout_times) if sum(rollout_times) else 0.0
            ),
            "refinement_steps_per_second": total_steps / measured_seconds if measured_seconds else 0.0,
            "record_time": numeric_summary(record_times),
            "save_time": numeric_summary(save_times),
            **tail,
        },
        "stages": stage_rows,
        "solver": {
            "backend_counts": dict(backend_counts),
            "solve_count": len(algebra_rows),
            "svd_calls_per_record": len(algebra_rows) / len(record_rows) if record_rows else 0.0,
            "jacobian_rows": numeric_summary([row["jacobian_rows"] for row in algebra_rows]),
            "jacobian_columns": numeric_summary([row["jacobian_columns"] for row in algebra_rows]),
            "effective_rank": numeric_summary([row["effective_rank"] for row in algebra_rows]),
            "condition_number": numeric_summary([row["condition_number"] for row in algebra_rows]),
            "orthogonality_error": numeric_summary(
                [row.get("orthogonality_error", 0.0) for row in algebra_rows]
            ),
            "reconstruction_error": numeric_summary(
                [row.get("reconstruction_error", 0.0) for row in algebra_rows]
            ),
        },
        "correlations": {
            "record_time_vs_atoms": pearson_correlation(record_times, [row["num_atoms"] for row in record_rows]),
            "record_time_vs_rotatable_bonds": pearson_correlation(record_times, [row["num_rotatable_bonds"] for row in record_rows]),
            "record_time_vs_jacobian_columns": pearson_correlation(record_times, [row["jacobian_columns"] for row in record_rows]),
            "record_time_vs_effective_rank": pearson_correlation(record_times, [row["effective_rank_mean"] for row in record_rows]),
            "record_time_vs_partial_file_bytes": pearson_correlation(record_times, [row["partial_file_bytes"] for row in record_rows]),
        },
        "batch_benchmark": batch_benchmark,
        "io_protocol_matrix": {
            "enabled": args.io_protocol_matrix,
            "record_count": len(partial_records),
            "records_computed_once_and_replayed": True,
            "formal_payload_scope_note": (
                "The exact current formal protocol uses the bundle's reduced manifest. "
                "It measures real record tensors and shared provenance construction, but "
                "does not reproduce the much larger source formal manifest size."
            ),
            "policies": io_policies,
        },
        "manifest_order_lookup": benchmark_manifest_order_lookup(
            [str(row["sample_id"]) for row in selected["records"]],
            repetitions=10_000,
        ),
        "torch_profiler": {
            "enabled": args.torch_profiler,
            "trace_directory": str(trace_dir.resolve()) if args.torch_profiler else None,
            "committed": False,
            "top_operators": operator_rows[:20],
        },
        "total_command_seconds": time.perf_counter() - started,
    }
    compact_json(payload, args.output_dir / "global4d_sampling_profile.json")
    write_csv(record_rows, args.output_dir / "global4d_sampling_profile.csv")
    if algebra_rows:
        flat_algebra = []
        for row in algebra_rows:
            flat = {key: value for key, value in row.items() if key not in {"timing", "attempted_backends"}}
            flat.update({f"timing_{key}": value for key, value in row["timing"].items()})
            flat["attempted_backends"] = ",".join(row["attempted_backends"])
            flat_algebra.append(flat)
        write_csv(flat_algebra, args.output_dir / "global4d_sampling_linear_algebra.csv")
    if operator_rows:
        write_csv(operator_rows, args.output_dir / "global4d_torch_profiler_operators.csv")
    (args.output_dir / "global4d_sampling_profile.md").write_text(_markdown(payload), encoding="utf-8")
    if not (args.disable_partial_save or args.partial_format == "disabled"):
        (args.output_dir / "profile_partial_samples.pt").unlink(missing_ok=True)
        (args.output_dir / "profile_sampling_state.json").unlink(missing_ok=True)
        final_profile_path.unlink(missing_ok=True)
        shutil.rmtree(profile_chunks, ignore_errors=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
