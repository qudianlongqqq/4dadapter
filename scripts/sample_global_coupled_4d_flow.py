#!/usr/bin/env python
"""Strictly label-free, resumable Global Coupled 4D rollout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.global_coupled_4d_sampling import (
    atomic_json_save,
    atomic_torch_save,
    checkpoint_inference_identity,
    configure_cpu_threads,
    file_sha256,
    resolve_device,
)
from etflow.commons.global4d_chunked_persistence import (
    cleanup_chunks,
    compact_sampling_state,
    ordered_sample_ids_sha256,
    scan_chunks,
    utc_now,
    validate_compact_state,
    write_chunk,
)
from etflow.commons.provenance import collect_run_provenance
from etflow.commons.run_state import update_run_state
from etflow.data.flexbond_eval_manifest import (
    build_manifest_aware_sample_payload,
    limit_manifest_molecules,
    load_eval_manifest,
    manifest_content_sha256,
    validate_sample_payload_provenance,
    validate_dataset_against_manifest,
)
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset
from etflow.models.global_coupled_4d_flow import (
    ABLATION_MODES,
    GlobalCoupled4DFlowLightningModule,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _atomic_trajectory(rows: list[dict], path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(rows[0]) if rows else ["sample_id", "rollout_step"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _profile_payload(
    rows: list[dict],
    *,
    dataset_load_time: float,
    thread_config: dict,
    partial_path: Path,
) -> dict:
    components = Counter()
    preparation = Counter()
    backends = Counter()
    devices = {}
    peak_gpu_memory = 0.0
    for row in rows:
        for key, value in row.get("mean_timing", {}).items():
            if key == "peak_gpu_memory":
                peak_gpu_memory = max(peak_gpu_memory, float(value))
            else:
                components[key] += float(value)
        for key, value in row.get("preparation_timing", {}).items():
            if isinstance(value, (int, float)) and key != "cache_hit":
                preparation[key] += float(value)
        backends.update(row.get("solver_backend_counts", {}))
        devices.update(row.get("devices", {}))
    count = len(rows)
    means = {key: value / count for key, value in components.items()} if count else {}
    preparation_means = (
        {key: value / count for key, value in preparation.items()} if count else {}
    )
    means.setdefault("rdkit_time", 0.0)
    means.setdefault("torch_linalg_solve_time", means.get("solve_time", 0.0))
    means.setdefault("torch_linalg_lstsq_time", means.get("lstsq_time", 0.0))
    means.setdefault("torch_linalg_svd_time", means.get("svd_time", 0.0))
    return {
        "created_at": _utc_now(),
        "profiled_molecules": count,
        "dataset_and_manifest_load_time": dataset_load_time,
        "mean_molecule_time": (
            sum(float(row["molecule_time"]) for row in rows) / count if count else 0.0
        ),
        "mean_refinement_step_time": (
            sum(float(row["mean_step_time"]) for row in rows) / count if count else 0.0
        ),
        "mean_component_seconds_per_step": means,
        "mean_coordinate_independent_preparation_seconds_per_molecule": preparation_means,
        "per_molecule": rows,
        "solver_backend_counts": dict(backends),
        "devices": devices,
        "peak_gpu_memory_bytes": int(peak_gpu_memory),
        "thread_configuration": thread_config,
        "partial_samples_path": str(partial_path.resolve()),
        "rdkit_operations_during_rollout": 0,
        "notes": [
            "RDKit is not called by this rollout; topology comes from cached tensor indices.",
            "Profile mode synchronizes CUDA only around measured regions.",
            "Non-profile rollout does not add per-step CUDA synchronization.",
        ],
    }


def _write_profile(payload: dict, json_path: Path, markdown_path: Path) -> None:
    atomic_json_save(payload, json_path)
    component_rows = "\n".join(
        f"| {key} | {value:.6f} |"
        for key, value in sorted(payload["mean_component_seconds_per_step"].items())
    )
    preparation_rows = "\n".join(
        f"| {key} | {value:.6f} |"
        for key, value in sorted(
            payload["mean_coordinate_independent_preparation_seconds_per_molecule"].items()
        )
    )
    molecule_rows = "\n".join(
        f"| {row['sample_id']} | {row['molecule_time']:.6f} | "
        f"{row['mean_step_time']:.6f} | {row['cpu_to_device_time']:.6f} | "
        f"{row['device_to_cpu_time']:.6f} |"
        for row in payload["per_molecule"]
    )
    text = f"""# Global Coupled 4D sampling profile

- Profiled molecules: {payload['profiled_molecules']}
- Profile source: {payload.get('profile_source', 'formal rollout')}
- Dataset/manifest load: {payload['dataset_and_manifest_load_time']:.6f} s
- Mean molecule: {payload['mean_molecule_time']:.6f} s
- Mean refinement step: {payload['mean_refinement_step_time']:.6f} s
- Devices: `{json.dumps(payload['devices'], sort_keys=True)}`
- Solver backends: `{json.dumps(payload['solver_backend_counts'], sort_keys=True)}`
- Peak GPU memory: {payload['peak_gpu_memory_bytes']} bytes
- Threads: `{json.dumps(payload['thread_configuration'], sort_keys=True)}`
- Partial payload: `{payload['partial_samples_path']}`
- RDKit rollout operations: 0

## Mean component time per step

| Component | Seconds |
| --- | ---: |
{component_rows}

## Coordinate-independent preparation per molecule

| Component | Seconds |
| --- | ---: |
{preparation_rows}

## Per molecule

| Sample | Total s | Mean step s | CPU→device s | Device→CPU s |
| --- | ---: | ---: | ---: | ---: |
{molecule_rows}
"""
    temporary = markdown_path.with_name(markdown_path.name + f".tmp.{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, markdown_path)


def _validate_resume_records(records: list[dict], rows: list[dict]) -> None:
    if len(records) > len(rows):
        raise ValueError("Partial payload contains more records than the selected manifest")
    for record, row in zip(records, rows):
        if str(record.get("sample_id")) != str(row["sample_id"]):
            raise ValueError("Partial payload is not an ordered manifest prefix")
        if str(record.get("source_mol_id", record.get("mol_id"))) != str(row["mol_id"]):
            raise ValueError(f"Partial payload molecule mismatch: {row['sample_id']}")
        if str(record.get("x_init_hash")) != str(row["x_init_hash"]):
            raise ValueError(f"Partial payload x_init_hash mismatch: {row['sample_id']}")


def _validate_completed_run_identity(payload: dict, run_identity: dict) -> None:
    stored = dict((payload.get("persistence") or {}).get("run_identity") or {})
    if stored:
        if stored != run_identity:
            raise ValueError("Existing final output belongs to a different sampling command")
        return

    provenance = dict(payload.get("provenance") or {})
    checkpoint = dict(provenance.get("checkpoint_identity") or {})
    if checkpoint.get("inference_sha256") != run_identity.get(
        "checkpoint_inference_sha256"
    ):
        raise ValueError("Existing final output checkpoint identity does not match")
    if provenance.get("config_sha256") != run_identity.get("config_sha256"):
        raise ValueError("Existing final output config identity does not match")
    records = list(payload.get("records") or [])
    expected_record_fields = {
        "alpha": run_identity.get("alpha"),
        "refinement_steps": run_identity.get("refinement_steps"),
        "max_displacement": run_identity.get("max_displacement"),
        "joint_mode": run_identity.get("joint_mode"),
    }
    for key, expected in expected_record_fields.items():
        if any(record.get(key) != expected for record in records):
            raise ValueError(f"Existing final output {key} does not match")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refinement_steps", type=int, default=10)
    parser.add_argument("--update_scale", type=float, default=0.5)
    parser.add_argument("--max_displacement", type=float, default=0.1)
    parser.add_argument("--max_coordinate_norm", type=float, default=1000.0)
    parser.add_argument("--max_molecules", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=int(os.environ.get("GLOBAL4D_CPU_THREADS", "4")),
    )
    parser.add_argument("--joint_mode", choices=ABLATION_MODES, default="full_4d")
    parser.add_argument("--save_trajectory_metrics", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_molecules", type=int, default=5)
    parser.add_argument(
        "--partial_format",
        choices=("chunked", "legacy", "disabled"),
        default="chunked",
    )
    parser.add_argument("--save_every_records", type=int, default=50)
    parser.add_argument("--cleanup_partial_chunks", action="store_true")
    parser.add_argument(
        "--profile_json",
        type=Path,
        default=Path("reports/global_coupled_4d_sampling_profile.json"),
    )
    parser.add_argument(
        "--profile_markdown",
        type=Path,
        default=Path("reports/global_coupled_4d_sampling_profile.md"),
    )
    args = parser.parse_args()
    if args.profile_molecules < 1:
        parser.error("--profile_molecules must be positive")
    if args.save_every_records < 1:
        parser.error("--save_every_records must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output.parent / "partial_samples.pt"
    chunks_path = args.output.parent / "partial_chunks"
    state_path = args.output.parent / "sampling_state.json"
    device = resolve_device(args.device)
    thread_config = configure_cpu_threads(args.cpu_threads)
    checkpoint_identity = checkpoint_inference_identity(args.checkpoint)
    run_identity = {
        "checkpoint_inference_sha256": checkpoint_identity["inference_sha256"],
        "checkpoint_global_step": checkpoint_identity["global_step"],
        "config_sha256": file_sha256(args.config),
        "manifest_sha256": None,
        "split": args.split,
        "alpha": float(args.update_scale),
        "refinement_steps": int(args.refinement_steps),
        "max_molecules": args.max_molecules,
        "max_displacement": args.max_displacement,
        "max_coordinate_norm": args.max_coordinate_norm,
        "joint_mode": args.joint_mode,
    }
    update_run_state(
        args.output.parent,
        "started",
        stage="sampling",
        output=str(args.output),
        partial_output=str(partial_path),
        device=device,
    )
    started = time.perf_counter()
    try:
        load_started = time.perf_counter()
        dataset = FlexBondInferenceDataset(args.cache_dir, args.split)
        manifest = load_eval_manifest(args.manifest)
        run_identity["manifest_sha256"] = manifest_content_sha256(manifest)
        selected_manifest = (
            limit_manifest_molecules(manifest, args.max_molecules)
            if args.max_molecules is not None
            else manifest
        )
        by_id = validate_dataset_against_manifest(dataset, selected_manifest)
        dataset_load_time = time.perf_counter() - load_started
        selected_rows = selected_manifest["records"]
        selected_order_hash = ordered_sample_ids_sha256(
            [str(row["sample_id"]) for row in selected_rows]
        )
        total = len(selected_rows)

        if args.output.is_file() and args.output.stat().st_size:
            completed = torch.load(args.output, map_location="cpu", weights_only=False)
            validate_sample_payload_provenance(
                completed,
                manifest=manifest,
                manifest_path=args.manifest,
                split=args.split,
                inference_cache_path=args.cache_dir,
                inference_by_id=by_id,
            )
            _validate_completed_run_identity(completed, run_identity)
            if [str(row.get("sample_id")) for row in completed["records"]] != [
                str(row["sample_id"]) for row in selected_rows
            ]:
                raise ValueError("Existing final output does not match the selected cohort")
            prior_state = {}
            if state_path.is_file():
                try:
                    prior_state = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    prior_state = {}
            completed_state = {
                **prior_state,
                "status": "COMPLETED",
                "updated_at": _utc_now(),
                "completed_count": len(completed["records"]),
                "total_count": total,
                "checkpoint_inference_sha256": run_identity[
                    "checkpoint_inference_sha256"
                ],
                "config_sha256": run_identity["config_sha256"],
                "manifest_sha256": run_identity["manifest_sha256"],
                "ordered_sample_ids_sha256": selected_order_hash,
                "output": str(args.output.resolve()),
                "partial_format": args.partial_format,
                "save_every_records": args.save_every_records,
                "eta_seconds": 0.0,
                "resumed_completed_output": True,
            }
            if args.partial_format != "legacy":
                completed_state.pop("completed_ordered_sample_ids", None)
            atomic_json_save(completed_state, state_path)
            update_run_state(
                args.output.parent,
                "completed",
                stage="sampling",
                output=str(args.output),
                num_records=len(completed["records"]),
                resumed_completed_output=True,
            )
            print(f"Existing validated output is complete: {args.output}", flush=True)
            return

        records: list[dict] = []
        trajectory: list[dict] = []
        profile_rows: list[dict] = []
        failed_molecules: list[dict] = []
        chunk_scan = None
        existing_state = {}
        if state_path.is_file():
            try:
                existing_state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing_state = {}
        started_at = str(existing_state.get("started_at") or utc_now())
        persistence_metrics = {
            "partial_payload_build_seconds": 0.0,
            "partial_serialization_seconds": 0.0,
            "state_json_seconds": 0.0,
            "partial_bytes_written": 0,
            "state_bytes_written": 0,
            "partial_write_count": 0,
            "state_write_count": 0,
        }
        for key, default in persistence_metrics.items():
            prior = existing_state.get("persistence_metrics", {}).get(key, default)
            persistence_metrics[key] = type(default)(prior)

        def save_sampling_state(payload: dict) -> None:
            state_started = time.perf_counter()
            atomic_json_save(payload, state_path)
            persistence_metrics["state_json_seconds"] += (
                time.perf_counter() - state_started
            )
            persistence_metrics["state_bytes_written"] += state_path.stat().st_size
            persistence_metrics["state_write_count"] += 1

        def chunk_state(**kwargs) -> dict:
            return {
                **compact_sampling_state(**kwargs),
                "persistence_metrics": dict(persistence_metrics),
            }

        if args.partial_format == "legacy":
            if chunks_path.exists() and any(chunks_path.iterdir()):
                raise ValueError("Refusing to mix legacy partials with chunked persistence")
        elif args.partial_format == "chunked":
            chunk_scan = scan_chunks(
                chunks_path,
                selected_rows=selected_rows,
                run_identity=run_identity,
            )
            if partial_path.is_file() and not (
                existing_state.get("partial_format") == "chunked"
                and existing_state.get("legacy_source_sha256")
            ):
                raise ValueError(
                    "Legacy partial_samples.pt detected; run "
                    "scripts/convert_legacy_partial_to_chunks.py or resume with "
                    "--partial_format legacy."
                )
            if existing_state.get("partial_format") == "chunked":
                validate_compact_state(
                    existing_state,
                    scan=chunk_scan,
                    run_identity=run_identity,
                    ordered_sample_ids_hash=selected_order_hash,
                    save_every_records=args.save_every_records,
                )
            records = list(chunk_scan.records)
            trajectory = list(chunk_scan.trajectory)
            profile_rows = list(chunk_scan.profile_rows)
            _validate_resume_records(records, selected_rows)
            save_sampling_state(
                chunk_state(
                    status="INITIALIZED" if not records else "PARTIAL",
                    completed_count=len(records),
                    total_count=total,
                    completed_chunk_count=chunk_scan.chunk_count,
                    current_chunk_size=0,
                    save_every_records=args.save_every_records,
                    run_identity=run_identity,
                    ordered_sample_ids_hash=selected_order_hash,
                    output=args.output,
                    device=device,
                    started_at=started_at,
                    latest_chunk_sha256=chunk_scan.latest_chunk_sha256,
                    legacy_source_sha256=existing_state.get("legacy_source_sha256"),
                )
            )
        elif partial_path.is_file() or (
            chunks_path.exists() and any(chunks_path.iterdir())
        ):
            raise ValueError("Disabled persistence cannot resume partial artifacts")

        if args.partial_format == "legacy" and partial_path.is_file():
            partial = torch.load(partial_path, map_location="cpu", weights_only=False)
            if partial.get("partial") is not True:
                raise ValueError("Refusing to resume from a payload not marked partial")
            if partial.get("run_identity") != run_identity:
                raise ValueError("Partial payload belongs to a different sampling command")
            records = list(partial.get("records", []))
            trajectory = list(partial.get("trajectory", []))
            profile_rows = list(partial.get("profile_rows", []))
            failed_molecules = list(partial.get("failed_molecules", []))
            _validate_resume_records(records, selected_rows)

        model = GlobalCoupled4DFlowLightningModule.load_from_checkpoint(
            args.checkpoint, map_location=device
        ).to(device).eval()
        backend_totals = Counter()
        for record in records:
            backend_totals.update(record.get("solver_backend_counts", {}))
        state = {
            "status": "running",
            "updated_at": _utc_now(),
            "checkpoint_path": checkpoint_identity["path"],
            "checkpoint_file_sha256": checkpoint_identity["file_sha256"],
            "checkpoint_inference_sha256": checkpoint_identity["inference_sha256"],
            "config_sha256": run_identity["config_sha256"],
            "manifest_sha256": run_identity["manifest_sha256"],
            "split": args.split,
            "alpha": args.update_scale,
            "refinement_steps": args.refinement_steps,
            "max_molecules": args.max_molecules,
            "completed_ordered_sample_ids": [str(row["sample_id"]) for row in records],
            "completed_count": len(records),
            "total_count": total,
            "current_molecule": None,
            "average_seconds_per_molecule": 0.0,
            "eta_seconds": None,
            "failed_molecules": failed_molecules,
            "solver_backend_counts": dict(backend_totals),
            "partial_samples_path": str(partial_path.resolve()),
            "device": device,
            "thread_configuration": thread_config,
        }

        pending_records: list[dict] = []
        pending_trajectory: list[dict] = []
        pending_profile_rows: list[dict] = []
        sealed_count = len(records)
        chunk_count = chunk_scan.chunk_count if chunk_scan is not None else 0
        latest_chunk_hash = (
            chunk_scan.latest_chunk_sha256 if chunk_scan is not None else None
        )

        for index in range(len(records), total):
            manifest_row = selected_rows[index]
            sample_id = str(manifest_row["sample_id"])
            elapsed = time.perf_counter() - started
            average = elapsed / len(records) if records else 0.0
            if args.partial_format == "legacy":
                state.update({
                    "status": "running",
                    "updated_at": _utc_now(),
                    "completed_ordered_sample_ids": [str(row["sample_id"]) for row in records],
                    "completed_count": len(records),
                    "current_molecule": sample_id,
                    "average_seconds_per_molecule": average,
                    "eta_seconds": average * (total - len(records)) if average else None,
                })
                save_sampling_state(state)
            molecule_started = time.perf_counter()
            try:
                transfer_started = time.perf_counter()
                data = by_id[sample_id].to(device)
                _sync(device)
                cpu_to_device_time = time.perf_counter() - transfer_started
                should_profile = args.profile and len(profile_rows) < args.profile_molecules
                refined, diagnostics = model.refine(
                    data,
                    args.refinement_steps,
                    args.update_scale,
                    args.max_displacement,
                    args.max_coordinate_norm,
                    args.joint_mode,
                    args.save_trajectory_metrics,
                    profile=should_profile,
                )
                transfer_started = time.perf_counter()
                atomic_numbers = data.atomic_numbers.detach().cpu()
                x_init = data.x_init.detach().cpu()
                x_refined = refined.detach().cpu() if diagnostics["stable"] else None
                _sync(device)
                device_to_cpu_time = time.perf_counter() - transfer_started
                record = {
                    "mol_id": data.mol_id,
                    "sample_id": data.sample_id,
                    "source_mol_id": data.source_mol_id,
                    "smiles": data.smiles,
                    "atomic_numbers": atomic_numbers,
                    "x_init": x_init,
                    "x_init_hash": str(manifest_row["x_init_hash"]),
                    "x_refined": x_refined,
                    "num_rotatable_bonds": int(data.num_rotatable_bonds.item()),
                    "method_name": "global_coupled_4d_adapter",
                    "motion_mode": model.motion_mode,
                    "status": "success" if diagnostics["stable"] else "failed",
                    "checkpoint_path": checkpoint_identity["path"],
                    "checkpoint_inference_sha256": checkpoint_identity["inference_sha256"],
                    "config_path": str(Path(args.config).resolve()),
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
                records.append(record)
                pending_records.append(record)
                backend_totals.update(diagnostics.get("solver_backend_counts", {}))
                for row in diagnostics["trajectory"]:
                    trajectory_row = {"sample_id": sample_id, **row}
                    trajectory.append(trajectory_row)
                    pending_trajectory.append(trajectory_row)
                molecule_time = time.perf_counter() - molecule_started
                if should_profile:
                    profile_row = {
                        "sample_id": sample_id,
                        "molecule_time": molecule_time,
                        "mean_step_time": diagnostics["mean_step_time"],
                        "step_times": diagnostics["step_times"],
                        "mean_timing": diagnostics["mean_timing"],
                        "preparation_timing": diagnostics["preparation_timing"],
                        "cpu_to_device_time": cpu_to_device_time,
                        "device_to_cpu_time": device_to_cpu_time,
                        "solver_backend_counts": diagnostics["solver_backend_counts"],
                        "devices": diagnostics["devices"],
                        "topology_cache_hit_rate": diagnostics["topology_cache_hit_rate"],
                        "linear_algebra": diagnostics.get("linear_algebra", []),
                    }
                    profile_rows.append(profile_row)
                    pending_profile_rows.append(profile_row)
            except Exception as exc:
                failure = {
                    "sample_id": sample_id,
                    "error": repr(exc),
                    "time": _utc_now(),
                }
                failed_molecules.append(failure)
                if args.partial_format == "chunked":
                    save_sampling_state(
                        chunk_state(
                            status="FAILED",
                            completed_count=sealed_count,
                            total_count=total,
                            completed_chunk_count=chunk_count,
                            current_chunk_size=len(pending_records),
                            save_every_records=args.save_every_records,
                            run_identity=run_identity,
                            ordered_sample_ids_hash=selected_order_hash,
                            output=args.output,
                            device=device,
                            started_at=started_at,
                            latest_chunk_sha256=latest_chunk_hash,
                            latest_error=failure,
                            average_seconds_per_record=average,
                            eta_seconds=(
                                average * (total - sealed_count) if average else None
                            ),
                            legacy_source_sha256=existing_state.get(
                                "legacy_source_sha256"
                            ),
                        )
                    )
                else:
                    state.update({
                        "status": "failed",
                        "updated_at": _utc_now(),
                        "failed_molecules": failed_molecules,
                    })
                    save_sampling_state(state)
                raise

            elapsed = time.perf_counter() - started
            average = elapsed / len(records)
            should_seal = (
                len(pending_records) >= args.save_every_records
                or len(records) == total
            )
            if args.partial_format == "legacy" and should_seal:
                partial_build_started = time.perf_counter()
                completed_manifest = {
                    **selected_manifest,
                    "records": selected_rows[: len(records)],
                }
                partial_payload = build_manifest_aware_sample_payload(
                    records=records,
                    manifest=manifest,
                    manifest_path=args.manifest,
                    selected_manifest=completed_manifest,
                    split=args.split,
                    inference_cache_path=args.cache_dir,
                    inference_by_id=by_id,
                    extra={
                        "partial": True,
                        "run_identity": run_identity,
                        "trajectory": trajectory,
                        "profile_rows": profile_rows,
                        "failed_molecules": failed_molecules,
                    },
                )
                persistence_metrics["partial_payload_build_seconds"] += (
                    time.perf_counter() - partial_build_started
                )
                partial_save_started = time.perf_counter()
                atomic_torch_save(partial_payload, partial_path)
                persistence_metrics["partial_serialization_seconds"] += (
                    time.perf_counter() - partial_save_started
                )
                persistence_metrics["partial_bytes_written"] += (
                    partial_path.stat().st_size
                )
                persistence_metrics["partial_write_count"] += 1
                state.update({
                    "status": "partial" if len(records) < total else "finalizing",
                    "updated_at": _utc_now(),
                    "completed_ordered_sample_ids": [str(row["sample_id"]) for row in records],
                    "completed_count": len(records),
                    "current_molecule": None,
                    "average_seconds_per_molecule": average,
                    "eta_seconds": average * (total - len(records)),
                    "solver_backend_counts": dict(backend_totals),
                })
                state["persistence_metrics"] = dict(persistence_metrics)
                save_sampling_state(state)
                pending_records.clear()
                pending_trajectory.clear()
                pending_profile_rows.clear()
                sealed_count = len(records)
            elif args.partial_format == "chunked" and should_seal:
                partial_save_started = time.perf_counter()
                chunk_file, latest_chunk_hash, chunk_created = write_chunk(
                    chunks_path,
                    records=pending_records,
                    selected_rows=selected_rows,
                    chunk_index=chunk_count,
                    start=sealed_count,
                    run_identity=run_identity,
                    previous_chunk_sha256=latest_chunk_hash,
                    auxiliary={
                        "trajectory": pending_trajectory,
                        "profile_rows": pending_profile_rows,
                    },
                )
                persistence_metrics["partial_serialization_seconds"] += (
                    time.perf_counter() - partial_save_started
                )
                if chunk_created:
                    persistence_metrics["partial_bytes_written"] += (
                        chunk_file.stat().st_size
                    )
                    persistence_metrics["partial_write_count"] += 1
                chunk_count += 1
                sealed_count = len(records)
                pending_records.clear()
                pending_trajectory.clear()
                pending_profile_rows.clear()
                save_sampling_state(
                    chunk_state(
                        status="PARTIAL" if len(records) < total else "FINALIZING",
                        completed_count=sealed_count,
                        total_count=total,
                        completed_chunk_count=chunk_count,
                        current_chunk_size=0,
                        save_every_records=args.save_every_records,
                        run_identity=run_identity,
                        ordered_sample_ids_hash=selected_order_hash,
                        output=args.output,
                        device=device,
                        started_at=started_at,
                        latest_chunk_sha256=latest_chunk_hash,
                        average_seconds_per_record=average,
                        eta_seconds=average * (total - sealed_count),
                        legacy_source_sha256=existing_state.get(
                            "legacy_source_sha256"
                        ),
                    )
                )
            print(
                f"[{index + 1}/{total}] {sample_id} {molecule_time:.2f}s; "
                f"mean={average:.2f}s ETA={average * (total - len(records)):.1f}s; "
                f"backends={dict(backend_totals)}",
                flush=True,
            )
            if args.profile and profile_rows:
                profile_payload = _profile_payload(
                    profile_rows,
                    dataset_load_time=dataset_load_time,
                    thread_config=thread_config,
                    partial_path=partial_path,
                )
                _write_profile(profile_payload, args.profile_json, args.profile_markdown)

        final_merge_started = time.perf_counter()
        final_scan_seconds = 0.0
        partial_disk_bytes = 0
        if args.partial_format == "chunked":
            final_scan = scan_chunks(
                chunks_path,
                selected_rows=selected_rows,
                run_identity=run_identity,
            )
            final_scan_seconds = final_scan.scan_seconds
            partial_disk_bytes = final_scan.total_bytes
            if final_scan.completed_count != total:
                raise ValueError(
                    f"Chunk merge expected {total} records, found "
                    f"{final_scan.completed_count}"
                )
            records = list(final_scan.records)
            trajectory = list(final_scan.trajectory)
            profile_rows = list(final_scan.profile_rows)
            _validate_resume_records(records, selected_rows)
            backend_totals = Counter()
            for record in records:
                backend_totals.update(record.get("solver_backend_counts", {}))
        final_merge_seconds = (
            time.perf_counter() - final_merge_started
            if args.partial_format == "chunked"
            else 0.0
        )

        provenance = collect_run_provenance(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            cache_path=args.cache_dir,
        )
        provenance.update({
            "label_free": True,
            "joint_mode": args.joint_mode,
            "checkpoint_identity": checkpoint_identity,
            "config_sha256": run_identity["config_sha256"],
            "device": device,
            "thread_configuration": thread_config,
        })
        failures = sum(row["status"] != "success" for row in records)
        payload_build_started = time.perf_counter()
        payload = build_manifest_aware_sample_payload(
            records=records,
            manifest=manifest,
            manifest_path=args.manifest,
            selected_manifest=selected_manifest,
            split=args.split,
            inference_cache_path=args.cache_dir,
            inference_by_id=by_id,
            extra={
                "provenance": provenance,
                "failure_count": failures,
                "failure_rate": failures / len(records) if records else 0.0,
                "solver_backend_counts": dict(backend_totals),
                "persistence": {
                    "partial_format": args.partial_format,
                    "save_every_records": args.save_every_records,
                    "run_identity": run_identity,
                    "completed_chunk_count": chunk_count,
                    "resume_scan_seconds": (
                        chunk_scan.scan_seconds if chunk_scan is not None else 0.0
                    ),
                    "final_scan_seconds": final_scan_seconds,
                    "peak_partial_disk_bytes": partial_disk_bytes,
                    "metrics": dict(persistence_metrics),
                },
            },
        )
        final_payload_build_seconds = time.perf_counter() - payload_build_started
        final_save_started = time.perf_counter()
        atomic_torch_save(payload, args.output)
        final_save_seconds = time.perf_counter() - final_save_started
        finalization_seconds = time.perf_counter() - final_merge_started
        if args.save_trajectory_metrics:
            _atomic_trajectory(
                trajectory,
                args.output.with_name(args.output.stem + "_trajectory.csv"),
            )
        if args.partial_format == "legacy" and partial_path.exists():
            partial_path.unlink()
        total_time = time.perf_counter() - started
        if args.partial_format == "chunked":
            save_sampling_state(
                {
                    **chunk_state(
                        status="COMPLETED",
                        completed_count=total,
                        total_count=total,
                        completed_chunk_count=chunk_count,
                        current_chunk_size=0,
                        save_every_records=args.save_every_records,
                        run_identity=run_identity,
                        ordered_sample_ids_hash=selected_order_hash,
                        output=args.output,
                        device=device,
                        started_at=started_at,
                        latest_chunk_sha256=latest_chunk_hash,
                        average_seconds_per_record=(total_time / total if total else 0.0),
                        eta_seconds=0.0,
                        total_seconds=total_time,
                        legacy_source_sha256=existing_state.get(
                            "legacy_source_sha256"
                        ),
                    ),
                    "final_payload_build_seconds": final_payload_build_seconds,
                    "final_serialization_seconds": final_save_seconds,
                    "final_merge_seconds": final_merge_seconds,
                    "finalization_seconds": finalization_seconds,
                    "resume_scan_seconds": (
                        chunk_scan.scan_seconds if chunk_scan is not None else 0.0
                    ),
                    "peak_partial_disk_bytes": partial_disk_bytes,
                }
            )
            if args.cleanup_partial_chunks:
                cleanup_chunks(chunks_path)
        elif args.partial_format == "legacy":
            save_sampling_state({
                **state,
                "status": "completed",
                "updated_at": _utc_now(),
                "completed_count": total,
                "current_molecule": None,
                "eta_seconds": 0.0,
                "total_seconds": total_time,
                "output": str(args.output.resolve()),
                "partial_format": args.partial_format,
                "save_every_records": args.save_every_records,
                "final_payload_build_seconds": final_payload_build_seconds,
                "final_serialization_seconds": final_save_seconds,
                "final_merge_seconds": final_merge_seconds,
                "finalization_seconds": finalization_seconds,
                "persistence_metrics": dict(persistence_metrics),
            })
        else:
            save_sampling_state({
                "status": "COMPLETED",
                "updated_at": _utc_now(),
                "completed_count": total,
                "total_count": total,
                "checkpoint_inference_sha256": run_identity[
                    "checkpoint_inference_sha256"
                ],
                "config_sha256": run_identity["config_sha256"],
                "manifest_sha256": run_identity["manifest_sha256"],
                "ordered_sample_ids_sha256": selected_order_hash,
                "split": args.split,
                "device": device,
                "output": str(args.output.resolve()),
                "partial_format": args.partial_format,
                "save_every_records": args.save_every_records,
                "total_seconds": total_time,
                "final_payload_build_seconds": final_payload_build_seconds,
                "final_serialization_seconds": final_save_seconds,
                "final_merge_seconds": final_merge_seconds,
                "finalization_seconds": finalization_seconds,
                "persistence_metrics": dict(persistence_metrics),
            })
        update_run_state(
            args.output.parent,
            "completed",
            stage="sampling",
            output=str(args.output),
            num_records=len(records),
            failure_count=failures,
            total_seconds=total_time,
        )
    except Exception as exc:
        update_run_state(args.output.parent, "failed", stage="sampling", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
