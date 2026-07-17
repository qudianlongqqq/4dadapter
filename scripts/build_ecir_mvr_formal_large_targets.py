#!/usr/bin/env python
"""Build resumable formal-large sources and offline Minimal Validity Targets."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402

from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.formal_target_assets import (  # noqa: E402
    RuntimeTelemetry,
    atomic_json,
    build_source_manifests,
    build_summary,
    build_target,
    clear_failure,
    failure_count,
    finalize_manifests,
    load_config,
    record_failure,
    require_parquet_engine,
    summary_markdown,
    unresolved_failure_sample_ids,
    utc_now,
    validate_formal_assets,
    verify_stage_d_identities,
    write_asset_metadata_and_inventory,
)
from etflow.ecir.minimal_validity_target import MinimalValidityTargetBuilder  # noqa: E402


def _rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.sort_values(["molecule_id", "sample_id"]).to_dict("records")


def _state(
    output_root: Path,
    *,
    status: str,
    stage: str,
    started_at: str,
    planned_records: int,
    completed_records: int,
    successful_records: int,
    failed_records: int,
    skipped_records: int,
    error: str | None = None,
) -> None:
    atomic_json(
        {
            "schema_version": "ecir-mvr-formal-large-target-build-state-v1",
            "status": status,
            "stage": stage,
            "pid": os.getpid(),
            "started_at": started_at,
            "updated_at": utc_now(),
            "planned_records": int(planned_records),
            "completed_records": int(completed_records),
            "successful_records": int(successful_records),
            "failed_records": int(failed_records),
            "skipped_records": int(skipped_records),
            "test_records_read": 0,
            "error": error,
        },
        output_root / "logs" / "build_state.json",
    )


def _run_records(
    records: Iterable[Mapping[str, Any]],
    *,
    output_root: Path,
    builder: MinimalValidityTargetBuilder,
    identities: Mapping[str, Any],
    config_file_sha256: str,
    telemetry: RuntimeTelemetry,
    started_at: str,
    stage: str,
    planned_records: int,
    counters: dict[str, int],
) -> None:
    for source in records:
        started = time.perf_counter()
        try:
            _, skipped = build_target(
                source,
                output_root=output_root,
                builder=builder,
                identities=identities,
                config_file_sha256=config_file_sha256,
            )
            clear_failure(source, output_root)
            counters["successful"] += 1
            counters["skipped"] += int(skipped)
            success = True
        except Exception as error:
            record_failure(source, output_root, error)
            counters["failed"] += 1
            success = False
        counters["completed"] += 1
        seconds = time.perf_counter() - started
        telemetry.update(success=success, skipped=success and skipped, seconds=seconds)
        _state(
            output_root,
            status="RUNNING",
            stage=stage,
            started_at=started_at,
            planned_records=planned_records,
            completed_records=counters["completed"],
            successful_records=counters["successful"],
            failed_records=counters["failed"],
            skipped_records=counters["skipped"],
        )


def _write_reports(summary: Mapping[str, Any], report_dir: Path) -> None:
    atomic_json(summary, report_dir / "D1B_FORMAL_TARGET_BUILD_SUMMARY.json")
    from etflow.ecir.formal_target_assets import atomic_text

    atomic_text(
        summary_markdown(summary), report_dir / "D1B_FORMAL_TARGET_BUILD_SUMMARY.md"
    )


def _fail_before_target_build(
    *,
    telemetry: RuntimeTelemetry,
    output_root: Path,
    stage: str,
    started_at: str,
    planned_records: int,
    error: BaseException,
) -> None:
    telemetry.stop()
    _state(
        output_root,
        status="FAILED",
        stage=stage,
        started_at=started_at,
        planned_records=planned_records,
        completed_records=0,
        successful_records=0,
        failed_records=0,
        skipped_records=0,
        error=f"{type(error).__name__}: {error}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ecir_mvr_formal_large_minimal_targets.yaml"),
    )
    parser.add_argument("--input-cache", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/ecir_mvr")
    parser.add_argument("--pilot-records", type=int)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--no-auto-continue", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--gpu-index", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--retry-unresolved-only", action="store_true")
    parser.add_argument("--retry-sample-id", action="append", default=[])
    args = parser.parse_args()

    require_parquet_engine()
    config = load_config(args.config, output_root=args.output_root)
    if args.input_cache is not None:
        config["input_cache"] = str(args.input_cache)
    output_root = Path(config["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    identities = verify_stage_d_identities(config)
    identities["target_builder_config"] = dict(config["target_builder"])
    identities["config_file_sha256"] = config["config_file_sha256"]
    pilot_records = int(args.pilot_records or config["pilot_records"])
    if pilot_records != 100:
        raise ValueError("formal target pilot is frozen at exactly 100 train records")
    auto_continue = bool(config["auto_continue_after_pilot"])
    auto_continue = auto_continue and not args.no_auto_continue and not args.pilot_only
    if args.retry_unresolved_only and (args.pilot_only or args.no_auto_continue):
        raise ValueError("retry-unresolved mode cannot be combined with pilot options")
    if args.retry_sample_id and not args.retry_unresolved_only:
        raise ValueError("--retry-sample-id requires --retry-unresolved-only")
    retry_ids = unresolved_failure_sample_ids(output_root) if args.retry_unresolved_only else set()
    requested_retry_ids = set(map(str, args.retry_sample_id))
    if requested_retry_ids:
        missing_failures = requested_retry_ids - retry_ids
        if missing_failures:
            raise ValueError(
                f"requested sample ids are not unresolved failures: {sorted(missing_failures)}"
            )
        retry_ids = requested_retry_ids
    if args.retry_unresolved_only and not retry_ids:
        raise ValueError("no unresolved formal target failures are available to retry")
    expected_records = sum(
        int(config["splits"][split]["expected_molecules"])
        * int(config["splits"][split]["expected_records_per_molecule"])
        for split in ("train", "val")
    )
    total_records = (
        len(retry_ids)
        if args.retry_unresolved_only
        else (expected_records if auto_continue else pilot_records)
    )
    started_at = utc_now()
    started = time.monotonic()
    counters = {"completed": 0, "successful": 0, "failed": 0, "skipped": 0}
    telemetry = RuntimeTelemetry(
        output_root,
        total_records=total_records,
        interval=float(config["telemetry_interval_seconds"]),
        gpu_index=str(args.gpu_index),
    )
    telemetry.start()
    try:
        source_frames, source_metadata = build_source_manifests(
            config, resume=not args.no_resume
        )
    except BaseException as error:
        _fail_before_target_build(
            telemetry=telemetry,
            output_root=output_root,
            stage="source_manifests",
            started_at=started_at,
            planned_records=total_records,
            error=error,
        )
        raise
    try:
        actual_records = sum(len(frame) for frame in source_frames.values())
        if actual_records != expected_records:
            raise ValueError(
                f"formal source record count is {actual_records}; expected {expected_records}"
            )
        validity = ChemicalValidity(identities["validity_statistics_path"])
        builder = MinimalValidityTargetBuilder(validity, config["target_builder"])
    except BaseException as error:
        _fail_before_target_build(
            telemetry=telemetry,
            output_root=output_root,
            stage="target_builder_initialization",
            started_at=started_at,
            planned_records=total_records,
            error=error,
        )
        raise
    status = "FAILED"
    validation = None
    manifest_metadata: dict[str, Any] = {
        split: {
            "planned_records": len(frame),
            "completed_records": 0,
            "successful_records": 0,
            "failed_records": 0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
            "molecules": 0,
            "target_status_counts": {},
            "target_manifest_sha256": "",
            "pairing_manifest_sha256": "",
            "aggregate_target_sha256": "",
        }
        for split, frame in source_frames.items()
    }
    if args.retry_unresolved_only:
        try:
            source_lookup = {
                str(row["sample_id"]): row
                for frame in source_frames.values()
                for row in _rows(frame)
            }
            missing_sources = retry_ids - set(source_lookup)
            if missing_sources:
                raise ValueError(
                    f"unresolved failures have no source rows: {sorted(missing_sources)}"
                )
            retry_rows = [source_lookup[sample_id] for sample_id in sorted(retry_ids)]
            _run_records(
                retry_rows,
                output_root=output_root,
                builder=builder,
                identities=identities,
                config_file_sha256=config["config_file_sha256"],
                telemetry=telemetry,
                started_at=started_at,
                stage="retry_unresolved",
                planned_records=len(retry_rows),
                counters=counters,
            )
            manifest_metadata = finalize_manifests(
                output_root,
                source_frames,
                int(config["manifest_shard_size"]),
            )
            write_asset_metadata_and_inventory(
                output_root=output_root,
                source_frames=source_frames,
                source_metadata=source_metadata,
                manifest_metadata=manifest_metadata,
                identities=identities,
                config_file_sha256=config["config_file_sha256"],
            )
            validation = validate_formal_assets(
                output_root=output_root,
                source_frames=source_frames,
                identities=identities,
                require_complete=True,
                strict_sample_count=100,
            )
            atomic_json(validation, output_root / "statistics" / "validation.json")
            status = (
                "COMPLETED"
                if validation["decision"] == "D1B_FORMAL_TARGETS_READY"
                else "FAILED"
            )
            if status == "COMPLETED":
                write_asset_metadata_and_inventory(
                    output_root=output_root,
                    source_frames=source_frames,
                    source_metadata=source_metadata,
                    manifest_metadata=manifest_metadata,
                    identities=identities,
                    config_file_sha256=config["config_file_sha256"],
                    decision="D1B_FORMAL_TARGETS_READY",
                )
        except BaseException as error:
            _state(
                output_root,
                status="FAILED",
                stage="retry_unresolved_failed",
                started_at=started_at,
                planned_records=len(retry_ids),
                completed_records=counters["completed"],
                successful_records=counters["successful"],
                failed_records=max(counters["failed"], failure_count(output_root)),
                skipped_records=counters["skipped"],
                error=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            telemetry.stop()
            summary = build_summary(
                output_root=output_root,
                source_metadata=source_metadata,
                manifest_metadata=manifest_metadata,
                identities=identities,
                status=status,
                started_at=started_at,
                elapsed_seconds=time.monotonic() - started,
                validation=validation,
            )
            _write_reports(summary, args.report_dir)
        _state(
            output_root,
            status=status,
            stage="complete" if status == "COMPLETED" else "validation_failed",
            started_at=started_at,
            planned_records=len(retry_ids),
            completed_records=counters["completed"],
            successful_records=counters["successful"],
            failed_records=failure_count(output_root),
            skipped_records=counters["skipped"],
        )
        print(validation["decision"])
        if validation["decision"] != "D1B_FORMAL_TARGETS_READY":
            raise SystemExit(1)
        return
    try:
        pilot = _rows(source_frames["train"])[:pilot_records]
        _run_records(
            pilot,
            output_root=output_root,
            builder=builder,
            identities=identities,
            config_file_sha256=config["config_file_sha256"],
            telemetry=telemetry,
            started_at=started_at,
            stage="pilot100",
            planned_records=pilot_records,
            counters=counters,
        )
        manifest_metadata = finalize_manifests(
            output_root,
            source_frames,
            int(config["manifest_shard_size"]),
        )
        write_asset_metadata_and_inventory(
            output_root=output_root,
            source_frames=source_frames,
            source_metadata=source_metadata,
            manifest_metadata=manifest_metadata,
            identities=identities,
            config_file_sha256=config["config_file_sha256"],
        )
        pilot_validation = validate_formal_assets(
            output_root=output_root,
            source_frames={"train": pd.DataFrame(pilot), "val": source_frames["val"].iloc[0:0]},
            identities=identities,
            require_complete=False,
            strict_sample_count=pilot_records,
        )
        atomic_json(pilot_validation, output_root / "statistics" / "pilot100_validation.json")
        if (
            counters["failed"]
            or pilot_validation["decision"] != "D1B_FORMAL_TARGET_PILOT_PASS"
        ):
            raise RuntimeError("D1B_FORMAL_TARGET_PILOT_FAILED")
        print("D1B_FORMAL_TARGET_PILOT_PASS", flush=True)

        if auto_continue:
            pilot_ids = {str(row["sample_id"]) for row in pilot}
            for split in ("train", "val"):
                records = _rows(source_frames[split])
                if split == "train":
                    records = [
                        row for row in records if str(row["sample_id"]) not in pilot_ids
                    ]
                _run_records(
                    records,
                    output_root=output_root,
                    builder=builder,
                    identities=identities,
                    config_file_sha256=config["config_file_sha256"],
                    telemetry=telemetry,
                    started_at=started_at,
                    stage=f"full_{split}",
                    planned_records=sum(len(frame) for frame in source_frames.values()),
                    counters=counters,
                )
            manifest_metadata = finalize_manifests(
                output_root,
                source_frames,
                int(config["manifest_shard_size"]),
            )
            write_asset_metadata_and_inventory(
                output_root=output_root,
                source_frames=source_frames,
                source_metadata=source_metadata,
                manifest_metadata=manifest_metadata,
                identities=identities,
                config_file_sha256=config["config_file_sha256"],
            )
            validation = validate_formal_assets(
                output_root=output_root,
                source_frames=source_frames,
                identities=identities,
                require_complete=True,
                strict_sample_count=100,
            )
            atomic_json(validation, output_root / "statistics" / "validation.json")
            status = "COMPLETED" if validation["decision"] == "D1B_FORMAL_TARGETS_READY" else "FAILED"
            if status == "COMPLETED":
                write_asset_metadata_and_inventory(
                    output_root=output_root,
                    source_frames=source_frames,
                    source_metadata=source_metadata,
                    manifest_metadata=manifest_metadata,
                    identities=identities,
                    config_file_sha256=config["config_file_sha256"],
                    decision="D1B_FORMAL_TARGETS_READY",
                )
        else:
            validation = pilot_validation
            status = "PILOT_COMPLETED"
    except BaseException as error:
        _state(
            output_root,
            status="FAILED",
            stage="failed",
            started_at=started_at,
            planned_records=total_records,
            completed_records=counters["completed"],
            successful_records=counters["successful"],
            failed_records=max(counters["failed"], failure_count(output_root)),
            skipped_records=counters["skipped"],
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        telemetry.stop()
        summary = build_summary(
            output_root=output_root,
            source_metadata=source_metadata,
            manifest_metadata=manifest_metadata,
            identities=identities,
            status=status,
            started_at=started_at,
            elapsed_seconds=time.monotonic() - started,
            validation=validation,
        )
        _write_reports(summary, args.report_dir)

    _state(
        output_root,
        status=status,
        stage=(
            "complete"
            if status == "COMPLETED"
            else ("pilot_complete" if status == "PILOT_COMPLETED" else "validation_failed")
        ),
        started_at=started_at,
        planned_records=total_records,
        completed_records=counters["completed"],
        successful_records=counters["successful"],
        failed_records=failure_count(output_root),
        skipped_records=counters["skipped"],
    )
    print(validation["decision"])
    if auto_continue and validation["decision"] != "D1B_FORMAL_TARGETS_READY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
