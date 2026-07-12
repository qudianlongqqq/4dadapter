#!/usr/bin/env python
"""Measure current full-rewrite resume I/O against append-only chunks."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.commons.global4d_performance import (
    PROFILE_SCHEMA_VERSION,
    compact_json,
    pearson_correlation,
    run_save_policy_benchmark,
    synthetic_sample_record,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("--atoms", type=int, default=40)
    parser.add_argument("--target_records", type=int, default=23882)
    parser.add_argument("--save_every_records", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--output_dir", type=Path, default=Path("reports/global4d_sampling_io"))
    parser.add_argument("--keep_work_dir", action="store_true")
    args = parser.parse_args()
    if args.records < 1 or args.atoms < 1:
        parser.error("--records and --atoms must be positive")
    records = [synthetic_sample_record(index, args.atoms) for index in range(args.records)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="global4d_io_", dir=args.output_dir))
    results = []
    try:
        for mode in ("full_rewrite", "chunk"):
            for every in sorted(set(args.save_every_records)):
                if every < 1:
                    parser.error("save frequencies must be positive")
                result = run_save_policy_benchmark(
                    records,
                    work / f"{mode}_{every}",
                    save_every=every,
                    mode=mode,
                    state_writes_per_save=2 if mode == "full_rewrite" else 1,
                )
                for event in result["save_events"]:
                    event.update({"mode": mode, "save_every_records": every})
                results.append(result)
        single = run_save_policy_benchmark(
            records,
            work / "single_final_write",
            save_every=len(records),
            mode="full_rewrite",
            write_state=False,
        )
        event_rows = [
            event
            for result in results
            for event in result.pop("save_events", [])
        ]
        current_events = [
            row
            for row in event_rows
            if row["mode"] == "full_rewrite" and row["save_every_records"] == 1
        ]
        segment = max(1, len(current_events) // 10)
        event_total = lambda row: float(row["save_seconds"]) + float(row["state_seconds"])
        current_policy = next(
            row
            for row in results
            if row["mode"] == "full_rewrite" and row["save_every_records"] == 1
        )
        growth = {
            "record_index_vs_save_time_correlation": pearson_correlation(
                [row["completed_count"] for row in current_events],
                [event_total(row) for row in current_events],
            ),
            "record_index_vs_partial_bytes_correlation": pearson_correlation(
                [row["completed_count"] for row in current_events],
                [row["file_bytes"] for row in current_events],
            ),
            "first_10_percent_mean_event_seconds": sum(
                event_total(row) for row in current_events[:segment]
            ) / segment,
            "middle_50_percent_mean_event_seconds": sum(
                event_total(row)
                for row in current_events[len(current_events) // 4 : 3 * len(current_events) // 4]
            ) / max(len(current_events) // 2, 1),
            "last_10_percent_mean_event_seconds": sum(
                event_total(row) for row in current_events[-segment:]
            ) / segment,
            "cumulative_to_final_file_byte_amplification": (
                current_policy["total_serialized_bytes"]
                / current_policy["final_partial_bytes"]
            ),
            "target_records": args.target_records,
            "target_full_rewrite_byte_amplification": (args.target_records + 1) / 2,
            "host_specific_quadratic_time_extrapolation_seconds": (
                current_policy["total_seconds"]
                * (args.target_records / args.records) ** 2
            ),
            "extrapolation_warning": (
                "Quadratic extrapolation assumes the synthetic record size and this host's "
                "filesystem/fsync behavior; it is diagnostic, not a runtime promise."
            ),
        }
        single.pop("save_events", None)
        payload = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": "global4d_resume_io",
            "record_count": args.records,
            "atoms_per_record": args.atoms,
            "filesystem_note": "Results are host/filesystem-specific; compare ratios before absolute times.",
            "single_final_write": single,
            "policies": results,
            "growth_analysis": growth,
            "complexity": {
                "full_rewrite_every_record": "O(N^2) serialized bytes and fsync work",
                "full_rewrite_every_k": "O(N^2/k) serialized bytes",
                "append_only_chunks": "O(N) serialized bytes",
            },
        }
        compact_json(payload, args.output_dir / "global4d_sampling_io_benchmark.json")
        write_csv(results, args.output_dir / "global4d_sampling_io_benchmark.csv")
        write_csv(event_rows, args.output_dir / "global4d_sampling_io_events.csv")
        lines = [
            "# Global 4D sampling I/O benchmark",
            "",
            f"Synthetic records: {args.records}; atoms/record: {args.atoms}.",
            "",
            "| Mode | Save every | Saves | Total s | Save s | State s | Serialized MiB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in results:
            lines.append(
                f"| {row['mode']} | {row['save_every_records']} | {row['save_count']} | "
                f"{row['total_seconds']:.6f} | {row['save_seconds']:.6f} | "
                f"{row['state_seconds']:.6f} | {row['total_serialized_bytes'] / 2**20:.3f} |"
            )
        lines.extend([
            "",
            "The current sampler corresponds to `full_rewrite`, save every 1 record, plus two atomic JSON state writes per record.",
        ])
        (args.output_dir / "global4d_sampling_io_benchmark.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2))
    finally:
        if not args.keep_work_dir:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
