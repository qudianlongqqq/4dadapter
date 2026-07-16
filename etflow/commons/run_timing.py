"""Crash-safe wall/monotonic timing and heartbeat utilities for long runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .global_coupled_4d_sampling import atomic_json_save


INTERVAL_FIELDS = (
    "step_start", "step_end", "interval_seconds", "cumulative_training_seconds",
    "active_optimizer_seconds", "validation_seconds", "steps_per_second",
    "examples_per_second", "ETA_seconds", "ETA_finish_time",
    "cuda_peak_allocated_mib", "cuda_peak_reserved_mib", "gpu_utilization_mean",
    "gpu_utilization_p95",
)


def iso_now() -> str:
    """Return local ISO 8601 time including the UTC offset."""

    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class RunTiming:
    """Persist a multi-process timeline without mixing pipeline and training time."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeline_path = self.output_dir / "run_timeline.log"
        self.timing_path = self.output_dir / "timing.json"
        self.csv_path = self.output_dir / "timing.csv"

    def load(self) -> dict[str, Any]:
        if not self.timing_path.is_file():
            return {"schema_version": "ecir-mvr-run-timing-v1", "events": [], "segments": []}
        return json.loads(self.timing_path.read_text(encoding="utf-8"))

    def save(self, state: Mapping[str, Any]) -> None:
        atomic_json_save(dict(state), self.timing_path)

    def mark(self, event: str, **details: Any) -> dict[str, Any]:
        state = self.load()
        record = {
            "event": str(event), "timestamp": iso_now(),
            "unix_seconds": time.time(), "monotonic_seconds": time.monotonic(),
            **details,
        }
        state.setdefault("events", []).append(record)
        if event == "pipeline_start":
            state["pipeline_started_at"] = record["timestamp"]
            state["pipeline_started_unix"] = record["unix_seconds"]
            state["pipeline_started_monotonic"] = record["monotonic_seconds"]
        elif event == "pipeline_end":
            state["pipeline_finished_at"] = record["timestamp"]
            state["pipeline_finished_unix"] = record["unix_seconds"]
            start = state.get("pipeline_started_monotonic", record["monotonic_seconds"])
            state["pipeline_wall_seconds"] = record["monotonic_seconds"] - start
        self.save(state)
        with self.timeline_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record

    def event_seconds(self, start_event: str, end_event: str) -> float:
        events = self.load().get("events", [])
        starts: list[float] = []
        total = 0.0
        for event in events:
            if event["event"] == start_event:
                starts.append(float(event["monotonic_seconds"]))
            elif event["event"] == end_event and starts:
                total += float(event["monotonic_seconds"]) - starts.pop(0)
        return total

    def write_intervals(self, rows: list[Mapping[str, Any]]) -> None:
        lines = []
        if rows:
            from io import StringIO

            stream = StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=INTERVAL_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in INTERVAL_FIELDS})
            lines = stream.getvalue()
        else:
            lines = ",".join(INTERVAL_FIELDS) + "\n"
        _atomic_text(self.csv_path, lines)

    def finalize(
        self,
        *,
        completed_optimizer_steps: int,
        batch_size: int,
        active_optimizer_seconds: float,
        interval_rows: list[Mapping[str, Any]] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self.load()
        events = state.get("events", [])
        training_starts = [event for event in events if event["event"] == "training_process_start"]
        training_ends = [event for event in events if event["event"] == "training_process_end"]
        training_start = training_starts[0] if training_starts else None
        training_end = training_ends[-1] if training_ends else None
        training_wall = (
            float(training_end["monotonic_seconds"]) - float(training_start["monotonic_seconds"])
            if training_start and training_end else 0.0
        )
        steps = int(completed_optimizer_steps)
        active = float(active_optimizer_seconds)
        steps_per_second = steps / active if active > 0 else 0.0
        examples_per_second = steps * int(batch_size) / active if active > 0 else 0.0
        state.update({
            "training_started_at": training_start["timestamp"] if training_start else None,
            "training_finished_at": training_end["timestamp"] if training_end else None,
            "training_wall_seconds": training_wall,
            "active_optimizer_seconds": active,
            "validation_seconds": self.event_seconds("validation_start", "validation_end"),
            "checkpoint_io_seconds": self.event_seconds("checkpoint_save_start", "checkpoint_save_end"),
            "final_evaluation_seconds": self.event_seconds("final_evaluation_start", "final_evaluation_end"),
            "bootstrap_seconds": self.event_seconds("bootstrap_start", "bootstrap_end"),
            "report_seconds": self.event_seconds("report_generation_start", "report_generation_end"),
            "completed_optimizer_steps": steps,
            "mean_optimizer_steps_per_second": steps_per_second,
            "mean_examples_per_second": examples_per_second,
            "estimated_100k_active_seconds": 100000.0 / steps_per_second if steps_per_second > 0 else math.inf,
            "estimated_100k_active_hours": 100000.0 / steps_per_second / 3600.0 if steps_per_second > 0 else math.inf,
        })
        if extra:
            state.update(dict(extra))
        self.save(state)
        if interval_rows is not None:
            self.write_intervals(interval_rows)
        return state


def write_heartbeat(output_dir: str | Path, **values: Any) -> None:
    output = Path(output_dir)
    heartbeat = output / "heartbeat.json"
    current = json.loads(heartbeat.read_text(encoding="utf-8")) if heartbeat.is_file() else {}
    current.update(values)
    current.setdefault("pid", os.getpid())
    current["updated_at"] = iso_now()
    atomic_json_save(current, heartbeat)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    mark = sub.add_parser("mark")
    mark.add_argument("event")
    mark.add_argument("--details", default="{}")
    heartbeat = sub.add_parser("heartbeat")
    heartbeat.add_argument("status")
    heartbeat.add_argument("--error")
    args = parser.parse_args()
    if args.command == "mark":
        details = json.loads(args.details)
        RunTiming(args.output_dir).mark(args.event, **details)
    else:
        write_heartbeat(
            args.output_dir, status=args.status, latest_error=args.error,
            current_step=0, target_step=20000,
        )


if __name__ == "__main__":
    _main()
