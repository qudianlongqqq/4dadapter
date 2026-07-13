"""Low-overhead helpers for Global Coupled 4D performance audits.

This module deliberately contains no model code.  It is shared by diagnostic
scripts so that save-policy simulations and compact statistics can be tested
without loading CUDA, Lightning, or torch-geometric.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch

from etflow.commons.global_coupled_4d_sampling import atomic_json_save, atomic_torch_save


PROFILE_SCHEMA_VERSION = "1.0"


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(data),
        "min": min(data),
        "median": statistics.median(data),
        "mean": statistics.fmean(data),
        "p90": percentile(data, 0.90),
        "p99": percentile(data, 0.99),
        "max": max(data),
    }


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_delta = [float(value) - left_mean for value in left]
    right_delta = [float(value) - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


class StageAccumulator:
    """Accumulate host wall time without synchronizing CUDA implicitly."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, float | int]] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            row = self.rows.setdefault(
                name, {"calls": 0, "cpu_wall_seconds": 0.0, "cuda_seconds": 0.0}
            )
            row["calls"] = int(row["calls"]) + 1
            row["cpu_wall_seconds"] = float(row["cpu_wall_seconds"]) + elapsed

    def add(
        self,
        name: str,
        *,
        calls: int = 1,
        cpu_wall_seconds: float = 0.0,
        cuda_seconds: float = 0.0,
    ) -> None:
        row = self.rows.setdefault(
            name, {"calls": 0, "cpu_wall_seconds": 0.0, "cuda_seconds": 0.0}
        )
        row["calls"] = int(row["calls"]) + int(calls)
        row["cpu_wall_seconds"] = float(row["cpu_wall_seconds"]) + float(
            cpu_wall_seconds
        )
        row["cuda_seconds"] = float(row["cuda_seconds"]) + float(cuda_seconds)

    def compact(self, records: int, steps: int, total_seconds: float) -> list[dict]:
        output = []
        for name, row in self.rows.items():
            cpu = float(row["cpu_wall_seconds"])
            cuda = float(row["cuda_seconds"])
            output.append(
                {
                    "stage": name,
                    "calls": int(row["calls"]),
                    "cpu_wall_seconds": cpu,
                    "cuda_seconds": cuda,
                    "self_seconds": max(cpu - cuda, 0.0),
                    "seconds_per_record": cpu / records if records else 0.0,
                    "seconds_per_refinement_step": cpu / steps if steps else 0.0,
                    "wall_time_fraction": cpu / total_seconds if total_seconds else 0.0,
                }
            )
        return sorted(output, key=lambda item: item["cpu_wall_seconds"], reverse=True)


class CudaEventTimer:
    """Measure CUDA elapsed time and explicitly report synchronization cost."""

    def __init__(self, device: str, enabled: bool) -> None:
        self.device = torch.device(device)
        self.enabled = bool(enabled and self.device.type == "cuda")
        self.elapsed_seconds = 0.0
        self.synchronize_seconds = 0.0
        self._start = None
        self._end = None

    def __enter__(self):
        if self.enabled:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.enabled:
            self._end.record()
            sync_started = time.perf_counter()
            torch.cuda.synchronize(self.device)
            self.synchronize_seconds = time.perf_counter() - sync_started
            self.elapsed_seconds = float(self._start.elapsed_time(self._end)) / 1000.0


def synthetic_sample_record(index: int, atoms: int = 40) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(index)
    return {
        "mol_id": f"mol-{index // 100:06d}",
        "source_mol_id": f"mol-{index // 100:06d}",
        "sample_id": f"sample-{index:08d}",
        "x_init_hash": f"hash-{index:08d}",
        "method_name": "global_coupled_4d_adapter",
        "status": "success",
        "atomic_numbers": torch.full((atoms,), 6, dtype=torch.long),
        "x_init": torch.randn((atoms, 3), generator=generator),
        "x_refined": torch.randn((atoms, 3), generator=generator),
        "num_rotatable_bonds": max(atoms // 10, 1),
        "solver_backend_counts": {"svd_fallback": 10},
    }


def _partial_payload(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "partial": True,
        "records": list(records),
        "manifest_provenance": {
            "manifest_sha256": "synthetic",
            "ordered_sample_ids": [str(row["sample_id"]) for row in records],
            "sample_count": len(records),
        },
    }


def run_save_policy_benchmark(
    records: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    save_every: int,
    mode: str,
    write_state: bool = True,
    state_writes_per_save: int = 1,
) -> dict[str, Any]:
    """Benchmark full rewrites or append-only chunks with crash-safe writes."""

    if save_every < 1:
        raise ValueError("save_every must be positive")
    if mode not in {"full_rewrite", "chunk"}:
        raise ValueError("mode must be full_rewrite or chunk")
    if state_writes_per_save < 0:
        raise ValueError("state_writes_per_save must be non-negative")
    root.mkdir(parents=True, exist_ok=True)
    partial = root / "partial_samples.pt"
    chunks = root / "chunks"
    chunks.mkdir(exist_ok=True)
    state = root / "sampling_state.json"
    save_rows = []
    started = time.perf_counter()
    for end in range(save_every, len(records) + save_every, save_every):
        end = min(end, len(records))
        begin = int(save_rows[-1]["completed_count"]) if save_rows else 0
        if save_rows and end == save_rows[-1]["completed_count"]:
            break
        save_started = time.perf_counter()
        if mode == "full_rewrite":
            atomic_torch_save(_partial_payload(records[:end]), partial)
            written_path = partial
        else:
            written_path = chunks / f"records_{begin:08d}_{end:08d}.pt"
            atomic_torch_save(
                {"start": begin, "end": end, "records": list(records[begin:end])},
                written_path,
            )
        save_seconds = time.perf_counter() - save_started
        state_seconds = 0.0
        if write_state:
            state_started = time.perf_counter()
            for write_index in range(state_writes_per_save):
                state_payload = {
                    "status": "running" if write_index == 0 else "partial",
                    "completed_count": end,
                }
                if mode == "full_rewrite":
                    state_payload["completed_ordered_sample_ids"] = [
                        str(row["sample_id"]) for row in records[:end]
                    ]
                else:
                    state_payload["last_chunk"] = written_path.name
                    state_payload["chunk_count"] = len(save_rows) + 1
                atomic_json_save(
                    state_payload,
                    state,
                )
            state_seconds = time.perf_counter() - state_started
        save_rows.append(
            {
                "completed_count": end,
                "save_seconds": save_seconds,
                "state_seconds": state_seconds,
                "file_bytes": written_path.stat().st_size,
            }
        )
        if end == len(records):
            break
    total = time.perf_counter() - started
    return {
        "mode": mode,
        "record_count": len(records),
        "save_every_records": save_every,
        "save_count": len(save_rows),
        "state_writes_per_save": state_writes_per_save if write_state else 0,
        "total_seconds": total,
        "save_seconds": sum(float(row["save_seconds"]) for row in save_rows),
        "state_seconds": sum(float(row["state_seconds"]) for row in save_rows),
        "total_serialized_bytes": sum(int(row["file_bytes"]) for row in save_rows),
        "final_partial_bytes": partial.stat().st_size if partial.is_file() else 0,
        "chunk_bytes": sum(path.stat().st_size for path in chunks.glob("*.pt")),
        "save_events": save_rows,
    }


def run_current_full_rewrite_benchmark(
    records: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    payload_factory: Callable[[int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure the formal sampler's exact per-record persistence sequence.

    The current sampler writes a growing pre-record JSON state, constructs a
    manifest-aware payload for the complete ordered prefix, overwrites the
    partial ``.pt`` file, and then writes a growing post-record JSON state.
    ``payload_factory`` keeps the shared provenance implementation at the call
    site instead of duplicating it in this diagnostics module.
    """

    root.mkdir(parents=True, exist_ok=True)
    partial = root / "partial_samples.pt"
    state = root / "sampling_state.json"
    events = []
    payload_build_seconds = 0.0
    save_seconds = 0.0
    state_seconds = 0.0
    started = time.perf_counter()
    for end in range(1, len(records) + 1):
        state_started = time.perf_counter()
        atomic_json_save(
            {
                "status": "running",
                "completed_count": end - 1,
                "total_count": len(records),
                "completed_ordered_sample_ids": [
                    str(row["sample_id"]) for row in records[: end - 1]
                ],
            },
            state,
        )
        pre_state_seconds = time.perf_counter() - state_started

        build_started = time.perf_counter()
        payload = payload_factory(end)
        build_seconds = time.perf_counter() - build_started

        save_started = time.perf_counter()
        atomic_torch_save(payload, partial)
        event_save_seconds = time.perf_counter() - save_started

        state_started = time.perf_counter()
        atomic_json_save(
            {
                "status": "partial" if end < len(records) else "finalizing",
                "completed_count": end,
                "total_count": len(records),
                "completed_ordered_sample_ids": [
                    str(row["sample_id"]) for row in records[:end]
                ],
            },
            state,
        )
        post_state_seconds = time.perf_counter() - state_started
        event_state_seconds = pre_state_seconds + post_state_seconds

        payload_build_seconds += build_seconds
        save_seconds += event_save_seconds
        state_seconds += event_state_seconds
        events.append(
            {
                "completed_count": end,
                "payload_build_seconds": build_seconds,
                "save_seconds": event_save_seconds,
                "state_seconds": event_state_seconds,
                "file_bytes": partial.stat().st_size,
            }
        )
    return {
        "mode": "current_formal_full_rewrite",
        "record_count": len(records),
        "save_every_records": 1,
        "save_count": len(events),
        "state_writes_per_save": 2,
        "payload_build_seconds": payload_build_seconds,
        "save_seconds": save_seconds,
        "state_seconds": state_seconds,
        "total_seconds": time.perf_counter() - started,
        "total_serialized_bytes": sum(int(row["file_bytes"]) for row in events),
        "final_partial_bytes": partial.stat().st_size if partial.is_file() else 0,
        "chunk_bytes": 0,
        "save_events": events,
    }


def recover_record_chunks(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    expected = 0
    for path in sorted((root / "chunks").glob("records_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["start"]) != expected:
            raise ValueError(f"Non-contiguous chunk sequence at {path}")
        chunk = list(payload["records"])
        if int(payload["end"]) != expected + len(chunk):
            raise ValueError(f"Invalid chunk bounds at {path}")
        records.extend(chunk)
        expected += len(chunk)
    return records


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def compact_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write a report while rejecting accidental raw record payloads."""

    forbidden = {"records", "ordered_sample_ids", "completed_ordered_sample_ids"}

    def check(value: Any) -> None:
        if isinstance(value, Mapping):
            overlap = forbidden.intersection(value)
            if overlap:
                raise ValueError(f"Compact report contains large fields: {sorted(overlap)}")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(payload)
    atomic_json_save(payload, path)
