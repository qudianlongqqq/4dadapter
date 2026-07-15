#!/usr/bin/env python
"""RTX training-capacity benchmark for the real Serial Global4D model/cache."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.serial_global4d.cache import SerialGlobal4DResidualDataset
from etflow.serial_global4d.model import SerialGlobal4DResidualRefiner


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _gpu_utilization() -> float | None:
    try:
        value = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()[0]
        return float(value.strip())
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


def _cohort_indices(cache_root: Path) -> dict[str, list[int]]:
    manifest = json.loads((cache_root / "train_manifest.json").read_text("utf-8"))
    rows = manifest["records"]
    mixed = list(range(len(rows)))
    high = []
    for index, row in enumerate(rows):
        record = torch.load(
            cache_root / "train" / row["path"],
            map_location="cpu",
            weights_only=False,
        )
        if str(record.get("flexibility_cohort")) == "high":
            high.append(
                (
                    int(record["num_joints"]),
                    int(record["num_atoms"]),
                    int(record["num_edges"]),
                    index,
                )
            )
    # Cycle through a broad upper-complexity pool, not one favorable batch.
    high.sort(reverse=True)
    high_indices = [row[-1] for row in high[: max(512, len(high) // 3)]]
    return {"mixed": mixed, "high_complexity": high_indices}


def _new_model(device: str) -> SerialGlobal4DResidualRefiner:
    return SerialGlobal4DResidualRefiner().to(device)


def _run_candidate(
    dataset,
    indices: list[int],
    *,
    cohort: str,
    batch_size: int,
    warmup_steps: int,
    measured_steps: int,
    device: str,
) -> dict:
    torch.manual_seed(42)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(42),
    )
    iterator = iter(loader)
    model = _new_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=1.0e-6)
    total = warmup_steps + measured_steps
    timings: list[float] = []
    forward_times: list[float] = []
    backward_times: list[float] = []
    optimizer_times: list[float] = []
    grad_norms: list[float] = []
    gpu_util: list[float] = []
    measured_records = measured_atoms = measured_edges = measured_joints = 0
    finite_losses = finite_gradients = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started_all = time.perf_counter()
    try:
        for step in range(total):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            forward_started = started
            output = model.phase_a_loss(batch)
            torch.cuda.synchronize()
            forward_finished = time.perf_counter()
            output["loss"].backward()
            torch.cuda.synchronize()
            backward_finished = time.perf_counter()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gradients_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            finished = time.perf_counter()
            if step >= warmup_steps:
                timings.append(finished - started)
                forward_times.append(forward_finished - forward_started)
                backward_times.append(backward_finished - forward_finished)
                optimizer_times.append(finished - backward_finished)
                grad_norms.append(float(grad_norm.detach()))
                finite_losses += int(bool(torch.isfinite(output["loss"])))
                finite_gradients += int(gradients_finite)
                measured_records += int(batch.num_graphs)
                measured_atoms += int(batch.num_nodes)
                measured_edges += int(batch.edge_index.size(1))
                measured_joints += int(batch.rotatable_bond_index.size(1))
                utilization = _gpu_utilization()
                if utilization is not None:
                    gpu_util.append(utilization)
    except torch.OutOfMemoryError as error:
        del model, optimizer
        torch.cuda.empty_cache()
        return {
            "batch_size": batch_size,
            "accumulation": 1,
            "effective_batch_size": batch_size,
            "cohort": cohort,
            "status": "OOM",
            "oom": True,
            "error": str(error),
        }
    elapsed = sum(timings)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "batch_size": batch_size,
        "accumulation": 1,
        "effective_batch_size": batch_size,
        "cohort": cohort,
        "status": "PASS",
        "oom": False,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "actual_total_records": measured_records,
        "actual_total_atoms": measured_atoms,
        "actual_total_edges": measured_edges,
        "actual_total_joints": measured_joints,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "total_gpu_memory_bytes": total_memory,
        "reserved_fraction": peak_reserved / total_memory,
        "memory_headroom_bytes": total_memory - peak_reserved,
        "mean_step_seconds": statistics.mean(timings),
        "median_step_seconds": statistics.median(timings),
        "p95_step_seconds": _percentile(timings, 0.95),
        "mean_forward_seconds": statistics.mean(forward_times),
        "mean_backward_seconds": statistics.mean(backward_times),
        "mean_optimizer_seconds": statistics.mean(optimizer_times),
        "optimizer_steps_per_second": measured_steps / elapsed,
        "records_per_second": measured_records / elapsed,
        "atoms_per_second": measured_atoms / elapsed,
        "joints_per_second": measured_joints / elapsed,
        "finite_loss_fraction": finite_losses / measured_steps,
        "finite_gradient_fraction": finite_gradients / measured_steps,
        "mean_gradient_norm": statistics.mean(grad_norms),
        "max_gradient_norm": max(grad_norms),
        "mean_gpu_utilization_percent": (
            statistics.mean(gpu_util) if gpu_util else None
        ),
        "wall_seconds": time.perf_counter() - started_all,
    }


def _recommend(rows: list[dict]) -> dict:
    passed = [row for row in rows if row["status"] == "PASS"]
    mixed = {row["batch_size"]: row for row in passed if row["cohort"] == "mixed"}
    high = {
        row["batch_size"]: row for row in passed if row["cohort"] == "high_complexity"
    }
    common = sorted(set(mixed).intersection(high))
    safe = [
        size
        for size in common
        if max(mixed[size]["reserved_fraction"], high[size]["reserved_fraction"]) <= 0.8
        and min(
            mixed[size]["memory_headroom_bytes"], high[size]["memory_headroom_bytes"]
        )
        >= int(2.5 * 1024**3)
    ]
    max_throughput = max(
        common,
        key=lambda size: min(
            mixed[size]["records_per_second"], high[size]["records_per_second"]
        ),
    )
    best_throughput = min(
        mixed[max_throughput]["records_per_second"],
        high[max_throughput]["records_per_second"],
    )
    eligible = [
        size
        for size in safe
        if min(mixed[size]["records_per_second"], high[size]["records_per_second"])
        >= 0.95 * best_throughput
    ]
    recommended = min(eligible) if eligible else max(safe)
    return {
        "max_oom_free_batch": max(common),
        "max_safe_batch": max(safe),
        "max_throughput_batch": max_throughput,
        "recommended_training_batch": recommended,
        "recommended_accumulation": 1,
        "recommended_effective_batch": recommended,
        "recommended_lr": 2.0e-4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--batch_sizes", default="4,8,16,32,48,64,96,128")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measured_steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Serial capacity benchmark requires CUDA")
    dataset = SerialGlobal4DResidualDataset(args.cache_root, "train")
    cohorts = _cohort_indices(args.cache_root)
    rows = []
    for cohort, indices in cohorts.items():
        for batch_size in [int(value) for value in args.batch_sizes.split(",")]:
            try:
                row = _run_candidate(
                    dataset,
                    indices,
                    cohort=cohort,
                    batch_size=batch_size,
                    warmup_steps=args.warmup_steps,
                    measured_steps=args.measured_steps,
                    device=args.device,
                )
            except Exception as error:
                row = {
                    "batch_size": batch_size,
                    "accumulation": 1,
                    "effective_batch_size": batch_size,
                    "cohort": cohort,
                    "status": "ERROR",
                    "oom": False,
                    "error": repr(error),
                }
                torch.cuda.empty_cache()
            rows.append(row)
            print(json.dumps(row), flush=True)
    recommendation = _recommend(rows)
    payload = {
        "status": "COMPLETED",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "real_training_step": [
            "cache_read",
            "cuda_transfer",
            "jacobian_and_forward",
            "q_loss",
            "internal_loss",
            "backward",
            "gradient_clip",
            "optimizer_step",
            "zero_grad",
        ],
        "rows": rows,
        "recommendation": recommendation,
    }
    atomic_json_save(payload, args.output_json)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
