#!/usr/bin/env python
"""Full-training-step capacity benchmark for Gated Global4D V2.

Each (batch size, composition, repeat) runs in a fresh subprocess so a CUDA OOM
cannot contaminate later conditions.  The benchmark never writes to its cache.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch
import yaml
from torch_geometric.loader import DataLoader

from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset
from etflow.models.global4d_checkpoint import resolved_model_arguments
from etflow.models.global_coupled_4d_flow import GlobalCoupled4DFlowLightningModule


DEFAULT_BATCH_SIZES = (4, 8, 16, 32, 48, 64, 96, 128)
COMPOSITIONS = ("low_complexity", "mixed", "high_complexity")


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


class GPUUtilizationSampler:
    def __init__(self, interval_seconds: float = 0.2):
        self.interval_seconds = interval_seconds
        self.values: list[float] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                        "--id=0",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.values.append(float(completed.stdout.splitlines()[0].strip()))
            except Exception as exc:  # Utilization is diagnostic, not correctness.
                self.error = repr(exc)
                return
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return {
            "gpu_utilization_mean_percent": (
                sum(self.values) / len(self.values) if self.values else None
            ),
            "gpu_utilization_max_percent": max(self.values) if self.values else None,
            "gpu_utilization_samples": len(self.values),
            "gpu_utilization_error": self.error,
        }


def _complexity(data) -> tuple[int, int, int]:
    return (
        int(data.num_nodes),
        int(data.edge_index.size(1)),
        int(data.rotatable_bond_index.size(1)),
    )


def _load_composition(
    cache_dir: Path,
    split: str,
    composition: str,
    pool_records: int,
) -> list:
    dataset = FlexBondOptimizerDataset(cache_dir, split, validate=False)
    count = min(len(dataset), int(pool_records))
    pool = [dataset[index] for index in range(count)]
    pool.sort(key=lambda data: (sum(_complexity(data)), _complexity(data)))
    if composition == "low_complexity":
        return pool[: max(len(pool) // 3, 1)]
    if composition == "high_complexity":
        return pool[-max(len(pool) // 3, 1) :]
    return pool


def _infinite_batches(loader):
    while True:
        yield from loader


def _worker(args) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA worker started without CUDA availability")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_args = resolved_model_arguments(config)
    if model_args["fusion_mode"] != "gated_additive":
        raise ValueError("capacity benchmark requires fusion_mode=gated_additive")
    model = GlobalCoupled4DFlowLightningModule(**model_args).cuda().train()
    model.log_dict = lambda *unused_args, **unused_kwargs: None
    optimizer = model.configure_optimizers()
    records = _load_composition(
        Path(args.cache_dir), args.split, args.composition, args.pool_records
    )
    if len(records) < args.batch_size:
        repeats = math.ceil(args.batch_size / len(records))
        records = (records * repeats)[: args.batch_size]
    loader = DataLoader(
        records,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        exclude_keys=["x_ref_candidates", "reference_conformer_ptr", "metadata"],
    )
    batches = _infinite_batches(loader)
    optimizer.zero_grad(set_to_none=True)

    def optimizer_step(measure: bool) -> tuple[int, int, int, int, float]:
        step_records = step_atoms = step_edges = step_joints = 0
        last_loss = float("nan")
        for _ in range(args.accumulate_grad_batches):
            batch = next(batches).to("cuda", non_blocking=True)
            loss = model._shared_step(batch, "train") / args.accumulate_grad_batches
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            last_loss = float(loss.detach()) * args.accumulate_grad_batches
            if measure:
                step_records += int(batch.num_graphs)
                step_atoms += int(batch.num_nodes)
                step_edges += int(batch.edge_index.size(1))
                step_joints += int(batch.rotatable_bond_index.size(1))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return step_records, step_atoms, step_edges, step_joints, last_loss

    for _ in range(args.warmup_optimizer_steps):
        optimizer_step(False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sampler = GPUUtilizationSampler()
    sampler.start()
    measured_records = total_atoms = total_edges = total_joints = optimizer_steps = 0
    final_loss = float("nan")
    started = time.perf_counter()
    while measured_records < args.fixed_records_seen:
        row = optimizer_step(True)
        records_seen, atoms, edges, joints, final_loss = row
        measured_records += records_seen
        total_atoms += atoms
        total_edges += edges
        total_joints += joints
        optimizer_steps += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    utilization = sampler.stop()
    return {
        "batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "effective_batch_size": args.batch_size * args.accumulate_grad_batches,
        "composition": args.composition,
        "repeat": args.repeat,
        "seed": args.seed,
        "warmup_optimizer_steps": args.warmup_optimizer_steps,
        "optimizer_steps": optimizer_steps,
        "fixed_records_seen_target": args.fixed_records_seen,
        "records_seen": measured_records,
        "total_atoms_seen": total_atoms,
        "total_edges_seen": total_edges,
        "total_joints_seen": total_joints,
        "mean_atoms_per_record": total_atoms / measured_records,
        "mean_edges_per_record": total_edges / measured_records,
        "mean_joints_per_record": total_joints / measured_records,
        "elapsed_seconds": elapsed,
        "optimizer_steps_per_second": optimizer_steps / elapsed,
        "records_per_second": measured_records / elapsed,
        "peak_allocated_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_gpu_memory_bytes": torch.cuda.max_memory_reserved(),
        "peak_allocated_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_gpu_memory_mib": torch.cuda.max_memory_reserved() / 2**20,
        "final_loss": final_loss,
        "loss_finite": math.isfinite(final_loss),
        "oom": False,
        **_environment(),
        **utilization,
    }


def _write_reports(payload: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for row in payload["results"]:
        rows.append(
            "| {batch_size} | {accumulate_grad_batches} | {composition} | "
            "{records_seen} | {total_atoms_seen} | {total_edges_seen} | "
            "{total_joints_seen} | {records_per_second} | "
            "{peak_allocated_gpu_memory_mib} | {peak_reserved_gpu_memory_mib} | "
            "{loss_finite} | {oom} | {status} |".format(
                batch_size=row.get("batch_size"),
                accumulate_grad_batches=row.get("accumulate_grad_batches"),
                composition=row.get("composition"),
                records_seen=row.get("records_seen", ""),
                total_atoms_seen=row.get("total_atoms_seen", ""),
                total_edges_seen=row.get("total_edges_seen", ""),
                total_joints_seen=row.get("total_joints_seen", ""),
                records_per_second=(
                    f"{row['records_per_second']:.3f}"
                    if row.get("records_per_second") is not None
                    else ""
                ),
                peak_allocated_gpu_memory_mib=(
                    f"{row['peak_allocated_gpu_memory_mib']:.1f}"
                    if row.get("peak_allocated_gpu_memory_mib") is not None
                    else ""
                ),
                peak_reserved_gpu_memory_mib=(
                    f"{row['peak_reserved_gpu_memory_mib']:.1f}"
                    if row.get("peak_reserved_gpu_memory_mib") is not None
                    else ""
                ),
                loss_finite=(
                    "" if row.get("loss_finite") is None else row.get("loss_finite")
                ),
                oom="" if row.get("oom") is None else row.get("oom"),
                status=row.get("status", "measured"),
            )
        )
    markdown = [
        "# Gated Global4D V2 batch capacity benchmark",
        "",
        f"Status: **{payload['status']}**",
        "",
        f"Environment: `{json.dumps(payload['environment'], sort_keys=True)}`",
        "",
        "Every measured condition executes forward, loss, backward, optimizer.step, and zero_grad.",
        "",
        "| batch | accum | composition | records | atoms | edges | joints | records/s | peak allocated MiB | peak reserved MiB | finite | OOM | status |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        *rows,
        "",
    ]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gated_global4d_v2_pilot.yaml")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch_sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    parser.add_argument("--compositions", default=",".join(COMPOSITIONS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup_optimizer_steps", type=int, default=3)
    parser.add_argument("--fixed_records_seen", type=int, default=768)
    parser.add_argument("--pool_records", type=int, default=768)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("reports/GATED_GLOBAL4D_BATCH_BENCHMARK.json"),
    )
    parser.add_argument(
        "--output_markdown",
        type=Path,
        default=Path("reports/GATED_GLOBAL4D_BATCH_BENCHMARK.md"),
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--accumulate_grad_batches", type=int)
    parser.add_argument("--composition")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            result = _worker(args)
        except torch.cuda.OutOfMemoryError as exc:
            result = {
                "batch_size": args.batch_size,
                "accumulate_grad_batches": args.accumulate_grad_batches,
                "composition": args.composition,
                "repeat": args.repeat,
                "oom": True,
                "loss_finite": False,
                "status": "oom",
                "error": repr(exc),
                "peak_allocated_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_gpu_memory_bytes": torch.cuda.max_memory_reserved(),
                "peak_allocated_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_gpu_memory_mib": torch.cuda.max_memory_reserved() / 2**20,
                "gpu_utilization_error": "measurement interrupted by OOM",
                **_environment(),
            }
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            result = {
                "batch_size": args.batch_size,
                "accumulate_grad_batches": args.accumulate_grad_batches,
                "composition": args.composition,
                "repeat": args.repeat,
                "oom": True,
                "loss_finite": False,
                "status": "oom",
                "error": repr(exc),
                "peak_allocated_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_gpu_memory_bytes": torch.cuda.max_memory_reserved(),
                "peak_allocated_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_gpu_memory_mib": torch.cuda.max_memory_reserved() / 2**20,
                "gpu_utilization_error": "measurement interrupted by OOM",
                **_environment(),
            }
        if args.result is None:
            raise ValueError("worker requires --result")
        args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    for destination in (args.output_json, args.output_markdown):
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing benchmark report: {destination}"
            )
    environment = _environment()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value]
    compositions = [value for value in args.compositions.split(",") if value]
    results: list[dict[str, Any]] = []
    if not torch.cuda.is_available():
        for batch_size in batch_sizes:
            accumulate = 2 if batch_size == 4 else 1
            for composition in compositions:
                results.append(
                    {
                        "batch_size": batch_size,
                        "accumulate_grad_batches": accumulate,
                        "effective_batch_size": batch_size * accumulate,
                        "composition": composition,
                        "oom": None,
                        "loss_finite": None,
                        "status": "skipped_cuda_unavailable",
                    }
                )
        payload = {
            "status": "SKIPPED_CUDA_UNAVAILABLE",
            "environment": environment,
            "fixed_records_seen": args.fixed_records_seen,
            "results": results,
        }
        _write_reports(payload, args.output_json, args.output_markdown)
        print(json.dumps(payload, indent=2))
        return

    work = args.output_json.parent / ".gated_global4d_batch_benchmark_work"
    work.mkdir(parents=True, exist_ok=True)
    for batch_size in batch_sizes:
        accumulate = 2 if batch_size == 4 else 1
        for composition in compositions:
            for repeat in range(args.repeats):
                result_path = work / f"bs{batch_size}_{composition}_r{repeat}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--config", args.config,
                    "--cache_dir", args.cache_dir,
                    "--split", args.split,
                    "--batch_size", str(batch_size),
                    "--accumulate_grad_batches", str(accumulate),
                    "--composition", composition,
                    "--repeat", str(repeat),
                    "--seed", str(args.seed + repeat),
                    "--warmup_optimizer_steps", str(args.warmup_optimizer_steps),
                    "--fixed_records_seen", str(args.fixed_records_seen),
                    "--pool_records", str(max(args.pool_records, batch_size * 3)),
                    "--result", str(result_path),
                ]
                completed = subprocess.run(command, check=False)
                if not result_path.is_file():
                    results.append(
                        {
                            "batch_size": batch_size,
                            "accumulate_grad_batches": accumulate,
                            "composition": composition,
                            "repeat": repeat,
                            "oom": False,
                            "loss_finite": False,
                            "status": "worker_failed",
                            "returncode": completed.returncode,
                        }
                    )
                else:
                    results.append(json.loads(result_path.read_text(encoding="utf-8")))
    status = "COMPLETED" if all(row.get("status", "measured") != "worker_failed" for row in results) else "FAILED"
    payload = {
        "status": status,
        "environment": environment,
        "fixed_records_seen": args.fixed_records_seen,
        "results": results,
    }
    _write_reports(payload, args.output_json, args.output_markdown)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
