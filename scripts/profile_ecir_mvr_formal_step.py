#!/usr/bin/env python
"""Read-only step profiler for the formal-large D1-B training path."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import psutil  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.loader.dataloader import Collater  # noqa: E402

from etflow.commons.global_coupled_4d_sampling import atomic_json_save  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_dataset import (  # noqa: E402
    RuntimeCacheStatistics,
    runtime_statistics_identity,
)
from scripts.preflight_ecir_mvr_formal_large import (  # noqa: E402
    GpuSampler,
    external_memory_changed,
    percentile,
    query_compute_processes,
    query_gpu,
    resolve_gpu_selection,
    should_block_shared_gpu,
)
from scripts.train_ecir_mvr_medium_rescue_v2 import (  # noqa: E402
    _backward_loss,
    _build_training_components,
    _formal_asset_identities,
    _forward_loss,
    _learning_rate_at_step,
)
from scripts.train_ecir_mvr_run_a import _dataset, _seed  # noqa: E402


PROFILE_SCHEMA = "ecir-mvr-formal-step-profile-v1"
STATUS_COMPLETE = "D1B_FORMAL_PROFILE_COMPLETE"
STATUS_FAILED = "D1B_FORMAL_PROFILE_FAILED"
STATUS_BLOCKED = "D1B_FORMAL_PROFILE_BLOCKED_SHARED_GPU"
DEFAULT_OUTPUT_DIR = ROOT / "reports/ecir_mvr/formal_step_profile"
BASE_CONFIG = ROOT / "configs/ecir_mvr_formal_large_d1b_base.yaml"
SCIENTIFIC_TRAINING_FIELDS = (
    "learning_rate",
    "base_learning_rate",
    "lr_schedule",
    "warmup_steps",
    "warmup_start_lr",
    "peak_lr",
    "final_lr",
    "weight_decay",
    "gradient_clip_norm",
    "teacher_steps",
)
TIMING_FIELDS = (
    "dataloader_wait_seconds",
    "h2d_seconds",
    "model_forward_seconds",
    "loss_seconds",
    "backward_seconds",
    "optimizer_seconds",
    "scheduler_seconds",
    "total_optimizer_step_seconds",
)
RUNTIME_VARIANTS = (
    {
        "name": "baseline",
        "formal_adapter_lru_size": 0,
        "precompute_training_topology": False,
    },
    {
        "name": "optimized",
        "formal_adapter_lru_size": 512,
        "precompute_training_topology": True,
    },
)


def _parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise ValueError("integer list must contain nonnegative values")
    return result


def _parse_bool_list(value: str) -> list[bool]:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    values = []
    for item in value.split(","):
        key = item.strip().lower()
        if key:
            if key not in mapping:
                raise ValueError("boolean lists accept true,false,1,0")
            values.append(mapping[key])
    if not values:
        raise ValueError("boolean list is empty")
    return values


def profile_matrix(
    workers: Iterable[int],
    prefetch_factors: Iterable[int],
    persistent_workers: Iterable[bool],
    pin_memory: Iterable[bool],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen = set()
    for worker_count in workers:
        worker_count = int(worker_count)
        if worker_count < 0:
            raise ValueError("num_workers must be nonnegative")
        for pinned in pin_memory:
            if worker_count == 0:
                row = (0, None, False, bool(pinned))
                if row not in seen:
                    seen.add(row)
                    result.append(
                        {
                            "num_workers": 0,
                            "prefetch_factor": None,
                            "persistent_workers": False,
                            "pin_memory": bool(pinned),
                        }
                    )
                continue
            for prefetch in prefetch_factors:
                if int(prefetch) <= 0:
                    raise ValueError("prefetch_factor must be positive")
                for persistent in persistent_workers:
                    row = (
                        worker_count,
                        int(prefetch),
                        bool(persistent),
                        bool(pinned),
                    )
                    if row in seen:
                        continue
                    seen.add(row)
                    result.append(
                        {
                            "num_workers": worker_count,
                            "prefetch_factor": int(prefetch),
                            "persistent_workers": bool(persistent),
                            "pin_memory": bool(pinned),
                        }
                    )
    return result


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_directory(output_dir: Path, config: Mapping[str, Any]) -> Path:
    resolved = output_dir.resolve()
    protected = []
    for key in ("root", "train_sources", "val_sources", "train_targets", "val_targets"):
        value = config["data"].get(key)
        if value:
            candidate = Path(value).expanduser()
            protected.append(
                candidate.resolve() if candidate.suffix == "" else candidate.resolve().parent
            )
    if any(
        _is_relative_to(resolved, parent) or _is_relative_to(parent, resolved)
        for parent in protected
    ):
        raise ValueError("profile output must be outside formal source/target assets")
    return resolved


def validate_profile_config(config: Mapping[str, Any]) -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    if (
        config.get("experiment_name") != "ecir_mvr_formal_large_d1b_seed42"
        or int(config.get("seed", -1)) != 42
        or config.get("stage_d_method") != "explicit_bond"
        or config.get("model") != base.get("model")
        or config.get("loss") != base.get("loss")
    ):
        raise ValueError("formal D1-B scientific model or loss configuration changed")
    for key in SCIENTIFIC_TRAINING_FIELDS:
        if config["training"].get(key) != base["training"].get(key):
            raise ValueError(f"formal D1-B training semantic changed: {key}")
    data_keys = " ".join(str(key).lower() for key in config["data"])
    data_values = " ".join(
        str(value).lower().replace("\\", "/")
        for value in config["data"].values()
    )
    if "test" in data_keys or "test.parquet" in data_values or "/test/" in data_values:
        raise ValueError("profile configuration may not name a test asset")


def _assert_ready(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_path = Path(config["data"]["target_validation"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("decision") != "D1B_FORMAL_TARGETS_READY"
        or int(audit.get("test_records_read", -1)) != 0
        or not all(audit.get("criteria", {}).values())
    ):
        raise RuntimeError("formal targets are not a test-free READY")
    identities = _formal_asset_identities(dict(config))
    frozen = config.get("frozen_identities")
    if frozen and frozen != identities:
        raise RuntimeError("formal frozen identities differ from immutable assets")
    return audit, identities


def _next_batch(iterator, loader, dataset, epoch: int):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        dataset.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


class _ForwardTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.started = 0.0
        self.elapsed = 0.0

    def before(self, _module, _args) -> None:
        torch.cuda.synchronize(self.device)
        self.started = time.perf_counter()

    def after(self, _module, _args, _output) -> None:
        torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self.started


def _probe_data_pipeline(
    config: dict, micro_batch: int, probe_records: int
) -> dict[str, Any]:
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    dataset = _dataset(config, "train", validity)
    count = min(max(int(probe_records), int(micro_batch)), len(dataset))
    item_rows = []
    items = []
    for index in range(count):
        started = time.perf_counter()
        items.append(dataset[index])
        item_rows.append(time.perf_counter() - started)
    if not items:
        raise RuntimeError("data pipeline probe produced no train records")
    collate_items = items[: int(micro_batch)]
    if len(collate_items) != int(micro_batch):
        raise RuntimeError("dataset is smaller than the requested profile micro batch")
    collater = Collater(dataset, follow_batch=None, exclude_keys=None)
    started = time.perf_counter()
    batch = collater(collate_items)
    collate_seconds = time.perf_counter() - started
    result = {
        "dataset_item_probe_records": count,
        "dataset_item_mean_seconds": statistics.fmean(item_rows),
        "dataset_item_median_seconds": statistics.median(item_rows),
        "dataset_item_p95_seconds": percentile(item_rows, 0.95),
        "official_collate_batch_size": int(micro_batch),
        "official_collate_seconds": collate_seconds,
        "official_collate_separate_probe": True,
        "probe_batch_atoms": int(batch.num_nodes),
        "probe_batch_directed_edges": int(batch.edge_index.size(1)),
    }
    del batch, collate_items, items, dataset, validity
    gc.collect()
    return result


def summarize_steps(
    rows: list[Mapping[str, float]], *, warmup_steps: int, micro_batch: int
) -> dict[str, Any]:
    measured = rows[int(warmup_steps) :]
    if not measured:
        raise ValueError("warmup leaves no measured profile steps")
    result: dict[str, Any] = {"measured_optimizer_steps": len(measured)}
    for field in TIMING_FIELDS:
        values = [float(row[field]) for row in measured]
        result[f"{field}_mean"] = statistics.fmean(values)
        result[f"{field}_median"] = statistics.median(values)
        result[f"{field}_p95"] = percentile(values, 0.95)
    total = result["total_optimizer_step_seconds_mean"]
    result.update(
        {
            "records_per_second": float(micro_batch) / float(total),
            "loss_start": float(measured[0]["loss"]),
            "loss_end": float(measured[-1]["loss"]),
            "cpu_rss_peak_mib": max(float(row["cpu_rss_mib"]) for row in measured),
        }
    )
    return result


def summarize_cache_statistics(
    worker_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    workers = sorted(
        (dict(row) for row in worker_rows),
        key=lambda row: (int(row["worker_id"]), int(row["pid"])),
    )
    hits = sum(int(row["cache_hits"]) for row in workers)
    misses = sum(int(row["cache_misses"]) for row in workers)
    return {
        "schema_version": "ecir-mvr-runtime-cache-statistics-v1",
        "feature_version": "formal-static-runtime-cache-v1",
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate": hits / max(hits + misses, 1),
        "rdkit_adapter_build_count": sum(
            int(row["rdkit_adapter_build_count"]) for row in workers
        ),
        "topology_build_count": sum(
            int(row["topology_build_count"]) for row in workers
        ),
        "workers": workers,
    }


def comparison_result(
    baseline: Mapping[str, Any], optimized: Mapping[str, Any]
) -> dict[str, Any]:
    comparable = (
        baseline.get("status") == "PASS" and optimized.get("status") == "PASS"
    )
    return {
        "status": "PASS" if comparable else "INCOMPLETE",
        "records_per_second_speedup_ratio": (
            float(optimized["records_per_second"])
            / float(baseline["records_per_second"])
            if comparable
            else None
        ),
        "baseline_records_per_second": baseline.get("records_per_second"),
        "optimized_records_per_second": optimized.get("records_per_second"),
    }


def run_profile_setting(
    config: dict,
    setting: Mapping[str, Any],
    *,
    micro_batch: int,
    device: torch.device,
    physical_gpu_index: int,
    warmup_steps: int,
    measured_steps: int,
    item_probe_records: int,
    runtime_statistics=None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _seed(int(config["seed"]))
    phase_config = json.loads(json.dumps(config))
    phase_config["training"]["batch_size"] = int(micro_batch)
    phase_config["training"]["gradient_accumulation_steps"] = 1
    probe = _probe_data_pipeline(phase_config, micro_batch, item_probe_records)
    validity = ChemicalValidity(phase_config["data"]["validity_statistics"])
    dataset = _dataset(
        phase_config,
        "train",
        validity,
        runtime_statistics=runtime_statistics,
    )
    loader_kwargs = {
        "num_workers": int(setting["num_workers"]),
        "pin_memory": bool(setting["pin_memory"]),
    }
    if int(setting["num_workers"]) > 0:
        loader_kwargs.update(
            {
                "prefetch_factor": int(setting["prefetch_factor"]),
                "persistent_workers": bool(setting["persistent_workers"]),
            }
        )
    loader = DataLoader(
        dataset,
        batch_size=int(micro_batch),
        shuffle=False,
        **loader_kwargs,
    )
    loader_started = time.perf_counter()
    iterator = iter(loader)
    loader_initialization_seconds = time.perf_counter() - loader_started
    model, loss_fn, optimizer = _build_training_components(phase_config, device)
    model.train()
    forward_timer = _ForwardTimer(device)
    before_handle = model.register_forward_pre_hook(forward_timer.before)
    after_handle = model.register_forward_hook(forward_timer.after)
    sampler = GpuSampler(physical_gpu_index, interval_seconds=0.2)
    process = psutil.Process()
    epoch = 0
    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    sampler.start()
    try:
        for step in range(1, int(warmup_steps) + int(measured_steps) + 1):
            torch.cuda.synchronize(device)
            step_started = time.perf_counter()

            scheduler_started = time.perf_counter()
            learning_rate = _learning_rate_at_step(phase_config["training"], step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            scheduler_seconds = time.perf_counter() - scheduler_started

            load_started = time.perf_counter()
            batch, iterator, epoch = _next_batch(iterator, loader, dataset, epoch)
            dataloader_wait_seconds = time.perf_counter() - load_started

            transfer_started = time.perf_counter()
            batch = batch.to(
                device, non_blocking=bool(setting["pin_memory"])
            )
            torch.cuda.synchronize(device)
            h2d_seconds = time.perf_counter() - transfer_started

            optimizer.zero_grad(set_to_none=True)
            forward_timer.elapsed = 0.0
            forward_loss_started = time.perf_counter()
            losses = _forward_loss(model, loss_fn, batch)
            torch.cuda.synchronize(device)
            forward_loss_seconds = time.perf_counter() - forward_loss_started
            model_forward_seconds = forward_timer.elapsed
            loss_seconds = max(0.0, forward_loss_seconds - model_forward_seconds)

            backward_started = time.perf_counter()
            _backward_loss(losses, accumulation_steps=1)
            torch.cuda.synchronize(device)
            backward_seconds = time.perf_counter() - backward_started

            optimizer_started = time.perf_counter()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(phase_config["training"]["gradient_clip_norm"]),
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("non-finite profile gradient")
            optimizer.step()
            torch.cuda.synchronize(device)
            optimizer_seconds = time.perf_counter() - optimizer_started
            total_seconds = time.perf_counter() - step_started
            rows.append(
                {
                    "step": step,
                    "warmup": step <= int(warmup_steps),
                    "loss": float(losses["loss"].detach()),
                    "learning_rate": learning_rate,
                    "dataloader_wait_seconds": dataloader_wait_seconds,
                    "h2d_seconds": h2d_seconds,
                    "model_forward_seconds": model_forward_seconds,
                    "loss_seconds": loss_seconds,
                    "backward_seconds": backward_seconds,
                    "optimizer_seconds": optimizer_seconds,
                    "scheduler_seconds": scheduler_seconds,
                    "total_optimizer_step_seconds": total_seconds,
                    "cpu_rss_mib": process.memory_info().rss / 2**20,
                }
            )
        summary = summarize_steps(
            rows, warmup_steps=warmup_steps, micro_batch=micro_batch
        )
        gpu_samples = sampler.samples
        gpu_utilization = [
            float(row["gpu_utilization_percent"]) for row in gpu_samples
            if math.isfinite(float(row["gpu_utilization_percent"]))
        ]
        summary.update(
            {
                **dict(setting),
                **probe,
                "status": "PASS",
                "micro_batch_size": int(micro_batch),
                "warmup_optimizer_steps": int(warmup_steps),
                "loader_initialization_seconds": loader_initialization_seconds,
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                / 2**20,
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(device)
                / 2**20,
                "gpu_utilization_mean": (
                    statistics.fmean(gpu_utilization) if gpu_utilization else None
                ),
                "gpu_utilization_p50": percentile(gpu_utilization, 0.50),
                "gpu_utilization_p95": percentile(gpu_utilization, 0.95),
                "gpu_samples": len(gpu_samples),
                "nan_or_inf": False,
            }
        )
        return summary, rows
    finally:
        sampler.stop()
        before_handle.remove()
        after_handle.remove()
        del iterator, loader, dataset, validity, optimizer, loss_fn, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def benchmark_setting(*args, **kwargs) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    setting = args[1] if len(args) > 1 else kwargs["setting"]
    try:
        return run_profile_setting(*args, **kwargs)
    except torch.cuda.OutOfMemoryError as error:
        status = "OOM"
        message = str(error)
    except FloatingPointError as error:
        status = "NaN"
        message = str(error)
    except RuntimeError as error:
        status = "OOM" if "out of memory" in str(error).lower() else "ERROR"
        message = str(error)
    except Exception as error:
        status = "ERROR"
        message = f"{type(error).__name__}: {error}"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {**dict(setting), "status": status, "error": message}, []


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = (
        "setting_index",
        "variant",
        "step",
        "warmup",
        "loss",
        "learning_rate",
        *TIMING_FIELDS,
        "cpu_rss_mib",
    )
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "reports/ecir_mvr/D1B_FORMAL_RECOMMENDED_CONFIG.yaml",
    )
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=30)
    parser.add_argument("--item-probe-records", type=int, default=32)
    parser.add_argument("--num-workers", default="0,2,4,8,12")
    parser.add_argument("--prefetch-factors", default="2")
    parser.add_argument("--persistent-workers", default="true")
    parser.add_argument("--pin-memory", default="true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.micro_batch <= 0 or args.warmup_steps != 5 or args.profile_steps != 30:
        raise ValueError("formal step profile is fixed at 5 warmup + 30 measured steps")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_profile_config(config)
    audit, identities = _assert_ready(config)
    output_dir = validate_output_directory(args.output_dir, config)
    settings = profile_matrix(
        _parse_int_list(args.num_workers),
        _parse_int_list(args.prefetch_factors),
        _parse_bool_list(args.persistent_workers),
        _parse_bool_list(args.pin_memory),
    )
    selection = resolve_gpu_selection(
        args.gpu_index, os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    physical_gpu_index = selection["physical_gpu_index"]
    logical_cuda_index = selection["logical_cuda_index"]
    gpu_before = query_gpu(physical_gpu_index)
    external_before = query_compute_processes(str(gpu_before["gpu_uuid"]))
    report: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA,
        "status": STATUS_FAILED,
        "created_at_unix": time.time(),
        "config_path": str(args.config.resolve()),
        "commit_sha": _git_commit(),
        "formal_target_identity_sha256": identities[
            "formal_target_identity_sha256"
        ],
        "formal_asset_identities": identities,
        "target_validation_decision": audit["decision"],
        "test_records_read": 0,
        "formal_training_started": False,
        "formal_checkpoint_created": False,
        "formal_target_modified": False,
        "micro_batch_size": int(args.micro_batch),
        "warmup_optimizer_steps": int(args.warmup_steps),
        "measured_optimizer_steps": int(args.profile_steps),
        "runtime_variants": list(RUNTIME_VARIANTS),
        "comparisons": [],
        "settings": settings,
        "results": [],
        "gpu_before": gpu_before,
        "external_processes_at_start": external_before,
        "shared_gpu": bool(external_before),
        "allow_shared_gpu": bool(args.allow_shared_gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_index": physical_gpu_index,
        "logical_cuda_index": logical_cuda_index,
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "profile.json"
    if should_block_shared_gpu(
        external_before, allow_shared_gpu=args.allow_shared_gpu
    ):
        report["status"] = STATUS_BLOCKED
        atomic_json_save(report, report_path)
        print(STATUS_BLOCKED)
        raise SystemExit(2)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal step profiling")
    device = torch.device(f"cuda:{logical_cuda_index}")
    all_rows = []
    for index, setting in enumerate(settings):
        variant_results = {}
        for variant in RUNTIME_VARIANTS:
            variant_config = json.loads(json.dumps(config))
            variant_config.setdefault("data", {})[
                "runtime_optimizations"
            ] = {
                "formal_adapter_lru_size": int(
                    variant["formal_adapter_lru_size"]
                ),
                "precompute_training_topology": bool(
                    variant["precompute_training_topology"]
                ),
            }
            worker_count = max(1, int(setting["num_workers"]))
            shared_statistics = RuntimeCacheStatistics(
                worker_count,
                runtime_statistics_identity(
                    int(variant["formal_adapter_lru_size"]),
                    bool(variant["precompute_training_topology"]),
                ),
            )
            result, rows = benchmark_setting(
                variant_config,
                setting,
                micro_batch=args.micro_batch,
                device=device,
                physical_gpu_index=physical_gpu_index,
                warmup_steps=args.warmup_steps,
                measured_steps=args.profile_steps,
                item_probe_records=args.item_probe_records,
                runtime_statistics=shared_statistics,
            )
            result.update(
                {
                    "setting_index": index,
                    "variant": variant["name"],
                    "runtime_optimizations": {
                        key: value
                        for key, value in variant.items()
                        if key != "name"
                    },
                    "cache_statistics": summarize_cache_statistics(
                        shared_statistics.snapshot()
                    ),
                }
            )
            variant_results[str(variant["name"])] = result
            report["results"].append(result)
            all_rows.extend(
                {
                    **row,
                    "setting_index": index,
                    "variant": variant["name"],
                }
                for row in rows
            )
        comparison = comparison_result(
            variant_results["baseline"], variant_results["optimized"]
        )
        comparison.update({"setting_index": index, **dict(setting)})
        report["comparisons"].append(comparison)
    gpu_after = query_gpu(physical_gpu_index)
    external_after = query_compute_processes(str(gpu_before["gpu_uuid"]))
    report.update(
        {
            "status": (
                STATUS_COMPLETE
                if any(
                    row.get("status") == "PASS"
                    for row in report["comparisons"]
                )
                else STATUS_FAILED
            ),
            "gpu_after": gpu_after,
            "external_processes_at_end": external_after,
            "external_memory_changed": external_memory_changed(
                external_before, external_after
            ),
        }
    )
    _atomic_csv(output_dir / "profile_steps.csv", all_rows)
    atomic_json_save(report, report_path)
    print(report["status"])
    if report["status"] != STATUS_COMPLETE:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
