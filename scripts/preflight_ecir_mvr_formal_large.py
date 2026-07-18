#!/usr/bin/env python
"""Benchmark safe formal-large D1-B micro batches without starting training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from etflow.commons.global_coupled_4d_sampling import atomic_json_save  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.formal_runtime_readiness import (  # noqa: E402
    RUNTIME_REPORT,
    assert_runtime_ready_for_base,
    file_sha256 as runtime_file_sha256,
)
from scripts.train_ecir_mvr_medium_rescue_v2 import (  # noqa: E402
    _backward_loss,
    _build_training_components,
    _formal_asset_identities,
    _forward_loss,
    _learning_rate_at_step,
    _loader_settings,
)
from scripts.train_ecir_mvr_run_a import _dataset, _seed  # noqa: E402


STATUS_PASS = "D1B_FORMAL_PREFLIGHT_PASS"
STATUS_FAILED = "D1B_FORMAL_PREFLIGHT_FAILED"
STATUS_BLOCKED = "D1B_FORMAL_PREFLIGHT_BLOCKED_SHARED_GPU"
STATUS_CAPACITY_PASS = "D1B_FORMAL_CAPACITY_PASS"
STATUS_CAPACITY_FAILED = "D1B_FORMAL_CAPACITY_FAILED"
REPORT_SCHEMA = "ecir-mvr-formal-large-preflight-v1"
FORMAL64_REPORT_DIR = ROOT / "reports/ecir_mvr/formal64_preflight"
DEFAULT_REPORT_JSON = FORMAL64_REPORT_DIR / "D1B_FORMAL_PREFLIGHT.json"
DEFAULT_REPORT_MD = FORMAL64_REPORT_DIR / "D1B_FORMAL_PREFLIGHT.md"
DEFAULT_RECOMMENDED_CONFIG = (
    ROOT / "reports/ecir_mvr/D1B_FORMAL_RECOMMENDED_CONFIG.yaml"
)
MEASURED_TENSOR_FIELDS = (
    "x_input",
    "x_target",
    "active_mode_mask",
    "affected_atom_mask",
    "deterministic_error_features",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def candidate_pairs(
    target_effective_batch: int,
    candidate_micro_batches: Iterable[int] | None = None,
) -> list[dict[str, int]]:
    target = int(target_effective_batch)
    if target <= 0:
        raise ValueError("target effective batch must be positive")
    candidates = (
        [target // divisor for divisor in (1, 2, 4, 8, 16)]
        if candidate_micro_batches is None
        else [int(value) for value in candidate_micro_batches]
    )
    result = []
    for micro in sorted(set(candidates), reverse=True):
        if micro <= 0 or micro > target or target % micro:
            continue
        accumulation = target // micro
        result.append(
            {
                "micro_batch_size": micro,
                "gradient_accumulation_steps": accumulation,
                "effective_batch_size": micro * accumulation,
            }
        )
    if not result:
        raise ValueError("no micro batch exactly divides target effective batch")
    return result


def resolve_gpu_selection(
    requested_gpu_index: int, visible_devices: str | None
) -> dict[str, int]:
    requested = int(requested_gpu_index)
    if requested < 0:
        raise ValueError("GPU index must be nonnegative")
    if not visible_devices:
        return {"physical_gpu_index": requested, "logical_cuda_index": 0}
    entries = [value.strip() for value in visible_devices.split(",") if value.strip()]
    if not entries or any(not value.isdigit() for value in entries):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain numeric GPU indices for preflight"
        )
    physical = [int(value) for value in entries]
    if requested in physical:
        return {
            "physical_gpu_index": requested,
            "logical_cuda_index": physical.index(requested),
        }
    if requested < len(physical):
        return {
            "physical_gpu_index": physical[requested],
            "logical_cuda_index": requested,
        }
    raise ValueError(
        f"GPU {requested} is not available through CUDA_VISIBLE_DEVICES={visible_devices}"
    )


def output_paths(
    *,
    capacity_only: bool,
    target_effective_batch: int,
    report_json: Path | None,
    report_md: Path | None,
    recommended_config: Path | None,
) -> dict[str, Path]:
    if capacity_only:
        if report_json is not None or report_md is not None or recommended_config is not None:
            raise ValueError("capacity-only output paths are fixed and cannot be overridden")
        root = ROOT / (
            f"reports/ecir_mvr/capacity_effective{int(target_effective_batch)}"
        )
        return {
            "report_json": report_json or root / "D1B_FORMAL_CAPACITY.json",
            "report_md": report_md or root / "D1B_FORMAL_CAPACITY.md",
            "recommended_config": recommended_config
            or DEFAULT_RECOMMENDED_CONFIG,
        }
    if int(target_effective_batch) == 64:
        if report_json is not None or report_md is not None or recommended_config is not None:
            raise ValueError("formal64 preflight output paths are fixed")
        return {
            "report_json": DEFAULT_REPORT_JSON,
            "report_md": DEFAULT_REPORT_MD,
            "recommended_config": FORMAL64_REPORT_DIR
            / "D1B_FORMAL_PREFLIGHT_CANDIDATE_CONFIG.yaml",
        }
    root = ROOT / f"reports/ecir_mvr/preflight_effective{int(target_effective_batch)}"
    return {
        "report_json": report_json or root / "D1B_FORMAL_PREFLIGHT.json",
        "report_md": report_md or root / "D1B_FORMAL_PREFLIGHT.md",
        "recommended_config": recommended_config
        or root / "D1B_FORMAL_PREFLIGHT_CANDIDATE_CONFIG.yaml",
    }


def budget_definition(
    target_effective_batch: int, *, capacity_only: bool
) -> dict[str, Any]:
    effective = int(target_effective_batch)
    optimizer_steps = 12_500 if capacity_only else 1_600_000 // effective
    return {
        "budget_type": (
            "expanded_exploratory" if capacity_only else "matched_exposure"
        ),
        "effective_batch_size": effective,
        "optimizer_steps": optimizer_steps,
        "total_sample_exposures": effective * optimizer_steps,
        "scientific_equivalence_warning": (
            "Equal exposures do not imply scientific equivalence when optimizer "
            "update counts differ."
        ),
        "formal_scientific_equivalence": False,
    }


def result_status(*, capacity_only: bool, has_candidate: bool) -> str:
    if capacity_only:
        return STATUS_CAPACITY_PASS if has_candidate else STATUS_CAPACITY_FAILED
    return STATUS_PASS if has_candidate else STATUS_FAILED


def _parse_number(value: str) -> float:
    text = value.strip().replace("MiB", "").strip()
    return float(text) if text not in {"", "N/A", "[N/A]"} else math.nan


def query_gpu(gpu_index: int) -> dict[str, Any]:
    fields = (
        "uuid,name,memory.total,memory.used,memory.free,utilization.gpu,"
        "power.draw,temperature.gpu,driver_version"
    )
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_index),
        ],
        text=True,
        timeout=10,
    ).strip()
    values = [value.strip() for value in output.split(",")]
    if len(values) != 9:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {output}")
    keys = (
        "gpu_uuid",
        "gpu_name",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "gpu_utilization_percent",
        "power_draw_w",
        "temperature_c",
        "driver_version",
    )
    result = dict(zip(keys, values, strict=True))
    for key in keys[2:8]:
        result[key] = _parse_number(str(result[key]))
    result["gpu_index"] = int(gpu_index)
    return result


def query_compute_processes(gpu_uuid: str) -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except subprocess.CalledProcessError:
        return []
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",", 3)]
        if len(values) != 4 or values[3] != gpu_uuid:
            continue
        pid = int(values[0])
        if pid == os.getpid():
            continue
        rows.append(
            {
                "pid": pid,
                "process_name": values[1],
                "used_gpu_memory_mib": _parse_number(values[2]),
                "gpu_uuid": values[3],
            }
        )
    return sorted(rows, key=lambda row: row["pid"])


def external_memory_changed(
    before: Iterable[Mapping[str, Any]], after: Iterable[Mapping[str, Any]]
) -> bool:
    def snapshot(rows: Iterable[Mapping[str, Any]]) -> dict[int, float]:
        return {
            int(row["pid"]): float(row["used_gpu_memory_mib"])
            for row in rows
        }

    return snapshot(before) != snapshot(after)


def should_block_shared_gpu(
    external_processes: Iterable[Mapping[str, Any]], *, allow_shared_gpu: bool
) -> bool:
    return bool(list(external_processes)) and not bool(allow_shared_gpu)


def required_safety_margin_mib(baseline_free_mib: float) -> float:
    return max(4096.0, 0.10 * float(baseline_free_mib))


def percentile(values: Iterable[float], quantile: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, quantile)) if array.size else None


def summarize_measured_steps(
    rows: list[Mapping[str, float]], *, warmup_steps: int
) -> dict[str, Any]:
    measured = rows[int(warmup_steps) :]
    if not measured:
        raise ValueError("warmup leaves no measured optimizer steps")

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in measured]

    step_times = values("optimizer_step_time_seconds")
    losses = values("loss")
    return {
        "measured_optimizer_steps": len(measured),
        "avg_optimizer_step_time_seconds": statistics.fmean(step_times),
        "median_optimizer_step_time_seconds": statistics.median(step_times),
        "p95_optimizer_step_time_seconds": percentile(step_times, 0.95),
        "avg_dataloader_time_seconds": statistics.fmean(values("dataloader_time_seconds")),
        "avg_forward_time_seconds": statistics.fmean(values("forward_time_seconds")),
        "avg_backward_time_seconds": statistics.fmean(values("backward_time_seconds")),
        "avg_optimizer_time_seconds": statistics.fmean(values("optimizer_time_seconds")),
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "gpu_utilization_mean": statistics.fmean(values("gpu_utilization_percent")),
        "gpu_utilization_p50": percentile(values("gpu_utilization_percent"), 0.50),
        "gpu_utilization_p95": percentile(values("gpu_utilization_percent"), 0.95),
        "gpu_power_mean_w": statistics.fmean(values("power_draw_w")),
        "gpu_power_peak_w": max(values("power_draw_w")),
        "nvidia_smi_peak_memory_used_mib": max(values("memory_used_mib")),
        "cpu_rss_peak_mib": max(values("cpu_rss_mib")),
    }


def _next_batch(iterator, loader, dataset, epoch: int):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        dataset.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class GpuSampler:
    def __init__(self, gpu_index: int, interval_seconds: float = 0.2) -> None:
        self.gpu_index = int(gpu_index)
        self.interval_seconds = float(interval_seconds)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            row = query_gpu(self.gpu_index)
            row["cpu_rss_mib"] = psutil.Process().memory_info().rss / 2**20
            self.samples.append(row)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
            pass

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._sample()
        self._thread = None


def run_training_phase(
    config: dict,
    pair: Mapping[str, int],
    *,
    device: torch.device,
    gpu_index: int,
    optimizer_steps: int,
    warmup_steps: int,
) -> dict[str, Any]:
    _seed(int(config["seed"]))
    phase_config = json.loads(json.dumps(config))
    phase_config["training"]["batch_size"] = int(pair["micro_batch_size"])
    phase_config["training"]["gradient_accumulation_steps"] = int(
        pair["gradient_accumulation_steps"]
    )
    validity = ChemicalValidity(phase_config["data"]["validity_statistics"])
    dataset = _dataset(phase_config, "train", validity)
    loader = DataLoader(
        dataset,
        batch_size=int(pair["micro_batch_size"]),
        shuffle=False,
        **_loader_settings(phase_config),
    )
    iterator = iter(loader)
    model, loss_fn, optimizer = _build_training_components(phase_config, device)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    rows: list[dict[str, float]] = []
    epoch = 0
    process = psutil.Process()
    started = time.perf_counter()
    try:
        for step in range(1, int(optimizer_steps) + 1):
            torch.cuda.synchronize(device)
            step_started = time.perf_counter()
            dataloader_seconds = 0.0
            forward_seconds = 0.0
            backward_seconds = 0.0
            losses_for_step = []
            optimizer.zero_grad(set_to_none=True)
            learning_rate = _learning_rate_at_step(phase_config["training"], step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            for _ in range(int(pair["gradient_accumulation_steps"])):
                load_started = time.perf_counter()
                batch, iterator, epoch = _next_batch(
                    iterator, loader, dataset, epoch
                )
                batch = batch.to(
                    device,
                    non_blocking=bool(
                        phase_config["training"].get("pin_memory", True)
                    ),
                )
                torch.cuda.synchronize(device)
                dataloader_seconds += time.perf_counter() - load_started

                forward_started = time.perf_counter()
                losses = _forward_loss(model, loss_fn, batch)
                torch.cuda.synchronize(device)
                forward_seconds += time.perf_counter() - forward_started

                backward_started = time.perf_counter()
                _backward_loss(
                    losses,
                    accumulation_steps=int(pair["gradient_accumulation_steps"]),
                )
                torch.cuda.synchronize(device)
                backward_seconds += time.perf_counter() - backward_started
                losses_for_step.append(float(losses["loss"].detach()))

            optimizer_started = time.perf_counter()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(phase_config["training"]["gradient_clip_norm"]),
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("non-finite formal training gradient")
            optimizer.step()
            torch.cuda.synchronize(device)
            optimizer_seconds = time.perf_counter() - optimizer_started
            step_seconds = time.perf_counter() - step_started
            gpu = query_gpu(gpu_index)
            rows.append(
                {
                    "step": float(step),
                    "optimizer_step_time_seconds": step_seconds,
                    "dataloader_time_seconds": dataloader_seconds,
                    "forward_time_seconds": forward_seconds,
                    "backward_time_seconds": backward_seconds,
                    "optimizer_time_seconds": optimizer_seconds,
                    "loss": statistics.fmean(losses_for_step),
                    "gpu_utilization_percent": float(
                        gpu["gpu_utilization_percent"]
                    ),
                    "power_draw_w": float(gpu["power_draw_w"]),
                    "memory_used_mib": float(gpu["memory_used_mib"]),
                    "cpu_rss_mib": process.memory_info().rss / 2**20,
                }
            )
        summary = summarize_measured_steps(rows, warmup_steps=warmup_steps)
        summary.update(
            {
                "optimizer_steps": int(optimizer_steps),
                "total_sample_exposures": int(
                    pair["effective_batch_size"] * optimizer_steps
                ),
                "records_per_second": float(pair["effective_batch_size"])
                / summary["avg_optimizer_step_time_seconds"],
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                / 2**20,
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(device)
                / 2**20,
                "total_elapsed_seconds": time.perf_counter() - started,
                "nan_or_inf": False,
            }
        )
        return summary
    finally:
        del iterator, loader, dataset, optimizer, loss_fn, model
        _cleanup_cuda()


def benchmark_candidate(
    config: dict,
    pair: Mapping[str, int],
    *,
    device: torch.device,
    gpu_index: int,
    preflight_steps: int,
    warmup_steps: int,
    phase_runner: Callable[..., dict[str, Any]] = run_training_phase,
) -> dict[str, Any]:
    baseline_gpu = query_gpu(gpu_index)
    external_before = query_compute_processes(str(baseline_gpu["gpu_uuid"]))
    result = {
        **dict(pair),
        "status": "ERROR",
        "external_gpu_baseline_used_mib": baseline_gpu["memory_used_mib"],
        "external_gpu_baseline_free_mib": baseline_gpu["memory_free_mib"],
        "required_safety_margin_mib": required_safety_margin_mib(
            float(baseline_gpu["memory_free_mib"])
        ),
        "optimizer_steps": int(preflight_steps),
        "total_sample_exposures": int(
            pair["effective_batch_size"] * preflight_steps
        ),
    }
    sampler = GpuSampler(gpu_index)
    sampler.start()
    try:
        phase_runner(
            config,
            pair,
            device=device,
            gpu_index=gpu_index,
            optimizer_steps=2,
            warmup_steps=0,
        )
        measured = phase_runner(
            config,
            pair,
            device=device,
            gpu_index=gpu_index,
            optimizer_steps=preflight_steps,
            warmup_steps=warmup_steps,
        )
        result.update(measured)
        result["status"] = "PASS"
    except torch.cuda.OutOfMemoryError as error:
        result.update({"status": "OOM", "error": str(error), "nan_or_inf": False})
    except FloatingPointError as error:
        result.update({"status": "NaN", "error": str(error), "nan_or_inf": True})
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            result.update(
                {"status": "OOM", "error": str(error), "nan_or_inf": False}
            )
        else:
            result.update(
                {"status": "ERROR", "error": str(error), "nan_or_inf": False}
            )
    except Exception as error:
        result.update(
            {
                "status": "ERROR",
                "error": f"{type(error).__name__}: {error}",
                "nan_or_inf": False,
            }
        )
    finally:
        sampler.stop()
        if device.type == "cuda" and torch.cuda.is_available():
            result.setdefault(
                "torch_peak_allocated_mib",
                torch.cuda.max_memory_allocated(device) / 2**20,
            )
            result.setdefault(
                "torch_peak_reserved_mib",
                torch.cuda.max_memory_reserved(device) / 2**20,
            )
        _cleanup_cuda()
    end_gpu = query_gpu(gpu_index)
    external_after = query_compute_processes(str(end_gpu["gpu_uuid"]))
    result["external_processes_before"] = external_before
    result["external_processes_after"] = external_after
    result["external_memory_changed"] = external_memory_changed(
        external_before, external_after
    )
    sampled_peak = max(
        (
            float(sample["memory_used_mib"])
            for sample in sampler.samples
            if math.isfinite(float(sample["memory_used_mib"]))
        ),
        default=float(end_gpu["memory_used_mib"]),
    )
    peak_used = max(
        sampled_peak,
        float(
            result.get(
                "nvidia_smi_peak_memory_used_mib", end_gpu["memory_used_mib"]
            )
        ),
    )
    result["nvidia_smi_peak_memory_used_mib"] = peak_used
    result.setdefault(
        "cpu_rss_peak_mib",
        max(
            (float(sample["cpu_rss_mib"]) for sample in sampler.samples),
            default=psutil.Process().memory_info().rss / 2**20,
        ),
    )
    remaining = float(baseline_gpu["memory_total_mib"]) - peak_used
    result["remaining_memory_at_peak_mib"] = remaining
    result["memory_safe"] = bool(
        result["status"] == "PASS"
        and remaining >= float(result["required_safety_margin_mib"])
    )
    if result["status"] == "PASS":
        average_step = float(result["avg_optimizer_step_time_seconds"])
        result["estimated_25000_optimizer_steps_seconds"] = average_step * 25_000
        result["estimated_formal_budget_seconds"] = average_step * int(
            config["training"]["optimizer_steps"]
        )
        result["estimated_formal_optimizer_steps_seconds"] = float(
            result["avg_optimizer_step_time_seconds"]
        ) * int(config["training"]["optimizer_steps"])
    return result


def recommend_candidate(candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        dict(candidate)
        for candidate in candidates
        if candidate.get("status") == "PASS"
        and not candidate.get("nan_or_inf", False)
        and candidate.get("memory_safe") is True
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda candidate: (
            float(candidate["records_per_second"]),
            int(candidate["micro_batch_size"]),
        ),
    )


def _environment(gpu: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        **dict(gpu),
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# D1-B Formal-Large Preflight",
        "",
        f"Status: `{report['status']}`",
        "",
        f"- Shared GPU: `{str(report['shared_gpu']).lower()}`",
        f"- GPU index: {report['gpu']['gpu_index']}",
        f"- Baseline used MiB: {report['gpu']['memory_used_mib']}",
        f"- Baseline free MiB: {report['gpu']['memory_free_mib']}",
        f"- Target effective batch: {report['target_effective_batch']}",
        f"- Mode: `{report['mode']}`",
        f"- Budget optimizer steps: {report['formal_budget']['optimizer_steps']}",
        f"- Budget total sample exposures: {report['formal_budget']['total_sample_exposures']}",
        "",
        "Equal sample exposures do not make runs with different optimizer update counts scientifically equivalent.",
        "",
    ]
    for candidate in report.get("candidates", []):
        lines.extend(
            [
                f"## micro={candidate['micro_batch_size']} accum={candidate['gradient_accumulation_steps']}",
                "",
                f"- Status: `{candidate['status']}`",
                f"- Effective batch: {candidate['effective_batch_size']}",
                f"- Memory safe: `{str(candidate.get('memory_safe', False)).lower()}`",
                f"- Records/s: {candidate.get('records_per_second')}",
                f"- Peak allocated MiB: {candidate.get('torch_peak_allocated_mib')}",
                f"- Peak card used MiB: {candidate.get('nvidia_smi_peak_memory_used_mib')}",
                "",
            ]
        )
    if report.get("recommended"):
        recommended = report["recommended"]
        lines.extend(
            [
                "## Recommendation",
                "",
                f"- Micro batch: {recommended['micro_batch_size']}",
                f"- Gradient accumulation: {recommended['gradient_accumulation_steps']}",
                f"- Effective batch: {recommended['effective_batch_size']}",
                f"- Requires stable external memory: `{str(report['recommendation_requires_stable_external_memory']).lower()}`",
                "",
                (
                    "Formal64 finalizer command:"
                    if report.get("formal64_finalizer_required")
                    else "Formal command (manual confirmation required):"
                ),
                "",
                "```bash",
                report.get("formal64_finalizer_command")
                or report["formal_training_command"],
                "```",
            ]
        )
    if report.get("capacity_best_candidate"):
        candidate = report["capacity_best_candidate"]
        lines.extend(
            [
                "## Exploratory Capacity Result",
                "",
                f"- Micro batch: {candidate['micro_batch_size']}",
                f"- Gradient accumulation: {candidate['gradient_accumulation_steps']}",
                f"- Effective batch: {candidate['effective_batch_size']}",
                "- Formal scientific equivalence: `false`",
                "- Recommended config generated: `false`",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_recommended_config(
    config: dict,
    recommendation: Mapping[str, Any],
    identities: Mapping[str, Any],
    path: Path,
    report_json: Path,
) -> None:
    resolved = json.loads(json.dumps(config))
    effective = int(recommendation["effective_batch_size"])
    formal_steps = 1_600_000 // effective
    resolved["training"].update(
        {
            "batch_size": int(recommendation["micro_batch_size"]),
            "gradient_accumulation_steps": int(
                recommendation["gradient_accumulation_steps"]
            ),
            "effective_batch_size": effective,
            "optimizer_steps": formal_steps,
            "total_sample_exposures": effective * formal_steps,
            "checkpoint_steps": [
                formal_steps // 4,
                formal_steps // 2,
                3 * formal_steps // 4,
                formal_steps,
            ],
            "checkpoint_validation_steps": [
                formal_steps // 4,
                formal_steps // 2,
                3 * formal_steps // 4,
                formal_steps,
            ],
        }
    )
    resolved["frozen_identities"] = dict(identities)
    resolved["preflight"] = {
        "report": str(report_json.resolve()),
        "report_sha256": _sha256(report_json),
        "manual_training_confirmation_required": True,
    }
    _atomic_text(path, yaml.safe_dump(resolved, sort_keys=False))


def write_report_artifacts(
    report: Mapping[str, Any],
    *,
    config: dict,
    identities: Mapping[str, Any],
    report_json: Path,
    report_md: Path,
    recommended_config: Path,
    capacity_only: bool,
) -> None:
    atomic_json_save(dict(report), report_json)
    if report.get("recommended") is not None and not capacity_only:
        _write_recommended_config(
            config,
            report["recommended"],
            identities,
            recommended_config,
            report_json,
        )
    _atomic_text(report_md, report_markdown(report))


def _parse_candidates(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _validate_base_config(
    config: Mapping[str, Any],
    target_effective: int,
    *,
    capacity_only: bool = False,
) -> None:
    allowed_effective_batches = {256, 512} if capacity_only else {64, 128}
    if (
        config.get("experiment_name") != "ecir_mvr_formal_large_d1b_seed42"
        or int(config.get("seed", -1)) != 42
        or config.get("stage_d_method") != "explicit_bond"
        or float(config["model"].get("bond_explicit_alpha", -1.0)) != 1.0
        or int(config["training"].get("total_sample_exposures", -1)) != 1_600_000
        or target_effective not in allowed_effective_batches
    ):
        raise ValueError("formal-large D1-B base scientific configuration changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ecir_mvr_formal_large_d1b_base.yaml"),
    )
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--capacity-only", action="store_true")
    parser.add_argument("--target-effective-batch", type=int, default=64)
    parser.add_argument("--candidate-micro-batches")
    parser.add_argument("--preflight-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--recommended-config", type=Path)
    args = parser.parse_args()
    if args.preflight_steps != 100 or args.warmup_steps != 20:
        raise ValueError("formal preflight is frozen at 100 steps with 20 warmup steps")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate_base_config(
        config,
        args.target_effective_batch,
        capacity_only=args.capacity_only,
    )
    paths = output_paths(
        capacity_only=args.capacity_only,
        target_effective_batch=args.target_effective_batch,
        report_json=args.report_json,
        report_md=args.report_md,
        recommended_config=args.recommended_config,
    )
    identities = _formal_asset_identities(config)
    runtime_report = assert_runtime_ready_for_base(
        config, args.config, RUNTIME_REPORT
    )
    audit_path = Path(config["data"]["target_validation"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("decision") != "D1B_FORMAL_TARGETS_READY"
        or int(audit.get("test_records_read", -1)) != 0
        or not all(audit.get("criteria", {}).values())
    ):
        raise RuntimeError("formal targets are not a test-free READY")

    selection = resolve_gpu_selection(
        args.gpu_index, os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    physical_gpu_index = selection["physical_gpu_index"]
    logical_cuda_index = selection["logical_cuda_index"]
    gpu = query_gpu(physical_gpu_index)
    external = query_compute_processes(str(gpu["gpu_uuid"]))
    shared = bool(external)
    base_report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "created_at": _utc_now(),
        "status": (
            STATUS_CAPACITY_FAILED if args.capacity_only else STATUS_BLOCKED
        ),
        "mode": "capacity_only" if args.capacity_only else "formal_preflight",
        "capacity_only": bool(args.capacity_only),
        "shared_gpu": shared,
        "allow_shared_gpu": bool(args.allow_shared_gpu),
        "external_processes_at_start": external,
        "gpu": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_cuda_index": logical_cuda_index,
        "target_effective_batch": int(args.target_effective_batch),
        "candidate_pairs": candidate_pairs(
            args.target_effective_batch,
            _parse_candidates(args.candidate_micro_batches),
        ),
        "candidates": [],
        "commit_sha": _git_commit(),
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "formal_target_identity_sha256": identities[
            "formal_target_identity_sha256"
        ],
        "runtime_validation_report": str(RUNTIME_REPORT.resolve()),
        "runtime_validation_report_sha256": runtime_file_sha256(RUNTIME_REPORT),
        "runtime_validation_identity_sha256": runtime_report[
            "runtime_validation_identity_sha256"
        ],
        "frozen_identities": identities,
        "environment": _environment(gpu),
        "test_records_read": 0,
        "formal_budget": budget_definition(
            args.target_effective_batch, capacity_only=args.capacity_only
        ),
        "formal_training_started": False,
        "formal_checkpoint_created": False,
    }
    if should_block_shared_gpu(external, allow_shared_gpu=args.allow_shared_gpu):
        base_report["blocked_shared_gpu"] = True
        write_report_artifacts(
            base_report,
            config=config,
            identities=identities,
            report_json=paths["report_json"],
            report_md=paths["report_md"],
            recommended_config=paths["recommended_config"],
            capacity_only=args.capacity_only,
        )
        print(base_report["status"])
        raise SystemExit(2)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal preflight")
    device = torch.device(f"cuda:{logical_cuda_index}")
    results = []
    pinned_config = json.loads(json.dumps(config))
    pinned_config["frozen_identities"] = identities
    pinned_config["training"]["effective_batch_size"] = int(
        args.target_effective_batch
    )
    pinned_config["training"]["optimizer_steps"] = int(
        base_report["formal_budget"]["optimizer_steps"]
    )
    pinned_config["training"]["total_sample_exposures"] = int(
        base_report["formal_budget"]["total_sample_exposures"]
    )
    for pair in base_report["candidate_pairs"]:
        results.append(
            benchmark_candidate(
                pinned_config,
                pair,
                device=device,
                gpu_index=physical_gpu_index,
                preflight_steps=args.preflight_steps,
                warmup_steps=args.warmup_steps,
            )
        )
    best_candidate = recommend_candidate(results)
    status = result_status(
        capacity_only=args.capacity_only,
        has_candidate=best_candidate is not None,
    )
    recommendation = None if args.capacity_only else best_candidate
    report = {
        **base_report,
        "status": status,
        "candidates": results,
        "recommended": recommendation,
        "capacity_best_candidate": (
            best_candidate if args.capacity_only else None
        ),
        "recommendation_requires_stable_external_memory": shared,
        "benchmark_external_memory_changed": any(
            bool(result.get("external_memory_changed")) for result in results
        ),
    }
    if recommendation is not None:
        report["recommended_config_path"] = str(
            paths["recommended_config"].resolve()
        )
        if int(args.target_effective_batch) == 64:
            report["formal64_finalizer_required"] = True
            report["formal_training_command"] = None
            report["formal64_finalizer_command"] = (
                "python scripts/finalize_ecir_mvr_formal64_config.py "
                "--base-config configs/ecir_mvr_formal_large_d1b_base.yaml "
                f"--preflight-report {paths['report_json']} "
                "--output reports/ecir_mvr/D1B_FORMAL_RECOMMENDED_CONFIG.yaml"
            )
        else:
            report["formal_training_command"] = (
                f"CUDA_VISIBLE_DEVICES={physical_gpu_index} python "
                "scripts/train_ecir_mvr_medium_rescue_v2.py "
                f"--config {paths['recommended_config']} "
                f"--data_audit {audit_path} --device cuda:0"
            )
    write_report_artifacts(
        report,
        config=pinned_config,
        identities=identities,
        report_json=paths["report_json"],
        report_md=paths["report_md"],
        recommended_config=paths["recommended_config"],
        capacity_only=args.capacity_only,
    )
    print(status)
    successful_status = (
        STATUS_CAPACITY_PASS if args.capacity_only else STATUS_PASS
    )
    if status != successful_status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
