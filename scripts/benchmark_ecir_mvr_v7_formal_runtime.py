#!/usr/bin/env python3
"""Measure two formal batches without taking an optimizer step."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from etflow.ecir.bac_evaluation import attach_canonical_constraints  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_loss import MCVRLoss  # noqa: E402
from etflow.ecir.mvr_model import MCVRModel  # noqa: E402
from etflow.ecir.mvr_v7_formal import (  # noqa: E402
    build_v7_formal_model,
    file_sha256,
    load_v7_formal_config,
)
from scripts.run_ecir_mvr_v7_10k_validation import _build_items  # noqa: E402
from scripts.train_ecir_mvr_run_a import _dataset  # noqa: E402


V7_10K_EVALUATION_SECONDS = 3652.3279999999795
V7_10K_RECORDS = 30_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--v7-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v7_formal/runtime_benchmark.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("docs/MCVR_V7_RUNTIME_ESTIMATE.md"),
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, operation: Any) -> tuple[Any, float]:
    _sync(device)
    started = time.perf_counter()
    result = operation()
    _sync(device)
    return result, time.perf_counter() - started


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _markdown(report: dict[str, Any]) -> str:
    training = report["training_compute"]
    solver = report["v7_inference"]
    estimate = report["estimate"]
    return f"""# MCVR V7 Formal-Large Runtime Estimate

## Decision

`{report['decision']}`

This is a two-batch compute benchmark. It performs forward and backward for the
existing D1-B prior but does not call `optimizer.step()`. V7 remains
inference-only; its analytic Angle solver has no backward pass.

## Frozen inputs

- Device: `{report['device']}`
- GPU: `{report['gpu']}`
- Batches: `{report['batches']}`
- Batch size: `{report['batch_size']}`
- D1-B checkpoint SHA256: `{report['checkpoint_sha256']}`
- Training config SHA256: `{report['training_config_sha256']}`
- V7 wrapper config SHA256: `{report['v7_config_sha256']}`
- Train molecules: `50000`
- Validation molecules: `5000`

## Measurements

| Measurement | Mean seconds/batch |
|---|---:|
| D1-B training forward + loss | {training['forward_loss_mean_seconds']:.6f} |
| D1-B backward | {training['backward_mean_seconds']:.6f} |
| D1-B compute total | {training['total_mean_seconds']:.6f} |
| Frozen prior inference forward | {solver['prior_forward_mean_seconds']:.6f} |
| V7 total inference forward | {solver['v7_forward_mean_seconds']:.6f} |
| V7 solver + fusion overhead | {solver['solver_fusion_mean_seconds']:.6f} |

- Peak training CUDA allocated: `{training['peak_cuda_allocated_mib']:.2f} MiB`
- Peak V7 CUDA allocated: `{solver['peak_cuda_allocated_mib']:.2f} MiB`
- Solver calls: `{solver['angle_solver']['calls']}`
- Solver failures: `{solver['angle_solver']['solver_failure_count']}`

## Estimate

- Compute-only 25K prior estimate: `{estimate['prior_compute_only_hours']:.3f} h`
- V7 10K-record validation estimate: `{estimate['v7_validation_10k_records_hours']:.3f} h`
- Combined compute-only estimate: `{estimate['combined_compute_only_hours']:.3f} h`

The compute-only extrapolation excludes dataloader stalls, validation metrics,
checkpoint serialization, telemetry, and scheduler overhead. It must not be
presented as a wall-clock guarantee. The existing seed43 D1-B formal run is the
stronger operational reference and completed in about `3.66 h` wall time on an
RTX 5080. A conservative `7-10 h` scheduling window remains sufficient, but the
measured local evidence does not require that much active compute.

## Isolation

```text
optimizer_steps_taken=0
checkpoint_created=false
test_records_read=0
test_assets_opened=false
formal_test_run=false
```
"""


def main() -> None:
    args = parse_args()
    if args.batches != 2:
        raise ValueError("V7 formal runtime benchmark is frozen to two batches")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("V7 formal runtime benchmark requires CUDA")
    for name in (
        "training_config",
        "v7_config",
        "checkpoint",
        "output_json",
        "output_report",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    config = yaml.safe_load(args.training_config.read_text(encoding="utf-8"))
    wrapper = load_v7_formal_config(args.v7_config)
    data = config["data"]
    if any("test" in str(key).lower() for key in data):
        raise RuntimeError("V7 formal benchmark configuration names test data")
    if int(data["train_molecules"]) != 50_000 or int(data["val_molecules"]) != 5_000:
        raise RuntimeError("V7 formal benchmark data scale changed")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_sha = file_sha256(args.checkpoint)
    validity = ChemicalValidity(data["validity_statistics"])

    dataset = _dataset(config, "train", validity)
    batch_size = int(config["training"]["batch_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    training_prior = MCVRModel(**config["model"]).to(device)
    incompatible = training_prior.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("V7 formal benchmark prior strict-load failed")
    training_prior.train()
    loss_fn = MCVRLoss(config["loss"])
    forward_times: list[float] = []
    backward_times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    for _ in range(args.batches):
        batch = next(iterator).to(device)
        training_prior.zero_grad(set_to_none=True)
        losses, forward_seconds = _timed(
            device,
            lambda model=training_prior, selected=batch: loss_fn(model, selected),
        )
        _, backward_seconds = _timed(device, lambda: losses["loss"].backward())
        forward_times.append(forward_seconds)
        backward_times.append(backward_seconds)
    training_peak = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    )
    del training_prior, loader, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    source_frame = pd.read_parquet(data["train_sources"]).iloc[
        : batch_size * args.batches
    ]
    target_frame = pd.read_parquet(data["train_targets"])
    target_frame = target_frame[target_frame.sample_id.isin(source_frame.sample_id)]
    items = _build_items(
        source_frame,
        target_frame,
        validity,
        source_cache_root=Path(data.get("source_cache_root", data["root"])),
        target_cache_root=Path(data.get("target_cache_root", Path(data["root"]) / "minimal_targets")),
    )
    attach_canonical_constraints(
        items,
        validity,
        source_identity_sha256=wrapper["formal_identities"][
            "formal_source_identity_sha256"
        ],
    )
    v7 = build_v7_formal_model(checkpoint, wrapper, device=device).eval()
    v7.reset_statistics()
    prior_forward_times: list[float] = []
    v7_forward_times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(items), batch_size):
        batch = Batch.from_data_list(
            [item["data"] for item in items[start : start + batch_size]]
        ).to(device)
        t = batch.x_init.new_full((batch.num_graphs,), 0.5)
        with torch.inference_mode():
            _, prior_seconds = _timed(
                device, lambda: v7.prior(batch, batch.x_init, t)
            )
            _, v7_seconds = _timed(device, lambda: v7(batch, batch.x_init, t))
        prior_forward_times.append(prior_seconds)
        v7_forward_times.append(v7_seconds)
    v7_peak = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    )
    forward_mean = _mean(forward_times)
    backward_mean = _mean(backward_times)
    prior_inference_mean = _mean(prior_forward_times)
    v7_forward_mean = _mean(v7_forward_times)
    solver_mean = max(v7_forward_mean - prior_inference_mean, 0.0)
    prior_hours = 25_000 * (forward_mean + backward_mean) / 3600.0
    v7_validation_hours = (
        V7_10K_EVALUATION_SECONDS / V7_10K_RECORDS * 10_000 / 3600.0
    )
    solver_summary = v7.angle_solver_summary()
    report = {
        "schema_version": "mcvr-v7-formal-runtime-benchmark-v1",
        "decision": "V7_FORMAL_RUNTIME_BENCHMARK_COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "batches": args.batches,
        "batch_size": batch_size,
        "records": batch_size * args.batches,
        "checkpoint_sha256": checkpoint_sha,
        "training_config_sha256": file_sha256(args.training_config),
        "v7_config_sha256": file_sha256(args.v7_config),
        "training_compute": {
            "forward_loss_seconds": forward_times,
            "backward_seconds": backward_times,
            "forward_loss_mean_seconds": forward_mean,
            "backward_mean_seconds": backward_mean,
            "total_mean_seconds": forward_mean + backward_mean,
            "peak_cuda_allocated_mib": training_peak,
        },
        "v7_inference": {
            "prior_forward_seconds": prior_forward_times,
            "v7_forward_seconds": v7_forward_times,
            "prior_forward_mean_seconds": prior_inference_mean,
            "v7_forward_mean_seconds": v7_forward_mean,
            "solver_fusion_mean_seconds": solver_mean,
            "peak_cuda_allocated_mib": v7_peak,
            "angle_solver": solver_summary,
        },
        "estimate": {
            "prior_compute_only_hours": prior_hours,
            "v7_validation_10k_records_hours": v7_validation_hours,
            "combined_compute_only_hours": prior_hours + v7_validation_hours,
            "conservative_scheduling_window_hours": [7.0, 10.0],
        },
        "optimizer_steps_taken": 0,
        "checkpoint_created": False,
        "test_records_read": 0,
        "test_assets_opened": False,
        "formal_test_run": False,
    }
    _write_json(args.output_json, report)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
