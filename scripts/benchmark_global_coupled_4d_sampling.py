#!/usr/bin/env python
"""Numeric and performance regression benchmark for Global Coupled 4D rollout."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.global_coupled_4d_jacobian import build_global_coupled_4d_jacobian
from etflow.commons.global_coupled_4d_projection import project_orthogonal_residual
from etflow.commons.global_coupled_4d_sampling import (
    atomic_json_save,
    configure_cpu_threads,
    resolve_device,
)
from etflow.commons.global_coupled_4d_topology import build_global_coupled_4d_topology
from etflow.data.flexbond_eval_manifest import (
    limit_manifest_molecules,
    load_eval_manifest,
    validate_dataset_against_manifest,
)
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset
from etflow.models.global_coupled_4d_flow import GlobalCoupled4DFlowLightningModule


def _synthetic_batch(seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "x_init": torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.2, 0.0],
            [3.0, 1.0, 0.2],
            [4.0, 1.0, 1.0],
        ]),
        "node_attr": torch.randn(5, 10, generator=generator),
        "edge_index": torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ]),
        "edge_attr": torch.zeros(8, 1),
        "rotatable_bond_index": torch.tensor([[1, 2], [2, 3]]),
        "batch": torch.zeros(5, dtype=torch.long),
    }


def _to_device(batch, device: str):
    if isinstance(batch, dict):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    return batch.to(device)


def _run(model, batches, *, optimized: bool, refinement_steps: int, alpha: float):
    outputs = []
    diagnostics = []
    model.topology_cache.clear()
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.reset_peak_memory_stats(next(model.parameters()).device)
        torch.cuda.synchronize(next(model.parameters()).device)
    started = time.perf_counter()
    for batch in batches:
        refined, detail = model.refine(
            batch,
            refinement_steps=refinement_steps,
            update_scale=alpha,
            max_displacement=0.1,
            save_trajectory_metrics=True,
            profile=True,
            use_rollout_cache=optimized,
            optimized=optimized,
        )
        outputs.append(refined.detach().cpu())
        diagnostics.append(detail)
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.synchronize(next(model.parameters()).device)
        peak = torch.cuda.max_memory_allocated(next(model.parameters()).device)
    else:
        peak = 0
    elapsed = time.perf_counter() - started
    return outputs, diagnostics, elapsed, peak


def _thread_benchmark() -> list[dict]:
    pos = _synthetic_batch(0)["x_init"]
    topology = build_global_coupled_4d_topology(
        pos.size(0),
        _synthetic_batch(0)["edge_index"],
        _synthetic_batch(0)["rotatable_bond_index"],
    )
    jacobian, _ = build_global_coupled_4d_jacobian(pos, topology)
    vector = torch.randn_like(pos)
    original = torch.get_num_threads()
    rows = []
    for threads in dict.fromkeys((1, 2, 4, 8, original)):
        torch.set_num_threads(threads)
        started = time.perf_counter()
        for _ in range(200):
            project_orthogonal_residual(jacobian, vector)
        rows.append({"threads": threads, "seconds_200_projections": time.perf_counter() - started})
    torch.set_num_threads(original)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--refinement_steps", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/global_coupled_4d_sampling_benchmark.json"),
    )
    args = parser.parse_args()
    real_values = (args.checkpoint, args.config, args.cache_dir, args.manifest)
    if any(real_values) and not all(real_values):
        parser.error("checkpoint, config, cache_dir, and manifest must be provided together")
    device = resolve_device(args.device)
    threads = configure_cpu_threads(args.cpu_threads)

    if all(real_values):
        model = GlobalCoupled4DFlowLightningModule.load_from_checkpoint(
            args.checkpoint, map_location=device
        ).to(device).eval()
        dataset = FlexBondInferenceDataset(args.cache_dir, args.split)
        manifest = limit_manifest_molecules(load_eval_manifest(args.manifest), 5)
        by_id = validate_dataset_against_manifest(dataset, manifest)
        batches = [
            _to_device(by_id[str(row["sample_id"])], device)
            for row in manifest["records"]
        ]
        source = "checkpoint"
    else:
        torch.manual_seed(5)
        model = GlobalCoupled4DFlowLightningModule(
            hidden_dim=24,
            edge_hidden_dim=24,
            time_embedding_dim=16,
            num_layers=2,
        ).to(device).eval()
        batches = [_to_device(_synthetic_batch(index), device) for index in range(5)]
        source = "synthetic"

    results = []
    for molecule_count in (1, 5):
        selected = batches[:molecule_count]
        reference, reference_diagnostics, old_time, old_peak = _run(
            model,
            selected,
            optimized=False,
            refinement_steps=args.refinement_steps,
            alpha=args.alpha,
        )
        optimized, optimized_diagnostics, new_time, new_peak = _run(
            model,
            selected,
            optimized=True,
            refinement_steps=args.refinement_steps,
            alpha=args.alpha,
        )
        differences = [new - old for new, old in zip(optimized, reference)]
        result = {
            "molecule_count": molecule_count,
            "reference_total_seconds": old_time,
            "optimized_total_seconds": new_time,
            "speedup": old_time / new_time if new_time else None,
            "reference_seconds_per_molecule": old_time / molecule_count,
            "optimized_seconds_per_molecule": new_time / molecule_count,
            "reference_mean_step_seconds": sum(
                row["mean_step_time"] for row in reference_diagnostics
            ) / molecule_count,
            "optimized_mean_step_seconds": sum(
                row["mean_step_time"] for row in optimized_diagnostics
            ) / molecule_count,
            "reference_peak_memory_bytes": old_peak,
            "optimized_peak_memory_bytes": new_peak,
            "max_absolute_coordinate_difference": max(
                float(value.abs().max()) for value in differences
            ),
            "coordinate_rmsd_difference": (
                torch.cat([value.reshape(-1) for value in differences]).square().mean().sqrt().item()
            ),
            "reference_solver_backends": [
                row["solver_backend_counts"] for row in reference_diagnostics
            ],
            "optimized_solver_backends": [
                row["solver_backend_counts"] for row in optimized_diagnostics
            ],
            "reference_max_orthogonality_error": max(
                trajectory["orthogonality_error"]
                for row in reference_diagnostics
                for trajectory in row["trajectory"]
            ),
            "optimized_max_orthogonality_error": max(
                trajectory["orthogonality_error"]
                for row in optimized_diagnostics
                for trajectory in row["trajectory"]
            ),
            "reference_max_reconstruction_error": max(
                trajectory["reconstruction_error"]
                for row in reference_diagnostics
                for trajectory in row["trajectory"]
            ),
            "optimized_max_reconstruction_error": max(
                trajectory["reconstruction_error"]
                for row in optimized_diagnostics
                for trajectory in row["trajectory"]
            ),
        }
        results.append(result)

    payload = {
        "source": source,
        "device": device,
        "alpha": args.alpha,
        "refinement_steps": args.refinement_steps,
        "thread_configuration": threads,
        "thread_benchmark": _thread_benchmark(),
        "results": results,
    }
    atomic_json_save(payload, args.output)
    markdown = args.output.with_suffix(".md")
    lines = [
        "# Global Coupled 4D sampling benchmark",
        "",
        f"- Source: `{source}`",
        f"- Device: `{device}`",
        f"- Alpha: `{args.alpha}`",
        f"- Refinement steps: `{args.refinement_steps}`",
        "",
        "| Molecules | Reference s | Optimized s | Speedup | Max abs diff | RMSD diff |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['molecule_count']} | {row['reference_total_seconds']:.6f} | "
            f"{row['optimized_total_seconds']:.6f} | {row['speedup']:.3f} | "
            f"{row['max_absolute_coordinate_difference']:.3e} | "
            f"{row['coordinate_rmsd_difference']:.3e} |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
