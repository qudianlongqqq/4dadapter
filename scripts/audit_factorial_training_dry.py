"""Ephemeral J0-R1 CUDA step benchmark; never saves model or optimizer state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch

ROOT = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial")
sys.path.insert(0, str(ROOT))
os.environ["SIXS_FACTORIAL_RUN_NAMESPACE"] = "sixs_musigma_reliability_factorial_cuda"
os.environ["SIXS_FACTORIAL_DEVICE"] = "cuda"
import scripts.run_sixs_musigma_reliability_factorial as pipeline


def gpu_sample() -> dict[str, float]:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().split(",")
    return {"gpu_utilization_percent": float(raw[0]), "gpu_memory_used_mib": float(raw[1]), "gpu_memory_total_mib": float(raw[2])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("dry CUDA benchmark requires CUDA")
    config = pipeline.cfg()
    training = config["training"]
    device = torch.device("cuda")
    prepared, source_payload = pipeline.load_inputs()
    sources = pipeline.source_index(source_payload["train"])
    generator = pipeline.seed_all(config["seed"] + 20260829)
    model, reliability = pipeline.make_arm("J0-R1", device)
    final = torch.load(
        ROOT / "artifacts/ecir_mvr/sixs_musigma_reliability_factorial_cuda/J0_R1/final.ckpt",
        map_location="cpu", weights_only=False,
    )
    model.load_state_dict(final["model_state"], strict=True)
    assert reliability is not None
    reliability.load_state_dict(final["reliability_state"], strict=True)
    belief_optimizer, _, action_optimizer, _ = pipeline.optimizers(model, reliability)
    assert action_optimizer is not None

    stop = threading.Event()
    samples: list[dict[str, float]] = []
    psutil.cpu_percent(interval=None)

    def sampler():
        while not stop.is_set():
            try:
                sample = gpu_sample()
                sample["cpu_utilization_percent"] = float(psutil.cpu_percent(interval=None))
                sample["available_ram_bytes"] = int(psutil.virtual_memory().available)
                sample["timestamp"] = time.time()
                samples.append(sample)
            except Exception:
                pass
            stop.wait(.25)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    step_seconds: list[float] = []
    losses: list[tuple[float, float]] = []
    io_before = psutil.disk_io_counters()
    proc = psutil.Process()
    cpu_before = proc.cpu_times()
    try:
        for index in range(args.warmup + args.iterations):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            model.train(); reliability.train()
            graphs, bg, source, reference, _ = pipeline.sample_batch(
                prepared["train"], sources, generator, training["batch_molecules"], device,
            )
            ref_b, ref_a = pipeline.geometry_values(reference, bg)
            belief_optimizer.zero_grad(set_to_none=True)
            pred = model(bg, detach_sigma_features=False)
            belief, _ = pipeline.belief_loss(
                "J0", pred, ref_b.to(pred["bond_mu"]), ref_a.to(pred["angle_mu"]), graphs,
                training["beta_nll_beta"],
            )
            belief.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
            belief_optimizer.step()
            action_optimizer.zero_grad(set_to_none=True)
            pred_action = model(bg, detach_sigma_features=False)
            action = pipeline.action_proposal(
                source, graphs, pred_action, tau=config["action"]["tau_control_angstrom"],
                atom_cap=config["action"]["atom_cap_angstrom"], reliability_head=reliability,
            )
            action_loss = pipeline.action_loss(action, reference, graphs, pred_action)
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.backbone_parameters() + list(reliability.parameters()), training["gradient_clip"])
            action_optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            if index >= args.warmup:
                step_seconds.append(elapsed)
                losses.append((float(belief.detach()), float(action_loss.detach())))
    finally:
        stop.set(); thread.join(timeout=2)
    cpu_after = proc.cpu_times()
    io_after = psutil.disk_io_counters()
    result = {
        "schema_version": "sixs-factorial-training-dry-resource-v1",
        "ephemeral_only": True,
        "model_or_optimizer_saved": False,
        "arm_shape": "J0-R1",
        "iterations": args.iterations,
        "warmup": args.warmup,
        "step_seconds": {
            "mean": float(statistics.mean(step_seconds)),
            "median": float(statistics.median(step_seconds)),
            "p95": float(np.percentile(step_seconds, 95)),
            "min": float(min(step_seconds)), "max": float(max(step_seconds)),
        },
        "process_cpu_seconds": float((cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)),
        "process_rss_bytes": int(proc.memory_info().rss),
        "disk_read_bytes": int(io_after.read_bytes - io_before.read_bytes),
        "disk_write_bytes": int(io_after.write_bytes - io_before.write_bytes),
        "gpu_utilization_percent": {
            "mean": float(np.mean([x["gpu_utilization_percent"] for x in samples])),
            "max": float(np.max([x["gpu_utilization_percent"] for x in samples])),
        },
        "gpu_memory_used_mib": {
            "mean": float(np.mean([x["gpu_memory_used_mib"] for x in samples])),
            "max": float(np.max([x["gpu_memory_used_mib"] for x in samples])),
        },
        "system_cpu_utilization_percent": {
            "mean": float(np.mean([x["cpu_utilization_percent"] for x in samples])),
            "max": float(np.max([x["cpu_utilization_percent"] for x in samples])),
        },
        "minimum_available_ram_bytes": int(min(x["available_ram_bytes"] for x in samples)),
        "finite_losses": bool(all(np.isfinite(x) for pair in losses for x in pair)),
        "samples": len(samples),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
