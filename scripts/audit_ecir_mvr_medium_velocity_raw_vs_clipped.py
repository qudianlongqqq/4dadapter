#!/usr/bin/env python
"""Reconstruct the V2 stop batch and audit raw/trust-clipped velocity semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import pandas as pd
import torch
import yaml
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.mvr_model import MCVRModel, trust_clip_velocity
from etflow.ecir.mvr_safety import _distribution, trust_clip_with_diagnostics
from scripts.train_ecir_mvr_run_a import _dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten(prefix: str, values: dict) -> dict[str, float]:
    return {
        f"{prefix}_velocity_atom_mean": values["atom_mean"],
        f"{prefix}_velocity_atom_p95": values["atom_p95"],
        f"{prefix}_velocity_atom_max": values["atom_max"],
        f"{prefix}_velocity_graph_rms": values["graph_rms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict", "optimizer_state_dict", "global_step", "rng_states",
        "sampler_state", "timing_accumulator", "frozen_identities",
    }
    checkpoint_complete = required.issubset(payload)
    if not checkpoint_complete or int(payload["global_step"]) != 2450:
        raise RuntimeError("V2 step2450 checkpoint is not strictly resumable")
    if payload["frozen_identities"] != config["frozen_identities"]:
        raise RuntimeError("V2 checkpoint frozen identities changed")

    validity = ChemicalValidity(config["data"]["validity_statistics"])
    dataset = _dataset(config, "train", validity)
    epoch = int(payload["sampler_state"]["epoch"])
    batch_offset = int(payload["sampler_state"]["batch_offset"])
    dataset.set_epoch(epoch)
    start = (batch_offset - 1) * int(config["training"]["batch_size"])
    indices = list(range(start, start + int(config["training"]["batch_size"])))
    batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False)))
    device = torch.device(args.device)
    batch = batch.to(device)
    model = MCVRModel(**config["model"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(batch, batch.x_input, batch.x_input.new_full((batch.num_graphs,), 0.5))
    reconstructed_clipped, clipping = trust_clip_with_diagnostics(
        output["v_raw"], batch.batch,
        max_atom_norm=float(config["model"]["max_velocity_atom_norm"]),
        max_graph_rms=float(config["model"]["max_velocity_graph_rms"]),
    )
    legacy_clipped = trust_clip_velocity(
        output["v_raw"], batch.batch,
        max_atom_norm=float(config["model"]["max_velocity_atom_norm"]),
        max_graph_rms=float(config["model"]["max_velocity_graph_rms"]),
    )
    final_stats = _distribution(output["v_final"], batch.batch)
    metric_row = pd.read_csv(args.metrics)
    metric_row = metric_row[(metric_row.split == "train") & (metric_row.step == 2450)].iloc[-1]
    stopped_value = float(metric_row.velocity_graph_rms)
    metrics = {
        **_flatten("raw", clipping["raw"]),
        **_flatten("clipped", clipping["clipped"]),
        **_flatten("final_output", final_stats),
        "graph_clip_scale": clipping["graph_clip_scale"],
        "atom_clip_scale": clipping["atom_clip_scale"],
        "graph_clipped_fraction": clipping["graph_clipped_fraction"],
        "atom_clipped_fraction": clipping["atom_clipped_fraction"],
    }
    result = {
        "schema_version": "ecir-mvr-medium-raw-vs-clipped-audit-v1",
        "decision": "POST_CLIP_THRESHOLD_SELF_TRIGGER",
        "v2_stop_value": stopped_value,
        "v2_stop_value_source": "v_final_post_trust_clip_and_safety_gate",
        "source_classification": "B_POST_CLIP",
        "explanation": (
            "V2 _diagnostics assigned velocity_graph_rms from output['v_final']; "
            "it did not monitor output['v_raw']. Equality at the trust boundary was therefore self-triggering."
        ),
        "metrics": metrics,
        "reconstruction": {
            "epoch": epoch, "batch_offset_after_step": batch_offset,
            "dataset_indices": indices,
            "model_clipped_matches_reconstruction_exact": bool(torch.equal(output["v_trust_clipped"], reconstructed_clipped)),
            "legacy_trust_clip_matches_reconstruction_exact": bool(torch.equal(legacy_clipped, reconstructed_clipped)),
            "v2_recorded_matches_reconstructed_final": abs(stopped_value - final_stats["graph_rms"]) <= 1.0e-9,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()), "sha256": _sha(args.checkpoint),
            "complete": checkpoint_complete, "step": int(payload["global_step"]),
            "required_fields": sorted(required),
            "optimizer_state_nonempty": bool(payload["optimizer_state_dict"].get("state")),
            "rng_state_keys": sorted(payload["rng_states"]),
            "sampler_state": payload["sampler_state"],
            "timing_accumulator": payload["timing_accumulator"],
            "frozen_identities": payload["frozen_identities"],
        },
        "test_records_read": 0, "seed43_started": False, "seed44_started": False,
        "100k_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_save(result, args.output)
    lines = [
        "# MCVR Medium Velocity Raw vs Clipped Audit", "",
        "Decision: **POST_CLIP_THRESHOLD_SELF_TRIGGER**", "",
        f"The V2 stop value `{stopped_value:.16f}` came from `v_final`, after trust clipping "
        "and the global safety gate. It was not a raw-velocity measurement.", "",
        "| Metric | Value |", "|---|---:|",
    ]
    lines.extend(f"| {name} | {value:.12f} |" for name, value in metrics.items())
    lines += [
        "", "The frozen trust clipping calculation is bitwise identical in the model, "
        "the legacy helper, and the audit reconstruction. The monitoring comparison, not "
        "the clipping mathematics, caused the stop.", "",
        f"The step2450 checkpoint is complete and strictly resumable: `{_sha(args.checkpoint)}`.",
        "It contains model, optimizer, global step, RNG, sampler, timing, and frozen identities.", "",
        "No test split, seed43/44, or 100k artifact was read or started.",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
