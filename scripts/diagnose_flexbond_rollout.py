#!/usr/bin/env python
"""Label-free, coordinate-free rollout trajectory diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.provenance import collect_run_provenance
from etflow.commons.refinement_utils import clip_atom_displacement
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset
from etflow.models.flexbond_optimizer import FlexBondOptimizerLightningModule


FIELDS = (
    "sample_id", "rollout_step", "update_norm", "cartesian_norm",
    "raw_v4d_norm", "scaled_v4d_norm", "max_atom_displacement",
    "coordinate_finite", "bond_length_ratio", "failure_reason",
)


def _mean_atom_norm(value):
    return float(torch.linalg.norm(value, dim=-1).mean()) if value.numel() else 0.0


@torch.no_grad()
def trajectory_rows(model, data, steps: int, max_displacement: float | None):
    x = data.x_init.clone()
    edge = data.edge_index
    keep = edge[0] < edge[1]
    src, dst = edge[:, keep]
    initial_bonds = torch.linalg.norm(x[src] - x[dst], dim=-1).clamp_min(1e-6)
    rows = []
    for step in range(steps):
        t = x.new_tensor(step / max(steps - 1, 1))
        output = model(data, x, t)
        raw_update = output["v_final"] / steps
        update, _ = clip_atom_displacement(raw_update, max_displacement=max_displacement)
        candidate = x + update
        finite = bool(torch.isfinite(candidate).all())
        if initial_bonds.numel() and finite:
            ratio = torch.linalg.norm(candidate[src] - candidate[dst], dim=-1) / initial_bonds
            bond_ratio = float(torch.maximum(ratio.max(), 1.0 / ratio.min().clamp_min(1e-8)))
        else:
            bond_ratio = 1.0
        reason = "" if finite and bond_ratio < 3.0 else (
            "nonfinite_coordinate" if not finite else "bond_length_ratio_out_of_range"
        )
        scale = float(model.hparams.correction_scale)
        rows.append({
            "sample_id": str(data.sample_id), "rollout_step": step,
            "update_norm": _mean_atom_norm(update),
            "cartesian_norm": _mean_atom_norm(output["v_cart"]),
            "raw_v4d_norm": _mean_atom_norm(output["v_4d"]),
            "scaled_v4d_norm": _mean_atom_norm(scale * output["v_4d"]),
            "max_atom_displacement": float(torch.linalg.norm(update, dim=-1).max()),
            "coordinate_finite": finite, "bond_length_ratio": bond_ratio,
            "failure_reason": reason,
        })
        if reason:
            break
        x = candidate
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--refinement_steps", type=int, default=10, choices=(1, 5, 10, 20))
    parser.add_argument("--max_displacement", type=float)
    parser.add_argument("--correction_scale_override", type=float)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, provenance_path = args.output_dir / "trajectory_metrics.csv", args.output_dir / "provenance.json"
    if csv_path.exists() or provenance_path.exists():
        raise FileExistsError("Refusing to overwrite an existing rollout diagnostic.")
    model = FlexBondOptimizerLightningModule.load_from_checkpoint(
        args.checkpoint, map_location=args.device).to(args.device).eval()
    checkpoint_scale = float(model.hparams.correction_scale)
    if args.correction_scale_override is not None:
        model.hparams.correction_scale = float(args.correction_scale_override)
    dataset = FlexBondInferenceDataset(args.cache_dir, args.split, args.max_samples)
    rows = []
    for data in dataset:
        rows.extend(trajectory_rows(model, data.to(args.device), args.refinement_steps, args.max_displacement))
    with csv_path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    provenance = collect_run_provenance(
        config_path=args.config, checkpoint_path=args.checkpoint, cache_path=args.cache_dir)
    provenance.update({
        "label_free": True, "refinement_steps": args.refinement_steps,
        "max_displacement": args.max_displacement,
        "checkpoint_correction_scale": checkpoint_scale,
        "override_correction_scale": args.correction_scale_override,
        "effective_correction_scale": float(model.hparams.correction_scale),
        "trajectory_schema": list(FIELDS),
    })
    with provenance_path.open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    print(f"Wrote {len(rows)} label-free trajectory rows to {csv_path}")


if __name__ == "__main__":
    main()
