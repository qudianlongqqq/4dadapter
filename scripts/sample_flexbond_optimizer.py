#!/usr/bin/env python
"""Run stable Euler secondary refinement from cached upstream conformers."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch
import yaml

from etflow.commons.provenance import collect_run_provenance
from etflow.data.flexbond_eval_manifest import (
    limit_manifest_molecules,
    load_eval_manifest,
    validate_dataset_against_manifest,
)
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset
from etflow.models.flexbond_optimizer import FlexBondOptimizerLightningModule


def _alpha_tag(update_scale: float) -> str:
    return f"{float(update_scale):g}".replace("-", "m").replace(".", "p")


def output_path_with_alpha(path: Path, update_scale: float) -> Path:
    tag = f"alpha{_alpha_tag(update_scale)}"
    if tag in path.stem:
        return path
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def resolve_correction_scale(model, config_scale: float, override: float | None) -> dict:
    """Apply an explicit inference-only override; ``None`` preserves behavior."""

    checkpoint_scale = float(model.hparams.correction_scale)
    effective = checkpoint_scale if override is None else float(override)
    if not torch.isfinite(torch.tensor(effective)):
        raise ValueError("correction_scale_override must be finite.")
    if override is not None:
        model.hparams.correction_scale = effective
    return {
        "config_correction_scale": float(config_scale),
        "checkpoint_correction_scale": checkpoint_scale,
        "override_correction_scale": None if override is None else float(override),
        "effective_correction_scale": effective,
    }


def _bond_stability(data, refined: torch.Tensor) -> dict[str, float | bool]:
    edge = data.edge_index
    keep = edge[0] < edge[1]
    src, dst = edge[:, keep]
    before = torch.linalg.norm(data.x_init[src] - data.x_init[dst], dim=-1)
    after = torch.linalg.norm(refined[src] - refined[dst], dim=-1)
    if before.numel() == 0:
        return {"bond_stable": True, "max_bond_ratio": 1.0, "min_bond_ratio": 1.0}
    ratio = after / before.clamp_min(1.0e-6)
    return {
        "bond_stable": bool((ratio.max() < 3.0) & (ratio.min() > 0.3)),
        "max_bond_ratio": float(ratio.max()),
        "min_bond_ratio": float(ratio.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refinement_steps", type=int, default=10, choices=(1, 5, 10, 20))
    parser.add_argument("--step_size", type=float)
    parser.add_argument("--update_scale", "--alpha", type=float, default=1.0)
    parser.add_argument("--max_displacement", type=float)
    parser.add_argument("--adaptive_alpha_by_update_norm", action="store_true")
    parser.add_argument("--target_update_norm", type=float)
    parser.add_argument("--max_coordinate_norm", type=float, default=1000.0)
    parser.add_argument("--max_molecules", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--correction_scale_override", type=float)
    args = parser.parse_args()
    if args.adaptive_alpha_by_update_norm and (
        args.target_update_norm is None or args.target_update_norm <= 0
    ):
        parser.error(
            "--target_update_norm must be positive with "
            "--adaptive_alpha_by_update_norm"
        )
    model = FlexBondOptimizerLightningModule.load_from_checkpoint(
        args.checkpoint, map_location=args.device
    ).to(args.device).eval()
    with open(args.config, encoding="utf-8") as handle:
        resolved_config = yaml.safe_load(handle) or {}
    config_scale = float(resolved_config.get("model", {}).get(
        "correction_scale", model.hparams.correction_scale
    ))
    scale_metadata = resolve_correction_scale(
        model, config_scale, args.correction_scale_override
    )
    dataset = FlexBondInferenceDataset(args.cache_dir, args.split)
    manifest = load_eval_manifest(args.manifest)
    if args.max_molecules is not None:
        manifest = limit_manifest_molecules(manifest, args.max_molecules)
    by_id = validate_dataset_against_manifest(dataset, manifest)
    records = []
    method_name = (
        "cartesian_adapter"
        if model.optimizer_mode == "cartesian_optimizer"
        else "flexbond4d_adapter"
    )
    for manifest_row in manifest["records"]:
        data = by_id[str(manifest_row["sample_id"])]
        data = data.to(args.device)
        refined, stability = model.refine(
            data,
            refinement_steps=args.refinement_steps,
            step_size=args.step_size,
            update_scale=args.update_scale,
            max_displacement=args.max_displacement,
            adaptive_alpha_by_update_norm=args.adaptive_alpha_by_update_norm,
            target_update_norm=args.target_update_norm,
            max_coordinate_norm=args.max_coordinate_norm,
        )
        bond_stability = _bond_stability(data, refined)
        stable = stability["stable"] and bond_stability["bond_stable"]
        result = {
            "mol_id": data.mol_id,
            "sample_id": data.sample_id,
            "source_mol_id": data.source_mol_id,
            "smiles": data.smiles,
            "atomic_numbers": data.atomic_numbers.cpu(),
            "x_init": data.x_init.cpu(),
            "x_init_hash": str(manifest_row["x_init_hash"]),
            "x_refined": refined.cpu() if stable else None,
            "num_rotatable_bonds": int(data.num_rotatable_bonds.item()),
            "method_name": method_name,
            "optimizer_mode": model.optimizer_mode,
            "refinement_steps": args.refinement_steps,
            "update_scale": args.update_scale,
            "alpha": args.update_scale,
            "max_displacement": args.max_displacement,
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "config_path": str(Path(args.config).resolve()),
            **scale_metadata,
            "stable": stable,
            **stability,
            **bond_stability,
        }
        result["status"] = "success" if stable else "failed"
        result["failure_reason"] = None if stable else (
            f"failed_step={stability['failed_step']}, bond_stable={bond_stability['bond_stable']}"
        )
        records.append(result)
        if not stable:
            print(f"Skipping unstable refinement: {data.mol_id}")
    output_path = output_path_with_alpha(args.output, args.update_scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = collect_run_provenance(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        cache_path=args.cache_dir,
    )
    provenance["correction_scale"] = scale_metadata
    successes = sum(row["status"] == "success" for row in records)
    failure_count = len(records) - successes
    update_means = torch.tensor(
        [float(row["mean_update_norm"]) for row in records], dtype=torch.float64
    )
    update_medians = torch.tensor(
        [float(row["median_update_norm"]) for row in records], dtype=torch.float64
    )
    update_maxima = torch.tensor(
        [float(row["max_update_norm"]) for row in records], dtype=torch.float64
    )
    alpha_effective = torch.tensor(
        [float(row["alpha_eff"]) for row in records], dtype=torch.float64
    )
    raw_step_means = torch.tensor(
        [float(row["mean_step_update_norm_raw"]) for row in records],
        dtype=torch.float64,
    )
    applied_step_means = torch.tensor(
        [float(row["mean_step_update_norm_applied"]) for row in records],
        dtype=torch.float64,
    )
    clipping_fractions = torch.tensor(
        [float(row["clipping_fraction"]) for row in records], dtype=torch.float64
    )
    sample_summary = {
        "update_scale": float(args.update_scale),
        "alpha": float(args.update_scale),
        "alpha_eff": (
            float(alpha_effective.mean()) if alpha_effective.numel() else 0.0
        ),
        "adaptive_alpha_by_update_norm": args.adaptive_alpha_by_update_norm,
        "target_update_norm": args.target_update_norm,
        "max_displacement": args.max_displacement,
        "refinement_steps": args.refinement_steps,
        "mean_update_norm": float(update_means.mean()) if update_means.numel() else 0.0,
        "median_update_norm": (
            float(update_medians.median()) if update_medians.numel() else 0.0
        ),
        "max_update_norm": float(update_maxima.max()) if update_maxima.numel() else 0.0,
        "mean_step_update_norm_raw": (
            float(raw_step_means.mean()) if raw_step_means.numel() else 0.0
        ),
        "mean_step_update_norm_applied": (
            float(applied_step_means.mean()) if applied_step_means.numel() else 0.0
        ),
        "clipping_fraction": (
            float(clipping_fractions.mean()) if clipping_fractions.numel() else 0.0
        ),
        "failure_count": failure_count,
        "failure_rate": failure_count / len(records) if records else 0.0,
        **scale_metadata,
    }
    torch.save(
        {
            "records": records,
            "manifest": manifest,
            "provenance": provenance,
            "summary": sample_summary,
            **sample_summary,
        },
        output_path,
    )
    print(
        f"Saved {successes} refinements and {len(records) - successes} "
        f"failures to {output_path}"
    )


if __name__ == "__main__":
    main()
