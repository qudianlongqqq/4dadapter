#!/usr/bin/env python
"""Run stable Euler secondary refinement from cached upstream conformers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset
from etflow.models.flexbond_optimizer import FlexBondOptimizerLightningModule


def _reference_candidates(data) -> torch.Tensor:
    ptr = data.reference_conformer_ptr.detach().cpu().tolist()
    return torch.stack(
        [data.x_ref_candidates[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1)]
    )


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
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refinement_steps", type=int, default=10, choices=(1, 5, 10, 20))
    parser.add_argument("--step_size", type=float)
    parser.add_argument("--max_molecules", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = FlexBondOptimizerLightningModule.load_from_checkpoint(
        args.checkpoint, map_location=args.device
    ).to(args.device).eval()
    dataset = FlexBondOptimizerDataset(
        args.cache_dir, args.split, max_molecules=args.max_molecules
    )
    results, failures = [], []
    for data in dataset:
        data = data.to(args.device)
        refined, stability = model.refine(
            data,
            refinement_steps=args.refinement_steps,
            step_size=args.step_size,
        )
        bond_stability = _bond_stability(data, refined)
        stable = stability["stable"] and bond_stability["bond_stable"]
        result = {
            "mol_id": data.mol_id,
            "source_mol_id": data.source_mol_id,
            "smiles": data.smiles,
            "atomic_numbers": data.atomic_numbers.cpu(),
            "x_init": data.x_init.cpu(),
            "x_refined": refined.cpu(),
            "x_ref_candidates": _reference_candidates(data).cpu(),
            "num_rotatable_bonds": int(data.num_rotatable_bonds.item()),
            "method_name": model.optimizer_mode,
            "refinement_steps": args.refinement_steps,
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "config_path": str(Path(args.config).resolve()),
            "stable": stable,
            **stability,
            **bond_stability,
        }
        if stable:
            results.append(result)
        else:
            failures.append(result)
            print(f"Skipping unstable refinement: {data.mol_id}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"results": results, "failures": failures}, args.output)
    print(f"Saved {len(results)} refinements and {len(failures)} failures to {args.output}")


if __name__ == "__main__":
    main()
