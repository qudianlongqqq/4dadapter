#!/usr/bin/env python
"""Run the required SE(3) equivariance sanity check on one cache sample."""

from __future__ import annotations

import argparse

import torch

from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset
from etflow.models.flexbond_optimizer import FlexBondOptimizerLightningModule


def random_rotation(dtype, device) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=dtype, device=device))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoint")
    parser.add_argument("--mode", default="flexbond4d_hybrid_optimizer")
    parser.add_argument("--atol", type=float, default=2.0e-5)
    args = parser.parse_args()
    model = (
        FlexBondOptimizerLightningModule.load_from_checkpoint(args.checkpoint)
        if args.checkpoint
        else FlexBondOptimizerLightningModule(mode=args.mode)
    ).eval()
    dataset = FlexBondOptimizerDataset(args.cache_dir, args.split, max_molecules=100)
    data = dataset[0]
    if model.optimizer_mode != "cartesian_optimizer":
        for candidate in dataset:
            if candidate.rotatable_bond_index.size(1) > 0:
                data = candidate
                break
        else:
            raise SystemExit(
                "No rotatable-bond sample found; v_4d equivariance would be vacuous."
            )
    rotation = random_rotation(data.x_init.dtype, data.x_init.device)
    translation = torch.randn(1, 3, dtype=data.x_init.dtype)
    transformed = data.x_init @ rotation.transpose(0, 1) + translation
    with torch.no_grad():
        base = model(data, data.x_init, torch.tensor([0.37]))
        moved = model(data, transformed, torch.tensor([0.37]))
    passed = True
    for key in ("v_cart", "v_4d", "v_final"):
        expected = base[key] @ rotation.transpose(0, 1)
        error = torch.linalg.norm(moved[key] - expected, dim=-1)
        mean_error = float(error.mean())
        max_error = float(error.max())
        passed &= max_error <= args.atol
        print(f"{key}: mean_error={mean_error:.8g} max_error={max_error:.8g}")
    if not passed:
        raise SystemExit(f"FAIL: max equivariance error exceeded atol={args.atol}")
    print("PASS: v_cart, v_4d, and v_final are SE(3)-equivariant within tolerance.")


if __name__ == "__main__":
    main()
