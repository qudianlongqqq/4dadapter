#!/usr/bin/env python
"""Check Jacobian mapping and ridge least squares on a real cached topology."""

from __future__ import annotations

import argparse

import torch

from etflow.commons.flexbond_jacobian import (
    apply_bond_jacobian,
    identify_target_bonds,
    jacobian_sanity_check,
    solve_q_star_least_squares,
)
from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_search", type=int, default=100)
    args = parser.parse_args()
    dataset = FlexBondOptimizerDataset(args.cache_dir, args.split)
    for index in range(min(len(dataset), args.max_search)):
        data = dataset[index]
        target = identify_target_bonds(
            data.rotatable_bond_index,
            data.atom_bond_influence_index,
            min_affected_atoms=2,
        )
        if target["anchor_index"].numel():
            break
    else:
        raise SystemExit("No sample with a valid rotatable target bond was found.")
    q_true = torch.randn(target["anchor_index"].numel(), 4) * 0.05
    velocity, _ = apply_bond_jacobian(data.x_init, q_true, target)
    q_star, valid, solve_stats = solve_q_star_least_squares(
        data.x_init, velocity, target
    )
    jacobian_stats = jacobian_sanity_check(data.x_init, q_true, target)
    valid_error = (
        float((q_star[valid] - q_true[valid]).abs().max()) if valid.any() else float("nan")
    )
    print(f"mol_id={data.mol_id}")
    print(f"jacobian={jacobian_stats}")
    print(
        "least_squares="
        f"num_valid_bonds={solve_stats['num_valid_bonds']} "
        f"num_skipped_too_small={solve_stats['num_skipped_too_small']} "
        f"num_skipped_rank_deficient={solve_stats['num_skipped_rank_deficient']} "
        f"q_star_nan_count={solve_stats['q_star_nan_count']} "
        f"max_valid_recovery_error={valid_error:.8g}"
    )
    if not jacobian_stats["q_shape_ok"] or not jacobian_stats["correction_shape_ok"]:
        raise SystemExit("FAIL: Jacobian shape check failed.")
    if not jacobian_stats["finite"] or solve_stats["q_star_nan_count"]:
        raise SystemExit("FAIL: non-finite Jacobian or least-squares output.")
    print("PASS: bond-local Jacobian and robust least-squares checks succeeded.")


if __name__ == "__main__":
    main()
