#!/usr/bin/env python
"""Label-aware diagnosis of FlexBond-4D projection and learned-head quality.

This script intentionally uses ``FlexBondOptimizerDataset`` and must only be
run on train/validation diagnostic caches.  Normal inference remains label-free.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import torch

from etflow.commons.flexbond_diagnostics import projection_quality, vector_rms
from etflow.commons.flexbond_jacobian import (
    apply_bond_jacobian,
    build_bond_local_frame,
    solve_q_star_least_squares,
)
from etflow.commons.jacobian_4d_velocity import build_atom_jacobian
from etflow.commons.provenance import collect_run_provenance
from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset
from etflow.models.flexbond_optimizer import FlexBondOptimizerLightningModule


SAMPLE_FIELDS = (
    "sample_id", "molecule_id", "t", "num_atoms", "num_rotatable_bonds",
    "valid_target_bonds", "skipped_too_small", "skipped_rank_deficient",
    "skipped_by_cap", "residual_norm", "cartesian_only_error",
    "hybrid_cart_error", "hybrid_final_error", "oracle_raw_error",
    "oracle_scaled_error", "oracle_explained_ratio", "q_pred_error",
    "q_pred_norm", "q_star_norm", "v4d_pred_norm", "v4d_star_norm",
    "scaled_v4d_pred_norm", "scaled_v4d_star_norm",
    "cosine_pred_to_residual", "cosine_star_to_residual",
    "pred_norm_to_residual_ratio", "scaled_pred_norm_to_residual_ratio",
    "oracle_norm_to_residual_ratio", "hybrid_branch_delta",
    "joint_training_delta", "total_hybrid_gap", "q_star_nonfinite_count",
    "q_pred_nonfinite_count", "condition_number_median",
)

BOND_FIELDS = (
    "sample_id", "t", "bond_index", "bond_atom_id_anchor", "bond_atom_id_moving",
    "affected_atom_count", "condition_number", "rank", "q_star_s",
    "q_star_w1", "q_star_w2", "q_star_w3", "q_pred_s", "q_pred_w1",
    "q_pred_w2", "q_pred_w3", "q_star_norm", "q_pred_norm", "q_mse",
    "frame_valid", "solve_valid", "skip_reason",
)


def _write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows, key) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else 0.0


def _summary(samples: list[dict], bonds: list[dict], group: str) -> dict:
    valid = sum(bool(row["solve_valid"]) for row in bonds)
    conditions = np.asarray([
        float(row["condition_number"]) for row in bonds
        if math.isfinite(float(row["condition_number"]))
    ])
    explained = np.asarray([float(row["oracle_explained_ratio"]) for row in samples])
    branch = np.asarray([float(row["hybrid_branch_delta"]) for row in samples])
    return {
        "group": group,
        "sample_count": len(samples),
        "bond_count": len(bonds),
        "valid_bond_ratio": valid / len(bonds) if bonds else 0.0,
        "skipped_too_small": sum(int(row["skipped_too_small"]) for row in samples),
        "skipped_rank_deficient": sum(int(row["skipped_rank_deficient"]) for row in samples),
        "skipped_by_cap": sum(int(row["skipped_by_cap"]) for row in samples),
        "mean_oracle_explained_ratio": float(explained.mean()) if len(explained) else 0.0,
        "median_oracle_explained_ratio": float(np.median(explained)) if len(explained) else 0.0,
        "fraction_oracle_explained_gt_025": float((explained > .25).mean()) if len(explained) else 0.0,
        "fraction_oracle_explained_gt_05": float((explained > .5).mean()) if len(explained) else 0.0,
        "fraction_oracle_explained_gt_075": float((explained > .75).mean()) if len(explained) else 0.0,
        "mean_cosine_pred_to_residual": _mean(samples, "cosine_pred_to_residual"),
        "fraction_cosine_pred_negative": float(np.mean([float(r["cosine_pred_to_residual"]) < 0 for r in samples])) if samples else 0.0,
        "mean_scaled_pred_norm_to_residual_ratio": _mean(samples, "scaled_pred_norm_to_residual_ratio"),
        "fraction_4d_improved": float((branch < 0).mean()) if len(branch) else 0.0,
        "fraction_4d_worsened": float((branch > 0).mean()) if len(branch) else 0.0,
        "mean_hybrid_branch_delta": _mean(samples, "hybrid_branch_delta"),
        "mean_joint_training_delta": _mean(samples, "joint_training_delta"),
        "mean_total_hybrid_gap": _mean(samples, "total_hybrid_gap"),
        "q_star_nan_inf_count": sum(int(r["q_star_nonfinite_count"]) for r in samples),
        "q_pred_nan_inf_count": sum(int(r["q_pred_nonfinite_count"]) for r in samples),
        "condition_q25": float(np.quantile(conditions, .25)) if len(conditions) else "",
        "condition_q50": float(np.quantile(conditions, .5)) if len(conditions) else "",
        "condition_q75": float(np.quantile(conditions, .75)) if len(conditions) else "",
        "condition_q95": float(np.quantile(conditions, .95)) if len(conditions) else "",
    }


def _bond_details(x, targets, q_pred, q_star, solve_valid, conditions, sample_id, t_value):
    rows = []
    frame, frame_valid = build_bond_local_frame(
        x, targets["anchor_index"], targets["moving_index"],
        targets["affected_atom_index"], targets["affected_bond_index"],
    )
    for index in range(q_pred.size(0)):
        mask = targets["affected_bond_index"] == index
        atoms = targets["affected_atom_index"][mask]
        rank = 0
        if atoms.numel() and bool(frame_valid[index]):
            lever = x[atoms] - x[targets["anchor_index"][index]]
            jac = build_atom_jacobian(
                frame[index, :, 0].expand_as(lever),
                frame[index].expand(atoms.numel(), -1, -1), lever,
            ).reshape(-1, 4)
            rank = int(torch.linalg.matrix_rank(jac).item())
        finite_pred = bool(torch.isfinite(q_pred[index]).all())
        finite_star = bool(torch.isfinite(q_star[index]).all())
        if not bool(frame_valid[index]):
            reason = "invalid_frame"
        elif not math.isfinite(float(conditions[index])):
            reason = "nonfinite_condition"
        elif not bool(solve_valid[index]):
            reason = "rank_or_condition_or_norm"
        elif not finite_pred:
            reason = "nonfinite_q_pred"
        elif not finite_star:
            reason = "nonfinite_q_star"
        else:
            reason = ""
        original = int(targets["original_bond_index"][index])
        pred = torch.nan_to_num(q_pred[index].detach())
        star = torch.nan_to_num(q_star[index].detach())
        rows.append({
            "sample_id": sample_id, "t": t_value, "bond_index": original,
            "bond_atom_id_anchor": int(targets["anchor_index"][index]),
            "bond_atom_id_moving": int(targets["moving_index"][index]),
            "affected_atom_count": int(atoms.numel()),
            "condition_number": float(conditions[index]), "rank": rank,
            "q_star_s": float(star[0]), "q_star_w1": float(star[1]),
            "q_star_w2": float(star[2]), "q_star_w3": float(star[3]),
            "q_pred_s": float(pred[0]), "q_pred_w1": float(pred[1]),
            "q_pred_w2": float(pred[2]), "q_pred_w3": float(pred[3]),
            "q_star_norm": float(torch.linalg.norm(star)),
            "q_pred_norm": float(torch.linalg.norm(pred)),
            "q_mse": float((pred - star).square().mean()),
            "frame_valid": bool(frame_valid[index]),
            "solve_valid": bool(solve_valid[index]), "skip_reason": reason,
        })
    return rows


@torch.no_grad()
def diagnose_sample(data, hybrid, cartesian, t_value: float):
    device = data.x_init.device
    t = torch.tensor([t_value], device=device, dtype=data.x_init.dtype)
    x_t = (1 - t_value) * data.x_init + t_value * data.x_ref_aligned
    target = data.x_ref_aligned - data.x_init
    hybrid_out = hybrid(data, x_t, t)
    cart_out = cartesian(data, x_t, t)
    residual = target - hybrid_out["v_cart"]
    q_pred_raw = hybrid_out["q_b"]
    q_star_raw, solve_valid, stats = solve_q_star_least_squares(
        x_t, residual, hybrid_out["target_bonds"],
        ridge_eps=hybrid.hparams.ridge_eps,
        max_q_norm=hybrid.hparams.max_q_norm,
        max_condition=hybrid.hparams.max_condition,
    )
    q_pred = torch.nan_to_num(q_pred_raw)
    q_star = torch.nan_to_num(q_star_raw)
    v4d_pred, _ = apply_bond_jacobian(x_t, q_pred, hybrid_out["target_bonds"])
    v4d_star, _ = apply_bond_jacobian(x_t, q_star, hybrid_out["target_bonds"])
    scale = float(hybrid.hparams.correction_scale)
    v_final = hybrid_out["v_cart"] + scale * v4d_pred
    cart_error = vector_rms(cart_out["v_cart"] - target)
    hybrid_cart_error = vector_rms(hybrid_out["v_cart"] - target)
    hybrid_final_error = vector_rms(v_final - target)
    quality = projection_quality(residual, v4d_pred, v4d_star, correction_scale=scale)
    sample_id = str(getattr(data, "sample_id", data.mol_id))
    conditions = stats["condition_numbers"]
    finite_conditions = conditions[torch.isfinite(conditions)]
    valid_q = solve_valid & torch.isfinite(q_pred_raw).all(dim=-1)
    row = {
        "sample_id": sample_id, "molecule_id": str(data.mol_id), "t": t_value,
        "num_atoms": int(data.x_init.size(0)),
        "num_rotatable_bonds": int(data.num_rotatable_bonds.item()),
        "valid_target_bonds": int(solve_valid.sum()),
        "skipped_too_small": int(stats["num_skipped_too_small"]),
        "skipped_rank_deficient": int(stats["num_skipped_rank_deficient"]),
        "skipped_by_cap": int(hybrid_out["target_bonds"]["num_skipped_by_cap"]),
        "cartesian_only_error": cart_error, "hybrid_cart_error": hybrid_cart_error,
        "hybrid_final_error": hybrid_final_error,
        "q_pred_error": float((q_pred[valid_q] - q_star[valid_q]).square().mean()) if valid_q.any() else 0.0,
        "q_pred_norm": vector_rms(q_pred), "q_star_norm": vector_rms(q_star),
        "hybrid_branch_delta": hybrid_final_error - hybrid_cart_error,
        "joint_training_delta": hybrid_cart_error - cart_error,
        "total_hybrid_gap": hybrid_final_error - cart_error,
        "q_star_nonfinite_count": int(stats["q_star_nan_count"]),
        "q_pred_nonfinite_count": int((~torch.isfinite(q_pred_raw).all(dim=-1)).sum()),
        "condition_number_median": float(finite_conditions.median()) if finite_conditions.numel() else float("inf"),
        **quality,
    }
    bonds = _bond_details(x_t, hybrid_out["target_bonds"], q_pred_raw, q_star_raw,
                          solve_valid, conditions, sample_id, t_value)
    return row, bonds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid_checkpoint", required=True)
    parser.add_argument("--hybrid_config", required=True)
    parser.add_argument("--cartesian_checkpoint", required=True)
    parser.add_argument("--cartesian_config", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed_times", nargs="+", type=float, default=[.05, .1, .25, .5])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if any(not 0 <= value <= 1 for value in args.fixed_times):
        parser.error("--fixed_times values must lie in [0, 1].")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / name for name in (
        "sample_metrics.csv", "bond_metrics.csv", "summary.csv", "summary_by_t.csv",
        "summary_by_rotatable_count.csv", "summary_by_condition.csv", "provenance.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite an existing diagnostic output.")
    dataset = FlexBondOptimizerDataset(args.cache_dir, args.split)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[:args.max_samples]
    hybrid = FlexBondOptimizerLightningModule.load_from_checkpoint(
        args.hybrid_checkpoint, map_location=args.device).to(args.device).eval()
    cartesian = FlexBondOptimizerLightningModule.load_from_checkpoint(
        args.cartesian_checkpoint, map_location=args.device).to(args.device).eval()
    if hybrid.optimizer_mode != "flexbond4d_hybrid_optimizer":
        raise ValueError("--hybrid_checkpoint is not a hybrid optimizer checkpoint.")
    sample_rows, bond_rows = [], []
    for index in indices:
        data = dataset[index].to(args.device)
        for t_value in args.fixed_times:
            row, rows = diagnose_sample(data, hybrid, cartesian, t_value)
            sample_rows.append(row); bond_rows.extend(rows)
    _write_csv(outputs[0], sample_rows, SAMPLE_FIELDS)
    _write_csv(outputs[1], bond_rows, BOND_FIELDS)
    all_summary = [_summary(sample_rows, bond_rows, "all")]
    summary_fields = tuple(all_summary[0])
    _write_csv(outputs[2], all_summary, summary_fields)
    by_t = []
    for value in args.fixed_times:
        selected = [r for r in sample_rows if float(r["t"]) == value]
        by_t.append(_summary(selected, [b for b in bond_rows if float(b["t"]) == value], str(value)))
    _write_csv(outputs[3], by_t, summary_fields)
    by_rot = []
    for label, predicate in (("0-2", lambda n:n < 3), ("3-4", lambda n:3 <= n < 5),
                             ("5", lambda n:n == 5), ("6+", lambda n:n >= 6)):
        chosen = [r for r in sample_rows if predicate(int(r["num_rotatable_bonds"]))]
        ids = {r["sample_id"] for r in chosen}
        by_rot.append(_summary(chosen, [b for b in bond_rows if b["sample_id"] in ids], label))
    _write_csv(outputs[4], by_rot, summary_fields)
    by_condition = []
    bands = (("<1e3", 0, 1e3), ("1e3-1e5", 1e3, 1e5), ("1e5-1e6", 1e5, 1e6), (">=1e6", 1e6, math.inf))
    for label, low, high in bands:
        chosen_bonds = [b for b in bond_rows if low <= float(b["condition_number"]) < high]
        ids = {b["sample_id"] for b in chosen_bonds}
        by_condition.append(_summary([r for r in sample_rows if r["sample_id"] in ids], chosen_bonds, label))
    _write_csv(outputs[5], by_condition, summary_fields)
    provenance = collect_run_provenance(
        config_path=args.hybrid_config, checkpoint_path=args.hybrid_checkpoint,
        cache_path=args.cache_dir)
    provenance.update({
        "diagnostic_only_label_aware": True, "split": args.split,
        "seed": args.seed, "fixed_times": args.fixed_times,
        "max_samples": args.max_samples,
        "cartesian_checkpoint": str(Path(args.cartesian_checkpoint).resolve()),
        "cartesian_config": str(Path(args.cartesian_config).resolve()),
        "hybrid_correction_scale": float(hybrid.hparams.correction_scale),
        "sample_metrics_schema": list(SAMPLE_FIELDS), "bond_metrics_schema": list(BOND_FIELDS),
    })
    with outputs[6].open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    print(f"Wrote {len(sample_rows)} sample-time rows and {len(bond_rows)} bond rows to {args.output_dir}")


if __name__ == "__main__":
    main()
