#!/usr/bin/env python
"""Compare Cartesian and bond-local branch contributions across checkpoints."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch

from etflow.commons.flexbond_jacobian import (
    apply_bond_jacobian,
    solve_q_star_least_squares,
)
from etflow.data.flexbond_eval_manifest import (
    load_eval_manifest,
    validate_dataset_against_manifest,
)
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset
from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset
from etflow.models.flexbond_optimizer import FlexBondOptimizerLightningModule


METRICS = (
    "v_cart_norm",
    "v_4d_norm",
    "scaled_v_4d_norm",
    "v_final_norm",
    "v_4d_to_v_cart_ratio",
    "q_norm",
    "q_star_norm",
    "corr_to_target_ratio",
    "residual_explained_by_Jq",
    "residual_explained_by_raw_Jq",
)


def _step(path: Path, checkpoint: object) -> int:
    if isinstance(checkpoint, dict) and checkpoint.get("global_step") is not None:
        return int(checkpoint["global_step"])
    matches = re.findall(r"(?:step[=_-]?)?(\d+)", path.stem)
    return int(matches[-1]) if matches else -1


def _mean_atom_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.norm(value, dim=-1).mean())


def _mean_bond_norm(value: torch.Tensor, valid: torch.Tensor | None = None) -> float:
    if valid is not None:
        value = value[valid]
    return float(torch.linalg.norm(value, dim=-1).mean()) if value.numel() else 0.0


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inference_cache", required=True)
    parser.add_argument("--reference_cache", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    if not 0 <= args.time <= 1:
        raise ValueError("time must be in [0, 1]")

    manifest = load_eval_manifest(args.manifest)
    inference = validate_dataset_against_manifest(
        FlexBondInferenceDataset(args.inference_cache, args.split), manifest
    )
    reference_by_id = {
        str(data.mol_id): data
        for data in FlexBondOptimizerDataset(
            args.reference_cache, args.split, validate=True
        )
    }
    rows = []
    for checkpoint_path in args.checkpoints:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        raw_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        step = _step(checkpoint_path, raw_checkpoint)
        model = FlexBondOptimizerLightningModule.load_from_checkpoint(
            checkpoint_path, map_location=args.device
        ).to(args.device).eval()
        correction_scale = (
            float(model.hparams.correction_scale)
            if model.optimizer_mode == "flexbond4d_hybrid_optimizer"
            else (1.0 if model.optimizer_mode == "flexbond4d_only_optimizer" else 0.0)
        )
        with torch.no_grad():
            for manifest_row in manifest["records"]:
                sample_id = str(manifest_row["sample_id"])
                data = inference[sample_id].to(args.device)
                reference = reference_by_id.get(sample_id)
                if reference is None:
                    raise ValueError(f"Reference cache is missing {sample_id!r}")
                target = (
                    reference.x_ref_aligned.to(args.device)
                    - reference.x_init.to(args.device)
                )
                output = model(data, data.x_init, data.x_init.new_tensor([args.time]))
                raw_v_4d, _ = apply_bond_jacobian(
                    data.x_init, output["q_b"], output["target_bonds"]
                )
                scaled_correction = correction_scale * raw_v_4d
                v_final = output["v_cart"] + scaled_correction
                residual = target - output["v_cart"]
                q_star, q_valid, q_stats = solve_q_star_least_squares(
                    data.x_init,
                    residual,
                    output["target_bonds"],
                    ridge_eps=float(model.hparams.ridge_eps),
                    max_q_norm=float(model.hparams.max_q_norm),
                    max_condition=float(model.hparams.max_condition),
                )
                residual_mse = float(residual.square().mean())
                explained_scaled = 1.0 - float(
                    (residual - scaled_correction).square().mean()
                ) / max(residual_mse, 1.0e-12)
                explained_raw = 1.0 - float(
                    (residual - raw_v_4d).square().mean()
                ) / max(residual_mse, 1.0e-12)
                cart_norm = _mean_atom_norm(output["v_cart"])
                raw_corr_norm = _mean_atom_norm(raw_v_4d)
                target_norm = _mean_atom_norm(target)
                rows.append(
                    {
                        "checkpoint_name": checkpoint_path.name,
                        "checkpoint_path": str(checkpoint_path),
                        "step": step,
                        "mode": model.optimizer_mode,
                        "sample_id": sample_id,
                        "mol_id": str(manifest_row["mol_id"]),
                        "num_rotatable_bonds": int(
                            manifest_row["num_rotatable_bonds"]
                        ),
                        "time": args.time,
                        "correction_scale": correction_scale,
                        "v_cart_norm": cart_norm,
                        "v_4d_norm": raw_corr_norm,
                        "scaled_v_4d_norm": _mean_atom_norm(scaled_correction),
                        "v_final_norm": _mean_atom_norm(v_final),
                        "v_4d_to_v_cart_ratio": raw_corr_norm / max(cart_norm, 1.0e-12),
                        "q_norm": _mean_bond_norm(output["q_b"]),
                        "q_star_norm": _mean_bond_norm(q_star, q_valid),
                        "corr_to_target_ratio": _mean_atom_norm(scaled_correction)
                        / max(target_norm, 1.0e-12),
                        "residual_explained_by_Jq": explained_scaled,
                        "residual_explained_by_raw_Jq": explained_raw,
                        "num_valid_q_star_bonds": int(q_valid.sum()),
                        "q_star_nan_count": int(q_stats["q_star_nan_count"]),
                    }
                )

    summary = []
    keys = sorted({(row["checkpoint_path"], row["step"], row["mode"]) for row in rows})
    for checkpoint_path, step, mode in keys:
        selected = [row for row in rows if row["checkpoint_path"] == checkpoint_path]
        output = {
            "checkpoint_name": Path(checkpoint_path).name,
            "checkpoint_path": checkpoint_path,
            "step": step,
            "mode": mode,
            "num_samples": len(selected),
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            output[f"{metric}_mean"] = float(np.mean(values))
            output[f"{metric}_median"] = float(np.median(values))
        summary.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "branch_contribution_per_sample.csv", rows)
    _write(args.output_dir / "branch_contribution_summary.csv", summary)
    print(f"Wrote branch diagnostics for {len(rows)} checkpoint/sample pairs.")


if __name__ == "__main__":
    main()
