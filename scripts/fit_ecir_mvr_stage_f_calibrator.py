#!/usr/bin/env python
"""Fit and select the small Stage F calibrator using train/internal-check only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.ecir.feature_conditioned_confidence import (
    FeatureConditionedConfidenceCalibrator, calibrator_identity_payload,
    dataframe_feature_tensors, internal_check_priority, stage_f_loss,
    validate_stage_f_frame, validate_stage_f_manifest,
)


def _model_config(config):
    value = config["calibrator"]
    return {name: value[name] for name in (
        "hidden_dim", "num_layers", "bond_type_embedding_dim",
        "element_pair_embedding_dim", "element_pair_buckets", "time_embedding_dim",
        "max_bias", "epsilon", "dropout",
    )}


def selection_metrics(frame: pd.DataFrame, confidence: np.ndarray, *, threshold: float) -> dict[str, float]:
    wrong = frame.wrong_sign.to_numpy(bool)
    false = frame.zero_target.to_numpy(bool) | frame.already_valid_unsafe.to_numpy(bool)
    beneficial = frame.beneficial.to_numpy(bool)
    active = confidence >= float(threshold)
    dangerous = wrong | false
    tp = int((active & beneficial).sum())
    fp = int((active & dangerous).sum())
    return {
        "wrong_sign_activation": float(confidence[wrong].mean()) if wrong.any() else 0.0,
        "false_positive_activation": float(confidence[false].mean()) if false.any() else 0.0,
        "optimal_scale_mae": float(np.abs(confidence - frame.optimal_scale.to_numpy(float)).mean()),
        "beneficial_correction_capture": float((active & beneficial).sum() / max(beneficial.sum(), 1)),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(beneficial.sum(), 1),
        "auprc": float(average_precision_score(beneficial.astype(int), confidence)),
        "abstention_fraction": float((~active).mean()),
    }


def molecule_grouped_batch_indices(
    molecule_ids: torch.Tensor, *, batch_size: int, generator: torch.Generator,
    max_molecules: int = 64,
) -> torch.Tensor:
    groups = [
        torch.nonzero(molecule_ids == value, as_tuple=False).reshape(-1)
        for value in torch.unique(molecule_ids, sorted=True)
    ]
    molecule_count = min(int(max_molecules), len(groups), int(batch_size))
    selected_groups = torch.randperm(len(groups), generator=generator)[:molecule_count]
    rows_per_molecule = max(1, int(batch_size) // molecule_count)
    selected = []
    for group_index in selected_groups.tolist():
        group = groups[group_index]
        local = torch.randint(0, len(group), (rows_per_molecule,), generator=generator)
        selected.append(group[local])
    return torch.cat(selected)[:int(batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ecir_mvr_stage_f_feature_confidence.yaml"))
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = args.input_dir or Path(config["output_dir"])
    output = args.output_dir or source
    manifest = json.loads((source / "calibration_manifest.json").read_text(encoding="utf-8"))
    validate_stage_f_manifest(manifest)
    frame = pd.read_parquet(source / "calibration_dataset.parquet")
    validate_stage_f_frame(frame, manifest)
    fit, check = frame[frame.split.eq("fit")].copy(), frame[frame.split.eq("internal_check")].copy()
    model_config = _model_config(config)
    calibrator = FeatureConditionedConfidenceCalibrator(**model_config)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        calibrator.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    fit_features = dataframe_feature_tensors(fit)
    molecule_ids = torch.as_tensor(fit.molecule_code.to_numpy(np.int64, copy=True))
    optimal = torch.as_tensor(fit.optimal_scale.to_numpy(np.float64, copy=True))
    weight = torch.as_tensor(fit.scale_weight.to_numpy(np.float64, copy=True))
    wrong = torch.as_tensor(fit.wrong_sign.to_numpy(bool, copy=True))
    false = torch.as_tensor((fit.zero_target | fit.already_valid_unsafe).to_numpy(bool, copy=True))
    check_features = dataframe_feature_tensors(check)
    max_steps = int(args.max_steps or training["optimizer_steps"])
    checkpoint_steps = sorted({step for step in training["checkpoint_steps"] if step <= max_steps} | {max_steps})
    generator = torch.Generator().manual_seed(int(config["seed"]))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    batch_size = min(int(training["batch_size"]), len(fit))
    for step in range(1, max_steps + 1):
        indices = molecule_grouped_batch_indices(
            molecule_ids, batch_size=batch_size, generator=generator,
        )
        batch_features = {name: value[indices] for name, value in fit_features.items()}
        confidence = calibrator(batch_features)
        loss, parts = stage_f_loss(
            confidence, optimal_scale=optimal[indices], scale_weight=weight[indices],
            wrong_sign=wrong[indices], false_positive=false[indices],
            molecule_ids=molecule_ids[indices],
            lambda_wrong_sign=training["lambda_wrong_sign"],
            lambda_false_positive=training["lambda_false_positive"],
            lambda_overactivation=training["lambda_overactivation"],
            lambda_rank=training["lambda_rank"], smooth_l1_beta=training["smooth_l1_beta"],
            rank_margin=training["rank_margin"],
            max_rank_pairs_per_molecule=training["max_rank_pairs_per_molecule"],
        )
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        evaluate = step % int(training["internal_check_interval"]) == 0 or step in checkpoint_steps
        if evaluate:
            with torch.inference_mode():
                check_confidence = calibrator(check_features).cpu().numpy()
            metrics = selection_metrics(
                check, check_confidence,
                threshold=float(training["reporting_activation_threshold"]),
            )
            row = {"step": step, "loss": float(loss.detach()), **{name: float(value.detach()) for name, value in parts.items()}, **metrics}
            history.append(row)
            if best is None or internal_check_priority(metrics) < internal_check_priority(best["metrics"]):
                best = {"step": step, "metrics": metrics, "state": {name: value.detach().cpu().clone() for name, value in calibrator.state_dict().items()}}
        if step in checkpoint_steps:
            torch.save({"step": step, "calibrator_state_dict": calibrator.state_dict(), "model_config": model_config}, checkpoint_dir / f"step{step:06d}.ckpt")
    calibrator.load_state_dict(best["state"], strict=True)
    best_path = checkpoint_dir / "best_internal_check.ckpt"
    torch.save({"step": best["step"], "calibrator_state_dict": calibrator.state_dict(), "model_config": model_config}, best_path)
    payload = calibrator_identity_payload(
        calibrator, model_config=model_config, checkpoint_sha256=config["checkpoint"]["sha256"],
        training_molecule_identity_sha256=manifest["training_molecule_identity_sha256"],
        manifest_identity_sha256=manifest["manifest_identity_sha256"], selected_step=best["step"],
        selection_metrics=best["metrics"], smoke=bool(manifest.get("smoke")),
    )
    payload["selected_checkpoint"] = best_path.as_posix()
    atomic_json_save(payload, output / "calibrator.json")
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    print(json.dumps({
        "status": "SMOKE_COMPLETE" if manifest.get("smoke") else "COMPLETE",
        "optimizer_steps": max_steps, "selected_step": best["step"],
        "selection_priority": training["selection_priority"],
        "selection_metrics": best["metrics"], "validation_used": False,
        "test_records_read": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
