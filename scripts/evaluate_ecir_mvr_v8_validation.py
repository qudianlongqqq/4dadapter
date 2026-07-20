#!/usr/bin/env python
"""Evaluate a V8 checkpoint on validation with frozen V7 BAC safety semantics."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch
from torch_geometric.data import Batch

from etflow.ecir.bac_safety import BACSafetyConfig, select_safe_bac_proposal
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.mcvr_v8_full import MCVRV8FullRefiner
from etflow.ecir.mvr_dataset import _load_record_and_coordinates
from etflow.ecir.v8_constraint_normalization import FrozenResidualScales
from scripts.train_ecir_mvr_v8 import ISOLATION, _assert_manifest, _real_dataset


def _model(checkpoint: dict, device: torch.device) -> MCVRV8FullRefiner:
    config = checkpoint["resolved_config"]
    constraint = dict(config["constraint_layer"])
    for key in ("frozen_scales", "frozen_scales_sha256", "use_frozen_scales"):
        constraint.pop(key, None)
    unroll = int(constraint.pop("unroll_steps"))
    model = MCVRV8FullRefiner.from_d1_checkpoint(
        config["model"]["d1_checkpoint"],
        expected_sha256=config["model"]["d1_checkpoint_sha256"],
        error_state=config["error_state"],
        constraint_layer=constraint,
        residual_scales=FrozenResidualScales.from_mapping(checkpoint["residual_scales"]),
        unroll_steps=unroll,
        step_embedding_enabled=config["model"]["step_embedding_enabled"],
        error_state_enabled=config["error_state"]["enabled"],
        train_d1_backbone=config["model"]["train_d1_backbone"],
        train_d1_head=config["model"]["train_d1_head"],
        max_cumulative_atom_displacement=config["safety"]["max_atom_displacement"],
        max_cumulative_graph_rms=config["safety"]["graph_rms_limit"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-sources", type=Path, required=True)
    parser.add_argument("--val-targets", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path)
    parser.add_argument("--target-cache-root", type=Path)
    parser.add_argument("--validity-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    source_frame = _assert_manifest(args.val_sources.resolve(), "val")
    _assert_manifest(args.val_targets.resolve(), "val")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "mcvr-v8-full-v1-checkpoint-v1":
        raise RuntimeError("V8 validation checkpoint schema changed")
    for key, value in ISOLATION.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"V8 checkpoint isolation field changed: {key}")
    device = torch.device(args.device)
    model = _model(checkpoint, device)
    validity = ChemicalValidity(args.validity_statistics)
    dataset = _real_dataset(
        args.val_sources.resolve(),
        args.val_targets.resolve(),
        validity,
        source_cache_root=args.source_cache_root,
        target_cache_root=args.target_cache_root,
        source_identity="validation-only-not-used-for-training-scales",
    )
    count = min(len(dataset), int(args.max_records or len(dataset)))
    safety = BACSafetyConfig(
        max_atom_displacement=0.12,
        max_molecule_rms_displacement=0.06,
        enable_backtracking=True,
        objective_mode="weighted_thresholded_validity",
    )
    sums: dict[str, float] = defaultdict(float)
    reasons: Counter[str] = Counter()
    with torch.no_grad():
        for index in range(count):
            item = dataset[index]
            row = dataset.sources.iloc[index]
            target_row = dataset.targets.loc[row.sample_id]
            record, _ = _load_record_and_coordinates(
                row,
                dataset_index=index,
                target_path=Path(target_row.target_cache_path),
            )
            batch = Batch.from_data_list([item]).to(device)
            source = batch.x_input
            output = model(batch, source, source.new_tensor([0.5]))
            raw = output["x_final"].detach().cpu()
            deployed, decision = select_safe_bac_proposal(
                item.x_input,
                raw - item.x_input,
                record,
                validity,
                safety,
            )
            before = (
                decision["attempts"][0]["before"]
                if decision.get("attempts")
                else validity.evaluate(item.x_input, record)
            )
            after = validity.evaluate(deployed, record, baseline_coordinates=item.x_input)
            sums["accepted"] += float(decision["accepted"])
            for reason in decision.get("reasons", ()):
                reasons[str(reason)] += 1
            sums["bond_delta"] += after["bond_outlier_rate"] - before["bond_outlier_rate"]
            sums["angle_delta"] += after["angle_outlier_rate"] - before["angle_outlier_rate"]
            sums["clash_delta"] += after["clash_penetration"] - before["clash_penetration"]
            sums["ring_delta"] += after["ring_bond_outlier_rate"] - before["ring_bond_outlier_rate"]
            sums["weighted_bac_delta"] += (
                after["total_thresholded_validity_score"]
                - before["total_thresholded_validity_score"]
            )
            displacement = torch.linalg.vector_norm(deployed - item.x_input, dim=-1)
            sums["mean_displacement"] += float(displacement.mean())
            sums["max_atom_displacement"] += float(displacement.max())
            sums["target_loss"] += float(
                torch.nn.functional.smooth_l1_loss(deployed, item.x_target)
            )
            sums["chirality_preserved"] += float(after["chirality_preserved"])
            sums["solver_failure_count"] += float(
                sum(step["solver_failure"].sum() for step in output["step_outputs"])
            )
            sums["confidence_mean"] += float(output["bounded_prior_confidence"].mean())
            sums["solver_bond_contribution"] += float(
                output["step_outputs"][-1]["solver_bond_contribution"].mean()
            )
            sums["solver_angle_contribution"] += float(
                output["step_outputs"][-1]["solver_angle_contribution"].mean()
            )
            if (index + 1) % 50 == 0 or index + 1 == count:
                print(f"validation_progress={index + 1}/{count}", flush=True)
    result = {
        "schema_version": "mcvr-v8-validation-report-v1",
        "records": count,
        "metrics": {key: value / max(count, 1) for key, value in sums.items()},
        "rejection_reasons": dict(reasons),
        "evaluator_semantics": "frozen_v7_bac_safety_weighted_thresholded_validity",
        "validation_only": True,
        "source_split_counts": source_frame.split.value_counts().to_dict(),
        "rmsd_mat_cov_status": "not_available_without_reference-ensemble binding",
        **ISOLATION,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
