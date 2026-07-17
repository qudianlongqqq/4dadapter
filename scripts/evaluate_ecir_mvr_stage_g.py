#!/usr/bin/env python
"""Evaluate the selected Stage G calibrator once on validation data."""

from __future__ import annotations

import argparse
import json
import math
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

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.ecir.bounded_residual_confidence import (
    STAGE_G_METHOD,
    load_stage_g_calibrator,
    stage_g_decision,
    verify_stage_f_identity,
)
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.confidence_calibration import molecule_paired_bootstrap, strict_load_frozen_model
from etflow.ecir.failure_attribution import bond_observations, relative_improvement
from etflow.ecir.geometry import bond_lengths
from etflow.ecir.run_a_evaluation import (
    BOOTSTRAP_METRICS,
    build_clean_control_items,
    method_rows,
    summarize_groups,
)
from scripts.evaluate_ecir_mvr_stage_e0 import load_validation_items
from scripts.evaluate_ecir_mvr_stage_f import activation_audit, infer_stage_f_mode


METHODS = {
    "d1_b_original_confidence": "original",
    "sign_safe_only": "sign_safe",
    STAGE_G_METHOD: "feature",
}


@torch.inference_mode()
def infer_stage_g_mode(
    model,
    items,
    validity,
    *,
    mode: str,
    calibrator,
    device: torch.device,
    inference,
):
    """Run deployment inference without reading target/reference/source identity features."""

    return infer_stage_f_mode(
        model,
        items,
        validity,
        mode=mode,
        calibrator=calibrator,
        e0_calibrator=None,
        device=device,
        inference=inference,
    )


def beneficial_activation_recall(details: pd.DataFrame, items, *, threshold: float) -> float:
    targets = {}
    for item in items:
        prepared = item.get("prepared_validity")
        if prepared is None:
            raise RuntimeError("Stage G beneficial audit requires prepared validity metadata")
        targets[str(item["row"].sample_id)] = bond_lengths(
            item["minimal_target"], prepared["bonds"]
        ).numpy()
    beneficial, active = [], []
    for row in details.itertuples(index=False):
        residual = float(
            targets[str(row.record_id)][int(row.bond_index)] - row.current_bond_length
        )
        improves = (
            bool(row.sign_safe_mask)
            and abs(residual) > 1.0e-4
            and np.sign(row.predicted_residual) == np.sign(residual)
        )
        beneficial.append(improves)
        active.append(float(row.confidence) >= float(threshold))
    beneficial_array = np.asarray(beneficial, dtype=bool)
    active_array = np.asarray(active, dtype=bool)
    return float((active_array & beneficial_array).sum() / max(beneficial_array.sum(), 1))


def _transition(items, coordinates, validity, method):
    repaired = broken = total = 0
    for item, candidate in zip(items, coordinates):
        before = bond_observations(validity, item["input"], item["record"]).outlier.to_numpy(bool)
        after = bond_observations(validity, candidate, item["record"]).outlier.to_numpy(bool)
        repaired += int((before & ~after).sum())
        broken += int((~before & after).sum())
        total += len(before)
    return {
        "method": method,
        "bonds": total,
        "repaired_bonds": repaired,
        "newly_broken_bonds": broken,
        "broken_to_repaired_ratio": broken / max(repaired, 1),
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    pd.read_csv(temporary)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ecir_mvr_stage_g_bounded_residual.yaml")
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--bootstrap-draws", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    verify_stage_f_identity(config)
    source = args.input_dir or Path(config["output_dir"])
    output_dir = args.output_dir or source
    fit_result = json.loads((source / "fit_result.json").read_text(encoding="utf-8"))
    if fit_result.get("decision") == "STAGE_G_COLLAPSED":
        raise RuntimeError("Stage G has no non-collapsed checkpoint; validation is not permitted")
    payload = json.loads((source / "calibrator.json").read_text(encoding="utf-8"))
    if payload["checkpoint_sha256"] != config["checkpoint"]["sha256"]:
        raise RuntimeError("Stage G calibrator checkpoint identity changed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage G requested CUDA but CUDA is unavailable")
    calibrator = load_stage_g_calibrator(
        payload["selected_checkpoint"], payload, device=device
    )
    model, checkpoint = strict_load_frozen_model(
        config["checkpoint"]["path"],
        expected_sha256=config["checkpoint"]["sha256"],
        device=device,
    )
    if checkpoint["frozen_identities"] != config["frozen_identities"]:
        raise RuntimeError("Stage G frozen identities changed")
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = load_validation_items(
        config["data"]["val_sources"],
        config["data"]["val_targets"],
        validity,
        limit_records=args.limit_records,
    )
    for item in items:
        item["prepared_validity"] = validity._prepare(item["record"])
    coordinates, metadata, details = {}, {}, {}
    for label, mode in METHODS.items():
        values, extras, detail = infer_stage_g_mode(
            model,
            items,
            validity,
            mode=mode,
            calibrator=calibrator,
            device=device,
            inference=config["inference"],
        )
        coordinates[label] = values
        metadata[label] = extras
        details[label] = detail
    methods = {
        "upstream": [item["input"] for item in items],
        **coordinates,
        "minimal_target": [item["minimal_target"] for item in items],
    }
    rows = method_rows(items, methods, validity, metadata)
    summary, molecules = summarize_groups(rows, items, methods)
    transitions = [_transition(items, coordinates[name], validity, name) for name in METHODS]
    threshold = float(config["training"]["reporting_activation_threshold"])
    activation = {}
    for name in METHODS:
        activation[name] = {
            **activation_audit(details[name], items),
            "beneficial_activation_recall": beneficial_activation_recall(
                details[name], items, threshold=threshold
            ),
        }
    gate = config["gate"]
    draws = int(args.bootstrap_draws or gate["bootstrap_draws"])
    seed = int(args.seed if args.seed is not None else gate["bootstrap_seed"])
    all_molecules = molecules[molecules.group.eq("all")]
    bootstrap_upstream = molecule_paired_bootstrap(
        all_molecules,
        candidate=STAGE_G_METHOD,
        baseline="upstream",
        metrics=BOOTSTRAP_METRICS,
        draws=draws,
        seed=seed,
    )
    clean_items = build_clean_control_items(items, validity, limit=min(20, len(items)))
    clean_identity = math.nan
    if clean_items:
        for item in clean_items:
            item["prepared_validity"] = validity._prepare(item["record"])
        clean_values, _, _ = infer_stage_g_mode(
            model,
            clean_items,
            validity,
            mode="feature",
            calibrator=calibrator,
            device=device,
            inference=config["inference"],
        )
        clean_identity = float(
            np.mean(
                [
                    torch.equal(torch.as_tensor(value), torch.as_tensor(item["input"]))
                    for value, item in zip(clean_values, clean_items)
                ]
            )
        )
    all_rows = summary[summary.group.eq("all")].set_index("method")
    upstream = all_rows.loc["upstream"]
    candidate = all_rows.loc[STAGE_G_METHOD]
    target = all_rows.loc["minimal_target"]
    relative = relative_improvement(
        upstream.bond_outlier_rate, candidate.bond_outlier_rate
    )
    target_relative = relative_improvement(
        upstream.bond_outlier_rate, target.bond_outlier_rate
    )
    recovery = relative / max(target_relative, 1.0e-12)
    transition = {row["method"]: row for row in transitions}
    cancellation = float(
        np.mean([value["bond_cancellation_fraction"] for value in metadata[STAGE_G_METHOD]])
    )
    candidate_activation = activation[STAGE_G_METHOD]
    original_activation = activation["d1_b_original_confidence"]
    sign_safe_activation = activation["sign_safe_only"]
    criteria = {
        "01_abstention_fraction_lt_0p90": candidate_activation["abstention_fraction"]
        < gate["abstention_fraction_max_exclusive"],
        "02_beneficial_recall_ge_80pct_sign_safe_only": candidate_activation[
            "beneficial_activation_recall"
        ]
        >= gate["beneficial_recall_relative_floor"]
        * sign_safe_activation["beneficial_activation_recall"],
        "03_model_to_target_recovery_ge_0p25": recovery
        >= gate["model_to_target_recovery_min"],
        "04_bond_relative_improvement_ge_15pct": relative
        >= gate["bond_relative_improvement_min"],
        "05_newly_broken_le_177": transition[STAGE_G_METHOD]["newly_broken_bonds"]
        <= gate["newly_broken_max"],
        "06_cancellation_ratio_le_0p30": cancellation <= gate["cancellation_ratio_max"],
        "07_wrong_sign_activation_below_d1b": candidate_activation["wrong_sign_activation"]
        < original_activation["wrong_sign_activation"],
        "08_false_positive_activation_below_d1b": candidate_activation[
            "false_positive_activation"
        ]
        < original_activation["false_positive_activation"],
        "09_rmsd_noninferior": bootstrap_upstream["aligned_RMSD"]["mean"]
        <= gate["rmsd_mean_delta_max"]
        and bootstrap_upstream["aligned_RMSD"]["ci95_high"]
        <= gate["rmsd_ci_upper_max"],
        "10_mat_p_mat_r_noninferior": all(
            bootstrap_upstream[name]["mean"] <= gate["mat_mean_delta_max"]
            and bootstrap_upstream[name]["ci95_high"] <= gate["mat_ci_upper_max"]
            for name in ("MAT_P", "MAT_R")
        ),
        "11_cov_p_cov_r_noninferior": candidate.COV_P
        >= upstream.COV_P - gate["cov_absolute_drop_max"]
        and candidate.COV_R >= upstream.COV_R - gate["cov_absolute_drop_max"],
        "12_clean_identity_ge_0p90": math.isfinite(clean_identity)
        and clean_identity >= gate["clean_identity_fraction_min"],
        "13_validation_only_test_zero": int(args.limit_records or len(items)) == len(items)
        and int(config["test_records_read"]) == 0,
    }
    criteria = {name: bool(value) for name, value in criteria.items()}
    collapsed = candidate_activation["abstention_fraction"] >= float(
        config["training"]["collapsed_abstention_fraction"]
    )
    harms = (
        candidate.total_thresholded_validity_score
        > all_rows.loc["d1_b_original_confidence"].total_thresholded_validity_score
        or bootstrap_upstream["aligned_RMSD"]["ci95_high"] > gate["rmsd_ci_upper_max"]
        or transition[STAGE_G_METHOD]["newly_broken_bonds"] > gate["newly_broken_max"]
    )
    decision = (
        "STAGE_G_SMOKE_COMPLETE"
        if args.limit_records
        else stage_g_decision(criteria, collapsed=collapsed, harms=harms)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_save(
        {
            "schema_version": "ecir-mvr-stage-g-bootstrap-v1",
            "draws": draws,
            "seed": seed,
            "vs_upstream": bootstrap_upstream,
            "paired_methods": list(METHODS),
            "test_records_read": 0,
        },
        output_dir / "bootstrap.json",
    )
    _atomic_csv(summary, output_dir / "method_summary.csv")
    _atomic_csv(pd.DataFrame(transitions), output_dir / "bond_transition.csv")
    _atomic_csv(
        pd.DataFrame([{"method": name, **value} for name, value in activation.items()]),
        output_dir / "activation_summary.csv",
    )
    result = {
        "schema_version": "ecir-mvr-stage-g-validation-v1",
        "decision": decision,
        "stage_f_formal_decision_unchanged": "STAGE_F_HARMS",
        "smoke": bool(args.limit_records),
        "validation_only": True,
        "validation_records_read": len(items),
        "test_records_read": 0,
        "neural_training_run": False,
        "calibrator_training_only": True,
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "calibrator_identity_sha256": payload["calibrator_identity_sha256"],
        "selected_step": payload["selected_step"],
        "methods": list(METHODS),
        "metrics": {
            "model_to_target_recovery": recovery,
            "bond_relative_improvement": relative,
            "newly_broken_bonds": transition[STAGE_G_METHOD]["newly_broken_bonds"],
            "cancellation_ratio": cancellation,
            "clean_identity_fraction": clean_identity,
            "abstention_fraction": candidate_activation["abstention_fraction"],
            "beneficial_repair_recall": candidate_activation[
                "beneficial_activation_recall"
            ],
            "sign_safe_only_beneficial_repair_recall": sign_safe_activation[
                "beneficial_activation_recall"
            ],
        },
        "activation_audit": activation,
        "criteria": criteria,
        "pass": decision == "STAGE_G_BOUNDED_RESIDUAL_PASS",
        "formal_training_permitted": False,
        "stage_g_100k_permitted": False,
        "next_command": None,
        "next_commands": [],
    }
    atomic_json_save(result, output_dir / "validation_result.json")
    print(
        json.dumps(
            {
                "decision": decision,
                "smoke": bool(args.limit_records),
                "validation_records_read": len(items),
                "test_records_read": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
