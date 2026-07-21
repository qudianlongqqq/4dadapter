#!/usr/bin/env python
"""Build the strict V8 Full versus matched D1-only 12.5K paired report."""

# ruff: noqa: E402

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np

from etflow.ecir.v8_validation_cache import ISOLATION, atomic_json, iter_prediction_records
from scripts.evaluate_ecir_mvr_v8_prediction_cache import _memberships
from scripts.report_mcvr_v8_12p5k_freeze_and_baselines import (
    COHORTS,
    HIGHER_IS_BETTER,
    METRICS,
    REPORT_DIR,
    SET_METRICS,
    TOLERANCE,
    applicable,
    direction,
    paired_statistics,
    read_json,
    sha256,
)


V8_RUN = ROOT / "diagnostics/ecir_mvr/v8_full_v1/formal_large_200k/full_seed43"
MATCHED_RUN = (
    ROOT / "diagnostics/ecir_mvr/v8_full_v1/matched_d1_formal_large_12p5k/d1_seed43"
)


def membership_map() -> dict[str, dict[str, bool]]:
    result = {}
    source_manifest = (
        ROOT
        / "diagnostics/ecir_mvr/validation_cache/formal_large_seed43/source/prediction_manifest.json"
    )
    for row in iter_prediction_records(source_manifest):
        item = row["item"]
        result[str(row["sample_id"])] = {
            **_memberships(item),
            "chirality_applicable": bool(item.protected_chirality_constraint_index.numel()),
        }
    return result


def timing(run: Path, evaluation: dict[str, Any]) -> dict[str, Any]:
    status = read_json(run / "status.json")
    train_rows = [
        json.loads(line)
        for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "reported_process_elapsed_seconds": status.get("elapsed_seconds"),
        "seconds_per_step_at_last_training_log": train_rows[-1].get("seconds_per_step"),
        "finite_loss_spike_count": status.get("finite_loss_spike_count", 0),
        "consecutive_solver_failure_steps": status.get(
            "consecutive_solver_failure_steps", 0
        ),
        "evaluation_timing": evaluation.get("timing"),
    }


def main() -> None:
    v8_report_path = V8_RUN / "validation_cache/step012500/full/evaluation.json"
    d1_report_path = MATCHED_RUN / "validation_cache/step012500/full/evaluation.json"
    v8_report, d1_report = read_json(v8_report_path), read_json(d1_report_path)
    v8_status, d1_status = read_json(V8_RUN / "status.json"), read_json(MATCHED_RUN / "status.json")
    if v8_status["status"] != "MCVR_V8_FULL_V1_FORMAL_LARGE_12P5K_COMPLETED":
        raise RuntimeError("V8 Full 12.5K status changed")
    if d1_status["status"] != "MCVR_V8_MATCHED_D1_FORMAL_LARGE_12P5K_COMPLETED":
        raise RuntimeError("matched D1 12.5K is not completed")
    v8_rows, d1_rows = v8_report["per_record_metrics"], d1_report["per_record_metrics"]
    identity = [(row["record_index"], row["sample_id"]) for row in v8_rows]
    if [(row["record_index"], row["sample_id"]) for row in d1_rows] != identity:
        raise RuntimeError("V8/matched D1 record identity or order differs")
    memberships = membership_map()
    comparisons: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        comparisons[cohort] = {}
        for metric in METRICS:
            selected = [
                index
                for index, row in enumerate(v8_rows)
                if memberships[row["sample_id"]][cohort]
                and applicable(metric, memberships[row["sample_id"]])
            ]
            values = np.asarray(
                [float(v8_rows[index][metric]) - float(d1_rows[index][metric]) for index in selected],
                dtype=np.float64,
            )
            stats = paired_statistics(values, draws=10000)
            tolerance = TOLERANCE.get(metric, 1.0e-12)
            if metric in HIGHER_IS_BETTER:
                wins, losses = int((values > tolerance).sum()), int((values < -tolerance).sum())
            else:
                wins, losses = int((values < -tolerance).sum()), int((values > tolerance).sum())
            ties = int(len(values) - wins - losses)
            low, high = stats["bootstrap_ci95_low"], stats["bootstrap_ci95_high"]
            significant = low is not None and (low > 0 or high < 0)
            effect = direction(metric, stats["paired_mean_difference"])
            result = {
                **stats,
                "win_count": wins,
                "tie_count": ties,
                "loss_count": losses,
                "applicable_record_count": len(selected),
                "significance_status": (
                    f"SIGNIFICANT_{effect}" if significant else "NOT_SIGNIFICANT"
                ),
                "effect_direction": effect,
            }
            comparisons[cohort][metric] = result
            csv_rows.append({"cohort": cohort, "metric": metric, **result})
    summaries = {
        "V8 Full 12.5K": {**v8_report["metrics"], **v8_report["set_metrics"]},
        "Matched D1-only 12.5K": {**d1_report["metrics"], **d1_report["set_metrics"]},
    }
    primary = comparisons["natural"]["weighted_bac_delta"]
    conclusion = (
        "V8 constraints provide improvement beyond continued D1 training."
        if primary["significance_status"] == "SIGNIFICANT_V8_12P5K_better"
        else "V8 Full did not establish a significant primary advantage over matched D1-only."
    )
    output = {
        "schema_version": "mcvr-v8-full-vs-matched-d1-12p5k-v1",
        "status": "COMPLETED",
        "matching_contract": {
            "seed": 43,
            "records": 10000,
            "record_identity_and_order_equal": True,
            "effective_batch": 64,
            "total_exposure": 800000,
            "optimizer_steps": 12500,
            "planned_scheduler_horizon": 200000,
            "scheduler_provenance": "original_200k_schedule",
            "evaluator_equal": v8_report["evaluator_semantics"]
            == d1_report["evaluator_semantics"],
            "checkpoint_selection_rule": "exact_step12500",
        },
        "method_summaries": summaries,
        "paired_comparisons": comparisons,
        "bootstrap_draws": 10000,
        "training_stability_and_timing": {
            "V8 Full 12.5K": timing(V8_RUN, v8_report),
            "Matched D1-only 12.5K": timing(MATCHED_RUN, d1_report),
        },
        "checkpoint_sha256": {
            "V8 Full 12.5K": sha256(V8_RUN / "checkpoints/step012500.ckpt"),
            "Matched D1-only 12.5K": sha256(
                MATCHED_RUN / "checkpoints/step012500.ckpt"
            ),
        },
        "full_evaluation_sha256": {
            "V8 Full 12.5K": sha256(v8_report_path),
            "Matched D1-only 12.5K": sha256(d1_report_path),
        },
        "conclusion": conclusion,
        **ISOLATION,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT_DIR / "V8_FULL_VS_MATCHED_D1_12P5K.json", output)
    columns = sorted({key for row in csv_rows for key in row})
    with (REPORT_DIR / "V8_FULL_VS_MATCHED_D1_12P5K.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    summary_rows = []
    for method, values in summaries.items():
        summary_rows.append(
            "| " + " | ".join(
                [method]
                + [
                    f"{values[key]:.8g}"
                    for key in (
                        "accepted", "weighted_bac_delta", "bond_delta", "angle_delta",
                        "active_angle_delta", "ring_delta", "clash_delta",
                        "chirality_preserved", "mean_displacement", "rmsd", "MAT_P",
                        "MAT_R", "COV_P", "COV_R", "target_loss",
                    )
                ]
            ) + " |"
        )
    paired_rows = []
    for metric in METRICS:
        value = comparisons["natural"][metric]
        paired_rows.append(
            f"| {metric} | {value['paired_mean_difference']:.8g} | "
            f"{value['median_difference']:.8g} | [{value['bootstrap_ci95_low']:.8g}, "
            f"{value['bootstrap_ci95_high']:.8g}] | {value['win_count']}/"
            f"{value['tie_count']}/{value['loss_count']} | "
            f"{value['applicable_record_count']} | {value['significance_status']} |"
        )
    markdown = "\n".join(
        [
            "# V8 Full 12.5K vs matched D1-only 12.5K",
            "",
            "Both methods match Seed43, exposure 800000, batch 64, the original 200K "
            "schedule provenance, exact step12500 selection, ordered 10K validation records, "
            "and frozen evaluator.",
            "",
            "| Method | Accept | Weighted BAC | Bond | Angle | Active angle | Ring | Clash | Chirality | Mean disp. | RMSD | MAT-P | MAT-R | COV-P | COV-R | Target loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *summary_rows,
            "",
            "## Natural-cohort paired results (V8 minus matched D1)",
            "",
            "| Metric | Mean | Median | Bootstrap 95% CI | W/T/L | Applicable | Status |",
            "|---|---:|---:|---|---:|---:|---|",
            *paired_rows,
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
            "Cohort-specific results, timing, stability, cache hashes, and all 10,000-draw "
            "bootstrap results are retained in the JSON/CSV artifacts.",
        ]
    )
    (REPORT_DIR / "V8_FULL_VS_MATCHED_D1_12P5K.md").write_text(
        markdown + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETED", "conclusion": conclusion}, indent=2))


if __name__ == "__main__":
    main()
