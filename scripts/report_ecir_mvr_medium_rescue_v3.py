#!/usr/bin/env python
"""Close Rescue V3, report segmented timing, and optionally prepare seed43/44 commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import pandas as pd
import yaml

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.commons.run_timing import RunTiming, iso_now


def _fmt(value, digits=6):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _row(frame, group, method):
    values = frame[(frame.group == group) & (frame.method == method)]
    return None if values.empty else values.iloc[0]


def _write_seed_configs(config: dict, selection: dict) -> list[str]:
    commands = []
    for seed in (43, 44):
        generated = {
            **config,
            "experiment_name": f"ecir_mvr_medium_5k_500_run_a_seed{seed}_20k_replication",
            "seed": seed,
            "initialize_from_checkpoint": None,
            "resume_checkpoint": None,
            "provenance": {
                **config["provenance"],
                "replication_of_seed42_v3_selected_checkpoint": selection["selected_checkpoint"],
                "replication_of_seed42_v3_selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
                "training_from_scratch": True,
            },
            "output_dir": f"logs_ecir_mvr/medium/run_a_seed{seed}_20k_replication",
            "diagnostics_dir": f"diagnostics/ecir_mvr/medium/run_a_seed{seed}_20k_replication",
        }
        path = Path(f"configs/ecir_mvr_medium_5k_500_run_a_seed{seed}_20k.yaml")
        path.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
        commands.append(f"python scripts/train_ecir_mvr_medium_20k.py --config {path.as_posix()}")
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result_path = args.evaluation_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    metadata = json.loads((args.output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(args.evaluation_dir / "source_summary.csv")
    comparisons = pd.read_csv(args.selection.with_suffix(".csv"))
    intervals = pd.read_csv(args.output_dir / "timing.csv")
    raw_audit = json.loads(Path(config["provenance"]["raw_vs_clipped_audit"]).read_text(encoding="utf-8"))
    timing = RunTiming(args.output_dir)
    timing.mark("report_generation_start")
    commands = _write_seed_configs(config, selection) if result["gate"]["pass"] else []
    result.update({
        "checkpoint_selection": selection,
        "seed43_44_permitted": bool(result["gate"]["pass"]),
        "next_commands": commands,
        "next_command": commands[0] if commands else None,
        "100k_permitted": False,
    })
    atomic_json_save(result, result_path)
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    state_path = Path("reports/ecir_mvr/progressive_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "git_commit": current_commit,
        "current_stage": "MEDIUM_SEED42_RESCUE_V3_COMPLETE",
        "current_decision": result["decision"],
        "medium_rescue_v3_permitted": False,
        "medium_rescue_v3_started": True,
        "medium_rescue_v3_completed": bool(metadata["20k_completed"]),
        "medium_rescue_v3_training_status": metadata["status"],
        "medium_rescue_v3_completed_optimizer_steps": metadata["completed_steps"],
        "medium_rescue_v3_stop_reason": metadata.get("stop_reason"),
        "medium_rescue_v3_decision": result["decision"],
        "medium_rescue_v3_selected_checkpoint": selection["selected_checkpoint"],
        "medium_rescue_v3_selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "medium_rescue_v3_selected_step": selection["selected_step"],
        "seed43_44_permitted": bool(result["gate"]["pass"]),
        "seed43_started": False, "seed44_started": False,
        "20k_permitted": False,
        "20k_completed": bool(metadata["20k_completed"]),
        "100k_permitted": False, "100k_started": False,
        "test_records_read": 0, "next_command": commands[0] if commands else None,
        "next_commands": commands, "updated_at": iso_now(),
        "decision_reasons": [
            "V2 stop was audited as POST_CLIP_THRESHOLD_SELF_TRIGGER and trust clipping mathematics remained unchanged",
            f"V3 strictly resumed model, optimizer, RNG, sampler, and timing from step 2450 and ended with status {metadata['status']}",
            f"Gate 2 passed {sum(result['gate']['conditions'].values())} of {len(result['gate']['conditions'])} metric conditions",
            "Seed43/44 commands were generated only if Gate 2 passed; neither command was executed",
            "100k and test evaluation remained prohibited",
        ],
    })
    atomic_json_save(state, state_path)
    timing.mark("report_generation_end")
    timing.mark("pipeline_end", decision=result["decision"])
    final_timing = timing.finalize(
        completed_optimizer_steps=int(metadata["completed_steps"]), batch_size=8,
        active_optimizer_seconds=float(metadata["active_optimizer_seconds"]), interval_rows=None,
    )

    candidate, upstream = _row(summary, "all", "medium_accepted"), _row(summary, "all", "upstream")
    high, high_up = _row(summary, "rotatable_ge_6", "medium_accepted"), _row(summary, "rotatable_ge_6", "upstream")
    unseen, unseen_up = _row(summary, "unseen_update_scale_0.35", "medium_accepted"), _row(summary, "unseen_update_scale_0.35", "upstream")
    v3_intervals = intervals[intervals.step_end > 2450]
    lines = [
        "# MCVR Medium Seed42 Rescue V3 Final Report", "",
        f"Decision: **{result['decision']}**", "",
        f"Monitoring audit: **{raw_audit['decision']}**. Trust clipping mathematics was not changed.", "",
        "## Completion and segmented timing", "", "| Item | Value |", "|---|---:|",
        f"| Completed optimizer steps | {metadata['completed_steps']} / 20000 |",
        f"| Training status | {metadata['status']} |",
        f"| Stop reason | {metadata.get('stop_reason') or 'none'} |",
        f"| V3 added training wall seconds | {_fmt(final_timing['segment_v3']['training_wall_seconds'], 3)} |",
        f"| V3 added active optimizer seconds | {_fmt(final_timing['segment_v3']['active_optimizer_seconds'], 3)} |",
        f"| V2+V3 total training wall seconds | {_fmt(final_timing['training_wall_seconds_total'], 3)} |",
        f"| V2+V3 active optimizer seconds | {_fmt(final_timing['active_optimizer_seconds_total'], 3)} |",
        f"| V2+V3 validation seconds | {_fmt(final_timing['validation_seconds_total'], 3)} |",
        f"| V3 pipeline wall seconds | {_fmt(final_timing['pipeline_wall_seconds'], 3)} |",
        f"| Mean optimizer steps/s (cumulative) | {_fmt(final_timing['mean_optimizer_steps_per_second'], 4)} |",
        f"| Mean examples/s (cumulative) | {_fmt(final_timing['mean_examples_per_second'], 4)} |",
        f"| Estimated 100k active hours | {_fmt(final_timing['estimated_100k_active_hours'], 3)} |",
        f"| Resume checkpoint | {final_timing['resume_checkpoint']} |",
        f"| Resume step/reason | {final_timing['resume_step']} / {final_timing['resume_reason']} |",
        f"| Downtime seconds | {_fmt(final_timing['downtime_seconds'], 3)} |", "",
        "## Checkpoint selection", "",
        f"Formal-policy selected step: **{selection['selected_step']}** (`{selection['selected_checkpoint_sha256']}`).", "",
        f"Best overall step: **{selection['best_overall_step']}**; early: `{selection['best_overall_is_early']}`.", "",
        "| Step | Segment | Formal | Validity delta | RMSD delta | MAT-P | MAT-R | Identity |", "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparisons.sort_values("step").iterrows():
        lines.append(f"| {int(row.step)} | {row.segment} | {bool(row.formal_checkpoint)} | {_fmt(row.validity_delta)} | {_fmt(row.rmsd_delta)} | {_fmt(row.mat_p_delta)} | {_fmt(row.mat_r_delta)} | {_fmt(row.identity_fraction)} |")
    lines += [
        "", "## Final Gate metrics", "", "| Metric | Upstream | V3 accepted | Delta |", "|---|---:|---:|---:|",
        f"| Total validity | {_fmt(upstream.total_thresholded_validity_score)} | {_fmt(candidate.total_thresholded_validity_score)} | {_fmt(candidate.total_thresholded_validity_score-upstream.total_thresholded_validity_score)} |",
        f"| RMSD | {_fmt(upstream.aligned_RMSD)} | {_fmt(candidate.aligned_RMSD)} | {_fmt(candidate.aligned_RMSD-upstream.aligned_RMSD)} |",
        f"| MAT-P | {_fmt(upstream.MAT_P)} | {_fmt(candidate.MAT_P)} | {_fmt(candidate.MAT_P-upstream.MAT_P)} |",
        f"| MAT-R | {_fmt(upstream.MAT_R)} | {_fmt(candidate.MAT_R)} | {_fmt(candidate.MAT_R-upstream.MAT_R)} |",
        f"| COV-P | {_fmt(upstream.COV_P)} | {_fmt(candidate.COV_P)} | {_fmt(candidate.COV_P-upstream.COV_P)} |",
        f"| COV-R | {_fmt(upstream.COV_R)} | {_fmt(candidate.COV_R)} | {_fmt(candidate.COV_R-upstream.COV_R)} |", "",
        f"High-flex validity: `{_fmt(high_up.total_thresholded_validity_score)} -> {_fmt(high.total_thresholded_validity_score)}`.",
        f"Unseen validity: `{_fmt(unseen_up.total_thresholded_validity_score)} -> {_fmt(unseen.total_thresholded_validity_score)}`.",
        f"Clean identity: `{_fmt(result['clean_identity_fraction'])}`.",
        f"Gate conditions: **{sum(result['gate']['conditions'].values())}/{len(result['gate']['conditions'])}**.", "",
        f"Seed43/44 permitted: **{'yes' if result['gate']['pass'] else 'no'}**. Generated commands: `{commands}`.",
        "Seed43/44 were not executed. 100k and test evaluation were not run.", "",
        "## V3 interval timing", "", "| Step end | Interval seconds | Active seconds | Steps/s | Examples/s |", "|---:|---:|---:|---:|---:|",
    ]
    for _, row in v3_intervals.iterrows():
        lines.append(f"| {int(row.step_end)} | {_fmt(row.interval_seconds,3)} | {_fmt(row.active_optimizer_seconds,3)} | {_fmt(row.steps_per_second,4)} | {_fmt(row.examples_per_second,4)} |")
    report_path = Path("docs/MCVR_MEDIUM_SEED42_RESCUE_V3_REPORT.md")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    failed_conditions = [
        name for name, passed in result["gate"]["conditions"].items() if not passed
    ]
    Path("docs/MCVR_MEDIUM_SEED42_RESCUE_V3_GATE2.md").write_text(
        f"# MCVR Medium Seed42 Rescue V3 Gate 2\n\nDecision: **{result['decision']}**.\n\n"
        f"Conditions: {sum(result['gate']['conditions'].values())}/{len(result['gate']['conditions'])}.\n\n"
        f"Failed conditions: `{failed_conditions}`.\n\n"
        f"Training completed: {result['gate']['training_completed']}. Formal checkpoint: {selection['qualified_formal_checkpoint_exists']}.\n\n"
        "No test evaluation or 100k run was performed.\n", encoding="utf-8"
    )
    Path("docs/MCVR_MEDIUM_SEED42_RESCUE_V3_CHECKPOINT_SELECTION.md").write_text(
        f"# MCVR Medium Seed42 Rescue V3 Checkpoint Selection\n\nSelected formal step: **{selection['selected_step']}**.\n\n"
        f"Best overall step: **{selection['best_overall_step']}**.\n\n"
        f"Selected SHA256: `{selection['selected_checkpoint_sha256']}`.\n", encoding="utf-8"
    )
    print(json.dumps({"decision": result["decision"], "selected_step": selection["selected_step"], "commands": commands}, indent=2))


if __name__ == "__main__":
    main()
