#!/usr/bin/env python
"""Close the final Medium Seed42 training-schedule confirmation."""

from __future__ import annotations

import argparse
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


FAILURE_BOUNDARY = (
    "模型有统计显著且精度非劣的中等有效性，但未达到预注册10%核心改善门槛。"
)


def _fmt(value, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def _write_seed_configs(config: dict, selection: dict) -> list[str]:
    commands = []
    for seed in (43, 44):
        generated = {
            **config,
            "experiment_name": f"ecir_mvr_medium_5k_500_run_a_seed{seed}_schedule_v4_10k",
            "seed": seed,
            "initialize_from_checkpoint": None,
            "resume_checkpoint": None,
            "provenance": {
                **config["provenance"],
                "replication_of_seed42_schedule_v4_checkpoint": selection["selected_checkpoint"],
                "replication_of_seed42_schedule_v4_checkpoint_sha256": selection["selected_checkpoint_sha256"],
                "training_from_scratch": True,
            },
            "output_dir": f"logs_ecir_mvr/medium/run_a_seed{seed}_schedule_v4_10k",
            "diagnostics_dir": f"diagnostics/ecir_mvr/medium/run_a_seed{seed}_schedule_v4",
        }
        path = Path(f"configs/ecir_mvr_medium_5k_500_run_a_seed{seed}_schedule_v4_10k.yaml")
        path.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
        commands.append(f"python scripts/train_ecir_mvr_medium_20k.py --config {path.as_posix()}")
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result_path = args.diagnostics_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    metadata = json.loads((args.output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(args.diagnostics_dir / "source_summary.csv")
    comparisons = pd.read_csv(args.selection.with_suffix(".csv"))
    lr_history = pd.read_csv(args.diagnostics_dir / "lr_history.csv")
    timing = RunTiming(args.output_dir)
    timing.mark("report_generation_start")

    passed = result["decision"] == "MEDIUM_SEED42_SCHEDULE_V4_PASS"
    commands = _write_seed_configs(config, selection) if passed else []
    result.update({
        "checkpoint_selection": selection,
        "seed43_44_permitted": passed,
        "next_commands": commands,
        "next_command": commands[0] if commands else None,
        "100k_permitted": False,
        "100k_started": False,
        "rescue_v5_permitted": False,
        "gate_adjustment_permitted": False,
        "final_boundary": None if passed else FAILURE_BOUNDARY,
    })
    atomic_json_save(result, result_path)
    comparisons.to_csv(args.diagnostics_dir / "checkpoint_comparison.csv", index=False)

    state_path = Path("reports/ecir_mvr/progressive_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "current_stage": "MEDIUM_SEED42_SCHEDULE_V4_COMPLETE",
        "current_decision": result["decision"],
        "medium_schedule_v4_config": args.config.as_posix(),
        "medium_schedule_v4_started": True,
        "medium_schedule_v4_completed": metadata["status"] == "COMPLETED",
        "medium_schedule_v4_training_from_scratch": True,
        "medium_schedule_v4_training_status": metadata["status"],
        "medium_schedule_v4_completed_optimizer_steps": metadata["completed_steps"],
        "medium_schedule_v4_stop_reason": metadata.get("stop_reason"),
        "medium_schedule_v4_decision": result["decision"],
        "medium_schedule_v4_selected_checkpoint": selection["selected_checkpoint"],
        "medium_schedule_v4_selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "medium_schedule_v4_selected_step": selection["selected_step"],
        "medium_rescue_v5_permitted": False,
        "gate_adjustment_permitted": False,
        "seed43_44_permitted": passed,
        "seed43_started": False, "seed44_started": False,
        "10k_started": True, "10k_completed": metadata["status"] == "COMPLETED",
        "100k_permitted": False, "100k_started": False,
        "test_records_read": 0,
        "next_command": commands[0] if commands else None,
        "next_commands": commands,
        "next_command_executed": False,
        "updated_at": iso_now(),
        "decision_reasons": [
            "Schedule V4 started from step 0 and changed only the learning-rate schedule",
            f"Training ended with status {metadata['status']} at step {metadata['completed_steps']}",
            f"Original Gate 2 passed {sum(result['gate']['conditions'].values())} of {len(result['gate']['conditions'])} conditions",
            "No seed43, seed44, 100k, or test evaluation was run",
        ] + ([] if passed else [FAILURE_BOUNDARY]),
    })
    atomic_json_save(state, state_path)

    timing.mark("report_generation_end")
    timing.mark("pipeline_end", decision=result["decision"])
    final_timing = timing.finalize(
        completed_optimizer_steps=int(metadata["completed_steps"]), batch_size=8,
        active_optimizer_seconds=float(metadata["active_optimizer_seconds"]), interval_rows=None,
    )

    all_rows = summary[summary.group.eq("all")].set_index("method")
    candidate, upstream = all_rows.loc["medium_accepted"], all_rows.loc["upstream"]
    failed = [name for name, value in result["gate"]["conditions"].items() if not value]
    report = [
        "# MCVR Medium Seed42 Schedule V4 Report", "",
        f"Decision: **{result['decision']}**", "",
        "Schedule V4 trained from step 0 with 500-step warmup from `2e-5` to `2e-4`, then cosine decay to `2e-5` at step 10000.", "",
        "## Completion and timing", "", "| Item | Value |", "|---|---:|",
        f"| Training status | {metadata['status']} |",
        f"| Completed optimizer steps | {metadata['completed_steps']} / 10000 |",
        f"| Pipeline wall seconds | {_fmt(final_timing['pipeline_wall_seconds'], 3)} |",
        f"| Training wall seconds | {_fmt(final_timing['training_wall_seconds'], 3)} |",
        f"| Active optimizer seconds | {_fmt(final_timing['active_optimizer_seconds'], 3)} |",
        f"| Validation seconds | {_fmt(final_timing['validation_seconds'], 3)} |",
        f"| Checkpoint I/O seconds | {_fmt(final_timing['checkpoint_io_seconds'], 3)} |",
        f"| Optimizer steps/s | {_fmt(final_timing['mean_optimizer_steps_per_second'], 4)} |",
        f"| Samples/s | {_fmt(final_timing['mean_examples_per_second'], 4)} |",
        f"| Mean GPU utilization | {_fmt(final_timing.get('gpu_utilization_mean'), 3)}% |",
        f"| Peak GPU memory used | {_fmt(final_timing.get('peak_card_memory_used_mib'), 1)} MiB |", "",
        "## Checkpoint comparison", "", "| Step | LR | Qualified | Validity delta | Core improvement | Displacement | High-flex validity | Unseen validity |", "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparisons.sort_values("step").iterrows():
        qualified = bool(row.accuracy_noninferior) and bool(row.safety_qualified)
        report.append(
            f"| {int(row.step)} | {_fmt(row.learning_rate, 8)} | {qualified} | {_fmt(row.validity_delta)} | {_fmt(row.max_core_relative_improvement)} | {_fmt(row.mean_displacement)} | {_fmt(row.high_flex_validity_delta)} | {_fmt(row.unseen_validity_delta)} |"
        )
    report += [
        "", f"Selected checkpoint: step **{selection['selected_step']}** (`{selection['selected_checkpoint_sha256']}`).", "",
        "## Final Gate", "", "| Metric | Upstream | V4 accepted | Delta |", "|---|---:|---:|---:|",
        f"| Total validity | {_fmt(upstream.total_thresholded_validity_score)} | {_fmt(candidate.total_thresholded_validity_score)} | {_fmt(candidate.total_thresholded_validity_score-upstream.total_thresholded_validity_score)} |",
        f"| RMSD | {_fmt(upstream.aligned_RMSD)} | {_fmt(candidate.aligned_RMSD)} | {_fmt(candidate.aligned_RMSD-upstream.aligned_RMSD)} |",
        f"| MAT-P | {_fmt(upstream.MAT_P)} | {_fmt(candidate.MAT_P)} | {_fmt(candidate.MAT_P-upstream.MAT_P)} |",
        f"| MAT-R | {_fmt(upstream.MAT_R)} | {_fmt(candidate.MAT_R)} | {_fmt(candidate.MAT_R-upstream.MAT_R)} |",
        "", f"Conditions: **{sum(result['gate']['conditions'].values())}/{len(result['gate']['conditions'])}**. Failed: `{failed}`.", "",
        f"LR history contains {len(lr_history)} measurements at 50-step intervals.", "",
        "Seed43/44 were not executed. 100k and test evaluation were not run.",
    ]
    if not passed:
        report += ["", FAILURE_BOUNDARY, "", "No Rescue V5 or Gate adjustment is permitted."]
    Path("docs/MCVR_MEDIUM_SEED42_SCHEDULE_V4_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    gate_lines = [
        "# MCVR Medium Seed42 Schedule V4 Gate 2", "",
        f"Decision: **{result['decision']}**", "",
        f"Conditions: {sum(result['gate']['conditions'].values())}/{len(result['gate']['conditions'])}.", "",
        f"Failed conditions: `{failed}`.", "",
        f"Core relative improvements: `{result['gate']['relative_improvements']}`.", "",
        "The original 27 conditions and the preregistered 10% threshold were unchanged.", "",
        "No seed43, seed44, 100k, or test evaluation was run.",
    ]
    if not passed:
        gate_lines += ["", FAILURE_BOUNDARY]
    Path("docs/MCVR_MEDIUM_SEED42_SCHEDULE_V4_GATE2.md").write_text("\n".join(gate_lines) + "\n", encoding="utf-8")

    v3 = pd.read_csv("diagnostics/ecir_mvr/medium/run_a_seed42_rescue_v3/checkpoint_selection.csv")
    analysis = [
        "# MCVR Medium Training Schedule Analysis", "",
        "V4 tests whether the fixed `2e-4` learning rate overwrote early validity gains. Scientific content, data identities, model, losses, optimizer family, batch size, inference, safety, and Gate 2 remained frozen.", "",
        "## Registered schedules", "", "| Run | Initialization | LR schedule | Candidate steps |", "|---|---|---|---|",
        "| Rescue V3 | resumed at step 2450 | fixed 2e-4 | 5000, 10000, 15000, 20000 formal |",
        "| Schedule V4 | step 0 | 500-step warmup, cosine 2e-4 to 2e-5 | 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000 |", "",
        "## Outcome", "",
        f"V3 selected step 10000 with validity delta `{_fmt(v3.loc[v3.formal_checkpoint.astype(bool)].sort_values('validity_delta').iloc[0].validity_delta)}` and failed only the 10% core-improvement condition.", "",
        f"V4 selected step {selection['selected_step']} with validity delta `{_fmt(selection['selected_metrics']['validity_delta'])}` and maximum core relative improvement `{_fmt(selection['selected_metrics']['max_core_relative_improvement'])}`.", "",
        f"Final decision: **{result['decision']}**.",
    ]
    if not passed:
        analysis += ["", FAILURE_BOUNDARY, "", "The training-schedule hypothesis did not produce a preregistered Gate 2 pass; further Medium Seed42 rescue is closed."]
    Path("docs/MCVR_MEDIUM_TRAINING_SCHEDULE_ANALYSIS.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "selected_step": selection["selected_step"], "commands": commands}, indent=2))


if __name__ == "__main__":
    main()
