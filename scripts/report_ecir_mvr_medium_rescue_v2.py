#!/usr/bin/env python
"""Generate the final Rescue V2 report and close the progressive state."""

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

ROOT = bootstrap()

import numpy as np
import pandas as pd
import yaml

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.commons.run_timing import RunTiming, iso_now


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(value, digits=6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(numeric) else f"{numeric:.{digits}f}"


def _summary_row(summary: pd.DataFrame, group: str, method: str):
    rows = summary[(summary.group == group) & (summary.method == method)]
    return None if rows.empty else rows.iloc[0]


def _report_text(result, metadata, timing, intervals, comparison, summary) -> str:
    candidate = _summary_row(summary, "all", "medium_accepted")
    upstream = _summary_row(summary, "all", "upstream")
    high = _summary_row(summary, "rotatable_ge_6", "medium_accepted")
    unseen = _summary_row(summary, "unseen_update_scale_0.35", "medium_accepted")
    clean = _summary_row(summary, "clean_valid", "medium_accepted")
    selected_step = int(result.get("selected_checkpoint_step", result.get("completed_steps", 0)))
    lines = [
        "# MCVR Medium Seed42 Rescue V2 Final Report", "",
        f"Decision: **{result['decision']}**", "",
        "Rescue V2 preserved batch size 8, effective batch size 8, learning rate 0.0002, "
        "20,000 optimizer steps, model, loss, and data mixture. It changed only the invalid "
        "standalone velocity-growth stop semantics and allowed operational throughput controls.", "",
        "## Timing and completion", "",
        "| Item | Value |", "|---|---:|",
        f"| Pipeline wall seconds | {_fmt(timing.get('pipeline_wall_seconds'), 3)} |",
        f"| Training wall seconds | {_fmt(timing.get('training_wall_seconds'), 3)} |",
        f"| Active optimizer seconds | {_fmt(timing.get('active_optimizer_seconds'), 3)} |",
        f"| Validation seconds | {_fmt(timing.get('validation_seconds'), 3)} |",
        f"| Checkpoint I/O seconds | {_fmt(timing.get('checkpoint_io_seconds'), 3)} |",
        f"| Final evaluation seconds | {_fmt(timing.get('final_evaluation_seconds'), 3)} |",
        f"| Bootstrap seconds | {_fmt(timing.get('bootstrap_seconds'), 3)} |",
        f"| Report seconds | {_fmt(timing.get('report_seconds'), 3)} |",
        f"| Completed optimizer steps | {metadata['completed_steps']} / 20000 |",
        f"| Mean optimizer steps/s | {_fmt(timing.get('mean_optimizer_steps_per_second'), 4)} |",
        f"| Mean examples/s | {_fmt(timing.get('mean_examples_per_second'), 4)} |",
        f"| Estimated 100k active hours (estimate only) | {_fmt(timing.get('estimated_100k_active_hours'), 3)} |",
        f"| Automatic recovery occurred | {'yes' if metadata.get('resumed') else 'no'} |",
        f"| Selected checkpoint step | {selected_step} |", "",
        "## GPU and memory", "",
        "| Item | Value |", "|---|---:|",
        f"| Peak PyTorch allocated MiB | {_fmt(timing.get('peak_cuda_allocated_mib'), 1)} |",
        f"| Peak PyTorch reserved MiB | {_fmt(timing.get('peak_cuda_reserved_mib'), 1)} |",
        f"| Peak whole-card used MiB | {_fmt(timing.get('peak_card_memory_used_mib'), 1)} |",
        f"| GPU utilization mean | {_fmt(timing.get('gpu_utilization_mean'), 2)}% |",
        f"| GPU utilization p95 | {_fmt(timing.get('gpu_utilization_p95'), 2)}% |", "",
        "Low memory occupancy is not treated as evidence of an invalid run.", "",
        "## Checkpoint validation", "",
        "| Step | Validity delta | RMSD delta | MAT-P delta | MAT-R delta | High-flex validity delta | Unseen validity delta | Accuracy noninferior |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in comparison.sort_values("step").iterrows():
        lines.append(
            f"| {int(row.step)} | {_fmt(row.validity_delta)} | {_fmt(row.rmsd_delta)} | "
            f"{_fmt(row.mat_p_delta)} | {_fmt(row.mat_r_delta)} | {_fmt(row.high_flex_validity_delta)} | "
            f"{_fmt(row.unseen_validity_delta)} | {bool(row.accuracy_noninferior)} |"
        )
    lines += ["", "## Final selected-checkpoint metrics", ""]
    if candidate is not None and upstream is not None:
        lines += [
            "| Metric | Upstream | Rescue V2 accepted | Delta |", "|---|---:|---:|---:|",
            f"| Total thresholded validity | {_fmt(upstream.total_thresholded_validity_score)} | {_fmt(candidate.total_thresholded_validity_score)} | {_fmt(candidate.total_thresholded_validity_score - upstream.total_thresholded_validity_score)} |",
            f"| Aligned RMSD | {_fmt(upstream.aligned_RMSD)} | {_fmt(candidate.aligned_RMSD)} | {_fmt(candidate.aligned_RMSD - upstream.aligned_RMSD)} |",
            f"| MAT-P | {_fmt(upstream.MAT_P)} | {_fmt(candidate.MAT_P)} | {_fmt(candidate.MAT_P - upstream.MAT_P)} |",
            f"| MAT-R | {_fmt(upstream.MAT_R)} | {_fmt(candidate.MAT_R)} | {_fmt(candidate.MAT_R - upstream.MAT_R)} |",
            f"| COV-P | {_fmt(upstream.COV_P)} | {_fmt(candidate.COV_P)} | {_fmt(candidate.COV_P - upstream.COV_P)} |",
            f"| COV-R | {_fmt(upstream.COV_R)} | {_fmt(candidate.COV_R)} | {_fmt(candidate.COV_R - upstream.COV_R)} |",
        ]
    lines += [
        "", f"High-flex accepted validity: `{_fmt(high.total_thresholded_validity_score) if high is not None else 'n/a'}`.",
        f"Unseen-scale accepted validity: `{_fmt(unseen.total_thresholded_validity_score) if unseen is not None else 'n/a'}`.",
        f"Clean identity fraction: `{_fmt(result['clean_identity_fraction'])}` (clean summary unchanged fraction `{_fmt(clean.unchanged_fraction) if clean is not None else 'n/a'}`).", "",
        "## Gate 2", "",
        f"Passed conditions: **{sum(result['gate']['conditions'].values())}/{len(result['gate']['conditions'])}**.", "",
    ]
    failed = [name for name, passed in result["gate"]["conditions"].items() if not passed]
    lines.append("Failed conditions: " + (", ".join(f"`{name}`" for name in failed) if failed else "none") + ".")
    lines += [
        "", f"Seed43/44 permitted for a future separately authorized task: **{'yes' if result['gate']['pass'] else 'no'}**.",
        "Seed43 and seed44 were not run and no launch command was generated.",
        "100k remains prohibited and was not run. Test records read: 0.", "",
        "## Per-1000-step timing", "",
        "| Step end | Interval s | Active optimizer s | Steps/s | Examples/s | GPU mean/p95 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in intervals.iterrows():
        lines.append(
            f"| {int(row.step_end)} | {_fmt(row.interval_seconds, 3)} | {_fmt(row.active_optimizer_seconds, 3)} | "
            f"{_fmt(row.steps_per_second, 4)} | {_fmt(row.examples_per_second, 4)} | "
            f"{_fmt(row.gpu_utilization_mean, 1)} / {_fmt(row.gpu_utilization_p95, 1)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    timing = RunTiming(args.output_dir)
    timing.mark("report_generation_start")
    result_path = args.evaluation_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = json.loads((args.output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    intervals = pd.read_csv(args.output_dir / "timing.csv")
    comparison_path = Path(config["diagnostics_dir"]) / "checkpoint_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.is_file() else pd.DataFrame(columns=[
        "step", "validity_delta", "rmsd_delta", "mat_p_delta", "mat_r_delta",
        "high_flex_validity_delta", "unseen_validity_delta", "accuracy_noninferior",
    ])
    summary = pd.read_csv(args.evaluation_dir / "source_summary.csv")
    selected_payload = __import__("torch").load(result["checkpoint"], map_location="cpu", weights_only=False)
    result["selected_checkpoint_step"] = int(selected_payload["step"])
    atomic_json_save(result, result_path)

    state_path = Path("reports/ecir_mvr/progressive_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "current_stage": "MEDIUM_SEED42_RESCUE_V2_COMPLETE",
        "current_decision": result["decision"],
        "medium_rescue_v2_permitted": False,
        "medium_rescue_v2_started": True,
        "medium_rescue_v2_completed": bool(metadata.get("20k_completed")),
        "medium_rescue_v2_training_status": metadata["status"],
        "medium_rescue_v2_completed_optimizer_steps": metadata["completed_steps"],
        "medium_rescue_v2_stop_reason": metadata.get("stop_reason"),
        "medium_rescue_v2_decision": result["decision"],
        "medium_rescue_v2_selected_checkpoint": result["checkpoint"],
        "medium_rescue_v2_selected_checkpoint_sha256": result["checkpoint_sha256"],
        "seed43_44_permitted": bool(result["gate"]["pass"]),
        "seed43_started": False, "seed44_started": False,
        "20k_permitted": False, "100k_permitted": False, "100k_started": False,
        "test_records_read": 0, "next_command": None, "next_commands": [],
        "updated_at": iso_now(),
    })
    atomic_json_save(state, state_path)

    report_path = Path("docs/MCVR_MEDIUM_SEED42_RESCUE_V2_REPORT.md")
    selection_path = Path("docs/MCVR_MEDIUM_SEED42_RESCUE_V2_CHECKPOINT_SELECTION.md")
    gate_path = Path("docs/MCVR_MEDIUM_SEED42_RESCUE_V2_GATE2.md")
    draft_timing = timing.load()
    report_path.write_text(_report_text(result, metadata, draft_timing, intervals, comparison, summary), encoding="utf-8")
    selection_path.write_text(
        "# MCVR Medium Seed42 Rescue V2 Checkpoint Selection\n\n"
        f"Selected step: **{result['selected_checkpoint_step']}**.\n\n"
        f"Checkpoint SHA256: `{result['checkpoint_sha256']}`.\n\n"
        "Selection was restricted to validation checkpoints that passed frozen accuracy noninferiority.\n",
        encoding="utf-8",
    )
    failed = [name for name, passed in result["gate"]["conditions"].items() if not passed]
    gate_path.write_text(
        "# MCVR Medium Seed42 Rescue V2 Gate 2\n\n"
        f"Decision: **{result['decision']}**.\n\n"
        f"Conditions passed: {sum(result['gate']['conditions'].values())}/{len(result['gate']['conditions'])}.\n\n"
        f"Failed: {', '.join(failed) if failed else 'none'}.\n\n"
        "Validation-only; test records read: 0; 100k was not authorized.\n",
        encoding="utf-8",
    )
    timing.mark("report_generation_end")
    timing.mark("pipeline_end", decision=result["decision"])
    final_timing = timing.finalize(
        completed_optimizer_steps=int(metadata["completed_steps"]), batch_size=8,
        active_optimizer_seconds=float(metadata["active_optimizer_seconds"]),
        interval_rows=None,
    )
    report_path.write_text(_report_text(result, metadata, final_timing, intervals, comparison, summary), encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"], "report": str(report_path),
        "pipeline_wall_seconds": final_timing.get("pipeline_wall_seconds"),
        "selected_checkpoint_step": result["selected_checkpoint_step"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }, indent=2))


if __name__ == "__main__":
    main()
