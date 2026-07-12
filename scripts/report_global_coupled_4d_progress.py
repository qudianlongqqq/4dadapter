#!/usr/bin/env python
"""Crash-tolerant terminal and JSON/Markdown progress dashboard."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs_global_coupled_4d"
DIAGNOSTICS = ROOT / "diagnostics/global_coupled_4d"
REPORTS = ROOT / "reports"


def _read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _pid(name):
    path = LOG_ROOT / name
    try:
        value = int(path.read_text().strip())
        os.kill(value, 0)
        return value
    except Exception:
        return None


def _stage():
    if (LOG_ROOT / "COMPLETED").exists():
        return "COMPLETED"
    if (LOG_ROOT / "FAILED").exists():
        return "FAILED"
    current = LOG_ROOT / "CURRENT_STAGE"
    if current.is_file():
        value = current.read_text(encoding="utf-8", errors="replace").strip()
        if value:
            return value
    if (LOG_ROOT / "FORMAL_RUNNING").exists():
        return "FORMAL_TRAIN"
    if (LOG_ROOT / "SMOKE_EVAL_COMPLETED").exists():
        return "FORMAL_TRAIN"
    if (LOG_ROOT / "SMOKE_SAMPLE_COMPLETED").exists():
        return "SMOKE_EVAL"
    if (LOG_ROOT / "SMOKE_TRAIN_COMPLETED").exists():
        return "SMOKE_SAMPLE"
    if (LOG_ROOT / "ORACLE_PASSED").exists():
        return "SMOKE"
    if (LOG_ROOT / "TESTS_PASSED").exists():
        return "ORACLE"
    return "TEST"


def _latest_metrics():
    candidates = sorted(LOG_ROOT.glob("**/metrics.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return {}, ""
    path = candidates[-1]
    latest = {}
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                for key, value in row.items():
                    if value not in (None, ""):
                        latest[key] = value
    except Exception:
        return {}, str(path)
    return latest, str(path)


def _metric(metrics, name):
    for key in (name, name + "_step", name + "_epoch"):
        if key in metrics:
            return metrics[key]
    return None


def _latest_eval():
    candidates = sorted(DIAGNOSTICS.glob("**/summary.csv"), key=lambda path: path.stat().st_mtime)
    for path in reversed(candidates):
        try:
            with path.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            selected = next((row for row in rows if row.get("method") == "global_coupled_4d_adapter" and row.get("subset") == "all"), None)
            if selected:
                return selected, str(path)
        except Exception:
            pass
    return {}, ""


def _gpu_memory():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        ).strip().replace("\n", ",") or None
    except Exception:
        return None


def main():
    stage = _stage()
    if _pid("EVAL.pid") is not None:
        stage = "EVAL"
    elif _pid("SAMPLE.pid") is not None:
        stage = "SAMPLE"
    budget = _read_json(REPORTS / "reference_4d_training_budget.json")
    metrics, metrics_path = _latest_metrics()
    evaluation, evaluation_path = _latest_eval()
    oracle = {}
    oracle_path = DIAGNOSTICS / "oracle/summary.csv"
    if oracle_path.is_file():
        try:
            with oracle_path.open(encoding="utf-8-sig") as handle:
                oracle = next((row for row in csv.DictReader(handle) if row.get("subset") == "all"), {})
        except Exception:
            pass
    state_candidates = sorted(LOG_ROOT.glob("**/run_state.json"), key=lambda path: path.stat().st_mtime)
    state = _read_json(state_candidates[-1]) if state_candidates else {}
    step = int(float(_metric(metrics, "step") or state.get("global_step", 0) or 0))
    target = 5000
    checkpoints = sorted(LOG_ROOT.glob("**/*.ckpt"), key=lambda path: path.stat().st_mtime)
    failed = _read_json(LOG_ROOT / "FAILED", {}) if (LOG_ROOT / "FAILED").is_file() else {}
    sweep_count = len(list(DIAGNOSTICS.glob("checkpoint_sweep_5k/**/summary.csv")))
    ablation_count = len(list(DIAGNOSTICS.glob("ablation_5k/**/summary.csv")))
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.05)
    except Exception:
        cpu_percent = None
    master_pid, train_pid, sample_pid, eval_pid = (
        _pid("MASTER.pid"), _pid("TRAIN.pid"), _pid("SAMPLE.pid"), _pid("EVAL.pid")
    )
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(), "stage": stage,
        "current_pid": eval_pid or sample_pid or train_pid or master_pid,
        "master_pid": master_pid, "train_pid": train_pid,
        "sample_pid": sample_pid, "eval_pid": eval_pid,
        "runtime_seconds": int(time.time() - (LOG_ROOT / "RUNNING").stat().st_mtime) if (LOG_ROOT / "RUNNING").exists() else 0,
        "gpu_memory_mib": _gpu_memory(), "cpu_percent": cpu_percent,
        "global_step": step, "formal_target_step": target,
        "formal_percent": 100 * step / target if target else 0.0,
        "epoch": _metric(metrics, "epoch"), "current_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "smoke_status": {
            "train": "COMPLETED" if (LOG_ROOT / "SMOKE_TRAIN_COMPLETED").exists() else "PENDING_OR_RUNNING",
            "sample": "COMPLETED" if (LOG_ROOT / "SMOKE_SAMPLE_COMPLETED").exists() else "PENDING_OR_RUNNING",
            "eval": "COMPLETED" if (LOG_ROOT / "SMOKE_EVAL_COMPLETED").exists() else "PENDING_OR_RUNNING",
        },
        "formal_status": "COMPLETED" if (LOG_ROOT / "FORMAL_COMPLETED").exists() else ("RUNNING" if (LOG_ROOT / "FORMAL_RUNNING").exists() else "PENDING"),
        "resumed": state.get("resumed"), "reference_budget": budget,
        "budget_matched": (LOG_ROOT / "BUDGET_MATCHED").exists(),
        "train_final_loss": _metric(metrics, "train/final_loss"),
        "val_final_loss": _metric(metrics, "val/final_loss"),
        "val_internal_loss": _metric(metrics, "val/internal_loss"),
        "val_residual_loss": _metric(metrics, "val/residual_loss"),
        "oracle_explained_ratio": oracle.get("4d_explained_ratio", _metric(metrics, "val/oracle_internal_explained_ratio")),
        "predicted_internal_fraction": _metric(metrics, "val/internal_velocity_fraction"),
        "stretch_fraction": _metric(metrics, "val/stretch_energy_fraction"),
        "bending_fraction": _metric(metrics, "val/bending_energy_fraction"),
        "torsion_fraction": _metric(metrics, "val/torsion_energy_fraction"),
        "jacobian_rank": _metric(metrics, "val/jacobian_effective_rank"),
        "condition_number": _metric(metrics, "val/jacobian_condition_number"),
        "orthogonality_error": _metric(metrics, "val/projection_orthogonality_error"),
        "reconstruction_error": _metric(metrics, "val/projection_reconstruction_error"),
        "solver_fallback_rate": _metric(metrics, "val/solver_fallback_rate"),
        "rollout_rmsd": evaluation.get("rmsd_mean"), "COV-R": evaluation.get("COV-R"),
        "COV-P": evaluation.get("COV-P"), "MAT-R": evaluation.get("MAT-R"),
        "MAT-P": evaluation.get("MAT-P"), "failure_rate": evaluation.get("failure_rate"),
        "completed_checkpoint_evaluations": sweep_count,
        "completed_ablations": ablation_count,
        "next_stage": {"CHECK": "TEST", "TEST": "ORACLE", "ORACLE": "SMOKE", "SMOKE": "FORMAL_TRAIN", "FORMAL_TRAIN": "SAMPLE", "SAMPLE": "EVAL", "EVAL": "ABLATION", "ABLATION": "COMPLETED"}.get(stage, "NONE"),
        "latest_error": failed or state.get("error"), "metrics_path": metrics_path,
        "evaluation_path": evaluation_path,
    }
    lines = ["# Global Coupled 4D progress", ""] + [f"- {key}: `{value}`" for key, value in payload.items()]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "global_coupled_4d_latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORTS / "global_coupled_4d_latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
