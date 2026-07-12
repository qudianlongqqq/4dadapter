#!/usr/bin/env python
"""Compact, bounded progress report for the isolated formal-large pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs_formal_large"
DIAG = ROOT / "diagnostics/formal_large"


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_metrics(path: Path) -> dict:
    latest = {}
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                latest.update({key: value for key, value in row.items() if value not in (None, "")})
    except Exception:
        pass
    return latest


def _metric(metrics: dict, names: tuple[str, ...]):
    return next((metrics[name] for name in names if name in metrics), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if (LOG / "FORMAL_LARGE_FINAL_TEST_RUNNING").exists():
        stage, method = "final_test", None
    elif (LOG / "FORMAL_LARGE_CONFIRM30_RUNNING").exists():
        stage, method = "confirm30", None
    elif (LOG / "FORMAL_LARGE_SCREEN10_RUNNING").exists():
        stage, method = "screen10", None
    elif (LOG / "FORMAL_LARGE_FINAL_TEST_COMPLETED").exists():
        stage, method = "final_test_completed", None
    elif (LOG / "FORMAL_LARGE_CONFIRM30_COMPLETED").exists():
        stage, method = "confirm30_completed", None
    elif (LOG / "FORMAL_LARGE_SCREEN10_COMPLETED").exists():
        stage, method = "screen10_completed", None
    elif (LOG / "GLOBAL4D_TRAINING").exists():
        stage, method = "training", "global4d"
    elif (LOG / "CARTESIAN_TRAINING").exists():
        stage, method = "training", "cartesian"
    elif (LOG / "FORMAL_LARGE_TRAINING_COMPLETED").exists():
        stage, method = "training_completed", None
    elif (LOG / "DATA_READY").exists():
        stage, method = "data_ready", None
    else:
        stage, method = "pending", None

    run = None
    if method:
        run = LOG / ("cartesian_seed42_200k" if method == "cartesian" else "global4d_seed42_200k")
    state = _json(run / "run_state.json") if run else {}
    metrics = _latest_metrics(run / "metrics.csv") if run else {}
    step = int(state.get("global_step", _metric(metrics, ("step",)) or 0))
    checkpoint = state.get("checkpoint")
    sampling_states = sorted(
        DIAG.glob("**/sampling_state.json"), key=lambda path: path.stat().st_mtime
    )
    sampling = _json(sampling_states[-1]) if sampling_states else {}
    sampling_relative = (
        sampling_states[-1].parent.relative_to(DIAG) if sampling_states else None
    )
    if method is None and sampling_relative is not None and len(sampling_relative.parts) > 1:
        method = sampling_relative.parts[1]
    if (LOG / "FORMAL_LARGE_TRAINING_COMPLETED").exists() and step == 0:
        step = 200_000
    failed = _json(LOG / "FAILED")
    payload = {
        "stage": stage,
        "method": method,
        "global_step": step,
        "target_step": 200_000,
        "percent": 100.0 * step / 200_000,
        "checkpoint": checkpoint,
        "train_loss": _metric(metrics, ("train/loss_epoch", "train/loss_step", "train/loss")),
        "val_loss": _metric(metrics, ("val/loss", "val/final_loss")),
        "current_selection_combination": (
            str(sampling_relative) if sampling_relative is not None else None
        ),
        "record_progress": (
            f"{sampling.get('completed_count', 0)}/{sampling.get('total_count', 0)}"
            if sampling else None
        ),
        "eta_seconds": sampling.get("eta_seconds"),
        "screen_completed_groups": len(list((DIAG / "screen10").glob("**/eval/summary.csv"))),
        "confirm_completed_groups": len(list((DIAG / "confirm30").glob("**/eval/summary.csv"))),
        "final_test_status": (
            "completed" if (LOG / "FORMAL_LARGE_FINAL_TEST_COMPLETED").exists() else "pending"
        ),
        "latest_error": failed or None,
    }
    if args.compact:
        for key, value in payload.items():
            print(f"{key}: {value}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
