"""Run quadratic and sigmoid angular-mu schedule sweeps sequentially."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SweepExperiment:
    name: str
    angular_mu_schedule: str
    angular_mu_max: float
    angular_loss_weight: float
    angular_mu_sigmoid_k: float = 10.0
    angular_mu_sigmoid_t0: float = 0.5


EXPERIMENTS: Tuple[SweepExperiment, ...] = (
    SweepExperiment("quad_mu03_lam001", "quadratic", 0.3, 0.01),
    SweepExperiment("quad_mu03_lam003", "quadratic", 0.3, 0.03),
    SweepExperiment("quad_mu05_lam001", "quadratic", 0.5, 0.01),
    SweepExperiment("quad_mu05_lam003", "quadratic", 0.5, 0.03),
    SweepExperiment("quad_mu10_lam001", "quadratic", 1.0, 0.01),
    SweepExperiment("quad_mu10_lam003", "quadratic", 1.0, 0.03),
    SweepExperiment("sig_mu03_lam001", "sigmoid", 0.3, 0.01),
    SweepExperiment("sig_mu03_lam003", "sigmoid", 0.3, 0.03),
    SweepExperiment("sig_mu05_lam001", "sigmoid", 0.5, 0.01),
    SweepExperiment("sig_mu05_lam003", "sigmoid", 0.5, 0.03),
    SweepExperiment("sig_mu10_lam001", "sigmoid", 1.0, 0.01),
    SweepExperiment("sig_mu10_lam003", "sigmoid", 1.0, 0.03),
)

VALID_SCHEDULES = ("quadratic", "sigmoid")
RETRYABLE_STATUSES = {
    "failed",
    "no_metrics",
    "not_started_time_limit",
}

METRIC_COLUMNS: Tuple[str, ...] = (
    "train/loss",
    "train/flow_matching_loss",
    "train/angular/dot_tau_loss",
    "train/angular/mean_abs_dot_tau_pred",
    "train/angular/mean_abs_dot_tau_target",
    "train/angular/scaled_angular_to_res_ratio",
    "train/angular/mu_t_mean",
    "train/angular/mu_t_max",
    "train/angular/mu_t_min",
    "val/loss",
    "val/flow_matching_loss",
    "val/angular/dot_tau_loss",
    "val/angular/mean_abs_dot_tau_pred",
    "val/angular/mean_abs_dot_tau_target",
    "val/angular/scaled_angular_to_res_ratio",
    "val/angular/mu_t_mean",
    "val/angular/mu_t_max",
    "val/angular/mu_t_min",
)

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "experiment_name",
    "status",
    "returncode",
    "angular_mu_schedule",
    "angular_mu_max",
    "angular_loss_weight",
    "angular_mu_sigmoid_k",
    "angular_mu_sigmoid_t0",
    "max_steps",
    "batch_size",
    "seed",
    "output_dir",
    "metrics_path",
    "resolved_config",
    "best_checkpoint",
    "last_checkpoint",
    *METRIC_COLUMNS,
    "started_at",
    "finished_at",
    "elapsed_minutes",
    "error_message",
    "run_log",
)

RANKING_COLUMNS: Tuple[str, ...] = (
    "experiment_name",
    "angular_mu_schedule",
    "angular_mu_max",
    "angular_loss_weight",
    "val/flow_matching_loss",
    "val/loss",
    "val/angular/dot_tau_loss",
    "val/angular/mean_abs_dot_tau_pred",
    "val/angular/mean_abs_dot_tau_target",
    "val/angular/scaled_angular_to_res_ratio",
    "val/angular/mu_t_mean",
    "val/angular/mu_t_max",
    "status",
)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_schedules(value: str) -> Tuple[str, ...]:
    schedules = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not schedules:
        raise argparse.ArgumentTypeError("--schedules must contain at least one schedule")
    invalid = sorted(set(schedules).difference(VALID_SCHEDULES))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown schedules {invalid}; choose from {list(VALID_SCHEDULES)}"
        )
    return tuple(dict.fromkeys(schedules))


def _select_experiments(schedules: Sequence[str]) -> Tuple[SweepExperiment, ...]:
    selected = set(schedules)
    return tuple(
        experiment
        for experiment in EXPERIMENTS
        if experiment.angular_mu_schedule in selected
    )


def _should_start_experiment(
    previous_status: str,
    has_completed_artifacts: bool,
    rerun: bool,
    rerun_failed: bool,
) -> bool:
    if rerun:
        return True
    if rerun_failed and previous_status in RETRYABLE_STATUSES:
        return True
    if previous_status:
        return False
    return not has_completed_artifacts


def _find_latest_metrics(output_dir: Path) -> Optional[Path]:
    candidates = [path for path in output_dir.rglob("metrics.csv") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _last_non_nan_metrics(metrics_path: Optional[Path]) -> Dict[str, str]:
    values = {name: "" for name in METRIC_COLUMNS}
    if metrics_path is None:
        return values

    try:
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for name in METRIC_COLUMNS:
                    raw_value = row.get(name, "")
                    if raw_value is None:
                        continue
                    raw_value = raw_value.strip()
                    if not raw_value:
                        continue
                    try:
                        numeric_value = float(raw_value)
                    except ValueError:
                        continue
                    if not math.isnan(numeric_value):
                        values[name] = raw_value
    except (OSError, csv.Error) as exc:
        print(f"Warning: could not read metrics from {metrics_path}: {exc}")

    return values


def _checkpoint_paths(output_dir: Path) -> Tuple[str, str]:
    checkpoint_dir = output_dir / "checkpoints"
    last_checkpoint = checkpoint_dir / "last.ckpt"
    last_value = str(last_checkpoint) if last_checkpoint.is_file() else ""

    if not checkpoint_dir.is_dir():
        return "", last_value

    best_candidates = [
        path
        for path in checkpoint_dir.rglob("*.ckpt")
        if path.is_file() and path.name != "last.ckpt"
    ]
    if not best_candidates:
        return "", last_value

    best_checkpoint = max(best_candidates, key=lambda path: path.stat().st_mtime)
    return str(best_checkpoint), last_value


def _logged_best_checkpoint(output_dir: Path) -> str:
    log_path = output_dir / "run.log"
    if not log_path.is_file():
        return ""
    marker = "best checkpoint path:"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        if marker not in line:
            continue
        candidate = Path(line.split(marker, 1)[1].strip())
        if candidate.is_file():
            return str(candidate)
    return ""


def _collect_artifacts(output_dir: Path) -> Dict[str, str]:
    metrics_path = _find_latest_metrics(output_dir)
    best_checkpoint, last_checkpoint = _checkpoint_paths(output_dir)
    best_checkpoint = _logged_best_checkpoint(output_dir) or best_checkpoint
    resolved_config = output_dir / "config.resolved.yaml"
    artifacts = {
        "metrics_path": str(metrics_path) if metrics_path is not None else "",
        "resolved_config": str(resolved_config) if resolved_config.is_file() else "",
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
    }
    artifacts.update(_last_non_nan_metrics(metrics_path))
    return artifacts


def _has_valid_metrics(artifacts: Dict[str, str]) -> bool:
    raw_value = artifacts.get("val/flow_matching_loss", "")
    try:
        return math.isfinite(float(raw_value))
    except (TypeError, ValueError):
        return False


def _is_completed_existing(output_dir: Path) -> bool:
    artifacts = _collect_artifacts(output_dir)
    return bool(
        _has_valid_metrics(artifacts)
        and (output_dir / "checkpoints" / "last.ckpt").is_file()
        and (output_dir / "config.resolved.yaml").is_file()
    )


def _load_existing_summary(summary_path: Path) -> Dict[str, Dict[str, str]]:
    if not summary_path.is_file():
        return {}

    records: Dict[str, Dict[str, str]] = {}
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                experiment_name = row.get("experiment_name", "").strip()
                if experiment_name:
                    records[experiment_name] = row
    except (OSError, csv.Error) as exc:
        print(f"Warning: could not read existing summary {summary_path}: {exc}")
    return records


def _ordered_records(
    records: Dict[str, Dict[str, str]],
) -> Iterable[Dict[str, str]]:
    rows = list(records.values())
    rows.sort(key=lambda row: (_metric_sort_value(row), row.get("experiment_name", "")))
    yield from rows


def _write_summary(
    summary_path: Path,
    records: Dict[str, Dict[str, str]],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for record in _ordered_records(records):
            writer.writerow({name: record.get(name, "") for name in SUMMARY_COLUMNS})
    temporary_path.replace(summary_path)


def _base_record(
    experiment: SweepExperiment,
    args: argparse.Namespace,
    output_dir: Path,
    started_at: str,
) -> Dict[str, str]:
    record = {name: "" for name in SUMMARY_COLUMNS}
    record.update(
        {
            "experiment_name": experiment.name,
            "angular_mu_schedule": experiment.angular_mu_schedule,
            "angular_mu_max": str(experiment.angular_mu_max),
            "angular_loss_weight": str(experiment.angular_loss_weight),
            "angular_mu_sigmoid_k": str(experiment.angular_mu_sigmoid_k),
            "angular_mu_sigmoid_t0": str(experiment.angular_mu_sigmoid_t0),
            "max_steps": str(args.max_steps),
            "batch_size": str(args.batch_size),
            "seed": str(args.seed),
            "output_dir": str(output_dir),
            "started_at": started_at,
            "run_log": str(output_dir / "run.log"),
        }
    )
    return record


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _build_command(
    experiment: SweepExperiment,
    args: argparse.Namespace,
    project_root: Path,
    config_path: Path,
    output_dir: Path,
) -> List[str]:
    return [
        sys.executable,
        str(project_root / "scripts" / "train_angular.py"),
        "--config",
        str(config_path),
        "--output_dir",
        str(output_dir),
        "--max_steps",
        str(args.max_steps),
        "--batch_size",
        str(args.batch_size),
        "--angular_mu_schedule",
        experiment.angular_mu_schedule,
        "--angular_mu_max",
        str(experiment.angular_mu_max),
        "--angular_mu_sigmoid_k",
        str(experiment.angular_mu_sigmoid_k),
        "--angular_mu_sigmoid_t0",
        str(experiment.angular_mu_sigmoid_t0),
        "--use_angular_loss",
        "--angular_loss_weight",
        str(experiment.angular_loss_weight),
        "--accumulate_grad_batches",
        str(args.accumulate_grad_batches),
        "--val_check_interval",
        str(args.val_check_interval),
        "--limit_val_batches",
        str(args.limit_val_batches),
        "--log_every_n_steps",
        str(args.log_every_n_steps),
        "--seed",
        str(args.seed),
    ]


def _run_with_tee(
    command: Sequence[str],
    project_root: Path,
    log_path: Path,
    timeout_seconds: float,
) -> Tuple[int, bool]:
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        command_line = _format_command(command)
        log_handle.write(f"$ {command_line}\n")
        print(f"command: {command_line}", flush=True)

        process = subprocess.Popen(
            list(command),
            cwd=str(project_root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        timed_out = threading.Event()

        def terminate_on_timeout() -> None:
            if process.poll() is None:
                timed_out.set()
                process.kill()

        timer = threading.Timer(timeout_seconds, terminate_on_timeout)
        timer.daemon = True
        timer.start()
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_handle.write(line)
            process.stdout.close()
            return process.wait(), timed_out.is_set()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        finally:
            timer.cancel()


def _metric_sort_value(record: Dict[str, str]) -> float:
    raw_value = record.get("val/flow_matching_loss", "")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return math.inf
    return value if not math.isnan(value) else math.inf


def _display_value(column: str, value: str) -> str:
    if not value:
        return "-"
    if column in METRIC_COLUMNS:
        try:
            return f"{float(value):.6g}"
        except ValueError:
            return value
    return value


def _print_ranked_summary(records: Dict[str, Dict[str, str]]) -> None:
    rows = list(_ordered_records(records))
    rows.sort(key=lambda row: (_metric_sort_value(row), row.get("experiment_name", "")))

    print("\nSweep results sorted by val/flow_matching_loss:")
    if not rows:
        print("(no experiment results)")
        return

    display_rows = [
        [_display_value(column, row.get(column, "")) for column in RANKING_COLUMNS]
        for row in rows
    ]
    widths = [
        max(len(column), *(len(row[index]) for row in display_rows))
        for index, column in enumerate(RANKING_COLUMNS)
    ]
    print("  ".join(column.ljust(widths[index]) for index, column in enumerate(RANKING_COLUMNS)))
    print("  ".join("-" * width for width in widths))
    for row in display_rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be positive, got {value}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quadratic then sigmoid angular-mu schedule sweeps."
    )
    parser.add_argument(
        "--quadratic_config",
        default="configs/drugs-so3-angular-loss-bs8-quadratic.yaml",
    )
    parser.add_argument(
        "--sigmoid_config",
        default="configs/drugs-so3-angular-loss-bs8-sigmoid.yaml",
    )
    parser.add_argument(
        "--base_output_dir",
        default="logs_sweep/angular_schedule_sweep_5000steps",
    )
    parser.add_argument(
        "--schedules",
        default="quadratic,sigmoid",
        help="comma-separated schedules: quadratic, sigmoid, or both",
    )
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_check_interval", type=int, default=500)
    parser.add_argument("--limit_val_batches", type=int, default=10)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument(
        "--time_limit_hours",
        type=float,
        default=12.0,
        help="global wall-clock budget for this sweep invocation",
    )
    parser.add_argument("--rerun", action="store_true", help="rerun all selected groups")
    parser.add_argument(
        "--rerun_failed",
        action="store_true",
        help=(
            "rerun failed/no_metrics/not_started_time_limit groups; "
            "completed groups stay skipped"
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if isinstance(args.schedules, str):
        args.schedules = _parse_schedules(args.schedules)
    for name in (
        "max_steps",
        "batch_size",
        "val_check_interval",
        "limit_val_batches",
        "log_every_n_steps",
        "accumulate_grad_batches",
    ):
        _positive_int(getattr(args, name), name)
    if args.time_limit_hours <= 0:
        raise argparse.ArgumentTypeError(
            f"time_limit_hours must be positive, got {args.time_limit_hours}"
        )


def main() -> int:
    args = parse_args()
    try:
        _validate_args(args)
    except argparse.ArgumentTypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[1]
    selected_experiments = _select_experiments(args.schedules)
    raw_config_paths = {
        "quadratic": args.quadratic_config,
        "sigmoid": args.sigmoid_config,
    }
    config_paths: Dict[str, Path] = {}
    for schedule in args.schedules:
        raw_path = raw_config_paths[schedule]
        config_path = Path(raw_path).expanduser()
        if not config_path.is_absolute():
            cwd_candidate = (Path.cwd() / config_path).resolve()
            project_candidate = (project_root / config_path).resolve()
            config_path = cwd_candidate if cwd_candidate.is_file() else project_candidate
        else:
            config_path = config_path.resolve()
        if not config_path.is_file():
            print(f"error: config does not exist: {config_path}", file=sys.stderr)
            return 2
        config_paths[schedule] = config_path

    base_output_dir = Path(args.base_output_dir).expanduser()
    if not base_output_dir.is_absolute():
        base_output_dir = (Path.cwd() / base_output_dir).resolve()
    else:
        base_output_dir = base_output_dir.resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = base_output_dir / "summary.csv"
    records = _load_existing_summary(summary_path)
    _write_summary(summary_path, records)

    sweep_started_at = _timestamp()
    sweep_start = time.monotonic()
    sweep_deadline = sweep_start + args.time_limit_hours * 3600.0
    print(f"sweep started_at: {sweep_started_at}")
    print(f"selected schedules: {','.join(args.schedules)}")
    print(f"global time limit hours: {args.time_limit_hours}")
    print(f"base output dir: {base_output_dir}")

    current_schedule = None
    user_interrupted = False
    for experiment in selected_experiments:
        if experiment.angular_mu_schedule != current_schedule:
            current_schedule = experiment.angular_mu_schedule
            print(f"\nStarting {current_schedule} schedule round")

        output_dir = base_output_dir / experiment.name
        previous_record = records.get(experiment.name)
        previous_status = previous_record.get("status", "") if previous_record else ""
        has_completed_artifacts = _is_completed_existing(output_dir)
        should_start = _should_start_experiment(
            previous_status=previous_status,
            has_completed_artifacts=has_completed_artifacts,
            rerun=args.rerun,
            rerun_failed=args.rerun_failed,
        )

        print("\n" + "=" * 80)
        print(f"experiment: {experiment.name}")
        print(f"angular_mu_schedule: {experiment.angular_mu_schedule}")
        print(f"angular_mu_max: {experiment.angular_mu_max}")
        print(f"angular_loss_weight: {experiment.angular_loss_weight}")
        print(f"output_dir: {output_dir}")

        if not should_start:
            if previous_status in RETRYABLE_STATUSES or previous_status == "interrupted":
                print(f"status: {previous_status} (not rerun)")
                continue
            record = (
                dict(previous_record)
                if previous_record is not None
                else _base_record(experiment, args, output_dir, started_at="")
            )
            record.update(_collect_artifacts(output_dir))
            record["status"] = "skipped_existing"
            record["returncode"] = "0"
            record["finished_at"] = _timestamp()
            record["elapsed_minutes"] = "0.000"
            records[experiment.name] = record
            _write_summary(summary_path, records)
            print("status: skipped_existing")
            continue

        if time.monotonic() >= sweep_deadline:
            record = _base_record(experiment, args, output_dir, started_at="")
            record["status"] = "not_started_time_limit"
            record["error_message"] = (
                f"Global sweep time limit of {args.time_limit_hours} hours was "
                "reached before this experiment started."
            )
            record["finished_at"] = _timestamp()
            record["elapsed_minutes"] = "0.000"
            records[experiment.name] = record
            _write_summary(summary_path, records)
            print("status: not_started_time_limit")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = _timestamp()
        experiment_start = time.monotonic()
        record = _base_record(experiment, args, output_dir, started_at)
        print(f"start time: {started_at}")

        command = _build_command(
            experiment,
            args,
            project_root,
            config_paths[experiment.angular_mu_schedule],
            output_dir,
        )
        timed_out = False
        try:
            remaining_seconds = max(0.0, sweep_deadline - time.monotonic())
            returncode, timed_out = _run_with_tee(
                command,
                project_root,
                output_dir / "run.log",
                remaining_seconds,
            )
            record["returncode"] = str(returncode)
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            record["returncode"] = "130"
            record["error_message"] = "Sweep interrupted by user."
            user_interrupted = True
        except Exception as exc:
            record["status"] = "failed"
            record["returncode"] = "-1"
            record["error_message"] = (
                f"{type(exc).__name__}: {exc}; see {output_dir / 'run.log'}"
            )
            try:
                with (output_dir / "run.log").open("a", encoding="utf-8") as log_handle:
                    log_handle.write(record["error_message"] + "\n")
            except OSError:
                pass
            print(f"experiment failed to launch or monitor: {exc}")

        artifacts = _collect_artifacts(output_dir)
        record.update(artifacts)
        if not record["status"]:
            if timed_out:
                record["status"] = "failed"
                record["error_message"] = (
                    "Global sweep time limit was reached after training started."
                )
            elif record["returncode"] != "0":
                record["status"] = "failed"
                record["error_message"] = (
                    f"Training exited with return code {record['returncode']}; "
                    f"see {output_dir / 'run.log'}"
                )
            elif _has_valid_metrics(artifacts):
                record["status"] = "completed"
            else:
                record["status"] = "no_metrics"
                record["error_message"] = (
                    "Training returned 0 but no finite val/flow_matching_loss "
                    "was found."
                )

        record["finished_at"] = _timestamp()
        record["elapsed_minutes"] = (
            f"{(time.monotonic() - experiment_start) / 60.0:.3f}"
        )
        records[experiment.name] = record
        _write_summary(summary_path, records)

        print(f"status: {record['status']}")
        print(f"returncode: {record['returncode']}")
        print(f"finished_at: {record['finished_at']}")
        print(f"elapsed_minutes: {record['elapsed_minutes']}")
        print(f"summary updated: {summary_path}")
        if user_interrupted:
            break

    _write_summary(summary_path, records)
    print(f"\nsummary.csv: {summary_path}")
    _print_ranked_summary(records)
    return 130 if user_interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
