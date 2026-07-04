"""Run the first low-risk 4D Jacobian correction sweep sequentially."""

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
    enabled: bool
    correction_scale: float
    q_loss_weight: float


EXPERIMENTS: Tuple[SweepExperiment, ...] = (
    SweepExperiment("base", False, 0.0, 0.0),
    SweepExperiment("scale001_q0001", True, 0.01, 0.001),
    SweepExperiment("scale003_q0001", True, 0.03, 0.001),
    SweepExperiment("scale01_q0001", True, 0.1, 0.001),
    SweepExperiment("scale003_q0", True, 0.03, 0.0),
    SweepExperiment("scale003_q0003", True, 0.03, 0.003),
)

RETRYABLE_STATUSES = {"failed", "no_metrics", "not_started_time_limit"}
METRIC_COLUMNS: Tuple[str, ...] = (
    "train/flow_matching_loss_base",
    "train/flow_matching_loss",
    "train/loss",
    "train/jacobian_4d/q_loss",
    "train/jacobian_4d/corr_to_residual_ratio",
    "val/flow_matching_loss_base",
    "val/flow_matching_loss",
    "val/loss",
    "val/jacobian_4d/q_loss",
    "val/jacobian_4d/corr_to_residual_ratio",
    "val/jacobian_4d/num_selected_bonds",
    "val/jacobian_4d/num_valid_bonds",
    "val/jacobian_4d/skip_rate",
    "val/jacobian_4d/mean_abs_s_pred",
    "val/jacobian_4d/mean_abs_s_target",
    "val/jacobian_4d/mean_abs_omega_pred",
    "val/jacobian_4d/mean_abs_omega_target",
)
SUMMARY_COLUMNS: Tuple[str, ...] = (
    "experiment_name",
    "status",
    "seed",
    "use_jacobian_4d_correction",
    "correction_scale",
    "q_loss_weight",
    "corr_reg_weight",
    "max_steps",
    "batch_size",
    "output_dir",
    "metrics_path",
    "resolved_config",
    "best_checkpoint",
    "last_checkpoint",
    *METRIC_COLUMNS,
    "returncode",
    "started_at",
    "finished_at",
    "elapsed_minutes",
    "error_message",
    "run_log",
)
RANKING_COLUMNS: Tuple[str, ...] = (
    "experiment_name",
    "status",
    "correction_scale",
    "q_loss_weight",
    "val/flow_matching_loss_base",
    "val/flow_matching_loss",
    "val/loss",
    "val/jacobian_4d/q_loss",
    "val/jacobian_4d/corr_to_residual_ratio",
    "val/jacobian_4d/num_valid_bonds",
    "val/jacobian_4d/skip_rate",
)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _find_latest_metrics(output_dir: Path) -> Optional[Path]:
    paths = [path for path in output_dir.rglob("metrics.csv") if path.is_file()]
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def _last_non_nan_metrics(metrics_path: Optional[Path]) -> Dict[str, str]:
    values = {name: "" for name in METRIC_COLUMNS}
    if metrics_path is None:
        return values
    try:
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for name in METRIC_COLUMNS:
                    raw = (row.get(name) or "").strip()
                    if not raw:
                        continue
                    try:
                        numeric = float(raw)
                    except ValueError:
                        continue
                    if math.isfinite(numeric):
                        values[name] = raw
    except (OSError, csv.Error) as exc:
        print(f"Warning: could not read {metrics_path}: {exc}")
    return values


def _checkpoint_paths(output_dir: Path) -> Tuple[str, str]:
    checkpoint_dir = output_dir / "checkpoints"
    last = checkpoint_dir / "last.ckpt"
    last_value = str(last) if last.is_file() else ""
    if not checkpoint_dir.is_dir():
        return "", last_value
    candidates = [
        path
        for path in checkpoint_dir.rglob("*.ckpt")
        if path.is_file() and path.name != "last.ckpt"
    ]
    best = max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    return (str(best) if best else ""), last_value


def _logged_best_checkpoint(output_dir: Path) -> str:
    log_path = output_dir / "run.log"
    if not log_path.is_file():
        return ""
    marker = "best checkpoint path:"
    for line in reversed(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if marker in line:
            candidate = Path(line.split(marker, 1)[1].strip())
            if candidate.is_file():
                return str(candidate)
    return ""


def _collect_artifacts(output_dir: Path) -> Dict[str, str]:
    metrics_path = _find_latest_metrics(output_dir)
    best, last = _checkpoint_paths(output_dir)
    resolved = output_dir / "config.resolved.yaml"
    values = {
        "metrics_path": str(metrics_path) if metrics_path else "",
        "resolved_config": str(resolved) if resolved.is_file() else "",
        "best_checkpoint": _logged_best_checkpoint(output_dir) or best,
        "last_checkpoint": last,
    }
    values.update(_last_non_nan_metrics(metrics_path))
    return values


def _has_valid_metrics(artifacts: Dict[str, str]) -> bool:
    try:
        return math.isfinite(float(artifacts["val/flow_matching_loss"]))
    except (KeyError, TypeError, ValueError):
        return False


def _is_completed_existing(output_dir: Path) -> bool:
    artifacts = _collect_artifacts(output_dir)
    return bool(
        _has_valid_metrics(artifacts)
        and artifacts["last_checkpoint"]
        and artifacts["resolved_config"]
    )


def _load_summary(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        return {}
    records = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("experiment_name") or "").strip()
            if name:
                records[name] = row
    return records


def _metric_sort_value(record: Dict[str, str]) -> float:
    try:
        value = float(record.get("val/flow_matching_loss", ""))
    except (TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) else math.inf


def _ordered(records: Dict[str, Dict[str, str]]) -> Iterable[Dict[str, str]]:
    yield from sorted(
        records.values(),
        key=lambda row: (_metric_sort_value(row), row.get("experiment_name", "")),
    )


def _write_summaries(
    csv_path: Path, markdown_path: Path, records: Dict[str, Dict[str, str]]
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in _ordered(records):
            writer.writerow({name: row.get(name, "") for name in SUMMARY_COLUMNS})
    temporary.replace(csv_path)

    rows = list(_ordered(records))
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# 4D Jacobian sweep summary\n\n")
        handle.write("| " + " | ".join(RANKING_COLUMNS) + " |\n")
        handle.write("| " + " | ".join("---" for _ in RANKING_COLUMNS) + " |\n")
        for row in rows:
            values = [str(row.get(column, "")).replace("|", "\\|") for column in RANKING_COLUMNS]
            handle.write("| " + " | ".join(values) + " |\n")


def _base_record(
    experiment: SweepExperiment,
    args: argparse.Namespace,
    output_dir: Path,
    status: str = "",
) -> Dict[str, str]:
    record = {name: "" for name in SUMMARY_COLUMNS}
    record.update(
        {
            "experiment_name": experiment.name,
            "status": status,
            "seed": str(args.seed),
            "use_jacobian_4d_correction": str(experiment.enabled),
            "correction_scale": str(experiment.correction_scale),
            "q_loss_weight": str(experiment.q_loss_weight),
            "corr_reg_weight": str(args.corr_reg_weight),
            "max_steps": str(args.max_steps),
            "batch_size": str(args.batch_size),
            "output_dir": str(output_dir),
            "run_log": str(output_dir / "run.log"),
        }
    )
    return record


def _build_command(
    experiment: SweepExperiment,
    args: argparse.Namespace,
    project_root: Path,
    config_path: Path,
    output_dir: Path,
) -> List[str]:
    return [
        sys.executable,
        str(project_root / "scripts" / "train_jacobian_4d.py"),
        "--config",
        str(config_path),
        "--output_dir",
        str(output_dir),
        "--max_steps",
        str(args.max_steps),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--val_check_interval",
        str(args.val_check_interval),
        "--limit_val_batches",
        str(args.limit_val_batches),
        "--log_every_n_steps",
        str(args.log_every_n_steps),
        "--accumulate_grad_batches",
        str(args.accumulate_grad_batches),
        "--use_jacobian_4d_correction",
        str(experiment.enabled).lower(),
        "--jacobian_4d_correction_scale",
        str(experiment.correction_scale),
        "--jacobian_4d_q_loss_weight",
        str(experiment.q_loss_weight),
        "--jacobian_4d_corr_reg_weight",
        str(args.corr_reg_weight),
    ]


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _run_with_tee(
    command: Sequence[str], project_root: Path, log_path: Path, timeout: float
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

        def terminate() -> None:
            if process.poll() is None:
                timed_out.set()
                process.kill()

        timer = threading.Timer(timeout, terminate)
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


def _should_start(
    previous_status: str, completed_artifacts: bool, rerun: bool, rerun_failed: bool
) -> bool:
    if rerun:
        return True
    if rerun_failed and previous_status in RETRYABLE_STATUSES:
        return True
    if previous_status:
        return False
    return not completed_artifacts


def _print_summary(records: Dict[str, Dict[str, str]]) -> None:
    print("\nSweep results sorted by val/flow_matching_loss:")
    rows = list(_ordered(records))
    if not rows:
        print("(no results)")
        return
    display = [[row.get(column, "") or "-" for column in RANKING_COLUMNS] for row in rows]
    widths = [
        max(len(column), *(len(row[index]) for row in display))
        for index, column in enumerate(RANKING_COLUMNS)
    ]
    print("  ".join(column.ljust(widths[i]) for i, column in enumerate(RANKING_COLUMNS)))
    for row in display:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/drugs-so3-jacobian-4d-bs4.yaml")
    parser.add_argument("--base_output_dir", default="logs_sweep/jacobian_4d_5000steps")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_check_interval", type=int, default=500)
    parser.add_argument("--limit_val_batches", type=int, default=10)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--corr_reg_weight", type=float, default=0.0001)
    parser.add_argument("--time_limit_hours", type=float, default=12.0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--rerun_failed",
        action="store_true",
        help="rerun failed/no_metrics/not_started_time_limit groups only",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="print all six commands and exit"
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "max_steps",
        "batch_size",
        "val_check_interval",
        "limit_val_batches",
        "log_every_n_steps",
        "accumulate_grad_batches",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.time_limit_hours <= 0:
        raise ValueError("time_limit_hours must be positive.")
    if not math.isfinite(args.corr_reg_weight) or args.corr_reg_weight < 0:
        raise ValueError("corr_reg_weight must be finite and non-negative.")


def main() -> int:
    args = parse_args()
    try:
        _validate_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    if not config_path.is_file():
        print(f"error: config does not exist: {config_path}", file=sys.stderr)
        return 2
    base_output = Path(args.base_output_dir).expanduser()
    if not base_output.is_absolute():
        base_output = (Path.cwd() / base_output).resolve()

    if args.dry_run:
        for experiment in EXPERIMENTS:
            command = _build_command(
                experiment,
                args,
                project_root,
                config_path,
                base_output / experiment.name,
            )
            print(f"{experiment.name}: {_format_command(command)}")
        return 0

    base_output.mkdir(parents=True, exist_ok=True)
    summary_csv = base_output / "summary.csv"
    summary_md = base_output / "summary.md"
    records = _load_summary(summary_csv)
    deadline = time.monotonic() + args.time_limit_hours * 3600.0

    try:
        for experiment in EXPERIMENTS:
            output_dir = base_output / experiment.name
            previous = records.get(experiment.name, {})
            completed_artifacts = _is_completed_existing(output_dir)
            if not _should_start(
                previous.get("status", ""),
                completed_artifacts,
                args.rerun,
                args.rerun_failed,
            ):
                previous_status = previous.get("status", "")
                skip_status = (
                    "skipped_existing"
                    if completed_artifacts
                    or previous_status in {"completed", "skipped_existing"}
                    else previous_status or "skipped_existing"
                )
                record = _base_record(experiment, args, output_dir, skip_status)
                for name in (
                    "returncode",
                    "started_at",
                    "finished_at",
                    "elapsed_minutes",
                    "error_message",
                ):
                    record[name] = previous.get(name, "")
                record.update(_collect_artifacts(output_dir))
                records[experiment.name] = record
                _write_summaries(summary_csv, summary_md, records)
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                record = _base_record(
                    experiment, args, output_dir, "not_started_time_limit"
                )
                record["finished_at"] = _timestamp()
                record["error_message"] = "global sweep time limit reached before start"
                records[experiment.name] = record
                _write_summaries(summary_csv, summary_md, records)
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            record = _base_record(experiment, args, output_dir)
            record["started_at"] = _timestamp()
            started = time.monotonic()
            print("\n" + "=" * 80)
            print(f"experiment: {experiment.name}")
            command = _build_command(
                experiment, args, project_root, config_path, output_dir
            )
            try:
                returncode, timed_out = _run_with_tee(
                    command, project_root, output_dir / "run.log", remaining
                )
                record["returncode"] = str(returncode)
                artifacts = _collect_artifacts(output_dir)
                record.update(artifacts)
                if returncode != 0:
                    record["status"] = "failed"
                    reason = (
                        "global time limit interrupted the running command"
                        if timed_out
                        else f"training returned {returncode}"
                    )
                    record["error_message"] = reason
                elif _has_valid_metrics(artifacts):
                    record["status"] = "completed"
                else:
                    record["status"] = "no_metrics"
                    record["error_message"] = (
                        "training returned 0 but no valid validation metric was found"
                    )
            except KeyboardInterrupt:
                record["status"] = "interrupted"
                record["error_message"] = "user interrupted the sweep"
                raise
            except Exception as exc:
                record["status"] = "failed"
                record["returncode"] = "-1"
                record["error_message"] = f"{type(exc).__name__}: {exc}"
            finally:
                record["finished_at"] = _timestamp()
                record["elapsed_minutes"] = f"{(time.monotonic() - started) / 60.0:.3f}"
                records[experiment.name] = record
                _write_summaries(summary_csv, summary_md, records)
    except KeyboardInterrupt:
        print("Sweep interrupted by user.", file=sys.stderr)
        _print_summary(records)
        return 130

    _print_summary(records)
    print(f"summary.csv: {summary_csv}")
    print(f"summary.md: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
