"""Run multi-seed midstage replications for established ETFlow ablations."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class GroupSpec:
    name: str
    experiment_prefix: str
    trainer_script: str
    config_kind: str
    extra_args: Tuple[str, ...]


GROUP_SPECS: Tuple[GroupSpec, ...] = (
    GroupSpec(
        name="angular_baseline",
        experiment_prefix="angular_base",
        trainer_script="train_angular.py",
        config_kind="angular",
        extra_args=(
            "--angular_mu_schedule",
            "constant",
            "--angular_mu",
            "0.0",
            "--use_angular_loss",
            "--angular_loss_weight",
            "0.0",
        ),
    ),
    GroupSpec(
        name="angular_sigmoid_best",
        experiment_prefix="angular_sig_mu05_lam001",
        trainer_script="train_angular.py",
        config_kind="angular",
        extra_args=(
            "--angular_mu_schedule",
            "sigmoid",
            "--angular_mu_max",
            "0.5",
            "--angular_mu_sigmoid_k",
            "10.0",
            "--angular_mu_sigmoid_t0",
            "0.5",
            "--use_angular_loss",
            "--angular_loss_weight",
            "0.01",
        ),
    ),
    GroupSpec(
        name="bond_local_baseline",
        experiment_prefix="bond_all_lam0",
        trainer_script="train_bond_local.py",
        config_kind="bond_local",
        extra_args=("--bond_velocity_loss_weight", "0.0"),
    ),
    GroupSpec(
        name="bond_local_best",
        experiment_prefix="bond_all_lam001",
        trainer_script="train_bond_local.py",
        config_kind="bond_local",
        extra_args=("--bond_velocity_loss_weight", "0.001"),
    ),
)

METRIC_COLUMNS: Tuple[str, ...] = (
    "val/flow_matching_loss",
    "val/loss",
    "val/angular/dot_tau_loss",
    "val/angular/mean_abs_dot_tau_pred",
    "val/angular/mean_abs_dot_tau_target",
    "val/angular/scaled_angular_to_res_ratio",
    "val/bond_local/loss",
    "val/bond_local/q_pred_abs_mean",
    "val/bond_local/q_target_abs_mean",
    "val/bond_local/parallel_loss",
    "val/bond_local/perp_loss",
)

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "experiment_name",
    "group",
    "seed",
    "status",
    "returncode",
    "output_dir",
    "max_steps",
    "batch_size",
    *METRIC_COLUMNS,
    "best_checkpoint",
    "last_checkpoint",
    "metrics_path",
    "run_log",
    "started_at",
    "finished_at",
    "elapsed_minutes",
    "error_message",
)

SUCCESS_STATUSES = {"completed", "skipped_existing"}
GROUP_ORDER = {spec.name: index for index, spec in enumerate(GROUP_SPECS)}
COMPARISONS = (
    ("angular_sigmoid_best", "angular_baseline"),
    ("bond_local_best", "bond_local_baseline"),
)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _filename_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _experiment_name(spec: GroupSpec, seed: int) -> str:
    return f"{spec.experiment_prefix}_seed{seed}"


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
            for row in csv.DictReader(handle):
                for name in METRIC_COLUMNS:
                    raw_value = (row.get(name) or "").strip()
                    if not raw_value:
                        continue
                    try:
                        numeric_value = float(raw_value)
                    except ValueError:
                        continue
                    if math.isfinite(numeric_value):
                        values[name] = raw_value
    except (OSError, csv.Error) as exc:
        print(f"Warning: could not read metrics from {metrics_path}: {exc}")
    return values


def _checkpoint_paths(output_dir: Path, run_log: Optional[Path]) -> Tuple[str, str]:
    checkpoint_dir = output_dir / "checkpoints"
    last_checkpoint = checkpoint_dir / "last.ckpt"
    last_value = str(last_checkpoint) if last_checkpoint.is_file() else ""

    if run_log is not None and run_log.is_file():
        marker = "best checkpoint path:"
        try:
            lines = run_log.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        for line in reversed(lines):
            if marker not in line:
                continue
            candidate = Path(line.split(marker, 1)[1].strip())
            if candidate.is_file():
                return str(candidate), last_value

    if not checkpoint_dir.is_dir():
        return "", last_value
    candidates = [
        path
        for path in checkpoint_dir.rglob("*.ckpt")
        if path.is_file() and path.name != "last.ckpt"
    ]
    best_value = str(max(candidates, key=lambda path: path.stat().st_mtime)) if candidates else ""
    return best_value, last_value


def _collect_artifacts(output_dir: Path, run_log: Optional[Path] = None) -> Dict[str, str]:
    metrics_path = _find_latest_metrics(output_dir)
    best_checkpoint, last_checkpoint = _checkpoint_paths(output_dir, run_log)
    artifacts = {
        "metrics_path": str(metrics_path) if metrics_path is not None else "",
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
    }
    artifacts.update(_last_non_nan_metrics(metrics_path))
    return artifacts


def _has_valid_flow_metric(artifacts: Dict[str, str]) -> bool:
    try:
        return math.isfinite(float(artifacts.get("val/flow_matching_loss", "")))
    except (TypeError, ValueError):
        return False


def _is_completed_existing(
    output_dir: Path,
    previous_record: Optional[Dict[str, str]],
) -> bool:
    if previous_record is not None and previous_record.get("status") not in SUCCESS_STATUSES:
        return False
    artifacts = _collect_artifacts(output_dir)
    return bool(_has_valid_flow_metric(artifacts) and artifacts["last_checkpoint"])


def _load_existing_summary(summary_path: Path) -> Dict[str, Dict[str, str]]:
    if not summary_path.is_file():
        return {}
    records: Dict[str, Dict[str, str]] = {}
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                name = (row.get("experiment_name") or "").strip()
                if name:
                    records[name] = row
    except (OSError, csv.Error) as exc:
        print(f"Warning: could not read {summary_path}: {exc}")
    return records


def _record_sort_key(record: Dict[str, str]) -> Tuple[int, int, str]:
    try:
        seed = int(record.get("seed", ""))
    except ValueError:
        seed = sys.maxsize
    return (
        GROUP_ORDER.get(record.get("group", ""), len(GROUP_ORDER)),
        seed,
        record.get("experiment_name", ""),
    )


def _ordered_records(records: Dict[str, Dict[str, str]]) -> Iterable[Dict[str, str]]:
    yield from sorted(records.values(), key=_record_sort_key)


def _write_summary_csv(
    summary_path: Path,
    records: Dict[str, Dict[str, str]],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in _ordered_records(records):
            writer.writerow({name: record.get(name, "") for name in SUMMARY_COLUMNS})
    temporary_path.replace(summary_path)


def _finite_group_values(
    records: Dict[str, Dict[str, str]],
    group: str,
    metric: str,
) -> List[float]:
    values = []
    for record in records.values():
        if record.get("group") != group or record.get("status") not in SUCCESS_STATUSES:
            continue
        try:
            value = float(record.get(metric, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _mean_std(values: Sequence[float]) -> Tuple[float, float, int]:
    if not values:
        return math.nan, math.nan, 0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else math.nan
    return mean, std, len(values)


def _format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "—"


def _format_mean_std(mean: float, std: float, count: int) -> str:
    if count == 0:
        return "—"
    std_text = _format_float(std) if count > 1 else "n/a"
    return f"{_format_float(mean)} ± {std_text} (n={count})"


def _markdown_cell(value: object) -> str:
    return str(value if value not in (None, "") else "—").replace("|", "\\|")


def _write_summary_md(
    summary_path: Path,
    records: Dict[str, Dict[str, str]],
) -> None:
    lines = [
        "# ETFlow Midstage Multi-Seed Replications",
        "",
        f"Generated: {_timestamp()}",
        "",
    ]

    for spec in GROUP_SPECS:
        lines.extend(
            [
                f"## {spec.name}",
                "",
                "| Seed | Status | val/flow_matching_loss | val/loss | "
                "Auxiliary loss | Output directory |",
                "|---:|---|---:|---:|---:|---|",
            ]
        )
        group_records = [
            record for record in _ordered_records(records) if record.get("group") == spec.name
        ]
        auxiliary_metric = (
            "val/angular/dot_tau_loss"
            if spec.config_kind == "angular"
            else "val/bond_local/loss"
        )
        for record in group_records:
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        record.get("seed"),
                        record.get("status"),
                        record.get("val/flow_matching_loss"),
                        record.get("val/loss"),
                        record.get(auxiliary_metric),
                        record.get("output_dir"),
                    )
                )
                + " |"
            )
        if not group_records:
            lines.append("| — | — | — | — | — | — |")
        lines.append("")

    lines.extend(
        [
            "## Group-level mean ± std",
            "",
            "| Group | Metric | Mean ± std |",
            "|---|---|---:|",
        ]
    )
    for spec in GROUP_SPECS:
        for metric in METRIC_COLUMNS:
            mean, std, count = _mean_std(
                _finite_group_values(records, spec.name, metric)
            )
            if count:
                lines.append(
                    f"| {spec.name} | {metric} | {_format_mean_std(mean, std, count)} |"
                )

    lines.extend(
        [
            "",
            "## Method vs baseline",
            "",
            "Delta is method mean minus baseline mean. Relative reduction is "
            "(baseline − method) / baseline.",
            "",
            "| Comparison | Metric | Baseline mean | Method mean | Delta | "
            "Relative reduction |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for method_group, baseline_group in COMPARISONS:
        for metric in ("val/flow_matching_loss", "val/loss"):
            method_mean, _, method_count = _mean_std(
                _finite_group_values(records, method_group, metric)
            )
            baseline_mean, _, baseline_count = _mean_std(
                _finite_group_values(records, baseline_group, metric)
            )
            if method_count and baseline_count:
                delta = method_mean - baseline_mean
                relative = (
                    (baseline_mean - method_mean) / abs(baseline_mean) * 100.0
                    if baseline_mean != 0
                    else math.nan
                )
            else:
                delta = math.nan
                relative = math.nan
            lines.append(
                f"| {method_group} vs {baseline_group} | {metric} | "
                f"{_format_float(baseline_mean)} | {_format_float(method_mean)} | "
                f"{_format_float(delta)} | "
                f"{_format_float(relative, digits=2)}% |"
            )
    lines.append("")

    temporary_path = summary_path.with_suffix(".md.tmp")
    temporary_path.write_text("\n".join(lines), encoding="utf-8")
    temporary_path.replace(summary_path)


def _save_summaries(
    base_output_dir: Path,
    records: Dict[str, Dict[str, str]],
) -> None:
    _write_summary_csv(base_output_dir / "summary.csv", records)
    _write_summary_md(base_output_dir / "summary.md", records)


def _base_record(
    spec: GroupSpec,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
    run_log: Path,
    started_at: str,
) -> Dict[str, str]:
    record = {name: "" for name in SUMMARY_COLUMNS}
    record.update(
        {
            "experiment_name": _experiment_name(spec, seed),
            "group": spec.name,
            "seed": str(seed),
            "output_dir": str(output_dir),
            "max_steps": str(args.max_steps),
            "batch_size": str(args.batch_size),
            "run_log": str(run_log),
            "started_at": started_at,
        }
    )
    return record


def _build_command(
    spec: GroupSpec,
    seed: int,
    args: argparse.Namespace,
    project_root: Path,
    config_paths: Dict[str, Path],
    output_dir: Path,
) -> List[str]:
    command = [
        sys.executable,
        str(project_root / "scripts" / spec.trainer_script),
        "--config",
        str(config_paths[spec.config_kind]),
        "--output_dir",
        str(output_dir),
        "--max_steps",
        str(args.max_steps),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(seed),
        "--val_check_interval",
        str(args.val_check_interval),
        "--limit_val_batches",
        str(args.limit_val_batches),
        "--log_every_n_steps",
        str(args.log_every_n_steps),
    ]
    command.extend(spec.extra_args)
    return command


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _next_run_log(output_dir: Path) -> Path:
    default_path = output_dir / "run.log"
    if not default_path.exists():
        return default_path
    stem = f"run.rerun_{_filename_timestamp()}"
    candidate = output_dir / f"{stem}.log"
    counter = 1
    while candidate.exists():
        candidate = output_dir / f"{stem}_{counter}.log"
        counter += 1
    return candidate


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


def _append_error(errors_path: Path, experiment_name: str, message: str) -> None:
    with errors_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_timestamp()}] {experiment_name}: {message}\n")


def _resolve_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    project_candidate = (project_root / path).resolve()
    return cwd_candidate if cwd_candidate.is_file() else project_candidate


def _positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be positive, got {value}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed replications for angular sigmoid and bond-local best."
    )
    parser.add_argument(
        "--base_output_dir",
        default="logs_replicates/midstage_5000steps",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_check_interval", type=int, default=500)
    parser.add_argument("--limit_val_batches", type=int, default=10)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--time_limit_hours", type=float, default=12.0)
    parser.add_argument(
        "--angular_config",
        default="configs/drugs-so3-angular-loss-bs8-sigmoid.yaml",
    )
    parser.add_argument(
        "--bond_local_config",
        default="configs/drugs-so3-bond-local-bs8.yaml",
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "max_steps",
        "batch_size",
        "val_check_interval",
        "limit_val_batches",
        "log_every_n_steps",
    ):
        _positive_int(getattr(args, name), name)
    if not args.seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
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
    config_paths = {
        "angular": _resolve_path(args.angular_config, project_root),
        "bond_local": _resolve_path(args.bond_local_config, project_root),
    }
    for config_path in config_paths.values():
        if not config_path.is_file():
            print(f"error: config does not exist: {config_path}", file=sys.stderr)
            return 2

    base_output_dir = Path(args.base_output_dir).expanduser()
    if not base_output_dir.is_absolute():
        base_output_dir = (Path.cwd() / base_output_dir).resolve()
    else:
        base_output_dir = base_output_dir.resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = base_output_dir / "summary.csv"
    errors_path = base_output_dir / "errors.log"
    records = _load_existing_summary(summary_path)
    _save_summaries(base_output_dir, records)

    sweep_start = time.monotonic()
    sweep_deadline = sweep_start + args.time_limit_hours * 3600.0
    print(f"replications started_at: {_timestamp()}")
    print(f"seeds: {args.seeds}")
    print(f"global time limit hours: {args.time_limit_hours}")
    print(f"base output dir: {base_output_dir}")

    interrupted = False
    for spec in GROUP_SPECS:
        for seed in args.seeds:
            experiment_name = _experiment_name(spec, seed)
            output_dir = base_output_dir / experiment_name
            previous_record = records.get(experiment_name)

            print("\n" + "=" * 80)
            print(f"experiment: {experiment_name}")
            print(f"group: {spec.name}")
            print(f"seed: {seed}")
            print(f"output_dir: {output_dir}")

            if not args.rerun and _is_completed_existing(output_dir, previous_record):
                run_log_value = previous_record.get("run_log", "") if previous_record else ""
                run_log = Path(run_log_value) if run_log_value else None
                record = _base_record(
                    spec,
                    seed,
                    args,
                    output_dir,
                    run_log or output_dir / "run.log",
                    started_at=previous_record.get("started_at", "") if previous_record else "",
                )
                record.update(_collect_artifacts(output_dir, run_log))
                record["status"] = "skipped_existing"
                record["returncode"] = "0"
                record["finished_at"] = _timestamp()
                record["elapsed_minutes"] = "0.000"
                records[experiment_name] = record
                _save_summaries(base_output_dir, records)
                print("status: skipped_existing")
                continue

            if time.monotonic() >= sweep_deadline:
                run_log = output_dir / "run.log"
                record = _base_record(
                    spec, seed, args, output_dir, run_log, started_at=""
                )
                record["status"] = "not_started_time_limit"
                record["finished_at"] = _timestamp()
                record["elapsed_minutes"] = "0.000"
                record["error_message"] = (
                    "Global time limit was reached before this experiment started."
                )
                records[experiment_name] = record
                _save_summaries(base_output_dir, records)
                print("status: not_started_time_limit")
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            run_log = _next_run_log(output_dir)
            started_at = _timestamp()
            experiment_start = time.monotonic()
            record = _base_record(
                spec, seed, args, output_dir, run_log, started_at=started_at
            )
            command = _build_command(
                spec,
                seed,
                args,
                project_root,
                config_paths,
                output_dir,
            )
            timed_out = False
            try:
                remaining_seconds = max(0.0, sweep_deadline - time.monotonic())
                returncode, timed_out = _run_with_tee(
                    command,
                    project_root,
                    run_log,
                    remaining_seconds,
                )
                record["returncode"] = str(returncode)
            except KeyboardInterrupt:
                record["status"] = "interrupted"
                record["returncode"] = "130"
                record["error_message"] = "Replication sweep interrupted by user."
                interrupted = True
            except Exception as exc:
                record["status"] = "failed"
                record["returncode"] = "-1"
                record["error_message"] = f"{type(exc).__name__}: {exc}"

            artifacts = _collect_artifacts(output_dir, run_log)
            record.update(artifacts)
            if not record["status"]:
                if timed_out:
                    record["status"] = "failed"
                    record["error_message"] = (
                        "Global time limit was reached after training started."
                    )
                elif record["returncode"] != "0":
                    record["status"] = "failed"
                    record["error_message"] = (
                        f"Training exited with return code {record['returncode']}."
                    )
                elif _has_valid_flow_metric(artifacts):
                    record["status"] = "completed"
                else:
                    record["status"] = "failed"
                    record["error_message"] = (
                        "Training returned 0 but no finite val/flow_matching_loss was found."
                    )

            record["finished_at"] = _timestamp()
            record["elapsed_minutes"] = (
                f"{(time.monotonic() - experiment_start) / 60.0:.3f}"
            )
            records[experiment_name] = record
            if record["status"] not in SUCCESS_STATUSES:
                _append_error(errors_path, experiment_name, record["error_message"])
            _save_summaries(base_output_dir, records)

            print(f"status: {record['status']}")
            print(f"returncode: {record['returncode']}")
            print(f"summary.csv: {base_output_dir / 'summary.csv'}")
            print(f"summary.md: {base_output_dir / 'summary.md'}")
            if interrupted:
                break
        if interrupted:
            break

    _save_summaries(base_output_dir, records)
    print(f"\nsummary.csv: {base_output_dir / 'summary.csv'}")
    print(f"summary.md: {base_output_dir / 'summary.md'}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
