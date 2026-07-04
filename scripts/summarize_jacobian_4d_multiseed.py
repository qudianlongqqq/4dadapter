"""Summarize formal multi-seed Jacobian 4D training experiments."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METRICS = (
    "train/flow_matching_loss",
    "val/flow_matching_loss",
    "val/loss",
    "val/jacobian_4d/q_loss",
    "val/jacobian_4d/corr_to_residual_ratio",
)
FIELDS = (
    "seed",
    "experiment_name",
    "status",
    *METRICS,
    "best_checkpoint",
    "last_checkpoint",
    "improvement_vs_same_seed_base",
    "relative_improvement_vs_same_seed_base",
    "run_log",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_output_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seeds", default="43,44")
    parser.add_argument("--experiments", default="base,scale001_q0001")
    parser.add_argument(
        "--title", default="Jacobian 4D formal multiseed 100k summary"
    )
    return parser.parse_args()


def _finite(raw: str) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metrics_path(experiment_dir: Path) -> Optional[Path]:
    copied_path = experiment_dir / "metrics.csv"
    if copied_path.is_file():
        return copied_path
    candidates = [
        path for path in experiment_dir.rglob("metrics.csv") if path.is_file()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _read_metrics(path: Optional[Path]) -> Dict[str, str]:
    values = {name: "" for name in METRICS}
    if path is None:
        return values

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: List[Dict[str, str]] = list(csv.DictReader(handle))

    for row in rows:
        raw = (row.get("train/flow_matching_loss") or "").strip()
        if _finite(raw) is not None:
            values["train/flow_matching_loss"] = raw

    best_row = None
    best_value = math.inf
    for row in rows:
        raw = (row.get("val/flow_matching_loss") or "").strip()
        value = _finite(raw)
        if value is not None and value < best_value:
            best_value = value
            best_row = row
    if best_row is not None:
        for name in METRICS[1:]:
            raw = (best_row.get(name) or "").strip()
            if _finite(raw) is not None:
                values[name] = raw
    return values


def _checkpoints(experiment_dir: Path) -> Tuple[str, str]:
    checkpoint_dir = experiment_dir / "checkpoints"
    last_path = checkpoint_dir / "last.ckpt"
    last_checkpoint = str(last_path) if last_path.is_file() else ""

    run_log = experiment_dir / "run.log"
    if run_log.is_file():
        marker = "best checkpoint path:"
        lines = run_log.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            if marker in line:
                candidate = Path(line.split(marker, 1)[1].strip())
                if candidate.is_file():
                    return str(candidate), last_checkpoint
    return "", last_checkpoint


def _status(experiment_dir: Path) -> str:
    status_path = experiment_dir / ".run_status"
    if not status_path.is_file():
        return "unknown"
    return status_path.read_text(encoding="utf-8").strip() or "unknown"


def _display(value: object) -> str:
    text = str(value)
    numeric = _finite(text)
    return f"{numeric:.6g}" if numeric is not None else (text or "-")


def main() -> int:
    args = parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    experiments = tuple(
        name.strip() for name in args.experiments.split(",") if name.strip()
    )
    if not seeds:
        raise ValueError("--seeds must be non-empty")
    if not experiments or experiments[0] != "base":
        raise ValueError("--experiments must be non-empty and start with base")

    base_output = Path(args.base_output_dir).expanduser().resolve()
    if not base_output.is_dir():
        raise FileNotFoundError(f"Training output does not exist: {base_output}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else base_output
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        seed_rows = []
        for experiment_name in experiments:
            experiment_dir = base_output / f"seed{seed}" / experiment_name
            values = _read_metrics(_metrics_path(experiment_dir))
            best_checkpoint, last_checkpoint = _checkpoints(experiment_dir)
            run_log = experiment_dir / "run.log"
            row = {
                "seed": seed,
                "experiment_name": experiment_name,
                "status": _status(experiment_dir),
                **values,
                "best_checkpoint": best_checkpoint,
                "last_checkpoint": last_checkpoint,
                "improvement_vs_same_seed_base": "",
                "relative_improvement_vs_same_seed_base": "",
                "run_log": str(run_log) if run_log.is_file() else "",
            }
            seed_rows.append(row)

        seed_rows[0]["improvement_vs_same_seed_base"] = "0"
        seed_rows[0]["relative_improvement_vs_same_seed_base"] = "0"
        base_value = _finite(seed_rows[0]["val/flow_matching_loss"])
        if base_value is not None:
            for row in seed_rows[1:]:
                value = _finite(row["val/flow_matching_loss"])
                if value is not None:
                    improvement = base_value - value
                    row["improvement_vs_same_seed_base"] = f"{improvement:.12g}"
                    if base_value != 0:
                        row["relative_improvement_vs_same_seed_base"] = (
                            f"{improvement / base_value:.12g}"
                        )
        rows.extend(seed_rows)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {args.title}\n\n")
        handle.write("| " + " | ".join(FIELDS) + " |\n")
        handle.write("| " + " | ".join("---" for _ in FIELDS) + " |\n")
        for row in rows:
            handle.write(
                "| " + " | ".join(_display(row[field]) for field in FIELDS) + " |\n"
            )
        handle.write(
            "\nValidation fields use the row with the minimum "
            "val/flow_matching_loss; train loss is the latest finite value. "
            "Improvement is same-seed base minus experiment validation loss; "
            "relative improvement divides that value by the same-seed base.\n"
        )

    for row in rows:
        print(
            f"seed{row['seed']}/{row['experiment_name']}",
            row["status"],
            f"val/flow_matching_loss={_display(row['val/flow_matching_loss'])}",
            "improvement_vs_same_seed_base="
            f"{_display(row['improvement_vs_same_seed_base'])}",
            "relative_improvement_vs_same_seed_base="
            f"{_display(row['relative_improvement_vs_same_seed_base'])}",
        )
    print(f"summary.csv: {csv_path}")
    print(f"summary.md: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
