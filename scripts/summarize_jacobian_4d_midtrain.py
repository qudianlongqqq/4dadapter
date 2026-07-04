"""Summarize selected Jacobian 4D training experiments."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


EXPERIMENTS = ("base", "scale001_q0001", "scale003_q0003")
METRICS = (
    "train/flow_matching_loss",
    "val/flow_matching_loss",
    "val/loss",
    "val/jacobian_4d/q_loss",
    "val/jacobian_4d/corr_to_residual_ratio",
    "val/jacobian_4d/num_valid_bonds",
    "val/jacobian_4d/skip_rate",
)
FIELDS = (
    "experiment_name",
    "status",
    *METRICS,
    "improvement_vs_base",
    "best_checkpoint",
    "last_checkpoint",
    "metrics_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_output_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--experiments",
        default=",".join(EXPERIMENTS),
        help="comma-separated experiment directory names; base must be first",
    )
    parser.add_argument("--title", default="Jacobian 4D seed42 midtrain summary")
    return parser.parse_args()


def _finite(raw: str) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metrics_path(output_dir: Path) -> Optional[Path]:
    candidates = [path for path in output_dir.rglob("metrics.csv") if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _read_metrics(path: Optional[Path]) -> Dict[str, str]:
    values = {name: "" for name in METRICS}
    if path is None:
        return values
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle))

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


def _checkpoints(output_dir: Path) -> Tuple[str, str]:
    checkpoint_dir = output_dir / "checkpoints"
    last_path = checkpoint_dir / "last.ckpt"
    last_value = str(last_path) if last_path.is_file() else ""
    run_log = output_dir / "run.log"
    if run_log.is_file():
        marker = "best checkpoint path:"
        for line in reversed(
            run_log.read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            if marker in line:
                candidate = Path(line.split(marker, 1)[1].strip())
                if candidate.is_file():
                    return str(candidate), last_value
    if not checkpoint_dir.is_dir():
        return "", last_value
    candidates = [
        path
        for path in checkpoint_dir.glob("*.ckpt")
        if path.name != "last.ckpt"
    ]
    best = max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    return (str(best) if best else ""), last_value


def _display(value: str) -> str:
    numeric = _finite(value)
    return f"{numeric:.6g}" if numeric is not None else (value or "-")


def main() -> int:
    args = parse_args()
    experiments = tuple(
        name.strip() for name in args.experiments.split(",") if name.strip()
    )
    if not experiments or experiments[0] != "base":
        raise ValueError("--experiments must be non-empty and start with base")
    base_output = Path(args.base_output_dir).expanduser().resolve()
    if not base_output.is_dir():
        raise FileNotFoundError(f"Midtrain output does not exist: {base_output}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else base_output
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in experiments:
        experiment_dir = base_output / name
        metrics_path = _metrics_path(experiment_dir)
        values = _read_metrics(metrics_path)
        best_checkpoint, last_checkpoint = _checkpoints(experiment_dir)
        status = (
            "completed"
            if _finite(values["val/flow_matching_loss"]) is not None
            and bool(last_checkpoint)
            else "incomplete"
        )
        row = {
            "experiment_name": name,
            "status": status,
            **values,
            "improvement_vs_base": "",
            "best_checkpoint": best_checkpoint,
            "last_checkpoint": last_checkpoint,
            "metrics_path": str(metrics_path) if metrics_path else "",
        }
        rows.append(row)

    base_value = _finite(rows[0]["val/flow_matching_loss"])
    if base_value is not None:
        for row in rows:
            value = _finite(row["val/flow_matching_loss"])
            if value is not None:
                row["improvement_vs_base"] = str(base_value - value)

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
                "| "
                + " | ".join(_display(str(row[field])) for field in FIELDS)
                + " |\n"
            )
        handle.write(
            "\nValidation fields are taken from the row with the minimum "
            "val/flow_matching_loss; train loss is the latest finite value.\n"
        )

    for row in rows:
        print(
            row["experiment_name"],
            row["status"],
            f"val/flow_matching_loss={_display(row['val/flow_matching_loss'])}",
            f"improvement_vs_base={_display(row['improvement_vs_base'])}",
        )
    print(f"summary.csv: {csv_path}")
    print(f"summary.md: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
